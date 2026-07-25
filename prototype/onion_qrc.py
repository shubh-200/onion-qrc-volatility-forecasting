"""Multi-timescale quantum reservoir circuits.

The onion encoding assigns one bounded HAR feature to each band and repeats it
on every qubit in that band: daily volatility drives ``short``, weekly drives
``mid``, and monthly drives ``long``.  Canonical five-column prototype inputs
(``log_rv, rv_d, rv_w, rv_m, log_return``) and compact three-column HAR inputs
are both supported.

Reservoir feedback remains classical: every call to :meth:`build_circuit`
creates a fresh circuit and applies rotations derived from the previous
long-band measurements. No quantum state is carried between time steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit as _QuantumCircuit
    from qiskit.quantum_info import Statevector as _Statevector
except ImportError:  # Pure allocation/resource helpers remain usable.
    _QuantumCircuit = None
    _Statevector = None

QuantumCircuit: Any = _QuantumCircuit
Statevector: Any = _Statevector


ObservableTerm = Tuple[int, ...]


@dataclass
class OnionAllocation:
    short_n: int
    mid_n: int
    long_n: int
    short_qubits: List[int] = field(default_factory=list)
    mid_qubits: List[int] = field(default_factory=list)
    long_qubits: List[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.short_n + self.mid_n + self.long_n


def allocate_onion(n_qubits: int) -> OnionAllocation:
    """Allocate all qubits to non-empty short, mid, and long bands."""
    if n_qubits < 3:
        raise ValueError("onion allocation requires at least three qubits")

    short_n = max(1, n_qubits // 4)
    mid_n = max(1, n_qubits // 3)
    long_n = n_qubits - short_n - mid_n
    if long_n < 1:
        mid_n -= 1 - long_n
        long_n = 1

    short_qubits = list(range(short_n))
    mid_qubits = list(range(short_n, short_n + mid_n))
    long_qubits = list(range(short_n + mid_n, n_qubits))
    return OnionAllocation(
        short_n,
        mid_n,
        long_n,
        short_qubits,
        mid_qubits,
        long_qubits,
    )


@dataclass
class IsingParams:
    J: np.ndarray
    h: float
    trotter_steps: int = 4
    dt: float = 0.5


def random_ising_params(n_qubits: int, h: float = 0.3, seed: int = 42) -> IsingParams:
    rng = np.random.default_rng(seed)
    J = rng.normal(0, 1, (n_qubits, n_qubits))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return IsingParams(J=J, h=h)


def _require_qiskit() -> None:
    if QuantumCircuit is None:
        raise ImportError("Qiskit is required to build or simulate circuits")


def _topology_edges(n_qubits: int, topology: str) -> Tuple[Tuple[int, int], ...]:
    if topology == "fully_connected":
        return tuple(combinations(range(n_qubits), 2))
    if topology == "ring":
        edges = {
            (min(q, (q + 1) % n_qubits), max(q, (q + 1) % n_qubits))
            for q in range(n_qubits)
        }
        return tuple(sorted(edges))
    raise ValueError("topology must be 'fully_connected' or 'ring'")


def _state_probabilities(state: Any) -> np.ndarray:
    data = np.asarray(state.data)
    return np.abs(data) ** 2


def _expectations_from_probabilities(
    probabilities: np.ndarray,
    n_qubits: int,
    terms: Sequence[ObservableTerm],
) -> np.ndarray:
    expected_size = 1 << n_qubits
    if probabilities.size != expected_size:
        raise ValueError(f"expected {expected_size} basis probabilities")

    basis = np.arange(expected_size, dtype=np.uint64)
    output = np.empty(len(terms), dtype=float)
    for index, term in enumerate(terms):
        parity = np.zeros(expected_size, dtype=np.uint8)
        for qubit in term:
            parity ^= ((basis >> qubit) & 1).astype(np.uint8)
        output[index] = np.dot(probabilities, 1.0 - 2.0 * parity)
    return output


class OnionQRC:
    """Onion quantum reservoir with bounded-degree and readout controls.

    Args:
        observable_order: Include all Z products through order 1 or 2.  Passing
            3 is accepted as shorthand for order 2 plus all three-body terms.
        n_third_order: Number of deterministic lexicographically selected
            three-body Z products to append.  Ignored when ``observable_order``
            is 3, which selects every three-body term.
        third_order_terms: Explicit selected three-body products.  This is
            mutually exclusive with a nonzero ``n_third_order``.
    """

    def __init__(
        self,
        n_qubits: int,
        seed: int = 42,
        h_field: float = 0.3,
        trotter_steps: int = 4,
        dt: float = 0.5,
        alpha_short: float = 1.0,
        alpha_mid: float = 0.6,
        alpha_long: float = 0.3,
        memory_feedback_strength: float = 0.2,
        topology: str = "fully_connected",
        observable_order: int = 2,
        n_third_order: int = 0,
        third_order_terms: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        if trotter_steps < 1:
            raise ValueError("trotter_steps must be positive")
        if observable_order not in (1, 2, 3):
            raise ValueError("observable_order must be 1, 2, or 3")
        if n_third_order < 0:
            raise ValueError("n_third_order cannot be negative")
        if third_order_terms is not None and n_third_order:
            raise ValueError("choose explicit third_order_terms or n_third_order, not both")

        self.n_qubits = n_qubits
        self.alloc = allocate_onion(n_qubits)
        self.ising = random_ising_params(n_qubits, h_field, seed)
        self.ising.trotter_steps = trotter_steps
        self.ising.dt = dt
        self.alpha = {
            "short": alpha_short,
            "mid": alpha_mid,
            "long": alpha_long,
        }
        self.feedback_strength = memory_feedback_strength
        self.topology = topology
        self.observable_order = observable_order
        self._edges = _topology_edges(n_qubits, topology)
        self._memory_state: Optional[np.ndarray] = None

        all_triples = tuple(combinations(range(n_qubits), 3))
        if third_order_terms is not None:
            selected_triples = self._validate_third_order_terms(third_order_terms)
        elif observable_order == 3:
            selected_triples = all_triples
        else:
            selected_triples = all_triples[:n_third_order]
            if n_third_order > len(all_triples):
                raise ValueError("n_third_order exceeds the available Z products")

        terms: List[ObservableTerm] = [(q,) for q in range(n_qubits)]
        if observable_order >= 2:
            terms.extend(combinations(range(n_qubits), 2))
        terms.extend(selected_triples)
        self._observable_terms = tuple(terms)
        self._third_order_terms = selected_triples

    def _validate_third_order_terms(
        self, terms: Sequence[Tuple[int, int, int]]
    ) -> Tuple[Tuple[int, int, int], ...]:
        normalized = []
        for term in terms:
            candidate = tuple(sorted(term))
            if len(candidate) != 3 or len(set(candidate)) != 3:
                raise ValueError("third-order terms require three distinct qubits")
            if candidate[0] < 0 or candidate[-1] >= self.n_qubits:
                raise ValueError("third-order term contains an invalid qubit")
            normalized.append(candidate)
        if len(set(normalized)) != len(normalized):
            raise ValueError("third-order terms must be unique")
        return tuple(sorted(normalized))

    @property
    def interaction_edges(self) -> Tuple[Tuple[int, int], ...]:
        return self._edges

    @property
    def observable_terms(self) -> Tuple[ObservableTerm, ...]:
        """Z-product qubit tuples in returned observable order."""
        return self._observable_terms

    @property
    def third_order_terms(self) -> Tuple[Tuple[int, int, int], ...]:
        return self._third_order_terms

    @property
    def memory_state(self) -> Optional[np.ndarray]:
        """Return a defensive copy of the classical feedback state."""
        return None if self._memory_state is None else self._memory_state.copy()

    def reset_memory(self) -> None:
        self._memory_state = None

    def update_memory_from_observables(self, observables: Sequence[float]) -> None:
        """Update long-band feedback from statevector or shot-derived observables."""
        values = np.asarray(observables, dtype=float).reshape(-1)
        if len(values) < self.n_qubits or not np.isfinite(values).all():
            raise ValueError("observables must contain finite single-qubit Z values")
        self._memory_state = values[np.asarray(self.alloc.long_qubits)].copy()

    @staticmethod
    def _har_band_features(
        features: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(features, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("features cannot be empty")
        if values.size >= 5:
            # Legacy prototype layout: log_rv, rv_d, rv_w, rv_m, log_return.
            har_values = values[1:4]
        else:
            # Phase 3 layout starts with daily, weekly, and monthly log RV;
            # an optional fourth column contains the asset return.
            har_values = np.pad(values[:3], (0, max(0, 3 - values.size)), mode="edge")
        return np.clip(har_values, -1.0, 1.0)

    def _build_ising_layer(self, qc: Any, dt: float) -> None:
        for i, j in self._edges:
            angle = 2 * self.ising.J[i, j] * dt
            if abs(angle) > 1e-10:
                qc.rzz(angle, i, j)
        field_angle = 2 * self.ising.h * dt
        if abs(field_angle) > 1e-10:
            for q in range(self.n_qubits):
                qc.rx(field_angle, q)

    def _build_encoding(
        self, qc: Any, features: Sequence[float] | np.ndarray
    ) -> None:
        bounded = self._har_band_features(features)
        bands = (
            ("short", self.alloc.short_qubits, bounded[0]),
            ("mid", self.alloc.mid_qubits, bounded[1]),
            ("long", self.alloc.long_qubits, bounded[2]),
        )
        for band_name, qubits, feature in bands:
            angle = self.alpha[band_name] * np.arcsin(feature)
            for qubit in qubits:
                qc.ry(angle, qubit)

    def _build_memory_feedback(self, qc: Any) -> None:
        if self._memory_state is None:
            return
        for qubit, value in zip(self.alloc.long_qubits, self._memory_state):
            qc.ry(self.feedback_strength * value, qubit)

    def build_circuit(
        self, features: Sequence[float] | np.ndarray, measure: bool = False
    ) -> Any:
        """Build a fresh reservoir circuit, optionally with Z-basis measurement."""
        _require_qiskit()
        qc = QuantumCircuit(self.n_qubits, self.n_qubits if measure else 0)
        self._build_encoding(qc, features)
        self._build_memory_feedback(qc)
        layer_dt = self.ising.dt / self.ising.trotter_steps
        for _ in range(self.ising.trotter_steps):
            self._build_ising_layer(qc, layer_dt)
        if measure:
            qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def observables_from_statevector(self, state: Any) -> np.ndarray:
        """Extract configured Z-product expectations from a statevector."""
        return _expectations_from_probabilities(
            _state_probabilities(state), self.n_qubits, self._observable_terms
        )

    def _compute_observables(self, state: Any) -> np.ndarray:
        """Backward-compatible statevector observable helper."""
        return self.observables_from_statevector(state)

    def observables_from_counts(self, counts: Mapping[Any, float]) -> np.ndarray:
        """Extract configured Z products from Qiskit-style measurement counts.

        Register separators are ignored and both binary and hexadecimal keys
        are accepted.  Values may be integer shots or probability weights.
        """
        if not counts:
            raise ValueError("counts cannot be empty")
        weighted = np.zeros(len(self._observable_terms), dtype=float)
        total = float(sum(counts.values()))
        if total <= 0:
            raise ValueError("counts must have positive total weight")

        for raw_key, weight in counts.items():
            if isinstance(raw_key, (int, np.integer)):
                basis_state = int(raw_key)
            else:
                key = str(raw_key).replace(" ", "").replace("_", "")
                basis_state = int(key, 16) if key.lower().startswith("0x") else int(key, 2)
            if basis_state < 0 or basis_state >= (1 << self.n_qubits):
                raise ValueError(f"count key {raw_key!r} does not fit {self.n_qubits} qubits")
            for index, term in enumerate(self._observable_terms):
                parity = 0
                for qubit in term:
                    parity ^= (basis_state >> qubit) & 1
                weighted[index] += float(weight) * (1.0 - 2.0 * parity)
        return weighted / total

    # Singular alias is convenient for callers that process one counts mapping.
    observable_from_counts = observables_from_counts

    def step(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        """Simulate one fresh circuit and update the classical long-band memory."""
        _require_qiskit()
        qc = self.build_circuit(features)
        state = Statevector.from_instruction(qc)
        observables = self.observables_from_statevector(state)
        self.update_memory_from_observables(observables)
        return observables

    def run_sequence(self, feature_sequence: np.ndarray, warmup: int = 50) -> np.ndarray:
        if warmup < 0:
            raise ValueError("warmup cannot be negative")
        results = np.zeros((len(feature_sequence), self.n_observables))
        self.reset_memory()
        for time_index, features in enumerate(feature_sequence):
            results[time_index] = self.step(features)
        return results[warmup:]

    @property
    def n_observables(self) -> int:
        return len(self._observable_terms)

    def estimate_resources(
        self,
        include_feedback: Optional[bool] = None,
        measure: bool = False,
    ) -> Dict[str, Any]:
        """Return logical gate and width estimates for one fresh circuit."""
        if include_feedback is None:
            include_feedback = self._memory_state is not None
        encoding_ry = self.n_qubits
        feedback_ry = self.alloc.long_n if include_feedback else 0
        rx = self.n_qubits * self.ising.trotter_steps
        rzz = len(self._edges) * self.ising.trotter_steps
        measurements = self.n_qubits if measure else 0
        single_qubit_gates = encoding_ry + feedback_ry + rx
        return {
            "n_qubits": self.n_qubits,
            "n_classical_bits": self.n_qubits if measure else 0,
            "topology": self.topology,
            "interaction_edges": len(self._edges),
            "trotter_steps": self.ising.trotter_steps,
            "encoding_ry": encoding_ry,
            "feedback_ry": feedback_ry,
            "rx": rx,
            "rzz": rzz,
            "measurements": measurements,
            "single_qubit_gates": single_qubit_gates,
            "two_qubit_gates": rzz,
            "logical_gates": single_qubit_gates + rzz,
            "total_operations": single_qubit_gates + rzz + measurements,
            "n_observables": self.n_observables,
        }

    # Compatibility/discoverability aliases.
    resource_estimate = estimate_resources
    logical_gate_estimate = estimate_resources

    def band_info(self) -> str:
        a = self.alloc
        return (
            f"N={self.n_qubits}: "
            f"short={a.short_n} (q{a.short_qubits}), "
            f"mid={a.mid_n} (q{a.mid_qubits}), "
            f"long={a.long_n} (q{a.long_qubits}), "
            f"topology={self.topology}, obs={self.n_observables}"
        )


class SingleBandQRC:
    """Original single-band baseline retained for prototype compatibility."""

    def __init__(self, n_qubits: int, seed: int = 42, h_field: float = 0.3,
                 trotter_steps: int = 4, dt: float = 0.5, alpha: float = 0.6):
        self.n_qubits = n_qubits
        self.ising = random_ising_params(n_qubits, h_field, seed)
        self.ising.trotter_steps = trotter_steps
        self.ising.dt = dt
        self.alpha = alpha
        self._memory_state: Optional[np.ndarray] = None
        self._observable_terms = tuple((q,) for q in range(n_qubits)) + tuple(
            combinations(range(n_qubits), 2)
        )

    def reset_memory(self) -> None:
        self._memory_state = None

    def build_circuit(self, features: Sequence[float] | np.ndarray) -> Any:
        _require_qiskit()
        qc = QuantumCircuit(self.n_qubits)
        values = np.asarray(features).reshape(-1)
        for i in range(min(len(values), self.n_qubits)):
            angle = self.alpha * np.arcsin(np.clip(values[i], -1, 1))
            qc.ry(angle, i)
        if self._memory_state is not None:
            for i in range(min(len(self._memory_state), self.n_qubits)):
                qc.ry(0.2 * self._memory_state[i], i)
        layer_dt = self.ising.dt / self.ising.trotter_steps
        edges = combinations(range(self.n_qubits), 2)
        edges = tuple(edges)
        for _ in range(self.ising.trotter_steps):
            for i, j in edges:
                angle = 2 * self.ising.J[i, j] * layer_dt
                if abs(angle) > 1e-10:
                    qc.rzz(angle, i, j)
            field_angle = 2 * self.ising.h * layer_dt
            if abs(field_angle) > 1e-10:
                for i in range(self.n_qubits):
                    qc.rx(field_angle, i)
        return qc

    def _compute_observables(self, state: Any) -> np.ndarray:
        return _expectations_from_probabilities(
            _state_probabilities(state), self.n_qubits, self._observable_terms
        )

    def step(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        _require_qiskit()
        state = Statevector.from_instruction(self.build_circuit(features))
        observables = self._compute_observables(state)
        self._memory_state = observables[:self.n_qubits]
        return observables

    def run_sequence(self, feature_sequence: np.ndarray, warmup: int = 50) -> np.ndarray:
        results = np.zeros((len(feature_sequence), self.n_observables))
        self.reset_memory()
        for time_index, features in enumerate(feature_sequence):
            results[time_index] = self.step(features)
        return results[warmup:]

    @property
    def n_observables(self) -> int:
        return len(self._observable_terms)


if __name__ == "__main__":
    for n in (5, 10, 15, 20):
        qrc = OnionQRC(n, topology="ring")
        print(qrc.band_info(), qrc.estimate_resources())
