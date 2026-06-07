import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class OnionAllocation:
    short_n: int
    mid_n: int
    long_n: int
    short_qubits: List[int] = field(default_factory=list)
    mid_qubits: List[int] = field(default_factory=list)
    long_qubits: List[int] = field(default_factory=list)

    @property
    def total(self):
        return self.short_n + self.mid_n + self.long_n


def allocate_onion(n_qubits: int) -> OnionAllocation:
    short_n = max(1, n_qubits // 4)
    mid_n = max(1, n_qubits // 3)
    long_n = max(1, n_qubits - short_n - mid_n)
    total = short_n + mid_n + long_n
    if total > n_qubits:
        long_n = n_qubits - short_n - mid_n
    idx = 0
    sq = list(range(idx, idx + short_n)); idx += short_n
    mq = list(range(idx, idx + mid_n)); idx += mid_n
    lq = list(range(idx, idx + long_n))
    return OnionAllocation(short_n, mid_n, long_n, sq, mq, lq)


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


class OnionQRC:
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
    ):
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
        self._memory_state: Optional[np.ndarray] = None

    def reset_memory(self):
        self._memory_state = None

    def _build_ising_layer(self, qc: QuantumCircuit, dt: float):
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                angle = 2 * self.ising.J[i, j] * dt
                if abs(angle) > 1e-10:
                    qc.rzz(angle, i, j)
        for i in range(self.n_qubits):
            angle = 2 * self.ising.h * dt
            if abs(angle) > 1e-10:
                qc.rx(angle, i)

    def _build_encoding(self, qc: QuantumCircuit, features: np.ndarray):
        feature_idx = 0
        for band_name, qubits in [("short", self.alloc.short_qubits),
                                   ("mid", self.alloc.mid_qubits),
                                   ("long", self.alloc.long_qubits)]:
            for q in qubits:
                if feature_idx < len(features):
                    angle = self.alpha[band_name] * np.arcsin(np.clip(features[feature_idx], -1, 1))
                    qc.ry(angle, q)
                    feature_idx += 1

    def _build_memory_feedback(self, qc: QuantumCircuit):
        if self._memory_state is None:
            return
        long_qs = self.alloc.long_qubits
        for i, q in enumerate(long_qs):
            if i < len(self._memory_state):
                angle = self.feedback_strength * self._memory_state[i]
                qc.ry(angle, q)

    def build_circuit(self, features: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        self._build_encoding(qc, features)
        self._build_memory_feedback(qc)
        dt = self.ising.dt / self.ising.trotter_steps
        for _ in range(self.ising.trotter_steps):
            self._build_ising_layer(qc, dt)
        return qc

    def _compute_observables(self, state: Statevector) -> np.ndarray:
        probs = np.abs(state.data) ** 2
        n = self.n_qubits
        bits = np.arange(2 ** n)
        obs = []
        for i in range(n):
            bit_i = (bits >> i) & 1
            z_i = np.sum(probs * (1 - 2 * bit_i))
            obs.append(z_i)
        for i in range(n):
            for j in range(i + 1, n):
                bit_i = (bits >> i) & 1
                bit_j = (bits >> j) & 1
                zz_ij = np.sum(probs * (1 - 2 * (bit_i ^ bit_j)))
                obs.append(zz_ij)
        return np.array(obs)

    def step(self, features: np.ndarray) -> np.ndarray:
        qc = self.build_circuit(features)
        state = Statevector.from_instruction(qc)
        observables = self._compute_observables(state)
        long_qs = self.alloc.long_qubits
        memory_readout = np.zeros(len(long_qs))
        for i, q in enumerate(long_qs):
            memory_readout[i] = observables[q]
        self._memory_state = memory_readout
        return observables

    def run_sequence(self, feature_sequence: np.ndarray, warmup: int = 50) -> np.ndarray:
        n_steps = len(feature_sequence)
        n_obs = self.n_qubits + self.n_qubits * (self.n_qubits - 1) // 2
        results = np.zeros((n_steps, n_obs))
        self.reset_memory()
        for t in range(n_steps):
            results[t] = self.step(feature_sequence[t])
        return results[warmup:]

    @property
    def n_observables(self) -> int:
        return self.n_qubits + self.n_qubits * (self.n_qubits - 1) // 2

    def band_info(self) -> str:
        a = self.alloc
        return (
            f"N={self.n_qubits}: "
            f"short={a.short_n} (q{a.short_qubits}), "
            f"mid={a.mid_n} (q{a.mid_qubits}), "
            f"long={a.long_n} (q{a.long_qubits}), "
            f"obs={self.n_observables}"
        )


class SingleBandQRC:
    def __init__(self, n_qubits: int, seed: int = 42, h_field: float = 0.3,
                 trotter_steps: int = 4, dt: float = 0.5, alpha: float = 0.6):
        self.n_qubits = n_qubits
        self.ising = random_ising_params(n_qubits, h_field, seed)
        self.ising.trotter_steps = trotter_steps
        self.ising.dt = dt
        self.alpha = alpha
        self._memory_state: Optional[np.ndarray] = None

    def reset_memory(self):
        self._memory_state = None

    def build_circuit(self, features: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(min(len(features), self.n_qubits)):
            angle = self.alpha * np.arcsin(np.clip(features[i], -1, 1))
            qc.ry(angle, i)
        if self._memory_state is not None:
            for i in range(min(len(self._memory_state), self.n_qubits)):
                angle = 0.2 * self._memory_state[i]
                qc.ry(angle, i)
        dt = self.ising.dt / self.ising.trotter_steps
        for _ in range(self.ising.trotter_steps):
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    angle = 2 * self.ising.J[i, j] * dt
                    if abs(angle) > 1e-10:
                        qc.rzz(angle, i, j)
            for i in range(self.n_qubits):
                angle = 2 * self.ising.h * dt
                if abs(angle) > 1e-10:
                    qc.rx(angle, i)
        return qc

    def _compute_observables(self, state: Statevector) -> np.ndarray:
        probs = np.abs(state.data) ** 2
        n = self.n_qubits
        bits = np.arange(2 ** n)
        obs = []
        for i in range(n):
            bit_i = (bits >> i) & 1
            z_i = np.sum(probs * (1 - 2 * bit_i))
            obs.append(z_i)
        for i in range(n):
            for j in range(i + 1, n):
                bit_i = (bits >> i) & 1
                bit_j = (bits >> j) & 1
                zz_ij = np.sum(probs * (1 - 2 * (bit_i ^ bit_j)))
                obs.append(zz_ij)
        return np.array(obs)

    def step(self, features: np.ndarray) -> np.ndarray:
        qc = self.build_circuit(features)
        state = Statevector.from_instruction(qc)
        obs = self._compute_observables(state)
        self._memory_state = obs[:self.n_qubits]
        return obs

    def run_sequence(self, feature_sequence: np.ndarray, warmup: int = 50) -> np.ndarray:
        n_steps = len(feature_sequence)
        n_obs = self.n_qubits + self.n_qubits * (self.n_qubits - 1) // 2
        results = np.zeros((n_steps, n_obs))
        self.reset_memory()
        for t in range(n_steps):
            results[t] = self.step(feature_sequence[t])
        return results[warmup:]

    @property
    def n_observables(self) -> int:
        return self.n_qubits + self.n_qubits * (self.n_qubits - 1) // 2


if __name__ == "__main__":
    for n in [5, 10]:
        qrc = OnionQRC(n)
        print(qrc.band_info())
        dummy = np.random.randn(5)
        obs = qrc.step(dummy)
        print(f"  Observables shape: {obs.shape}, range: [{obs.min():.4f}, {obs.max():.4f}]")
