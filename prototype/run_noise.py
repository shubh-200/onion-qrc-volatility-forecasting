"""Shot-based sequential OnionQRC noise experiments using Qiskit Aer.

The default invocation is deliberately small: N=5, one seed, 256 shots, the
noiseless finite-shot case, and at most one case. Use ``--full-matrix`` to opt
into the Phase 3 N/noise/shot matrix, optionally capped with ``--max-cases``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Support module and documented direct-script execution.
    from .data_loader import DEFAULT_FEATURES, load_spx_rv
    from .onion_qrc import OnionQRC
    from .readout import VolQRCReadout, compute_metrics
    from .run_phase3 import (
        SPLIT_NAMES,
        _date_range,
        package_versions,
        prepare_phase3_data,
        write_artifacts,
    )
except ImportError:  # pragma: no cover - direct script execution.
    from data_loader import DEFAULT_FEATURES, load_spx_rv  # type: ignore
    from onion_qrc import OnionQRC  # type: ignore
    from readout import VolQRCReadout, compute_metrics  # type: ignore
    from run_phase3 import (  # type: ignore
        SPLIT_NAMES,
        _date_range,
        package_versions,
        prepare_phase3_data,
        write_artifacts,
    )


RESULTS_DIR = Path(__file__).parent / "results" / "noise"
FULL_N_QUBITS = (5, 10, 15, 20)
FULL_SEEDS = (42, 123, 2026)
FULL_SHOTS = (256, 512, 1024, 4096)
FULL_DEPOLARIZING = (0.001, 0.005, 0.01)
# A canonical low-error sweep; device-derived gamma values can be supplied via CLI.
FULL_AMPLITUDE_DAMPING = (0.001, 0.005, 0.01)
DEFAULT_BASIS_GATES = ("id", "rx", "ry", "rzz", "cx")


@dataclass(frozen=True)
class NoiseCase:
    n_qubits: int
    seed: int
    shots: int
    topology: str
    noise_kind: str
    noise_parameter: float | None

    @property
    def case_id(self) -> str:
        value = "none" if self.noise_parameter is None else f"{self.noise_parameter:g}"
        return (
            f"n{self.n_qubits}_seed{self.seed}_shots{self.shots}_"
            f"{self.topology}_{self.noise_kind}_{value}"
        )


def _validate_probability(values: Sequence[float], name: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if any(not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError(f"{name} values must be in [0, 1]")
    return normalized


def generate_cases(
    *,
    n_qubits: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    shots: Sequence[int] | None = None,
    topology: str = "ring",
    depolarizing_p: Sequence[float] | None = None,
    amplitude_damping_gamma: Sequence[float] | None = None,
    include_noiseless: bool = True,
    full_matrix: bool = False,
    max_cases: int | None = None,
) -> list[NoiseCase]:
    """Build deterministic cases, requiring explicit opt-in for the full plan matrix."""
    if topology not in {"ring", "fully_connected"}:
        raise ValueError("topology must be 'ring' or 'fully_connected'")

    selected_n = tuple(FULL_N_QUBITS if full_matrix else (n_qubits or (5,)))
    selected_seeds = tuple(FULL_SEEDS if full_matrix else (seeds or (42,)))
    selected_shots = tuple(FULL_SHOTS if full_matrix else (shots or (256,)))
    depolarizing = _validate_probability(
        FULL_DEPOLARIZING if full_matrix else (depolarizing_p or ()),
        "depolarizing p",
    )
    damping = _validate_probability(
        FULL_AMPLITUDE_DAMPING if full_matrix else (amplitude_damping_gamma or ()),
        "amplitude damping gamma",
    )

    if any(value < 3 for value in selected_n):
        raise ValueError("OnionQRC requires at least three qubits")
    if any(value < 1 for value in selected_shots):
        raise ValueError("shots must be positive")
    if not full_matrix and (
        set(FULL_N_QUBITS).issubset(selected_n)
        and set(FULL_DEPOLARIZING).issubset(depolarizing)
        and set(FULL_AMPLITUDE_DAMPING).issubset(damping)
    ):
        raise ValueError("the complete N/noise matrix requires --full-matrix")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")

    noise_specs: list[tuple[str, float | None]] = []
    if include_noiseless:
        noise_specs.append(("noiseless_shots", None))
    noise_specs.extend(("depolarizing", value) for value in depolarizing)
    noise_specs.extend(("amplitude_damping", value) for value in damping)
    if not noise_specs:
        raise ValueError("select at least one noiseless or noisy case")

    cases = [
        NoiseCase(n, seed, shot_count, topology, kind, parameter)
        for n in selected_n
        for seed in selected_seeds
        for shot_count in selected_shots
        for kind, parameter in noise_specs
    ]
    # Outside full mode, an omitted cap intentionally means one conservative case.
    effective_max = max_cases if max_cases is not None else (None if full_matrix else 1)
    return cases if effective_max is None else cases[:effective_max]


def build_noise_model(case: NoiseCase) -> tuple[Any | None, tuple[str, ...]]:
    """Create an Aer noise model whose errors match the transpilation basis."""
    if case.noise_kind == "noiseless_shots":
        return None, DEFAULT_BASIS_GATES

    try:
        from qiskit_aer.noise import (
            NoiseModel,
            amplitude_damping_error,
            depolarizing_error,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError("Noise simulation requires the optional 'qiskit-aer' package") from exc

    if case.noise_parameter is None:
        raise ValueError(f"{case.noise_kind} requires a noise parameter")
    parameter = float(case.noise_parameter)
    model = NoiseModel()
    if case.noise_kind == "depolarizing":
        model.add_all_qubit_quantum_error(depolarizing_error(parameter, 1), ["rx", "ry"])
        model.add_all_qubit_quantum_error(depolarizing_error(parameter, 2), ["rzz", "cx"])
    elif case.noise_kind == "amplitude_damping":
        one_qubit_error = amplitude_damping_error(parameter)
        model.add_all_qubit_quantum_error(one_qubit_error, ["rx", "ry"])
        model.add_all_qubit_quantum_error(
            one_qubit_error.tensor(one_qubit_error), ["rzz", "cx"]
        )
    else:
        raise ValueError(f"unknown noise kind: {case.noise_kind}")
    return model, tuple(model.basis_gates)


def run_shot_sequence(
    qrc: Any,
    features: np.ndarray,
    *,
    backend: Any,
    transpile_fn: Any,
    shots: int,
    simulation_seed: int,
    basis_gates: Sequence[str],
    optimization_level: int = 1,
    progress: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one causal sequence where shot-derived observables drive feedback."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2:
        raise ValueError("features must be a 2D array")
    if shots < 1:
        raise ValueError("shots must be positive")

    output = np.empty((len(values), qrc.n_observables), dtype=float)
    timings = {"circuit_build": 0.0, "transpilation": 0.0, "execution": 0.0,
               "observable_extraction": 0.0}
    depths: list[int] = []
    operation_totals: dict[str, int] = {}
    qrc.reset_memory()
    started_total = time.perf_counter()
    progress_every = max(1, len(values) // 10)

    for index, row in enumerate(values):
        started = time.perf_counter()
        circuit = qrc.build_circuit(row, measure=True)
        timings["circuit_build"] += time.perf_counter() - started

        started = time.perf_counter()
        compiled = transpile_fn(
            circuit,
            basis_gates=list(basis_gates),
            optimization_level=optimization_level,
            seed_transpiler=simulation_seed,
        )
        timings["transpilation"] += time.perf_counter() - started

        started = time.perf_counter()
        job = backend.run(
            compiled,
            shots=shots,
            seed_simulator=simulation_seed + index,
        )
        counts = job.result().get_counts(compiled)
        timings["execution"] += time.perf_counter() - started

        started = time.perf_counter()
        # Every configured Z and ZZ expectation is extracted from this one mapping.
        observables = qrc.observables_from_counts(counts)
        output[index] = observables
        qrc.update_memory_from_observables(observables)
        timings["observable_extraction"] += time.perf_counter() - started

        if hasattr(compiled, "depth"):
            depths.append(int(compiled.depth()))
        if hasattr(compiled, "count_ops"):
            for gate, count in compiled.count_ops().items():
                operation_totals[str(gate)] = operation_totals.get(str(gate), 0) + int(count)
        if progress and ((index + 1) % progress_every == 0 or index + 1 == len(values)):
            print(f"    shot reservoir {index + 1}/{len(values)}")

    timings["total"] = time.perf_counter() - started_total
    resources = {
        "circuits": len(values),
        "shots_per_circuit": int(shots),
        "total_shots": int(len(values) * shots),
        "transpiled_depth_max": max(depths) if depths else None,
        "transpiled_depth_mean": float(np.mean(depths)) if depths else None,
        "transpiled_operation_totals": operation_totals,
    }
    return output, {"runtime_breakdown_seconds": timings, "execution_resources": resources}


def _split_contiguous(values: np.ndarray, data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    train_end = len(data["y_train"])
    val_end = train_end + len(data["y_val"])
    return {
        "train": values[:train_end],
        "val": values[train_end:val_end],
        "test": values[val_end:],
    }


def _evaluate_case(
    case: NoiseCase,
    data: Mapping[str, Any],
    *,
    method: str,
    optimization_level: int,
    ridge_alpha: float,
    quick: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        noise_model, basis_gates = build_noise_model(case)
        backend = AerSimulator(
            method=method,
            noise_model=noise_model,
            seed_simulator=case.seed,
        )
        sequence = np.vstack([data[f"X_{name}"] for name in SPLIT_NAMES])
        qrc = OnionQRC(
            case.n_qubits,
            topology=case.topology,
            seed=case.seed,
            observable_order=2,
        )
        all_features, simulation_info = run_shot_sequence(
            qrc,
            sequence,
            backend=backend,
            transpile_fn=transpile,
            shots=case.shots,
            simulation_seed=case.seed,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            progress=True,
        )
        split_features = _split_contiguous(all_features, data)
        use_regime = len(np.unique(data["regime_train"])) >= 2
        readout = VolQRCReadout(
            ridge_alpha=ridge_alpha,
            use_regime=use_regime,
            regime_cv_splits=3 if quick else 5,
            classifier_pca_components=min(8, split_features["train"].shape[1]),
        )
        if use_regime:
            readout.fit(
                split_features["train"],
                data["y_train"],
                data["regime_train"],
                data["X_train"][:, :3],
            )
        else:
            readout.fit(
                split_features["train"],
                data["y_train"],
                X_classical=data["X_train"][:, :3],
            )
        # Validation and test labels are never passed to fit or predict.
        val_pred = readout.predict(split_features["val"], data["X_val"][:, :3])
        test_pred = readout.predict(split_features["test"], data["X_test"][:, :3])
        regime_metrics: dict[str, Any] = {}
        if use_regime and readout._regime_fitted:
            regime_metrics = {
                "validation": readout.regime_metrics(
                    split_features["val"], data["regime_val"]
                ),
                "test": readout.regime_metrics(
                    split_features["test"], data["regime_test"]
                ),
            }
        return {
            "model": "OnionQRC-shot-noise",
            "status": "ok",
            "case_id": case.case_id,
            "runtime_seconds": time.perf_counter() - started,
            "validation_metrics": compute_metrics(data["y_val"], val_pred, is_log_rv=True),
            "test_metrics": compute_metrics(data["y_test"], test_pred, is_log_rv=True),
            "regime_metrics": regime_metrics,
            "validation_dates": _date_range(data["target_index_val"]),
            "test_dates": _date_range(data["target_index_test"]),
            "n_qubits": case.n_qubits,
            "topology": case.topology,
            "seed": case.seed,
            "shots": case.shots,
            "noise_kind": case.noise_kind,
            "noise_parameter": case.noise_parameter,
            "depolarizing_p": (
                case.noise_parameter if case.noise_kind == "depolarizing" else None
            ),
            "amplitude_damping_gamma": (
                case.noise_parameter if case.noise_kind == "amplitude_damping" else None
            ),
            "backend": "qiskit_aer.AerSimulator",
            "simulation_method": method,
            "basis_gates": list(basis_gates),
            "optimization_level": optimization_level,
            "logical_resources": qrc.estimate_resources(
                include_feedback=True, measure=True
            ),
            **simulation_info,
        }
    except Exception as exc:
        return {
            "model": "OnionQRC-shot-noise",
            "status": "failed",
            "case_id": case.case_id,
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"{type(exc).__name__}: {exc}",
            "n_qubits": case.n_qubits,
            "topology": case.topology,
            "seed": case.seed,
            "shots": case.shots,
            "noise_kind": case.noise_kind,
            "noise_parameter": case.noise_parameter,
            "backend": "qiskit_aer.AerSimulator",
        }


def _versions() -> dict[str, str | None]:
    versions = package_versions()
    try:
        versions["qiskit-aer"] = importlib.metadata.version("qiskit-aer")
    except importlib.metadata.PackageNotFoundError:
        versions["qiskit-aer"] = None
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a short chronological tail")
    parser.add_argument("--quick-rows", type=int, default=90)
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--n-qubits", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--shots", type=int, nargs="+", default=None)
    parser.add_argument("--topology", choices=("ring", "fully_connected"), default="ring")
    parser.add_argument("--depolarizing-p", type=float, nargs="*", default=None)
    parser.add_argument("--amplitude-damping-gamma", type=float, nargs="*", default=None)
    parser.add_argument(
        "--no-noiseless-shots",
        action="store_false",
        dest="include_noiseless",
        help="Exclude the finite-shot case without a gate noise model",
    )
    parser.set_defaults(include_noiseless=True)
    parser.add_argument("--tanh-scale", type=float, default=2.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=1)
    parser.add_argument(
        "--method",
        choices=("automatic", "statevector", "matrix_product_state"),
        default="automatic",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = generate_cases(
            n_qubits=args.n_qubits,
            seeds=args.seeds,
            shots=args.shots,
            topology=args.topology,
            depolarizing_p=args.depolarizing_p,
            amplitude_damping_gamma=args.amplitude_damping_gamma,
            include_noiseless=args.include_noiseless,
            full_matrix=args.full_matrix,
            max_cases=args.max_cases,
        )
    except ValueError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    print(f"[noise] selected {len(cases)} case(s)")
    print("[noise] loading real SPX realized-volatility data")
    data = prepare_phase3_data(
        load_spx_rv(allow_synthetic=False),
        tanh_scale=args.tanh_scale,
        quick=args.quick,
        quick_rows=args.quick_rows,
    )
    print(
        "[noise] strict split sizes: "
        + ", ".join(f"{name}={len(data[f'y_{name}'])}" for name in SPLIT_NAMES)
    )

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[noise] case {index}/{len(cases)}: {case.case_id}")
        results.append(
            _evaluate_case(
                case,
                data,
                method=args.method,
                optimization_level=args.optimization_level,
                ridge_alpha=args.ridge_alpha,
                quick=args.quick,
            )
        )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": "run_noise",
        "source": "real_spx_rv",
        "quick": args.quick,
        "quick_rows": args.quick_rows if args.quick else None,
        "full_matrix": args.full_matrix,
        "max_cases": args.max_cases,
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "features": list(DEFAULT_FEATURES),
        "context": 1,
        "horizon": 1,
        "tanh_scale": args.tanh_scale,
        "split_sizes": {name: len(data[f"y_{name}"]) for name in SPLIT_NAMES},
        "split_dates": {
            name: _date_range(data[f"target_index_{name}"]) for name in SPLIT_NAMES
        },
        "versions": _versions(),
        "command": [sys.executable, *sys.argv],
    }
    json_path, csv_path = write_artifacts(results, args.output_dir, metadata)
    print(f"[noise] JSON: {json_path}")
    print(f"[noise] CSV:  {csv_path}")
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
