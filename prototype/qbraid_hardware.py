"""Explicit, spend-capped qBraid preparation and submission utilities.

The default command is offline: it selects a deterministic balanced date panel,
builds/transpiles circuits, exports QASM, and estimates cost. Live device discovery
requires ``--live-discovery``. Submission additionally requires ``--submit`` and a
``QBRAID_API_KEY`` environment variable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from .data_loader import load_spx_rv
    from .onion_qrc import OnionQRC
    from .run_phase3 import _native, package_versions, prepare_phase3_data
except ImportError:  # pragma: no cover - direct script execution.
    from data_loader import load_spx_rv  # type: ignore
    from onion_qrc import OnionQRC  # type: ignore
    from run_phase3 import _native, package_versions, prepare_phase3_data  # type: ignore


DEFAULT_EXPORT_DIR = Path(__file__).parent / "hardware_exports"
API_KEY_ENV = "QBRAID_API_KEY"


@dataclass(frozen=True)
class CostEstimate:
    circuits: int
    shots_per_circuit: int
    total_shots: int
    cost_per_shot_usd: float
    fixed_cost_per_circuit_usd: float
    estimated_total_usd: float

    def as_dict(self) -> dict[str, Any]:
        return _native(self.__dict__)


def select_balanced_panel(
    dates: Any,
    regimes: Any,
    per_regime: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Select an equal, deterministic chronological panel from each regime.

    A seeded permutation prevents systematic edge-date selection. The same sorted
    integer positions can then be reused for every circuit configuration.
    """
    date_index = pd.DatetimeIndex(dates)
    labels = np.asarray(regimes)
    if len(date_index) != len(labels) or len(labels) == 0:
        raise ValueError("dates and regimes must be non-empty and equally sized")
    if not date_index.is_monotonic_increasing or not date_index.is_unique:
        raise ValueError("dates must be unique and chronological")
    if per_regime < 1:
        raise ValueError("per_regime must be positive")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_regime:
            raise ValueError(
                f"regime {label!r} has {len(candidates)} dates; {per_regime} required"
            )
        selected.extend(rng.permutation(candidates)[:per_regime].tolist())
    return np.asarray(sorted(selected), dtype=int)


def estimate_cost(
    n_circuits: int,
    shots: int,
    *,
    cost_per_shot_usd: float = 0.0,
    fixed_cost_per_circuit_usd: float = 0.0,
) -> CostEstimate:
    if n_circuits < 1 or shots < 1:
        raise ValueError("n_circuits and shots must be positive")
    if cost_per_shot_usd < 0 or fixed_cost_per_circuit_usd < 0:
        raise ValueError("cost inputs cannot be negative")
    total_shots = n_circuits * shots
    total = total_shots * cost_per_shot_usd + n_circuits * fixed_cost_per_circuit_usd
    return CostEstimate(
        circuits=n_circuits,
        shots_per_circuit=shots,
        total_shots=total_shots,
        cost_per_shot_usd=float(cost_per_shot_usd),
        fixed_cost_per_circuit_usd=float(fixed_cost_per_circuit_usd),
        estimated_total_usd=float(total),
    )


def enforce_spend_cap(estimate: CostEstimate, spend_cap_usd: float) -> None:
    if spend_cap_usd < 0:
        raise ValueError("spend_cap_usd cannot be negative")
    if estimate.estimated_total_usd > spend_cap_usd:
        raise RuntimeError(
            f"Estimated cost ${estimate.estimated_total_usd:.4f} exceeds "
            f"spend cap ${spend_cap_usd:.4f}; nothing was submitted"
        )


def discover_devices(*, live: bool = False, api_key: str | None = None) -> list[dict[str, Any]]:
    """Discover qBraid devices only after an explicit live opt-in."""
    if not live:
        return []
    try:
        runtime = importlib.import_module("qbraid.runtime")
    except ImportError as exc:
        raise ImportError("Live discovery requires the optional 'qbraid' package") from exc

    provider_class = runtime.QbraidProvider
    provider = provider_class(api_key=api_key) if api_key else provider_class()
    devices = provider.get_devices()
    output = []
    for device in devices:
        profile = getattr(device, "profile", None)
        output.append({
            "id": str(getattr(device, "id", getattr(profile, "device_id", device))),
            "status": str(getattr(device, "status", "unknown")),
            "profile": str(profile) if profile is not None else None,
        })
    return output


def _qasm_text(circuit: Any) -> tuple[str, str]:
    try:
        from qiskit import qasm3

        return qasm3.dumps(circuit), "openqasm3"
    except (ImportError, AttributeError):
        if hasattr(circuit, "qasm"):
            return circuit.qasm(), "openqasm2"
        raise RuntimeError("Installed Qiskit cannot export this circuit to QASM")


def prepare_circuits(
    features: np.ndarray,
    dates: Any,
    *,
    n_qubits: Sequence[int],
    topology: str,
    seeds: Sequence[int],
    output_dir: Path,
    optimization_level: int = 1,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Build, transpile, and export measured circuits without contacting qBraid."""
    try:
        from qiskit import transpile
    except ImportError as exc:
        raise ImportError("Circuit preparation requires the optional 'qiskit' package") from exc

    values = np.asarray(features, dtype=float)
    date_index = pd.DatetimeIndex(dates)
    if values.ndim != 2 or len(values) != len(date_index):
        raise ValueError("features must be a 2D array aligned with dates")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    circuits: list[Any] = []
    for n in n_qubits:
        for seed in seeds:
            qrc = OnionQRC(n, topology=topology, seed=seed)
            qrc.reset_memory()
            for row_index, (row, date) in enumerate(zip(values, date_index)):
                circuit = qrc.build_circuit(row, measure=True)
                compiled = transpile(circuit, optimization_level=optimization_level)
                qasm, qasm_format = _qasm_text(compiled)
                filename = f"onion_n{n}_seed{seed}_{date.strftime('%Y%m%d')}_{row_index:03d}.qasm"
                path = output_dir / filename
                path.write_text(qasm, encoding="utf-8")
                manifest.append({
                    "file": str(path),
                    "date": date.isoformat(),
                    "n_qubits": int(n),
                    "seed": int(seed),
                    "topology": topology,
                    "qasm_format": qasm_format,
                    "depth": int(compiled.depth()),
                    "operations": {str(k): int(v) for k, v in compiled.count_ops().items()},
                    "resources": qrc.estimate_resources(include_feedback=False, measure=True),
                })
                circuits.append(compiled)
    return manifest, circuits


def submit_circuits(
    circuits: Sequence[Any],
    *,
    device_id: str,
    shots: int,
    submit: bool = False,
    api_key_env: str = API_KEY_ENV,
) -> list[dict[str, Any]]:
    """Submit only with both the explicit flag and an environment API key."""
    if not submit:
        return []
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"--submit requires environment variable {api_key_env}")
    if not device_id:
        raise ValueError("--submit requires --device-id")
    try:
        runtime = importlib.import_module("qbraid.runtime")
    except ImportError as exc:
        raise ImportError("Submission requires the optional 'qbraid' package") from exc

    provider = runtime.QbraidProvider(api_key=api_key)
    device = provider.get_device(device_id)
    jobs = []
    for circuit in circuits:
        job = device.run(circuit, shots=shots)
        jobs.append({
            "job_id": str(getattr(job, "id", getattr(job, "job_id", job))),
            "device_id": device_id,
            "shots": int(shots),
        })
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, nargs="+", default=[5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--topology", choices=("ring", "fully_connected"), default="ring")
    parser.add_argument("--per-regime", type=int, default=2)
    parser.add_argument("--panel-seed", type=int, default=42)
    parser.add_argument("--tanh-scale", type=float, default=2.0)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--cost-per-shot-usd", type=float, default=0.0)
    parser.add_argument("--fixed-cost-per-circuit-usd", type=float, default=0.0)
    parser.add_argument("--spend-cap-usd", type=float, default=25.0)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--device-id")
    parser.add_argument("--live-discovery", action="store_true")
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(API_KEY_ENV)
    if args.live_discovery:
        print("[hardware] live qBraid discovery explicitly enabled")
        print(json.dumps(discover_devices(live=True, api_key=api_key), indent=2))

    print("[hardware] loading real data and selecting balanced test-date panel")
    data = prepare_phase3_data(load_spx_rv(allow_synthetic=False), tanh_scale=args.tanh_scale)
    positions = select_balanced_panel(
        data["target_index_test"], data["regime_test"], args.per_regime,
        seed=args.panel_seed,
    )
    panel_features = data["X_test"][positions]
    panel_dates = data["target_index_test"][positions]
    n_circuits = len(panel_dates) * len(args.n_qubits) * len(args.seeds)
    estimate = estimate_cost(
        n_circuits,
        args.shots,
        cost_per_shot_usd=args.cost_per_shot_usd,
        fixed_cost_per_circuit_usd=args.fixed_cost_per_circuit_usd,
    )
    enforce_spend_cap(estimate, args.spend_cap_usd)
    print(
        f"[hardware] estimated ${estimate.estimated_total_usd:.4f} for "
        f"{estimate.circuits} circuits / {estimate.total_shots} shots "
        f"(cap ${args.spend_cap_usd:.4f})"
    )

    started = time.perf_counter()
    manifest, circuits = prepare_circuits(
        panel_features,
        panel_dates,
        n_qubits=args.n_qubits,
        topology=args.topology,
        seeds=args.seeds,
        output_dir=args.output_dir,
        optimization_level=args.optimization_level,
    )
    jobs = submit_circuits(
        circuits,
        device_id=args.device_id or "",
        shots=args.shots,
        submit=args.submit,
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend_intent": "qbraid_hardware" if args.submit else "offline_qiskit_transpile_only",
        "submitted": bool(args.submit),
        "device_id": args.device_id,
        "balanced_panel_dates": [date.isoformat() for date in panel_dates],
        "cost_estimate": estimate.as_dict(),
        "spend_cap_usd": args.spend_cap_usd,
        "runtime_seconds": time.perf_counter() - started,
        "versions": package_versions(),
        "circuits": manifest,
        "jobs": jobs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_native(payload), indent=2, sort_keys=True), encoding="utf-8")
    print(f"[hardware] manifest: {manifest_path}")
    if not args.submit:
        print("[hardware] offline preparation complete; no circuits submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
