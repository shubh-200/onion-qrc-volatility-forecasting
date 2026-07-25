"""Phase 3 integration runner for classical and noiseless OnionQRC experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:  # Support both ``python -m gic.prototype.run_phase3`` and direct execution.
    from .baselines import (
        EGARCHBaseline,
        ESNBaseline,
        GARCHBaseline,
        LSTMBaseline,
        RandomFeatureRidgeBaseline,
    )
    from .data_loader import DEFAULT_FEATURES, load_spx_rv, split_data
    from .onion_qrc import OnionQRC
    from .readout import (
        VolQRCReadout,
        block_bootstrap_confidence_interval,
        compute_metrics,
        mincer_zarnowitz,
    )
except ImportError:  # pragma: no cover - exercised by the documented direct CLI.
    from baselines import (  # type: ignore
        EGARCHBaseline,
        ESNBaseline,
        GARCHBaseline,
        LSTMBaseline,
        RandomFeatureRidgeBaseline,
    )
    from data_loader import DEFAULT_FEATURES, load_spx_rv, split_data  # type: ignore
    from onion_qrc import OnionQRC  # type: ignore
    from readout import (  # type: ignore
        VolQRCReadout,
        block_bootstrap_confidence_interval,
        compute_metrics,
        mincer_zarnowitz,
    )


RESULTS_DIR = Path(__file__).parent / "results" / "phase3"
CACHE_DIR = Path(__file__).parent / ".phase3_cache"
SPLIT_NAMES = ("train", "val", "test")
BACKEND = "qiskit_statevector_noiseless"


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "qiskit", "arch", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def bound_har_features(X: np.ndarray, tanh_scale: float = 2.0) -> np.ndarray:
    """Use the only/last time step and smoothly map scaled HAR inputs to (-1, 1)."""
    if tanh_scale <= 0:
        raise ValueError("tanh_scale must be positive")
    values = np.asarray(X, dtype=float)
    if values.ndim == 3:
        values = values[:, -1, :]
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("X must contain at least three features")
    bounded = values.copy()
    bounded[:, :3] = np.tanh(bounded[:, :3] / tanh_scale)
    return bounded


def prepare_phase3_data(
    df: pd.DataFrame,
    *,
    tanh_scale: float = 2.0,
    quick: bool = False,
    quick_rows: int = 240,
) -> dict[str, Any]:
    """Create strict chronological train/validation/test arrays from real-data shape."""
    if quick:
        if quick_rows < 30:
            raise ValueError("quick_rows must be at least 30")
        df = df.tail(min(len(df), quick_rows)).copy()
    split = split_data(df, context=1, features=DEFAULT_FEATURES)
    for name in SPLIT_NAMES:
        split[f"X_raw_{name}"] = np.asarray(split[f"X_{name}"])[:, -1, :].copy()
        split[f"X_{name}"] = bound_har_features(split[f"X_{name}"], tanh_scale)
    split["frame"] = df
    split["tanh_scale"] = float(tanh_scale)
    return split


def _date_range(index: Sequence[Any]) -> dict[str, str | None]:
    if len(index) == 0:
        return {"start": None, "end": None}
    return {
        "start": str(pd.Timestamp(index[0])),
        "end": str(pd.Timestamp(index[-1])),
    }


def _model_result(
    model: str,
    status: str,
    runtime_seconds: float,
    *,
    validation: Mapping[str, Any] | None = None,
    test: Mapping[str, Any] | None = None,
    reason: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    result = {
        "model": model,
        "status": status,
        "runtime_seconds": float(runtime_seconds),
        "validation_metrics": _native(validation) if validation is not None else None,
        "test_metrics": _native(test) if test is not None else None,
    }
    if reason:
        result["reason"] = reason
    result.update(_native(metadata))
    return result


def _evaluate_predictions(
    model: str,
    started: float,
    data: Mapping[str, Any],
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    **metadata: Any,
) -> dict[str, Any]:
    validation_metrics = compute_metrics(data["y_val"], val_pred, is_log_rv=True)
    test_metrics = compute_metrics(data["y_test"], test_pred, is_log_rv=True)
    return _model_result(
        model,
        "ok",
        time.perf_counter() - started,
        validation=validation_metrics,
        test=test_metrics,
        validation_dates=_date_range(data["target_index_val"]),
        test_dates=_date_range(data["target_index_test"]),
        validation_mincer_zarnowitz=mincer_zarnowitz(data["y_val"], val_pred),
        test_mincer_zarnowitz=mincer_zarnowitz(data["y_test"], test_pred),
        test_rmse_block_bootstrap=block_bootstrap_confidence_interval(
            data["y_test"], test_pred,
            metric="rmse", n_bootstrap=200, random_state=42,
        ),
        validation_predictions=np.asarray(val_pred),
        test_predictions=np.asarray(test_pred),
        **metadata,
    )


def _run_ridge_family(
    data: Mapping[str, Any],
    model: Any,
    name: str,
    *,
    columns: slice | Sequence[int] = slice(None),
    feature_prefix: str = "X",
) -> dict[str, Any]:
    started = time.perf_counter()
    model.fit(data[f"{feature_prefix}_train"][:, columns], data["y_train"])
    return _evaluate_predictions(
        name,
        started,
        data,
        model.predict(data[f"{feature_prefix}_val"][:, columns]),
        model.predict(data[f"{feature_prefix}_test"][:, columns]),
    )


def _run_esn(data: Mapping[str, Any], size: int, seed: int, quick: bool) -> dict[str, Any]:
    started = time.perf_counter()
    effective_size = min(size, 80) if quick else size
    model = ESNBaseline(n_reservoir=effective_size, seed=seed)
    washout = min(10 if quick else 50, max(0, len(data["X_train"]) // 4))
    model.fit(data["X_train"], data["y_train"], washout=washout)
    # Calls are chronological: validation advances the state before test prediction.
    val_pred = model.predict(data["X_val"])
    test_pred = model.predict(data["X_test"])
    return _evaluate_predictions(
        f"ESN-{size}", started, data, val_pred, test_pred,
        configured_reservoir_size=size,
        effective_reservoir_size=effective_size,
        quick_reduction=bool(quick and effective_size != size),
        washout=washout,
    )


def _run_lstm(data: Mapping[str, Any], quick: bool) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch  # noqa: F401
    except ImportError:
        return _model_result(
            "LSTM", "skipped", time.perf_counter() - started,
            reason="Optional dependency 'torch' is not installed",
        )
    seq_len = min(20 if quick else 60, max(2, len(data["X_train"]) // 4))
    model = LSTMBaseline(
        hidden_size=16 if quick else 64,
        n_layers=1 if quick else 2,
        epochs=2 if quick else 50,
        seq_len=seq_len,
        seed=42,
    )
    model.fit(
        data["X_train"], data["y_train"],
        X_val=data["X_val"], y_val=data["y_val"],
    )
    val_context = np.vstack([data["X_train"][-seq_len:], data["X_val"]])
    test_history = np.vstack([data["X_train"], data["X_val"]])
    test_context = np.vstack([test_history[-seq_len:], data["X_test"]])
    val_pred = model.predict(val_context)[-len(data["y_val"]):]
    test_pred = model.predict(test_context)[-len(data["y_test"]):]
    return _evaluate_predictions(
        "LSTM", started, data, val_pred, test_pred,
        sequence_length=seq_len,
        epochs_trained=model.epochs_trained_,
    )


def _run_arch_models(data: Mapping[str, Any], quick: bool) -> list[dict[str, Any]]:
    try:
        importlib.import_module("arch")
    except ImportError:
        reason = "Optional dependency 'arch' is not installed; rolling GARCH/EGARCH skipped"
        return [
            _model_result(name, "skipped", 0.0, reason=reason)
            for name in ("GARCH(1,1)", "EGARCH(1,1,1)")
        ]

    frame = data["frame"]
    returns = frame["price_return"].astype(float)
    first_target = pd.Timestamp(data["target_index_val"][0])
    start = int(returns.index.get_loc(first_target))
    window = min(60 if quick else 252, start)
    results = []
    for name, model in (("GARCH(1,1)", GARCHBaseline()), ("EGARCH(1,1,1)", EGARCHBaseline())):
        started = time.perf_counter()
        try:
            forecast = model.rolling_forecast(
                returns, window=max(2, window), mode="rolling", start=start
            )
            log_forecast = pd.Series(
                np.log(np.maximum(np.asarray(forecast), 1e-12)),
                index=returns.index[start:],
            )
            val_pred = log_forecast.reindex(data["target_index_val"]).to_numpy()
            test_pred = log_forecast.reindex(data["target_index_test"]).to_numpy()
            results.append(_evaluate_predictions(
                name, started, data, val_pred, test_pred,
                return_source="aligned_price_return", rolling_window=max(2, window),
            ))
        except Exception as exc:
            results.append(_model_result(
                name, "failed", time.perf_counter() - started,
                reason=f"{type(exc).__name__}: {exc}",
            ))
    return results


def run_baselines(data: Mapping[str, Any], *, seed: int = 42, quick: bool = False) -> list[dict[str, Any]]:
    """Run every classical model on exactly the prepared split arrays."""
    results: list[dict[str, Any]] = []

    print("[baselines] persistence")
    started = time.perf_counter()
    scaler = data["scaler"]
    val_raw = scaler.inverse_transform(data["X_raw_val"])
    test_raw = scaler.inverse_transform(data["X_raw_test"])
    results.append(_evaluate_predictions(
        "Persistence", started, data, val_raw[:, 0], test_raw[:, 0]
    ))

    print("[baselines] causal HAR ridge")
    results.append(_run_ridge_family(
        data,
        Ridge(alpha=1.0),
        "HAR-Ridge",
        columns=slice(0, 3),
        feature_prefix="X_raw",
    ))

    print("[baselines] random-feature ridge 210")
    results.append(_run_ridge_family(
        data,
        RandomFeatureRidgeBaseline(n_features=210, seed=seed),
        "RandomFeatureRidge-210",
    ))

    for size in (210, 500):
        print(f"[baselines] ESN {size}")
        results.append(_run_esn(data, size, seed, quick))

    print("[baselines] rolling GARCH/EGARCH")
    results.extend(_run_arch_models(data, quick))
    print("[baselines] LSTM")
    results.append(_run_lstm(data, quick))
    return results


def _cache_key(data: Mapping[str, Any], n_qubits: int, topology: str, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.vstack([data[f"X_{s}"] for s in SPLIT_NAMES])).tobytes())
    digest.update(
        f"{BACKEND}|N={n_qubits}|topology={topology}|seed={seed}|observable_order=2".encode()
    )
    return digest.hexdigest()[:20]


def reservoir_features(
    data: Mapping[str, Any],
    *,
    n_qubits: int,
    topology: str,
    seed: int,
    cache_dir: Path,
    use_cache: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute/cache one continuous, noiseless statevector reservoir sequence."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"onion_{_cache_key(data, n_qubits, topology, seed)}.npz"
    if use_cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            return cached["features"], {
                "cache": "hit", "cache_path": str(cache_path),
                "reservoir_runtime_seconds": float(cached["runtime_seconds"]),
            }

    sequence = np.vstack([data[f"X_{name}"] for name in SPLIT_NAMES])
    qrc = OnionQRC(n_qubits, topology=topology, seed=seed, observable_order=2)
    output = np.empty((len(sequence), qrc.n_observables), dtype=float)
    qrc.reset_memory()
    started = time.perf_counter()
    progress_every = max(1, len(sequence) // 10)
    for index, row in enumerate(sequence):
        output[index] = qrc.step(row)
        if (index + 1) % progress_every == 0 or index + 1 == len(sequence):
            print(f"    reservoir {index + 1}/{len(sequence)}")
    runtime = time.perf_counter() - started
    np.savez_compressed(cache_path, features=output, runtime_seconds=np.asarray(runtime))
    return output, {
        "cache": "miss", "cache_path": str(cache_path),
        "reservoir_runtime_seconds": runtime,
    }


def _split_contiguous(values: np.ndarray, data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    train_end = len(data["y_train"])
    val_end = train_end + len(data["y_val"])
    return {
        "train": values[:train_end],
        "val": values[train_end:val_end],
        "test": values[val_end:],
    }


def run_simulator(
    data: Mapping[str, Any],
    *,
    n_qubits: Sequence[int] = (5, 10, 15, 20),
    topology: str = "ring",
    seeds: Sequence[int] = (42, 123, 2026),
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
    quick: bool = False,
) -> list[dict[str, Any]]:
    """Run noiseless statevector OnionQRC configurations with causal readouts."""
    try:
        import qiskit  # noqa: F401
    except ImportError:
        return [_model_result(
            "OnionQRC", "skipped", 0.0,
            reason="Optional dependency 'qiskit' is not installed; statevector simulation skipped",
            backend=BACKEND,
        )]

    results = []
    for n in n_qubits:
        if n < 3:
            raise ValueError("OnionQRC requires at least three qubits")
        for seed in seeds:
            print(f"[simulator] OnionQRC N={n} topology={topology} seed={seed} backend={BACKEND}")
            started = time.perf_counter()
            try:
                all_features, cache_info = reservoir_features(
                    data, n_qubits=n, topology=topology, seed=seed,
                    cache_dir=cache_dir, use_cache=use_cache,
                )
                features = _split_contiguous(all_features, data)
                qrc = OnionQRC(n, topology=topology, seed=seed, observable_order=2)
                use_regime = len(np.unique(data["regime_train"])) >= 2
                readout = VolQRCReadout(
                    ridge_alpha=1.0,
                    use_regime=use_regime,
                    regime_cv_splits=3 if quick else 5,
                    classifier_pca_components=min(8, features["train"].shape[1]),
                )
                if use_regime:
                    readout.fit(
                        features["train"], data["y_train"],
                        data["regime_train"], data["X_train"][:, :3],
                    )
                else:
                    readout.fit(
                        features["train"], data["y_train"],
                        X_classical=data["X_train"][:, :3],
                    )
                val_pred = readout.predict(features["val"], data["X_val"][:, :3])
                test_pred = readout.predict(features["test"], data["X_test"][:, :3])
                regime_metrics: dict[str, Any] = {}
                if use_regime and readout._regime_fitted:
                    regime_metrics = {
                        "validation": readout.regime_metrics(features["val"], data["regime_val"]),
                        "test": readout.regime_metrics(features["test"], data["regime_test"]),
                    }
                results.append(_evaluate_predictions(
                    "OnionQRC", started, data, val_pred, test_pred,
                    n_qubits=n,
                    topology=topology,
                    seed=seed,
                    backend=BACKEND,
                    noise_model="none",
                    resource_counts=qrc.estimate_resources(include_feedback=True, measure=False),
                    regime_metrics=regime_metrics,
                    **cache_info,
                ))
            except Exception as exc:
                results.append(_model_result(
                    "OnionQRC", "failed", time.perf_counter() - started,
                    reason=f"{type(exc).__name__}: {exc}", n_qubits=n,
                    topology=topology, seed=seed, backend=BACKEND,
                ))
    return results


def _csv_row(result: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, Mapping):
            if key in {"validation_metrics", "test_metrics"}:
                prefix = "val" if key.startswith("validation") else "test"
                for metric, metric_value in value.items():
                    row[f"{prefix}_{metric}"] = _native(metric_value)
            else:
                row[key] = json.dumps(_native(value), sort_keys=True)
        elif isinstance(value, (list, tuple, np.ndarray)):
            row[key] = json.dumps(_native(value))
        else:
            row[key] = _native(value)
    return row


def write_artifacts(
    results: Sequence[Mapping[str, Any]],
    output_dir: Path,
    run_metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"phase3_{stamp}.json"
    csv_path = output_dir / f"phase3_{stamp}.csv"
    payload = {"metadata": _native(run_metadata), "results": _native(results)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    rows = [_csv_row(result) for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baselines", "simulator", "all"))
    parser.add_argument("--quick", action="store_true", help="Use a small chronological tail for CI")
    parser.add_argument("--quick-rows", type=int, default=240)
    parser.add_argument("--tanh-scale", type=float, default=2.0)
    parser.add_argument(
        "--n-qubits", type=int, nargs="+", default=None,
        help="Defaults to 5 in quick mode, otherwise 5 10 15 20",
    )
    parser.add_argument("--topology", choices=("ring", "fully_connected"), default="ring")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Defaults to 42 in quick mode, otherwise 42 123 2026",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    n_qubits = args.n_qubits if args.n_qubits is not None else ([5] if args.quick else [5, 10, 15, 20])
    seeds = args.seeds if args.seeds is not None else ([42] if args.quick else [42, 123, 2026])
    print("[phase3] loading real SPX realized-volatility data")
    df = load_spx_rv(allow_synthetic=False)
    data = prepare_phase3_data(
        df, tanh_scale=args.tanh_scale, quick=args.quick, quick_rows=args.quick_rows
    )
    print(
        "[phase3] strict split sizes: "
        + ", ".join(f"{name}={len(data[f'y_{name}'])}" for name in SPLIT_NAMES)
    )
    results: list[dict[str, Any]] = []
    if args.mode in {"baselines", "all"}:
        results.extend(run_baselines(data, seed=seeds[0], quick=args.quick))
    if args.mode in {"simulator", "all"}:
        results.extend(run_simulator(
            data,
            n_qubits=n_qubits,
            topology=args.topology,
            seeds=seeds,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            quick=args.quick,
        ))

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "quick": args.quick,
        "source": "real_spx_rv",
        "features": list(DEFAULT_FEATURES),
        "context": 1,
        "horizon": 1,
        "tanh_scale": args.tanh_scale,
        "n_qubits": n_qubits,
        "seeds": seeds,
        "topology": args.topology,
        "split_sizes": {name: len(data[f"y_{name}"]) for name in SPLIT_NAMES},
        "split_dates": {name: _date_range(data[f"target_index_{name}"]) for name in SPLIT_NAMES},
        "versions": package_versions(),
        "command": [sys.executable, *sys.argv],
    }
    json_path, csv_path = write_artifacts(results, args.output_dir, metadata)
    print(f"[phase3] JSON: {json_path}")
    print(f"[phase3] CSV:  {csv_path}")
    for result in results:
        suffix = f" N={result['n_qubits']}" if "n_qubits" in result else ""
        print(f"  {result['model']}{suffix}: {result['status']} ({result['runtime_seconds']:.6f}s)")
    return 0 if all(result["status"] != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
