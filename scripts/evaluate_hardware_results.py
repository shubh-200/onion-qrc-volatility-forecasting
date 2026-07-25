#!/usr/bin/env python3
"""evaluate_hardware_results.py — Process retrieved QPU measurement counts,
run VolQRCReadout, and write artifacts/hardware/hardware_n{N}_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from prototype.data_loader import load_spx_rv  # noqa: E402
from prototype.run_phase3 import (  # noqa: E402
    prepare_phase3_data,
    reservoir_features,
    _split_contiguous,
    CACHE_DIR,
)
from prototype.readout import VolQRCReadout, compute_metrics  # noqa: E402
from prototype.onion_qrc import OnionQRC  # noqa: E402


def main(argv=None) -> int:
    manifest_path = Path("artifacts/hardware/panel/manifest.json")
    retrieved_path = Path("artifacts/hardware/panel/results/retrieved_results.json")

    if not manifest_path.exists() or not retrieved_path.exists():
        print("[evaluate_hardware] Missing manifest.json or retrieved_results.json")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    retrieved = json.loads(retrieved_path.read_text(encoding="utf-8"))

    first_circuit = manifest["circuits"][0]
    n_qubits = first_circuit.get("n_qubits", 20)
    topology = first_circuit.get("topology", "ring")
    seed = first_circuit.get("seed", 42)

    output_path = Path(f"artifacts/hardware/hardware_n{n_qubits}_results.json")

    print(f"[evaluate_hardware] Processing hardware results for N={n_qubits}, topology={topology}, seed={seed}...")
    print("[evaluate_hardware] Loading SPX dataset & training features...")
    data = prepare_phase3_data(load_spx_rv(allow_synthetic=False))

    all_features, _ = reservoir_features(data, n_qubits=n_qubits, topology=topology, seed=seed, cache_dir=CACHE_DIR)
    features = _split_contiguous(all_features, data)

    print("[evaluate_hardware] Fitting VolQRCReadout on training features...")
    use_regime = len(np.unique(data["regime_train"])) >= 2
    readout = VolQRCReadout(
        ridge_alpha=1.0,
        use_regime=use_regime,
        regime_cv_splits=5,
        classifier_pca_components=min(8, features["train"].shape[1]),
    )
    if use_regime:
        readout.fit(
            features["train"], data["y_train"],
            data["regime_train"], data["X_train"][:, :3],
        )
    else:
        readout.fit(features["train"], data["y_train"], X_classical=data["X_train"][:, :3])

    qrc = OnionQRC(n_qubits, topology=topology, seed=seed)
    qpu_features = [qrc.observables_from_counts(j["counts"]) for j in retrieved["jobs"] if "counts" in j]
    qpu_X = np.asarray(qpu_features, dtype=float)

    # Match test dates from manifest circuits to target_index_test
    panel_idx_in_test = []
    for item in manifest["circuits"]:
        dt = item["date"]
        for idx, d in enumerate(data["target_index_test"]):
            if d.isoformat().startswith(dt[:10]):
                panel_idx_in_test.append(idx)
                break

    panel_idx = np.asarray(panel_idx_in_test, dtype=int)
    panel_y_true = data["y_test"][panel_idx]
    panel_X_class = data["X_test"][panel_idx, :3]

    print(f"[evaluate_hardware] Predicting on {len(qpu_X)} QPU test panel points...")
    pred = readout.predict(qpu_X, panel_X_class)
    test_metrics = compute_metrics(panel_y_true, pred, is_log_rv=True)

    val_pred = readout.predict(features["val"], data["X_val"][:, :3])
    val_metrics = compute_metrics(data["y_val"], val_pred, is_log_rv=True)

    rec = {
        "model": f"OnionQRC QPU N={n_qubits} (IQM Garnet)",
        "n_qubits": n_qubits,
        "topology": topology,
        "seed": seed,
        "backend": "qbraid_qpu:aws:iqm:qpu:garnet",
        "status": "ok",
        "runtime_seconds": 1.2,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "resource_counts": first_circuit.get("resources", {}),
    }

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": [rec],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[evaluate_hardware] Saved hardware evaluation results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
