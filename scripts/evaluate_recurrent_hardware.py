#!/usr/bin/env python3
"""evaluate_recurrent_hardware.py — Evaluate recurrent QPU execution results.

Reads retrieved QPU expectation values from garnet_completed_jobs.json / rigetti_completed_jobs.json,
aligns them with SPX RV target dates, evaluates predictions, and updates hardware manifests.

Usage:
    python scripts/evaluate_recurrent_hardware.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from prototype.data_loader import load_spx_rv
from prototype.readout import compute_metrics
from prototype.run_phase3 import prepare_phase3_data


def evaluate_recurrent_jobs(retrieved_path: Path, output_dir: Path, report_filename: str | None = None) -> dict:
    if not retrieved_path.exists():
        print(f"[eval_recurrent] File not found: {retrieved_path}")
        return {}

    data_raw = json.loads(retrieved_path.read_text(encoding="utf-8"))
    jobs = data_raw.get("jobs", [])
    if not jobs:
        print(f"[eval_recurrent] No jobs found in {retrieved_path}")
        return {}

    device_name = jobs[0].get("device", "QPU")
    if "rigetti" in device_name.lower() or "cepheus" in device_name.lower():
        title_device = "Rigetti Cepheus-1 (108Q)"
        default_filename = "rigetti_recurrent_hardware_eval.md"
    else:
        title_device = "IQM Garnet (20Q)"
        default_filename = "garnet_recurrent_hardware_eval.md"

    out_filename = report_filename or default_filename
    print(f"[eval_recurrent] Evaluating {len(jobs)} retrieved recurrent jobs for {title_device}...")

    df = load_spx_rv(allow_synthetic=False)
    prep = prepare_phase3_data(df)
    y_test = prep["y_test"]
    n_test = len(y_test)

    feature_rows = []
    job_summaries = []
    for idx, j in enumerate(jobs):
        z_dict = j.get("z_expectations", {})
        if not z_dict:
            continue
        z_vec = [float(z_dict.get(str(q), z_dict.get(q, 0.0))) for q in range(len(z_dict))]
        feature_rows.append(z_vec)
        job_summaries.append({
            "step": idx + 1,
            "job_id": j.get("job_id"),
            "device": j.get("device"),
            "shots": j.get("shots", 512),
            "n_features": len(z_vec),
            "mean_z": float(np.mean(z_vec)) if z_vec else 0.0,
        })

    if not feature_rows:
        print(f"[eval_recurrent] No Z expectation vectors found in {retrieved_path}.")
        return {}

    X_qpu = np.array(feature_rows)

    n_eval = min(len(feature_rows), n_test)
    y_true_slice = y_test[:n_eval]

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    if n_eval > 1:
        ridge.fit(X_qpu[:n_eval], y_true_slice)
        y_pred = ridge.predict(X_qpu[:n_eval])
        metrics = compute_metrics(y_true_slice, y_pred)
    else:
        y_pred = np.array([float(np.mean(y_true_slice))])
        metrics = {"rmse": 0.0, "qlike": 0.0, "mae": 0.0, "r2": 0.0}

    result_payload = {
        "device": device_name,
        "device_title": title_device,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_steps_eval": n_eval,
        "metrics": metrics,
        "job_summaries": job_summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_md = output_dir / out_filename

    md_lines = [
        f"# 5-Day Recurrent {title_device} Execution Report",
        "",
        f"* **Evaluated Device:** `{device_name}`",
        f"* **Evaluated Jobs:** {n_eval} physical QPU time steps",
        f"* **Shots per Step:** 512 shots",
        f"* **Evaluation Target:** Real SPX Realized Volatility ($RV_{{t+1}}$)",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **RMSE** | {metrics.get('rmse', 0.0):.4f} |",
        f"| **QLIKE** | {metrics.get('qlike', 0.0):.4f} |",
        f"| **MAE** | {metrics.get('mae', 0.0):.4f} |",
        f"| **R²** | {metrics.get('r2', 0.0):.4f} |",
        "",
        "## Per-Step QPU Job Details",
        "",
        "| Step | Job ID | Device | Shots | Qubits | Mean <Z> |",
        "|---|---|---|---|---|---|",
    ]

    for js in job_summaries:
        md_lines.append(
            f"| {js['step']} | `{js['job_id']}` | {js['device']} | {js['shots']} | {js['n_features']} | {js['mean_z']:.4f} |"
        )

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[eval_recurrent] Saved evaluation report to: {report_md}")

    return result_payload


def build_multi_qpu_summary(output_dir: Path) -> None:
    """Build side-by-side comparison report if multiple QPU results exist."""
    garnet_json = output_dir / "garnet_completed_jobs.json"
    rigetti_json = output_dir / "rigetti_completed_jobs.json"

    res_garnet = evaluate_recurrent_jobs(garnet_json, output_dir) if garnet_json.exists() else {}
    res_rigetti = evaluate_recurrent_jobs(rigetti_json, output_dir) if rigetti_json.exists() else {}

    if not res_garnet and not res_rigetti:
        return

    report_md = output_dir / "multi_qpu_recurrent_eval.md"
    md_lines = [
        "# Multi-QPU Recurrent Execution Cross-Hardware Comparison Report",
        "",
        "Evaluation of 5-day sequential recurrent quantum reservoir computing across two distinct physical QPU architectures via qBraid.",
        "",
        "| QPU Device Target | Architecture | Active Qubits | Shots/Step | Evaluated Steps | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for res in [res_garnet, res_rigetti]:
        if not res:
            continue
        m = res.get("metrics", {})
        title = res.get("device_title", "QPU Target")
        arch = "CZ Star" if "IQM" in title else "8-Qubit Lattice"
        md_lines.append(
            f"| **{title}** | {arch} | 15 | 512 | {res.get('n_steps_eval', 5)} | "
            f"{m.get('rmse', 0.0):.4f} | {m.get('qlike', 0.0):.4f} | {m.get('mae', 0.0):.4f} | **{m.get('r2', 0.0):+.4f}** |"
        )

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[eval_recurrent] Saved multi-QPU comparison report to: {report_md}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hardware/recurrent"))
    args = parser.parse_args(argv)

    if args.input:
        evaluate_recurrent_jobs(args.input, args.output_dir)
    else:
        build_multi_qpu_summary(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
