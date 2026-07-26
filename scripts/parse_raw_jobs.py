#!/usr/bin/env python3
"""parse_raw_jobs.py — Parse raw QPU job JSON files and evaluate recurrent QPU execution.

Scans artifacts/hardware/recurrent/raw_jobs/*.json, separates jobs by QPU target
(IQM Garnet vs Rigetti Cepheus-1), writes garnet_completed_jobs.json and
rigetti_completed_jobs.json, and generates hardware evaluation reports.

Usage:
    python scripts/parse_raw_jobs.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from scripts.evaluate_recurrent_hardware import build_multi_qpu_summary, evaluate_recurrent_jobs
from scripts.find_and_retrieve_qpu import compute_z_expectation


def _save_payload(jobs: list[dict], output_file: Path) -> None:
    payload = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "total_retrieved": len(jobs),
        "jobs": jobs,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[parse_raw_jobs] Saved {len(jobs)} QPU jobs to: {output_file}")


def parse_raw_jobs(raw_dir: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    if not raw_dir.exists():
        print(f"[parse_raw_jobs] Directory not found: {raw_dir}")
        return [], []

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        print(f"[parse_raw_jobs] No JSON files found in {raw_dir}")
        return [], []

    print(f"[parse_raw_jobs] Found {len(json_files)} raw QPU job result files in {raw_dir}")

    iqm_jobs = []
    rigetti_jobs = []

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            status = data.get("status", "COMPLETED")
            result_data = data.get("resultData", {})
            counts = result_data.get("measurementCounts") or data.get("measurementCounts") or data.get("counts")

            if not counts and "measurement_counts" in data:
                counts = data["measurement_counts"]

            if not counts:
                print(f"  [WARN] Skipping {jf.name}: no measurementCounts found")
                continue

            job_id = data.get("job_id") or jf.name.replace("-result.json", "").replace("_", ":")
            shots = sum(counts.values())
            z_exp = compute_z_expectation(counts)

            device_name = (
                data.get("device_id")
                or data.get("device")
                or ("aws:rigetti:qpu:cepheus-1-108q" if "rigetti" in jf.name.lower() or "cepheus" in jf.name.lower() else "aws:iqm:qpu:garnet")
            )

            job_entry = {
                "job_id": str(job_id),
                "status": str(status),
                "device": str(device_name),
                "created_at": data.get("timeStamps", {}).get("createdAt", ""),
                "shots": shots,
                "counts_summary": {
                    "total_shots": shots,
                    "unique_bitstrings": len(counts),
                },
                "counts": counts,
                "z_expectations": z_exp,
            }

            if "rigetti" in str(device_name).lower() or "cepheus" in str(device_name).lower():
                rigetti_jobs.append(job_entry)
            else:
                iqm_jobs.append(job_entry)
            print(f"  [OK] Processed {jf.name} -> Job ID: {job_id} ({shots} shots, {len(counts)} bitstrings, target: {device_name})")
        except Exception as exc:
            print(f"  [ERROR] Failed to parse {jf.name}: {exc}")

    if iqm_jobs:
        garnet_out = output_dir / "garnet_completed_jobs.json"
        _save_payload(iqm_jobs, garnet_out)
        evaluate_recurrent_jobs(garnet_out, output_dir, report_filename="garnet_recurrent_hardware_eval.md")

    if rigetti_jobs:
        rigetti_out = output_dir / "rigetti_completed_jobs.json"
        _save_payload(rigetti_jobs, rigetti_out)
        evaluate_recurrent_jobs(rigetti_out, output_dir, report_filename="rigetti_recurrent_hardware_eval.md")

    build_multi_qpu_summary(output_dir)
    return iqm_jobs, rigetti_jobs


def main() -> int:
    raw_dir = Path("artifacts/hardware/recurrent/raw_jobs")
    out_dir = Path("artifacts/hardware/recurrent")

    iqm_jobs, rigetti_jobs = parse_raw_jobs(raw_dir, out_dir)
    if iqm_jobs or rigetti_jobs:
        print("[parse_raw_jobs] Full multi-QPU recurrent evaluation complete!")
        return 0
    else:
        print("[parse_raw_jobs] No QPU jobs processed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
