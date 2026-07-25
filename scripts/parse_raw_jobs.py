#!/usr/bin/env python3
"""parse_raw_jobs.py — Parse raw QPU job JSON files and evaluate recurrent QPU execution.

Scans artifacts/hardware/recurrent/raw_jobs/*.json, extracts measurement counts,
computes <Z_i> expectation values, writes garnet_completed_jobs.json,
and generates evaluation report.

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

from scripts.evaluate_recurrent_hardware import evaluate_recurrent_jobs
from scripts.find_and_retrieve_qpu import compute_z_expectation


def parse_raw_jobs(raw_dir: Path, output_file: Path) -> list[dict]:
    if not raw_dir.exists():
        print(f"[parse_raw_jobs] Directory not found: {raw_dir}")
        return []

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        print(f"[parse_raw_jobs] No JSON files found in {raw_dir}")
        return []

    print(f"[parse_raw_jobs] Found {len(json_files)} raw QPU job result files in {raw_dir}")

    jobs = []
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

            job_entry = {
                "job_id": str(job_id),
                "status": str(status),
                "device": "aws:iqm:qpu:garnet",
                "created_at": data.get("timeStamps", {}).get("createdAt", ""),
                "shots": shots,
                "counts_summary": {
                    "total_shots": shots,
                    "unique_bitstrings": len(counts),
                },
                "counts": counts,
                "z_expectations": z_exp,
            }
            jobs.append(job_entry)
            print(f"  [OK] Processed {jf.name} -> Job ID: {job_id} ({shots} shots, {len(counts)} bitstrings)")
        except Exception as exc:
            print(f"  [ERROR] Failed to parse {jf.name}: {exc}")

    payload = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "total_retrieved": len(jobs),
        "jobs": jobs,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[parse_raw_jobs] Successfully saved {len(jobs)} QPU jobs to: {output_file}")
    return jobs


def main() -> int:
    raw_dir = Path("artifacts/hardware/recurrent/raw_jobs")
    out_file = Path("artifacts/hardware/recurrent/garnet_completed_jobs.json")

    jobs = parse_raw_jobs(raw_dir, out_file)
    if jobs:
        evaluate_recurrent_jobs(out_file, Path("artifacts/hardware/recurrent"))
        print("[parse_raw_jobs] Full recurrent QPU evaluation complete!")
        return 0
    else:
        print("[parse_raw_jobs] Failed to process any raw jobs.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
