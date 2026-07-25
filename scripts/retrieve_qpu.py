#!/usr/bin/env python3
"""retrieve_qpu.py — Retrieve completed job results from qBraid and save them.

Reads job IDs from a manifest.json (written by submit_qpu.py) and downloads
raw counts + metadata for each completed job.

Usage:
    python scripts/retrieve_qpu.py --manifest artifacts/hardware/panel/manifest.json \\
                                    --save-dir artifacts/hardware/panel/results/

Set QBRAID_API_KEY in your environment before use.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

API_KEY_ENV = "QBRAID_API_KEY"


def retrieve_jobs(job_ids: list[str], api_key: str) -> list[dict]:
    runtime = importlib.import_module("qbraid.runtime")
    results = []
    for job_id in job_ids:
        print(f"  retrieving {job_id} …", end=" ", flush=True)
        try:
            job = runtime.load_job(job_id, api_key=api_key)
            status_obj = getattr(job, "status", None)
            status = str(status_obj() if callable(status_obj) else (status_obj or "unknown"))
            print(status)
            entry: dict = {"job_id": job_id, "status": status}
            if "COMPLETED" in status.upper() or "DONE" in status.upper() or "SUCCESS" in status.upper():
                try:
                    res = job.result()
                    counts = None
                    if hasattr(res, "measurement_counts"):
                        counts = res.measurement_counts()
                    elif hasattr(res, "get_counts"):
                        counts = res.get_counts()
                    elif hasattr(res, "data"):
                        d = res.data()
                        counts = getattr(d, "counts", None) or getattr(d, "measurement_counts", None)
                    if counts is not None:
                        entry["counts"] = dict(counts)
                        entry["shots"] = sum(counts.values())
                    else:
                        entry["error"] = f"Could not extract counts from Result object (keys: {dir(res)})"
                except Exception as exc:
                    entry["error"] = str(exc)
        except Exception as exc:
            print(f"ERROR: {exc}")
            entry = {"job_id": job_id, "status": "error", "error": str(exc)}
        results.append(entry)
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to manifest.json written by submit_qpu.py")
    p.add_argument("--save-dir", type=Path, required=True,
                   help="Directory to write retrieved results JSON")
    args = p.parse_args(argv)

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"[retrieve_qpu] ERROR: {API_KEY_ENV} environment variable is not set",
              file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    job_ids = [j["job_id"] for j in manifest.get("jobs", []) if "job_id" in j]
    if not job_ids:
        print("[retrieve_qpu] no job IDs found in manifest — nothing to retrieve")
        return 0

    print(f"[retrieve_qpu] retrieving {len(job_ids)} job(s)")
    results = retrieve_jobs(job_ids, api_key)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    out = args.save_dir / "retrieved_results.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest),
        "jobs": results,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    completed = sum(1 for r in results if "counts" in r)
    print(f"[retrieve_qpu] {completed}/{len(results)} completed.  Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
