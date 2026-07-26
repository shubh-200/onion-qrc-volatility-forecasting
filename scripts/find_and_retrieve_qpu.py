#!/usr/bin/env python3
"""find_and_retrieve_qpu.py — Find and retrieve recent completed qBraid QPU jobs.

Queries qBraid runtime API for recent jobs, extracts measurement counts,
computes observable expectation values, and saves JSON artifacts.

Usage:
    python scripts/find_and_retrieve_qpu.py [--limit 20] [--job-ids JOB_ID ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

API_KEY_ENV = "QBRAID_API_KEY"
DEFAULT_API_KEY = "qbr_97dd6f449d019064ef31133055b72fa0824444eb820a36c5ae1ccd9731b09b63"


def fetch_recent_jobs(limit: int = 20, api_key: str | None = None) -> list[dict]:
    """List recent jobs from qBraid account."""
    try:
        import qbraid.runtime as qrt
    except ImportError:
        print("ERROR: qbraid package is not installed.", file=sys.stderr)
        return []

    # Attempt fetching using qbraid get_jobs or Service
    jobs_raw = []
    providers_to_try = []

    for prov_name in ["aws", "qbraid", "rigetti", "openquantum"]:
        try:
            if hasattr(qrt, "load_provider"):
                p = qrt.load_provider(prov_name, api_key=api_key)
                providers_to_try.append(p)
        except Exception:
            pass

    try:
        if hasattr(qrt, "QbraidProvider"):
            providers_to_try.append(qrt.QbraidProvider(api_key=api_key))
    except Exception:
        pass

    for prov in providers_to_try:
        try:
            if hasattr(prov, "get_jobs"):
                pj = prov.get_jobs(limit=limit)
                if pj:
                    jobs_raw.extend(pj)
        except Exception:
            pass

    job_records = []
    print(f"[qbraid] Found {len(jobs_raw)} recent jobs in qBraid account")

    for j in jobs_raw:
        job_id = getattr(j, "id", None) or getattr(j, "job_id", str(j))
        status_val = "UNKNOWN"
        try:
            st = getattr(j, "status", None)
            if callable(st):
                status_val = str(st())
            elif st is not None:
                status_val = str(st)
        except Exception:
            pass

        device_name = getattr(j, "device_id", None) or getattr(j, "target", "unknown")
        created_at = getattr(j, "created_at", None) or getattr(j, "date", "")

        record = {
            "job_id": str(job_id),
            "status": status_val,
            "device": str(device_name),
            "created_at": str(created_at),
            "raw_job": j,
        }
        job_records.append(record)
        print(f"  Job ID: {record['job_id']} | Device: {record['device']} | Status: {record['status']}")

    return job_records


def retrieve_job_counts(job_record: dict, api_key: str | None = None) -> dict:
    """Retrieve result counts for a specific job."""
    import qbraid.runtime as qrt
    job_id = job_record["job_id"]
    try:
        if hasattr(qrt, "load_job"):
            job = qrt.load_job(job_id, api_key=api_key)
        else:
            job = job_record["raw_job"]

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
            counts_dict = dict(counts)
            shots = sum(counts_dict.values())
            job_record["counts"] = counts_dict
            job_record["shots"] = shots
            print(f"  [OK] Retrieved counts for {job_id} ({shots} shots, {len(counts_dict)} unique bitstrings)")
        else:
            job_record["error"] = "Counts object not found on result"
            print(f"  [WARN] Could not extract counts for {job_id}")
    except Exception as exc:
        job_record["error"] = str(exc)
        print(f"  [ERROR] Failed to fetch result for {job_id}: {exc}")

    return job_record


def compute_z_expectation(counts: dict[str, int]) -> dict[int, float]:
    """Compute single-qubit <Z_i> expectation values from measurement bitstrings."""
    if not counts:
        return {}
    total_shots = sum(counts.values())
    # find max bitstring length
    num_qubits = max(len(bs.replace(" ", "")) for bs in counts.keys())
    z_exp = {q: 0.0 for q in range(num_qubits)}

    for bs, cnt in counts.items():
        clean_bs = bs.replace(" ", "")
        # Qiskit bitstring order is little-endian: bs[0] is highest qubit index
        for q in range(len(clean_bs)):
            # qubit index q corresponds to position len(clean_bs) - 1 - q from left
            bit_char = clean_bs[len(clean_bs) - 1 - q]
            val = 1.0 if bit_char == '0' else -1.0
            z_exp[q] += val * cnt

    for q in z_exp:
        z_exp[q] /= total_shots
    return z_exp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", type=str, default=None, help="qBraid API key (overrides QBRAID_API_KEY env var)")
    parser.add_argument("--limit", type=int, default=20, help="Number of recent jobs to query")
    parser.add_argument("--job-ids", nargs="+", default=None, help="Explicit list of qBraid job IDs")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hardware/recurrent"))
    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get(API_KEY_ENV) or DEFAULT_API_KEY
    if not api_key:
        print(f"[find_and_retrieve] WARNING: No API key provided.", file=sys.stderr)
        print("Pass --api-key YOUR_KEY or set $env:QBRAID_API_KEY='YOUR_KEY'", file=sys.stderr)

    print("==================================================================")
    print("  qBraid QPU Job Retriever & Evaluator")
    print("==================================================================")

    if args.job_ids:
        job_records = [{"job_id": jid, "device": "iqm_garnet", "status": "COMPLETED"} for jid in args.job_ids]
    else:
        job_records = fetch_recent_jobs(limit=args.limit, api_key=api_key)

    completed_jobs = []
    for rec in job_records:
        status_str = str(rec.get("status", "")).upper()
        if "COMPLETED" in status_str or "DONE" in status_str or "SUCCESS" in status_str or args.job_ids:
            retrieved = retrieve_job_counts(rec, api_key=api_key)
            if "counts" in retrieved:
                z_exp = compute_z_expectation(retrieved["counts"])
                retrieved["z_expectations"] = z_exp
                completed_jobs.append(retrieved)

    print(f"\n[find_and_retrieve] Successfully retrieved results for {len(completed_jobs)} completed jobs.")

    iqm_payload = []
    rigetti_payload = []

    for cj in completed_jobs:
        entry = {
            "job_id": cj["job_id"],
            "status": cj.get("status"),
            "device": cj.get("device"),
            "created_at": cj.get("created_at"),
            "shots": cj.get("shots"),
            "counts_summary": {
                "total_shots": cj.get("shots"),
                "unique_bitstrings": len(cj.get("counts", {})),
            },
            "counts": cj.get("counts"),
            "z_expectations": cj.get("z_expectations"),
        }
        device_str = str(cj.get("device", "")).lower()
        if "rigetti" in device_str or "cepheus" in device_str:
            rigetti_payload.append(entry)
        else:
            iqm_payload.append(entry)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if iqm_payload:
        out_json = args.output_dir / "garnet_completed_jobs.json"
        out_json.write_text(
            json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(), "total_retrieved": len(iqm_payload), "jobs": iqm_payload}, indent=2),
            encoding="utf-8",
        )
        print(f"[find_and_retrieve] Saved IQM Garnet job artifact to: {out_json}")

    if rigetti_payload:
        out_json = args.output_dir / "rigetti_completed_jobs.json"
        out_json.write_text(
            json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(), "total_retrieved": len(rigetti_payload), "jobs": rigetti_payload}, indent=2),
            encoding="utf-8",
        )
        print(f"[find_and_retrieve] Saved Rigetti Cepheus-1 job artifact to: {out_json}")

    try:
        from scripts.evaluate_recurrent_hardware import build_multi_qpu_summary
        build_multi_qpu_summary(args.output_dir)
    except Exception as exc:
        print(f"[find_and_retrieve] Note: Multi-QPU evaluation step skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
