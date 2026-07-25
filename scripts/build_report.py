#!/usr/bin/env python3
"""build_report.py — Assemble summary tables from all Phase 3 JSON artifacts.

Reads JSON results from:
  - prototype/results/phase3/  (baselines + simulator runs)
  - artifacts/simulator/       (scaled runs)
  - artifacts/hardware/        (QPU runs)

Outputs:
  - artifacts/manifests/summary_table.csv  (machine-readable)
  - artifacts/manifests/summary_table.md   (judge-facing Markdown table)

Usage:
    python scripts/build_report.py [--results-dirs DIR [DIR ...]]
                                    [--output-dir artifacts/manifests]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

_DEFAULT_RESULTS_DIRS = [
    "prototype/results",
    "artifacts/simulator",
    "artifacts/hardware",
]


def _collect_jsons(dirs: list[Path]) -> list[dict]:
    raw_records = []
    for d in dirs:
        if not d.exists():
            continue
        for jf in sorted(d.rglob("*.json")):
            try:
                payload = json.loads(jf.read_text(encoding="utf-8"))
                for rec in payload.get("results", [payload]):
                    rec["_source_file"] = str(jf)
                    raw_records.append(rec)
            except Exception:
                continue

    grouped = {}
    for rec in raw_records:
        model = rec.get("model") or rec.get("name")
        if not model:
            continue
        key = (
            model,
            str(rec.get("n_qubits", "—")),
            str(rec.get("topology", "—")),
            str(rec.get("seed", "—")),
            str(rec.get("backend", "—")),
        )
        if key not in grouped:
            grouped[key] = rec
        else:
            existing = grouped[key]
            if existing.get("status") == "skipped" and rec.get("status") == "ok":
                grouped[key] = rec
            elif rec.get("status") == existing.get("status"):
                grouped[key] = rec
    return list(grouped.values())


def _fmt(v, decimals=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _model_row(rec: dict) -> dict | None:
    model = rec.get("model") or rec.get("name")
    if not model:
        return None
    test = rec.get("test_metrics") or rec.get("test") or {}
    val = rec.get("validation_metrics") or rec.get("validation") or {}
    return {
        "Model": model,
        "N_qubits": rec.get("n_qubits", "—"),
        "Topology": rec.get("topology", "—"),
        "Seed": rec.get("seed", "—"),
        "Backend": rec.get("backend", "—"),
        "Val RMSE": _fmt(val.get("rmse")),
        "Val QLIKE": _fmt(val.get("qlike")),
        "Val MAE": _fmt(val.get("mae")),
        "Test RMSE": _fmt(test.get("rmse")),
        "Test QLIKE": _fmt(test.get("qlike")),
        "Test MAE": _fmt(test.get("mae")),
        "Test R2": _fmt(test.get("r2")),
        "N_obs": test.get("n_obs", "—"),
        "Runtime_s": _fmt(rec.get("runtime_seconds"), decimals=1),
        "Status": rec.get("status", "ok"),
    }


def _rows_to_md(rows: list[dict]) -> str:
    if not rows:
        return "_No results found yet._\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "—")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dirs", type=Path, nargs="+",
                   default=[Path(d) for d in _DEFAULT_RESULTS_DIRS])
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/manifests"))
    args = p.parse_args(argv)

    records = _collect_jsons(args.results_dirs)
    rows = [r for rec in records for r in [_model_row(rec)] if r is not None]
    rows.sort(key=lambda r: (str(r["Model"]), str(r["N_qubits"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    import csv
    csv_path = args.output_dir / "summary_table.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[build_report] CSV: {csv_path}  ({len(rows)} rows)")
    else:
        print("[build_report] No result records found yet.")

    # Markdown
    md_path = args.output_dir / "summary_table.md"
    md_path.write_text(_rows_to_md(rows), encoding="utf-8")
    print(f"[build_report] Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
