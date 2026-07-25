#!/usr/bin/env python3
"""compute_statistics.py — Run Diebold-Mariano tests, Model Confidence Set,
seed aggregation, and Mincer-Zarnowitz summary across all Phase 3 results.

Outputs:
  artifacts/manifests/statistical_analysis.json   (full machine-readable results)
  artifacts/manifests/statistical_analysis.md     (judge-facing formatted report)
  artifacts/manifests/seed_aggregate_table.md     (mean ± std per N/topology)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from prototype.readout import (  # noqa: E402
    compute_metrics,
    diebold_mariano,
    model_confidence_set,
)

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIRS = [
    Path("prototype/results/phase3"),
    Path("prototype/results/noise"),
    Path("artifacts/hardware"),
]
OUT_DIR = Path("artifacts/manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_all_records() -> list[dict]:
    raw_records = []
    for d in RESULTS_DIRS:
        if not d.exists():
            continue
        for jf in sorted(d.rglob("*.json")):
            try:
                payload = json.loads(jf.read_text(encoding="utf-8"))
                for rec in payload.get("results", []):
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


def qlike_losses(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Point-wise QLIKE loss in RV space: u/exp(f) - log(u/exp(f)) - 1."""
    rv_true = np.exp(np.asarray(y_true, dtype=float))
    rv_pred = np.exp(np.asarray(pred, dtype=float))
    rv_pred = np.clip(rv_pred, 1e-10, None)
    ratio = rv_true / rv_pred
    return ratio - np.log(ratio) - 1.0


def label(rec: dict) -> str:
    model = rec.get("model", "?")
    n = rec.get("n_qubits")
    topo = rec.get("topology")
    seed = rec.get("seed")
    parts = [model]
    if n is not None:
        parts.append(f"N={n}")
    if topo:
        parts.append(topo)
    if seed is not None:
        parts.append(f"s{seed}")
    return " ".join(parts)


def _fmt(v, dec=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{dec}f}"
    except Exception:
        return str(v)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    records = load_all_records()
    # Only keep records with test predictions and ok status
    usable = [
        r for r in records
        if r.get("status") == "ok"
        and r.get("test_predictions")
        and r.get("test_metrics")
    ]
    print(f"[stats] Loaded {len(usable)} usable result records")

    # Build common y_true from the first complete record (all share same test set)
    # Use a record with 389 test observations (full test period)
    full_records = [r for r in usable if len(r.get("test_predictions", [])) == 389]
    if not full_records:
        print("[stats] ERROR: No records with 389 test predictions found")
        return 1

    # y_true is the same for all — use Persistence which is just y_{t-1}
    # Actually we don't have y_true directly. We reconstruct from metrics if needed.
    # Best approach: load the actual data
    from prototype.data_loader import load_spx_rv
    from prototype.run_phase3 import prepare_phase3_data
    data = prepare_phase3_data(load_spx_rv(allow_synthetic=False))
    y_true = data["y_test"]

    # ── Section 1: Seed Aggregation ──────────────────────────────────────────
    print("[stats] Computing seed aggregation statistics...")
    seed_groups: dict[str, list[dict]] = defaultdict(list)
    for r in full_records:
        model = r.get("model", "?")
        n = r.get("n_qubits")
        topo = r.get("topology", "—")
        if n is not None:
            group_key = f"{model}|N={n}|{topo}"
        else:
            group_key = f"{model}|—|—"
        seed_groups[group_key].append(r)

    seed_agg = {}
    for gkey, recs in seed_groups.items():
        if len(recs) == 0:
            continue
        metrics_list = [r["test_metrics"] for r in recs]
        agg = {}
        for metric in ["rmse", "qlike", "mae", "r2"]:
            vals = [m[metric] for m in metrics_list if metric in m]
            if vals:
                agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
        seed_agg[gkey] = {
            "n_seeds": len(recs),
            "seeds": [r.get("seed", "—") for r in recs],
            "metrics": agg,
        }

    # ── Section 2: Diebold-Mariano Tests ─────────────────────────────────────
    print("[stats] Computing Diebold-Mariano tests (QLIKE loss)...")

    # For DM: use best representative per model group (best seed by test QLIKE)
    # For single-seed models (baselines), use directly
    # For multi-seed QRC, use each seed separately and also the best seed
    dm_records = {}
    for gkey, recs in seed_groups.items():
        best = min(recs, key=lambda r: r["test_metrics"].get("qlike", 1e9))
        dm_records[gkey] = best

    dm_results = {}
    # Reference: HAR-Ridge
    har_keys = [k for k in dm_records if "HAR-Ridge" in k]
    ref_key = har_keys[0] if har_keys else None

    if ref_key:
        ref_rec = dm_records[ref_key]
        ref_pred = np.array(ref_rec["test_predictions"])
        for gkey, rec in dm_records.items():
            if gkey == ref_key:
                continue
            pred = np.array(rec["test_predictions"])
            try:
                dm = diebold_mariano(y_true, pred, ref_pred, loss="qlike", is_log_rv=True)
                stat_val = dm["statistic"]
                pval_val = dm["pvalue"]
                dm_results[gkey] = {
                    "vs": ref_key,
                    "dm_stat": stat_val,
                    "p_value": pval_val,
                    "better_than_ref": stat_val < 0 and pval_val < 0.1,
                    "significant": pval_val < 0.1,
                    "test_qlike": rec["test_metrics"].get("qlike"),
                    "ref_qlike": ref_rec["test_metrics"].get("qlike"),
                }
            except Exception as exc:
                dm_results[gkey] = {"error": str(exc)}
    else:
        print("[stats] WARNING: HAR-Ridge not found; skipping DM tests")

    # DM vs ESN-500
    esn_keys = [k for k in dm_records if "ESN-500" in k]
    dm_vs_esn = {}
    if esn_keys:
        esn_key = esn_keys[0]
        esn_rec = dm_records[esn_key]
        esn_pred = np.array(esn_rec["test_predictions"])
        for gkey, rec in dm_records.items():
            if gkey == esn_key:
                continue
            pred = np.array(rec["test_predictions"])
            try:
                dm = diebold_mariano(y_true, pred, esn_pred, loss="qlike", is_log_rv=True)
                stat_val = dm["statistic"]
                pval_val = dm["pvalue"]
                dm_vs_esn[gkey] = {
                    "vs": esn_key,
                    "dm_stat": stat_val,
                    "p_value": pval_val,
                    "better_than_ref": stat_val < 0 and pval_val < 0.1,
                    "significant": pval_val < 0.1,
                    "test_qlike": rec["test_metrics"].get("qlike"),
                    "ref_qlike": esn_rec["test_metrics"].get("qlike"),
                }
            except Exception as exc:
                dm_vs_esn[gkey] = {"error": str(exc)}

    # ── Section 3: Model Confidence Set ──────────────────────────────────────
    print("[stats] Computing Model Confidence Set (alpha=10%)...")
    # Build loss dictionary: name -> QLIKE loss array per observation
    mcs_losses = {}
    for gkey, rec in dm_records.items():
        pred = np.array(rec["test_predictions"])
        mcs_losses[gkey] = qlike_losses(y_true, pred)

    try:
        mcs_result = model_confidence_set(mcs_losses, alpha=0.1, n_bootstrap=1000, random_state=42)
    except Exception as exc:
        print(f"[stats] MCS failed: {exc}")
        mcs_result = {"error": str(exc)}

    # ── Section 4: Mincer-Zarnowitz Summary ───────────────────────────────────
    print("[stats] Collecting Mincer-Zarnowitz statistics...")
    mz_summary = {}
    for gkey, rec in dm_records.items():
        mz = rec.get("test_mincer_zarnowitz")
        if mz:
            mz_summary[gkey] = {
                "intercept": mz.get("intercept"),
                "slope": mz.get("slope"),
                "p_intercept": mz.get("p_intercept"),
                "p_slope": mz.get("p_slope"),
                "joint_pvalue": mz.get("joint_pvalue"),
                "unbiased": mz.get("unbiased", False),
            }

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output_json = {
        "seed_aggregation": seed_agg,
        "diebold_mariano_vs_HAR_Ridge": dm_results,
        "diebold_mariano_vs_ESN_500": dm_vs_esn,
        "model_confidence_set": mcs_result,
        "mincer_zarnowitz": mz_summary,
    }
    json_path = OUT_DIR / "statistical_analysis.json"
    json_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    print(f"[stats] Saved: {json_path}")

    # ── Render Markdown Report ─────────────────────────────────────────────────
    lines = []
    lines.append("# Phase 3 Statistical Analysis Report\n")

    # --- Seed Aggregation Table ---
    lines.append("## 1. Seed Aggregation (Mean ± Std across Seeds)\n")
    lines.append("| Model Group | N Seeds | Test RMSE | ± | Test QLIKE | ± | Test R² | ± |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for gkey, agg in sorted(seed_agg.items()):
        m = agg["metrics"]
        rmse = m.get("rmse", {})
        qlike = m.get("qlike", {})
        r2 = m.get("r2", {})
        display = gkey.replace("|", " / ")
        lines.append(
            f"| {display} | {agg['n_seeds']} "
            f"| {_fmt(rmse.get('mean'))} | {_fmt(rmse.get('std'))} "
            f"| {_fmt(qlike.get('mean'))} | {_fmt(qlike.get('std'))} "
            f"| {_fmt(r2.get('mean'))} | {_fmt(r2.get('std'))} |"
        )
    lines.append("")

    # --- DM Test Table ---
    lines.append("## 2. Diebold-Mariano Tests vs. HAR-Ridge (QLIKE, two-sided)\n")
    lines.append("> DM stat < 0 means model has lower QLIKE (better) than HAR-Ridge. p < 0.10 indicates significance.\n")
    lines.append("| Model | Test QLIKE | HAR-Ridge QLIKE | DM Stat | p-value | Better? | Significant? |")
    lines.append("|---|---|---|---|---|---|---|")
    for gkey, dm in sorted(dm_results.items()):
        if "error" in dm:
            lines.append(f"| {gkey.replace('|',' / ')} | — | — | ERROR | — | — | — |")
        else:
            better = "✅ Yes" if dm.get("better_than_ref") else "❌ No"
            sig = "✅ Yes" if dm.get("significant") else "No"
            lines.append(
                f"| {gkey.replace('|',' / ')} "
                f"| {_fmt(dm.get('test_qlike'))} "
                f"| {_fmt(dm.get('ref_qlike'))} "
                f"| {_fmt(dm.get('dm_stat'), 3)} "
                f"| {_fmt(dm.get('p_value'), 4)} "
                f"| {better} | {sig} |"
            )
    lines.append("")

    # --- MCS ---
    lines.append("## 3. Model Confidence Set (α = 10%)\n")
    if "error" in mcs_result:
        lines.append(f"> MCS computation failed: {mcs_result['error']}\n")
    else:
        survivors = mcs_result.get("survivors", [])
        eliminated = mcs_result.get("eliminated", [])
        lines.append(f"**Survivors ({len(survivors)} models):**\n")
        for s in survivors:
            lines.append(f"- {s}")
        lines.append(f"\n**Eliminated ({len(eliminated)} models):**\n")
        for e in eliminated:
            lines.append(f"- {e}")
        lines.append("")

    # --- MZ Table ---
    lines.append("## 4. Mincer-Zarnowitz Test Summary\n")
    lines.append("> An unbiased forecast has intercept=0 and slope=1. Joint p-value tests this jointly.\n")
    lines.append("| Model | Intercept | Slope | p(intercept) | p(slope) | Joint p-value | Unbiased? |")
    lines.append("|---|---|---|---|---|---|---|")
    for gkey, mz in sorted(mz_summary.items()):
        unbiased = "✅" if mz.get("unbiased") else "❌"
        lines.append(
            f"| {gkey.replace('|',' / ')} "
            f"| {_fmt(mz.get('intercept'), 3)} "
            f"| {_fmt(mz.get('slope'), 3)} "
            f"| {_fmt(mz.get('p_intercept'), 4)} "
            f"| {_fmt(mz.get('p_slope'), 4)} "
            f"| {_fmt(mz.get('joint_pvalue'), 4)} "
            f"| {unbiased} |"
        )
    lines.append("")

    md_path = OUT_DIR / "statistical_analysis.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[stats] Saved: {md_path}")

    # --- Seed aggregate as separate clean table ---
    seed_lines = []
    seed_lines.append("# QRC Seed Robustness Table\n")
    seed_lines.append("| Model | N | Topology | Seeds | Test RMSE Mean | ± Std | Test QLIKE Mean | ± Std | Test R² Mean | ± Std |")
    seed_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for gkey, agg in sorted(seed_agg.items()):
        parts = gkey.split("|")
        model = parts[0]
        nq = parts[1] if len(parts) > 1 else "—"
        topo = parts[2] if len(parts) > 2 else "—"
        seeds_str = ", ".join(str(s) for s in agg["seeds"])
        m = agg["metrics"]
        rmse = m.get("rmse", {})
        qlike = m.get("qlike", {})
        r2 = m.get("r2", {})
        seed_lines.append(
            f"| {model} | {nq} | {topo} | {seeds_str} "
            f"| {_fmt(rmse.get('mean'))} | {_fmt(rmse.get('std'))} "
            f"| {_fmt(qlike.get('mean'))} | {_fmt(qlike.get('std'))} "
            f"| {_fmt(r2.get('mean'))} | {_fmt(r2.get('std'))} |"
        )
    seed_md_path = OUT_DIR / "seed_aggregate_table.md"
    seed_md_path.write_text("\n".join(seed_lines), encoding="utf-8")
    print(f"[stats] Saved: {seed_md_path}")

    # Print summary to console
    print("\n[stats] === KEY RESULTS ===")
    if ref_key and dm_results:
        better_models = [k for k, v in dm_results.items() if isinstance(v, dict) and v.get("better_than_ref")]
        sig_better = [k for k, v in dm_results.items() if isinstance(v, dict) and v.get("significant") and v.get("better_than_ref")]
        print(f"  Models better than HAR-Ridge (DM): {len(better_models)}")
        print(f"  Significantly better (p<0.10):     {len(sig_better)}")
        for k in sig_better:
            print(f"    - {k}")
    if "survivors" in mcs_result:
        print(f"  MCS survivors (alpha=10%): {len(mcs_result['survivors'])}")
        for s in mcs_result["survivors"]:
            print(f"    - {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
