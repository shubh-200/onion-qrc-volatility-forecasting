#!/usr/bin/env python3
"""run_ablations.py — Execute required Phase 3 ablation studies:
  1. Observable-Order Ablation (Singles vs. Singles+Pairs)
  2. Regime-Gating Ablation (No Regime vs. Causal Regime vs. Oracle Regime)
  3. Quantum Regime Kernel Ablation (Linear SVM vs. RBF SVM vs. IQP Quantum Kernel)

Outputs:
  artifacts/manifests/ablations.json
  artifacts/manifests/ablations.md
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
from prototype.readout import (  # noqa: E402
    VolQRCReadout,
    compute_metrics,
    QuantumKernelClassifier,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.svm import SVC  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def _fmt(v, dec=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{dec}f}"
    except Exception:
        return str(v)


def main(argv=None) -> int:
    out_dir = Path("artifacts/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[ablations] Loading SPX dataset...")
    data = prepare_phase3_data(load_spx_rv(allow_synthetic=False))

    results_dict = {}
    md_lines = []
    md_lines.append("# Phase 3 Ablation Studies Report\n")

    # ──────────────────────────────────────────────────────────────────────────
    # ABLATION 1: Observable-Order Ablation (Singles vs. Singles+Pairs)
    # ──────────────────────────────────────────────────────────────────────────
    print("[ablations] Running Ablation 1: Observable-Order (Singles vs. Singles+Pairs)...")
    obs_ablation = {}
    configs = [
        (5, "ring", 42),
        (10, "ring", 2026),
        (15, "ring", 2026),
    ]

    for n, topo, seed in configs:
        all_feat, _ = reservoir_features(data, n_qubits=n, topology=topo, seed=seed, cache_dir=CACHE_DIR)
        feats = _split_contiguous(all_feat, data)

        # 1A. Singles Only (first n columns: Z_i)
        X_tr_singles = feats["train"][:, :n]
        X_val_singles = feats["val"][:, :n]
        X_test_singles = feats["test"][:, :n]

        r_singles = VolQRCReadout(ridge_alpha=1.0, use_regime=True, regime_cv_splits=5, classifier_pca_components=min(8, n))
        r_singles.fit(X_tr_singles, data["y_train"], data["regime_train"], data["X_train"][:, :3])
        pred_singles = r_singles.predict(X_test_singles, data["X_test"][:, :3])
        m_singles = compute_metrics(data["y_test"], pred_singles, is_log_rv=True)

        # 1B. Full (Singles + Pairs)
        r_full = VolQRCReadout(ridge_alpha=1.0, use_regime=True, regime_cv_splits=5, classifier_pca_components=min(8, feats["train"].shape[1]))
        r_full.fit(feats["train"], data["y_train"], data["regime_train"], data["X_train"][:, :3])
        pred_full = r_full.predict(feats["test"], data["X_test"][:, :3])
        m_full = compute_metrics(data["y_test"], pred_full, is_log_rv=True)

        key = f"N={n}_{topo}_seed{seed}"
        obs_ablation[key] = {
            "n_qubits": n,
            "singles_only": m_singles,
            "singles_and_pairs": m_full,
            "r2_gain": m_full["r2"] - m_singles["r2"],
            "qlike_reduction": m_singles["qlike"] - m_full["qlike"],
        }

    results_dict["observable_order_ablation"] = obs_ablation

    md_lines.append("## 1. Observable-Order Ablation Study (§4.4 / §12 Table 5)")
    md_lines.append("> Evaluates predictions using single-qubit expectation values $\\langle Z_i \\rangle$ vs. combined single + pair correlations $\\langle Z_i Z_j \\rangle$.\n")
    md_lines.append("| Config | Feature Set | Features | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ | R² Gain |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for key, item in obs_ablation.items():
        n = item["n_qubits"]
        ms = item["singles_only"]
        mf = item["singles_and_pairs"]
        gain = item["r2_gain"]
        md_lines.append(f"| {key} | Singles Only ($Z_i$) | {n} | {_fmt(ms['rmse'])} | {_fmt(ms['qlike'])} | {_fmt(ms['mae'])} | {_fmt(ms['r2'])} | — |")
        md_lines.append(f"| {key} | Singles + Pairs ($Z_i, Z_i Z_j$) | {n + n*(n-1)//2} | {_fmt(mf['rmse'])} | {_fmt(mf['qlike'])} | {_fmt(mf['mae'])} | {_fmt(mf['r2'])} | **+{_fmt(gain)}** |")
    md_lines.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # ABLATION 2: Regime-Gating Ablation (No Regime vs. Causal vs. Oracle)
    # ──────────────────────────────────────────────────────────────────────────
    print("[ablations] Running Ablation 2: Regime-Gating (No Regime vs. Causal vs. Oracle)...")
    all_feat_15, _ = reservoir_features(data, n_qubits=15, topology="ring", seed=2026, cache_dir=CACHE_DIR)
    feats_15 = _split_contiguous(all_feat_15, data)

    # 2A. No Regime
    r_no_reg = VolQRCReadout(ridge_alpha=1.0, use_regime=False)
    r_no_reg.fit(feats_15["train"], data["y_train"], X_classical=data["X_train"][:, :3])
    pred_no_reg = r_no_reg.predict(feats_15["test"], data["X_test"][:, :3])
    m_no_reg = compute_metrics(data["y_test"], pred_no_reg, is_log_rv=True)

    # 2B. Causal Regime
    r_causal = VolQRCReadout(ridge_alpha=1.0, use_regime=True, regime_cv_splits=5)
    r_causal.fit(feats_15["train"], data["y_train"], data["regime_train"], data["X_train"][:, :3])
    pred_causal = r_causal.predict(feats_15["test"], data["X_test"][:, :3])
    m_causal = compute_metrics(data["y_test"], pred_causal, is_log_rv=True)

    # 2C. Oracle Regime (upper bound using true test regime labels)
    # Fit individual Ridge models on true training regime subsets and predict test using true test regime labels
    unique_regimes = np.unique(data["regime_train"])
    oracle_models = {}
    for reg in unique_regimes:
        idx_tr = np.where(data["regime_train"] == reg)[0]
        X_reg_tr = np.hstack([feats_15["train"][idx_tr], data["X_train"][idx_tr, :3]])
        y_reg_tr = data["y_train"][idx_tr]
        from prototype.readout import RidgeReadout
        rr = RidgeReadout(alpha=1.0)
        rr.fit(X_reg_tr, y_reg_tr)
        oracle_models[reg] = rr

    pred_oracle = np.zeros_like(data["y_test"])
    for reg in unique_regimes:
        idx_te = np.where(data["regime_test"] == reg)[0]
        if len(idx_te) > 0:
            X_reg_te = np.hstack([feats_15["test"][idx_te], data["X_test"][idx_te, :3]])
            pred_oracle[idx_te] = oracle_models[reg].predict(X_reg_te)

    m_oracle = compute_metrics(data["y_test"], pred_oracle, is_log_rv=True)

    regime_ablation = {
        "no_regime": m_no_reg,
        "causal_predicted_regime": m_causal,
        "oracle_regime_upper_bound": m_oracle,
    }
    results_dict["regime_gating_ablation"] = regime_ablation

    md_lines.append("## 2. Regime-Gating Ablation Study (§3.3 / §12 Table 4)")
    md_lines.append("> Evaluates OnionQRC N=15 across no regime gating, causally predicted RBF regime gating, and oracle regime gating.\n")
    md_lines.append("| Gating Mode | Target Leakage? | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ |")
    md_lines.append("|---|---|---|---|---|---|")
    md_lines.append(f"| No Regime Signal | None | {_fmt(m_no_reg['rmse'])} | {_fmt(m_no_reg['qlike'])} | {_fmt(m_no_reg['mae'])} | {_fmt(m_no_reg['r2'])} |")
    md_lines.append(f"| **Causally Predicted Regime (Production)** | None | **{_fmt(m_causal['rmse'])}** | **{_fmt(m_causal['qlike'])}** | **{_fmt(m_causal['mae'])}** | **{_fmt(m_causal['r2'])}** |")
    md_lines.append(f"| Oracle Regime (Upper Bound) | Yes (True $RV_{{t+1}}$ label) | {_fmt(m_oracle['rmse'])} | {_fmt(m_oracle['qlike'])} | {_fmt(m_oracle['mae'])} | {_fmt(m_oracle['r2'])} |")
    md_lines.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # ABLATION 3: Quantum Regime Kernel vs. Classical Kernels (§8 / §12 Table 7)
    # ──────────────────────────────────────────────────────────────────────────
    print("[ablations] Running Ablation 3: Quantum Regime Kernel (IQP vs. Linear vs. RBF)...")
    all_feat_10, _ = reservoir_features(data, n_qubits=10, topology="ring", seed=42, cache_dir=CACHE_DIR)
    feats_10 = _split_contiguous(all_feat_10, data)

    # Standardize & PCA reduce quantum observables to 4 components for fast kernel evaluation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(feats_10["train"])
    X_te_sc = scaler.transform(feats_10["test"])

    pca = PCA(n_components=4, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)

    y_tr_clf = data["regime_train"]
    y_te_clf = data["regime_test"]

    # 3A. Linear SVM
    clf_lin = SVC(kernel="linear", random_state=42)
    clf_lin.fit(X_tr_pca, y_tr_clf)
    p_lin = clf_lin.predict(X_te_pca)

    # 3B. RBF SVM
    clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    clf_rbf.fit(X_tr_pca, y_tr_clf)
    p_rbf = clf_rbf.predict(X_te_pca)

    # 3C. IQP Quantum Kernel SVM
    clf_iqp = QuantumKernelClassifier(n_qubits=4, C=1.0)
    clf_iqp.fit(X_tr_pca, y_tr_clf)
    p_iqp = clf_iqp.predict(X_te_pca)

    def _eval_clf(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        return {
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "macro_f1": float(f1),
            "macro_precision": float(p),
            "macro_recall": float(r),
        }

    kernel_ablation = {
        "linear_svm": _eval_clf(y_te_clf, p_lin),
        "rbf_svm": _eval_clf(y_te_clf, p_rbf),
        "iqp_quantum_kernel": _eval_clf(y_te_clf, p_iqp),
    }
    results_dict["regime_kernel_ablation"] = kernel_ablation

    md_lines.append("## 3. Quantum Regime Kernel Ablation Study (§8 / §12 Table 7)")
    md_lines.append("> Compares Linear SVM, RBF SVM, and IQP Feature-Map Quantum Kernel SVM on regime classification.\n")
    md_lines.append("| Kernel Classifier | Accuracy ↑ | Balanced Accuracy ↑ | Macro F1 ↑ | Macro Precision | Macro Recall |")
    md_lines.append("|---|---|---|---|---|---|")
    for kname, km in kernel_ablation.items():
        label_str = kname.replace("_", " ").title()
        if "Iqp" in label_str:
            label_str = "**IQP Quantum Kernel SVM**"
        md_lines.append(f"| {label_str} | {_fmt(km['accuracy'])} | {_fmt(km['balanced_accuracy'])} | {_fmt(km['macro_f1'])} | {_fmt(km['macro_precision'])} | {_fmt(km['macro_recall'])} |")
    md_lines.append("")

    # Save JSON & Markdown
    json_path = out_dir / "ablations.json"
    json_path.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
    print(f"[ablations] Saved: {json_path}")

    md_path = out_dir / "ablations.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ablations] Saved: {md_path}")

    print("[ablations] All ablation studies completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
