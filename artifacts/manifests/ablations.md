# Phase 3 Ablation Studies Report

## 1. Observable-Order Ablation Study (§4.4 / §12 Table 5)
> Evaluates predictions using single-qubit expectation values $\langle Z_i \rangle$ vs. combined single + pair correlations $\langle Z_i Z_j \rangle$.

| Config | Feature Set | Features | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ | R² Gain |
|---|---|---|---|---|---|---|---|
| N=5_ring_seed42 | Singles Only ($Z_i$) | 5 | 0.3673 | 0.0645 | 0.2995 | 0.6075 | — |
| N=5_ring_seed42 | Singles + Pairs ($Z_i, Z_i Z_j$) | 15 | 0.3746 | 0.0663 | 0.3063 | 0.5917 | **+-0.0158** |
| N=10_ring_seed2026 | Singles Only ($Z_i$) | 10 | 0.3787 | 0.0676 | 0.3089 | 0.5828 | — |
| N=10_ring_seed2026 | Singles + Pairs ($Z_i, Z_i Z_j$) | 55 | 0.3749 | 0.0664 | 0.3064 | 0.5912 | **+0.0084** |
| N=15_ring_seed2026 | Singles Only ($Z_i$) | 15 | 0.3662 | 0.0642 | 0.3003 | 0.6098 | — |
| N=15_ring_seed2026 | Singles + Pairs ($Z_i, Z_i Z_j$) | 120 | 0.3460 | 0.0591 | 0.2835 | 0.6518 | **+0.0420** |

## 2. Regime-Gating Ablation Study (§3.3 / §12 Table 4)
> Evaluates OnionQRC N=15 across no regime gating, causally predicted RBF regime gating, and oracle regime gating.

| Gating Mode | Target Leakage? | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ |
|---|---|---|---|---|---|
| No Regime Signal | None | 0.3479 | 0.0596 | 0.2849 | 0.6479 |
| **Causally Predicted Regime (Production)** | None | **0.3539** | **0.0618** | **0.2886** | **0.6356** |
| Oracle Regime (Upper Bound) | Yes (True $RV_{t+1}$ label) | 0.2613 | 0.0319 | 0.2054 | 0.8014 |

## 3. Quantum Regime Kernel Ablation Study (§8 / §12 Table 7)
> Compares Linear SVM, RBF SVM, and IQP Feature-Map Quantum Kernel SVM on regime classification.

| Kernel Classifier | Accuracy ↑ | Balanced Accuracy ↑ | Macro F1 ↑ | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| Linear Svm | 0.8123 | 0.6314 | 0.6472 | 0.6679 | 0.6314 |
| Rbf Svm | 0.7609 | 0.5936 | 0.5900 | 0.5866 | 0.5936 |
| **IQP Quantum Kernel SVM** | 0.6761 | 0.5710 | 0.5390 | 0.5326 | 0.5710 |
