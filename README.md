# VolQRC — Volatility Quantum Reservoir Computing for GIC 2026

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/deploy-to-qbraid.svg)](https://account.qbraid.com/launch?gitHubUrl=https://github.com/shubh-200/onion-qrc-volatility-forecasting)

* **Team Name:** VolQRC Team
* **Project Title:** VolQRC: Volatility Quantum Reservoir Computing for SPX Realized Variance Forecasting
* **Challenge Track:** GIC 2026 Phase 3 — Open Innovation / Quantum Applications

---

## Executive Summary

VolQRC implements a temporal **Quantum Reservoir Computer (QRC)** using a multi-band **Onion Qubit Allocation** and transverse Ising Trotter dynamics to forecast financial realized volatility ($RV_{t+1}$) on S&P 500 market data. It encodes short-, mid-, and long-band realized volatility features into separate qubit bands of a transverse-field Ising chain, and extracts cross-correlation observables ($\langle Z_i \rangle, \langle Z_i Z_j \rangle$) through a causal, volatility-regime-aware Ridge readout layer.

This repository provides full end-to-end reproducibility for:
1. **Classical & Econometric Baselines:** Persistence, HAR-Ridge, ESN-210, ESN-500, RandomFeatureRidge-210, GARCH(1,1), EGARCH(1,1,1), and PyTorch LSTM.
2. **Quantum Simulator Benchmarks:** Noiseless statevector simulations across $N \in \{5, 10, 15\}$ qubits, ring vs. fully connected topologies, and 3 random seeds.
3. **Phase 3 Ablation Studies:** Observable-Order ($\langle Z_i \rangle$ vs. $\langle Z_i Z_j \rangle$), Regime-Gating (No Signal vs. Causal vs. Oracle), and Quantum Regime Kernel (Linear vs. RBF vs. IQP Quantum Kernel).
4. **Physical QPU Hardware Runs:** 20-qubit execution validation on **IQM Garnet** physical quantum processor (via qBraid).
5. **Statistical Diagnostics:** Diebold-Mariano QLIKE loss tests, Mincer-Zarnowitz regressions with HAC covariance, seed aggregation, and Model Confidence Sets (MCS).

---

## Headline Performance Table

Below are the benchmark results evaluated out-of-sample on $N_\text{obs} = 389$ test days:

| Model Class | Model | N Qubits | Topology | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Baseline** | **HAR-Ridge** | — | — | **0.3148** | **0.0537** | **0.2529** | **0.7118** | ok |
| **Naive Baseline** | **Persistence** | — | — | 0.3320 | 0.0586 | 0.2585 | 0.6794 | ok |
| **Classical Reservoir** | **ESN-210** | — | — | 0.3335 | 0.0581 | 0.2675 | 0.6764 | ok |
| **Quantum Reservoir** | **OnionQRC** | **15** | **ring** | **0.3460** | **0.0591** | **0.2835** | **0.6518** | ok |
| **Quantum Reservoir** | **OnionQRC** | 10 | fully_connected | 0.3524 | 0.0606 | 0.2887 | 0.6387 | ok |
| **Quantum Reservoir** | **OnionQRC** | 10 | ring | 0.3592 | 0.0624 | 0.2955 | 0.6246 | ok |
| **Classical Random** | **RandomFeatureRidge-210** | — | — | 0.3607 | 0.0635 | 0.2892 | 0.6216 | ok |
| **Classical Reservoir** | **ESN-500** | — | — | 0.3666 | 0.0690 | 0.2971 | 0.6091 | ok |
| **Quantum Reservoir** | **OnionQRC** | 5 | ring | 0.3746 | 0.0663 | 0.3063 | 0.5917 | ok |
| **Deep Learning** | **LSTM (PyTorch)** | — | — | 0.6565 | 0.1973 | 0.5607 | -0.2538 | ok |
| **Econometric** | **GARCH(1,1)** | — | — | 4.5988 | 137.0708 | 4.4923 | -60.5255 | ok |
| **Econometric** | **EGARCH(1,1,1)** | — | — | 38.9553 | 2.67e+08 | 7.5407 | -4413.75 | ok |
| **Hardware QPU** | **OnionQRC QPU** | 15 | ring | 2.3208 | 1.4033 | 2.2972 | -13.9400 | ok |
| **Hardware QPU** | **OnionQRC QPU** | 20 | ring | 1.7739 | 0.9286 | 1.7428 | -7.7286 | ok |

---

## Repository Layout

```text
gic/
├── run_all.py                     # Master one-command reproduction runner for judges
├── pyproject.toml                 # Package setup and build specification
├── requirements.txt               # Complete dependency list (including PyTorch for LSTM)
├── src/volqrc/                    # Core VolQRC package
│   ├── __init__.py                # Package API exports
│   ├── data.py                    # Causal HAR data loader, sliding windows, train/val/test splits
│   ├── circuits.py                # Onion allocation, Trotter Ising evolution, Z/ZZ observables
│   ├── readout.py                 # Ridge readout, cross-fitted causal regime gating, IQP kernel
│   ├── baselines.py               # HAR, ESN, GARCH, EGARCH, LSTM, Persistence, RandomFeature
│   ├── metrics.py                 # RMSE, MAE, QLIKE, R², Mincer-Zarnowitz, Diebold-Mariano, MCS
│   └── backends/                  # Execution backends (statevector, noisy Aer, qBraid QPU)
├── prototype/                     # Underlying prototype modules & caching engine
├── scripts/                       # Executable CLI scripts
│   ├── prepare_data.py            # Prepares SPX dataset & verifies checksums
│   ├── run_baselines.py           # Runs all 8 classical baselines
│   ├── run_scaling.py             # Runs OnionQRC quantum simulator scaling (N=5,10,15)
│   ├── run_ablations.py           # Runs Phase 3 ablation studies
│   ├── run_noise.py               # Runs Aer simulator shot noise sweeps
│   ├── submit_qpu.py              # Submits physical 20-qubit jobs to qBraid QPU
│   ├── retrieve_qpu.py            # Retrieves QPU job outputs
│   ├── evaluate_hardware_results.py # Evaluates QPU results against simulator benchmarks
│   ├── compute_statistics.py      # Computes DM tests, MZ regressions, and seed stats
│   └── build_report.py            # Rebuilds summary markdown and CSV tables
├── tests/                         # Automated unit tests (13 tests verifying 9 causal rules)
├── configs/                       # Hyperparameter and QPU configuration files
│   ├── phase3.yaml                # Simulator experiment configuration
│   └── qpu_iqm.yaml               # IQM Garnet hardware budget & panel specs
└── artifacts/                     # Output manifests & hardware artifacts
    ├── manifests/                 # summary_table.md, ablations.md, statistical_analysis.md
    └── hardware/                  # Pre-saved IQM Garnet QPU job execution files
```

---

## Setup Instructions

### Environment Creation (qBraid or Local)
Recommended Python version: **Python 3.10 – 3.12**.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gic.git
cd gic

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## Reproduction Guide for Judges

### Mode A: One-Command Reproduction (Recommended)
Judges can run the entire pipeline end-to-end (unit tests, baselines, simulator scaling, ablations, QPU evaluations, statistics, and report rebuilding) using:

```bash
python run_all.py
```

### Mode B: Step-by-Step Pipeline Execution

```bash
# 1. Run Automated Unit Tests (Verifies 9 Causal & Verification Rules)
python -m pytest

# 2. Train & Evaluate Classical Baselines (HAR, ESN, GARCH, EGARCH, LSTM)
python scripts/run_baselines.py

# 3. Execute Quantum Simulator Scaling (N = 5, 10, 15 across seeds)
python scripts/run_scaling.py

# 4. Execute Phase 3 Ablation Studies (Observables, Regime Gating, IQP Kernel)
python scripts/run_ablations.py

# 5. Evaluate Physical 20-Qubit QPU Hardware Results (IQM Garnet)
python scripts/evaluate_hardware_results.py

# 6. Compute Statistical Diagnostics & Update Summary Reports
python scripts/compute_statistics.py
python scripts/build_report.py
```

---

## Expected Inputs and Outputs

* **Input Data:** `rv_dataset.csv` (Oxford-Man Realized Library S&P 500 realized variance dataset, split chronologically into 60% training, 20% validation, 20% test).
* **Output Manifests:**
  * `artifacts/manifests/summary_table.md` & `summary_table.csv`: Final headline result tables.
  * `artifacts/manifests/ablations.md` & `ablations.json`: Detailed ablation study reports.
  * `artifacts/manifests/statistical_analysis.md` & `statistical_analysis.json`: Diebold-Mariano tests, Mincer-Zarnowitz regressions, seed statistics, and Model Confidence Sets.
  * `artifacts/hardware/`: Archived physical QPU bitstring counts and expectation values from IQM Garnet.

---

## Hardware QPU Reproduction (No API Key Required for Judging)

To enable immediate judge evaluation without requiring judges to spend qBraid API credits:
* `python scripts/evaluate_hardware_results.py` automatically reads the pre-saved, verified hardware execution outputs archived in `artifacts/hardware/` (`hardware_n15_results.json` and `hardware_n20_results.json`).

*(Optional)* If you want to submit fresh jobs to a live QPU via qBraid:
```bash
export QBRAID_API_KEY="your_qbraid_api_key"
python scripts/submit_qpu.py --mode panel --device-id iqm_garnet --submit
python scripts/retrieve_qpu.py
```

---

## Known Limitations and Assumptions

1. **Classically Intractable $N=20$ Simulator:** Full time-series statevector simulation of $N=20$ fully connected reservoirs over 2,500 continuous daily time steps is classically intractable. Therefore, $N=20$ is evaluated via a pre-registered 24-date physical QPU hardware panel on **IQM Garnet**.
2. **QPU Execution:** Hardware results rely on pre-archived execution manifests in `artifacts/hardware/` so judges do not require active qBraid hardware credits.
3. **Causal Alignment:** All model features strictly use data available through day $t$ to forecast day $t+1$ realized variance ($RV_{t+1}$). Regime thresholds and scalers are fitted on training data only.

---

## Citation

```bibtex
@article{volqrc2026,
  title={VolQRC: Volatility Quantum Reservoir Computing with Onion Allocation for SPX Realized Variance Forecasting},
  author={VolQRC Team},
  journal={GIC 2026 Phase 3 Submission},
  year={2026}
}
```
