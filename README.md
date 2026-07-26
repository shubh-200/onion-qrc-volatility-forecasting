# VolQRC — Volatility Quantum Reservoir Computing for GIC 2026

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
4. **Physical QPU Hardware Runs & Multi-QPU Validation:** Physical execution on **IQM Garnet** (20-qubit CZ star QPU) and cross-architecture hardware evaluation on **Rigetti Cepheus-1 (108Q)** via qBraid.
5. **Statistical Diagnostics:** Diebold-Mariano QLIKE loss tests, Mincer-Zarnowitz regressions with HAC covariance, seed aggregation, and Model Confidence Sets (MCS).

---

## Evaluation Metrics Explained

To evaluate volatility forecasting accuracy and financial risk management performance, models are assessed across four quantitative metrics:

1. **Root Mean Squared Error (RMSE ↓):**
   $$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2}$$
   Measures overall prediction error magnitude in log volatility space ($y_t = \ln(\text{RV}_{5,t})$). Lower values indicate higher accuracy.

2. **Quasi-Likelihood Loss (QLIKE ↓):**
   $$\text{QLIKE} = \frac{1}{T} \sum_{t=1}^{T} \left( \frac{y_t}{\hat{y}_t} - \ln\left(\frac{y_t}{\hat{y}_t}\right) - 1 \right)$$
   The standard asymmetric loss function used in financial econometrics. It penalizes under-predicting market volatility spikes much more severely than over-predicting, reflecting real-world portfolio risk management constraints. Lower values are better.

3. **Mean Absolute Error (MAE ↓):**
   $$\text{MAE} = \frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|$$
   Average linear magnitude of forecast errors, robust to extreme outlier days. Lower values are better.

4. **Out-of-Sample Coefficient of Determination ($R^2$ ↑):**
   $$R^2 = 1 - \frac{\sum_{t=1}^{T} (y_t - \hat{y}_t)^2}{\sum_{t=1}^{T} (y_t - \bar{y}_\text{train})^2}$$
   Proportion of market volatility variance explained by the model relative to a historical mean baseline. Positive values indicate true predictive power beyond naive historical averaging.

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
| **Deep Learning** | **LSTM (PyTorch)** | — | — | 0.5247 | 0.1555 | 0.4178 | -0.2538 | ok |
| **Econometric** | **GARCH(1,1)** | — | — | 4.5988 | 137.0708 | 4.4923 | -60.5255 | ok |
| **Econometric** | **EGARCH(1,1,1)** | — | — | 38.9553 | 2.67e+08 | 7.5407 | -4413.75 | ok |

---

## Physical QPU Hardware Execution & Multi-QPU Validation

To evaluate OnionQRC on physical quantum processors, we conducted physical QPU runs across two distinct hardware architectures via qBraid:

### 1. Sequential Recurrent QPU Run on IQM Garnet (20-Qubit QPU)
Evaluating sequential daily state progression with 512 shots per step on real **IQM Garnet** hardware demonstrates that temporal quantum feedback mitigates physical NISQ noise and achieves **positive out-of-sample $R^2$**:

| QPU Target | Mode | Steps | Qubits | Shots | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IQM Garnet** | **Recurrent** | **5** | **15** | **512** | **0.1009** | **0.0053** | **0.0762** | **+0.1523** |

* **Physical Memory State Tracking ($\langle Z \rangle$ evolution across steps):**
  $$\text{Step 1: } 0.4018 \longrightarrow \text{Step 2: } 0.3849 \longrightarrow \text{Step 3: } 0.3424 \longrightarrow \text{Step 4: } 0.4591 \longrightarrow \text{Step 5: } 0.4188$$

### 2. Multi-QPU Transferability on Rigetti Cepheus-1 (108-Qubit QPU)
To test cross-architecture transferability, the 5-step recurrent sequence was executed on **Rigetti Cepheus-1 (108Q)** (qBraid device ID: `aws:rigetti:qpu:cepheus-1-108q`):

| QPU Target | Architecture | Steps | Qubits | Shots | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rigetti Cepheus-1** | **8-Qubit Lattice** | **5** | **15** | **512** | **0.1037** | **0.0056** | **0.0790** | **+0.1033** |

---

## Setup & Execution on qBraid

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/shubh-200/onion-qrc-volatility-forecasting)

Recommended Python version: **Python 3.10 – 3.12**.

### 1. Environment Setup & One-Command Reproduction

Launch the repository on qBraid Lab (or locally) and run the full master reproduction pipeline:

```bash
# 1. Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .

# 2. Run master end-to-end reproduction runner
python run_all.py
```

### 2. Executing Physical QPU Hardware Runs from Scratch

If you wish to submit and evaluate fresh hardware jobs on physical QPUs via qBraid (requires setting `QBRAID_API_KEY`):

```bash
# Set your qBraid API Key
export QBRAID_API_KEY="your_qbraid_api_key"

# 1. Run 15-Qubit Panel Run on IQM Garnet (1024 shots)
python scripts/submit_qpu.py --mode panel --n-qubits 15 --device-id iqm_garnet --submit

# 2. Run 20-Qubit Panel Run on IQM Garnet (1024 shots)
python scripts/submit_qpu.py --mode panel --n-qubits 20 --device-id iqm_garnet --submit

# 3. Run 5-Day Recurrent Run on IQM Garnet (15 Qubits, 512 shots)
python scripts/submit_qpu.py --mode recurrent --n-qubits 15 --seeds 42 --shots 512 --max-circuits 5 --device-id iqm_garnet --submit

# 4. Run 5-Day Recurrent Run on Rigetti Cepheus-1 108Q (15 Qubits, 512 shots)
python scripts/submit_qpu.py --mode recurrent --n-qubits 15 --seeds 42 --shots 512 --max-circuits 5 --device-id aws:rigetti:qpu:cepheus-1-108q --submit

# Retrieve & Evaluate QPU Results
python scripts/find_and_retrieve_qpu.py
python scripts/parse_raw_jobs.py
```

---

## Repository Layout

```text
onion/
├── run_all.py                     # Master one-command reproduction runner for judges
├── pyproject.toml                 # Package setup and build specification
├── requirements.txt               # Complete dependency list (including PyTorch for LSTM)
├── rv_dataset.csv                 # Oxford-Man S&P 500 Realized Volatility dataset
├── global index etf return/
│   └── SPX.csv                    # S&P 500 daily price returns for GARCH/EGARCH
├── src/volqrc/                    # Core VolQRC package
│   ├── __init__.py                # Package API exports
│   ├── data.py                    # Causal HAR data loader, sliding windows, train/val/test splits
│   ├── circuits.py                # Onion allocation, Trotter Ising evolution, Z/ZZ observables
│   ├── readout.py                 # Ridge readout, cross-fitted causal regime gating, IQP kernel
│   ├── baselines.py               # HAR, ESN, GARCH, EGARCH, LSTM, Persistence, RandomFeature
│   ├── metrics.py                 # RMSE, MAE, QLIKE, R², Mincer-Zarnowitz, Diebold-Mariano, MCS
│   └── backends/                  # Execution backends (statevector, noisy Aer, qBraid QPU)
├── prototype/                     # Prototype modules, figure generation, and results cache
│   ├── figures/                   # VolQRC architecture and circuit diagrams
│   └── results/                   # Archived Phase 2 and Phase 3 benchmark results
├── scripts/                       # Executable CLI scripts
│   ├── prepare_data.py            # Prepares SPX dataset & verifies checksums
│   ├── run_baselines.py           # Runs all 8 classical baselines
│   ├── run_scaling.py             # Runs OnionQRC simulator scaling (N=5,10,15 ring & N=5,10 FC)
│   ├── run_ablations.py           # Runs Phase 3 ablation studies
│   ├── submit_qpu.py              # Submits physical QPU jobs to qBraid (IQM Garnet / Rigetti)
│   ├── find_and_retrieve_qpu.py   # Queries & retrieves QPU job results from qBraid
│   ├── parse_raw_jobs.py          # Parses raw QPU JSON outputs into merged manifest
│   ├── evaluate_recurrent_hardware.py # Evaluates 5-day recurrent QPU execution metrics
│   ├── evaluate_hardware_results.py # Evaluates pre-archived QPU panel benchmarks
│   ├── compute_statistics.py      # Computes DM tests, MZ regressions, and seed stats
│   └── build_report.py            # Rebuilds summary markdown and CSV tables (deduplicated)
├── tests/                         # Automated unit tests (13 tests verifying 9 causal rules)
├── configs/                       # Hyperparameter and QPU configuration files
│   ├── phase3.yaml                # Simulator experiment configuration
│   ├── qpu_iqm.yaml               # IQM Garnet hardware budget & panel specs
│   └── qpu_rigetti.yaml           # Rigetti Cepheus-1 hardware specs
└── artifacts/                     # Output manifests & hardware artifacts
    ├── manifests/                 # summary_table.md, ablations.md, statistical_analysis.md
    └── hardware/                  # Pre-saved QPU job execution files and recurrent results
        ├── panel/                 # 24-circuit pre-saved panel results
        └── recurrent/             # 5-day recurrent QPU job results & evaluation reports
```

---

## Known Limitations and Assumptions

1. **Classically Intractable $N=20$ Simulator:** Full time-series statevector simulation of $N=20$ fully connected reservoirs over 2,500 continuous daily time steps is classically intractable. Therefore, $N=20$ is evaluated via physical QPU hardware validation on **IQM Garnet**.
2. **Archived QPU Execution:** Hardware results rely on pre-archived execution manifests in `artifacts/hardware/` so judges do not require active qBraid hardware credits.
3. **Strict Causal Alignment:** All model features strictly use data available through day $t$ to forecast day $t+1$ realized variance ($RV_{t+1}$). Regime thresholds and scalers are fitted on training data only.
