# VolQRC - Volatility Quantum Reservoir Computing for GIC 2026

* **Team Name:** Shubham Barge
* **Project Title:** VolQRC: Volatility Quantum Reservoir Computing for SPX Realized Variance Forecasting
* **Challenge Track:** Track A: Financial Volatility Prediction

---

## Executive Summary

VolQRC implements a temporal **Quantum Reservoir Computer (QRC)** using a multi-band **Onion Qubit Allocation** and transverse Ising Trotter dynamics to forecast financial realized volatility ($RV_{t+1}$) on S&P 500 market data. It encodes short-, mid-, and long-band realized volatility features into separate qubit bands of a transverse-field Ising chain, and extracts cross-correlation observables ($\langle Z_i \rangle, \langle Z_i Z_j \rangle$) through a causal, volatility-regime-aware Ridge readout layer.

This repository provides full end-to-end reproducibility for:
1. **Classical & Econometric Baselines:** Persistence, HAR-Ridge, ESN-210, ESN-500, RandomFeatureRidge-210, GARCH(1,1), EGARCH(1,1,1), and PyTorch LSTM.
2. **Quantum Simulator Benchmarks:** Noiseless statevector simulations across $N \in \{5, 10, 15, 20\}$ qubits, ring vs. fully connected topologies, and 3 random seeds.
3. **Phase 3 Ablation Studies:** Observable-Order ($\langle Z_i \rangle$ vs. $\langle Z_i Z_j \rangle$), Regime-Gating (No Signal vs. Causal vs. Oracle), and Quantum Regime Kernel (Linear vs. RBF vs. IQP Quantum Kernel).
4. **Physical QPU Hardware Runs & Multi-QPU Validation:** Physical execution on **IQM Garnet** (20-qubit CZ star QPU) and cross-architecture hardware evaluation on **Rigetti Cepheus-1 (108Q)** via qBraid.
5. **Statistical Diagnostics:** Diebold-Mariano QLIKE loss tests, Mincer-Zarnowitz regressions with HAC covariance, seed aggregation, and Model Confidence Sets (MCS).
6. **Recurrent NISQ Noise Filtering:** Demonstrates that sequential state re-encoding across daily time steps acts as an intrinsic physical noise filter, achieving positive out-of-sample $R^2$ on real QPU hardware (**IQM Garnet** $R^2 = +0.1523$, **Rigetti Cepheus-1** $R^2 = +0.1033$) without explicit error mitigation.

---

## Setup & Execution on qBraid

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="250">](https://account.qbraid.com?gitHubUrl=https://github.com/shubh-200/onion-qrc-volatility-forecasting)

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
python scripts/submit_qpu.py --mode panel --n-qubits 15 --device-id aws:iqm:qpu:garnet --submit

# 2. Run 20-Qubit Panel Run on IQM Garnet (1024 shots)
python scripts/submit_qpu.py --mode panel --n-qubits 20 --device-id aws:iqm:qpu:garnet --submit

# 3. Run 5-Day Recurrent Run on IQM Garnet (15 Qubits, 512 shots)
python scripts/submit_qpu.py --mode recurrent --n-qubits 15 --seeds 42 --shots 512 --max-circuits 5 --device-id aws:iqm:qpu:garnet --submit

# 4. Run 5-Day Recurrent Run on Rigetti Cepheus-1 108Q (15 Qubits, 512 shots)
python scripts/submit_qpu.py --mode recurrent --n-qubits 15 --seeds 42 --shots 512 --max-circuits 5 --device-id aws:rigetti:qpu:cepheus-1-108q --submit

# Retrieve & Evaluate QPU Results
python scripts/find_and_retrieve_qpu.py
python scripts/parse_raw_jobs.py
```

---

## Expected Inputs and Outputs

* **Input Data:**
  * `rv_dataset.csv`: S&P 500 5-minute realized variance dataset (60% train, 20% validation, 20% test).
  * `global index etf return/SPX.csv`: Daily S&P 500 log price returns for GARCH/EGARCH.
* **Output Manifests:**
  * `artifacts/manifests/summary_table.md` & `summary_table.csv`: Final deduplicated headline result tables.
  * `artifacts/manifests/ablations.md` & `ablations.json`: Detailed ablation study reports.
  * `artifacts/manifests/statistical_analysis.md` & `statistical_analysis.json`: Diebold-Mariano tests, Mincer-Zarnowitz regressions, seed statistics, and Model Confidence Sets (MCS).
  * `artifacts/hardware/recurrent/multi_qpu_recurrent_eval.md`: Multi-QPU cross-hardware evaluation report.

---

## Core Results

Below are the benchmark results evaluated out-of-sample across classical baselines, quantum simulator scaling, and physical hardware QPU runs:

| Category | Model / Execution | N Qubits | Topology | Backend / Mode | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Baseline** | **HAR-Ridge** | — | — | Linear | **0.3148** | **0.0537** | **0.2529** | **0.7118** | ok |
| **Naive Baseline** | **Persistence** | — | — | Baseline | 0.3320 | 0.0586 | 0.2585 | 0.6794 | ok |
| **Classical Reservoir**| **ESN-210** | — | — | Reservoir | 0.3335 | 0.0581 | 0.2675 | 0.6764 | ok |
| **Classical Random** | **RandomFeatureRidge-210**| — | — | Random Projection | 0.3607 | 0.0635 | 0.2892 | 0.6216 | ok |
| **Classical Reservoir**| **ESN-500** | — | — | Reservoir | 0.3666 | 0.0690 | 0.2971 | 0.6091 | ok |
| **Deep Learning** | **LSTM (PyTorch)** | — | — | PyTorch | 0.5247 | 0.1555 | 0.4178 | -0.2538 | ok |
| **Econometric** | **GARCH(1,1)** | — | — | Maximum Likelihood| 4.5988 | 137.0708 | 4.4923 | -60.5255 | ok |
| **Econometric** | **EGARCH(1,1,1)** | — | — | Maximum Likelihood| 38.9553 | 2.67e+08 | 7.5407 | -4413.75 | ok |
| **OnionQRC (N=5)** | **OnionQRC** | 5 | ring | Statevector Sim | 0.3746 | 0.0663 | 0.3063 | 0.5917 | ok |
| **OnionQRC (N=5)** | **OnionQRC** | 5 | fully_connected | Statevector Sim | 0.3783 | 0.0673 | 0.3096 | 0.5838 | ok |
| **OnionQRC (N=10)** | **OnionQRC** | 10 | ring | Statevector Sim | 0.3592 | 0.0624 | 0.2955 | 0.6246 | ok |
| **OnionQRC (N=10)** | **OnionQRC** | 10 | fully_connected | Statevector Sim | 0.3524 | 0.0606 | 0.2887 | 0.6387 | ok |
| **OnionQRC (N=15)** | **OnionQRC** | **15** | **ring** | **Statevector Sim** | **0.3460** | **0.0591** | **0.2835** | **0.6518** | ok |
| **OnionQRC (N=15)** | **OnionQRC Recurrent (5-days)** | **15** | **CZ Star** | **IQM Garnet QPU**| **0.1009** | **0.0053** | **0.0762** | **+0.1523**| ok |
| **OnionQRC (N=15)** | **OnionQRC Recurrent (5-days)** | **15** | **8Q Lattice** | **Rigetti Cepheus-1**| **0.1037** | **0.0056** | **0.0790** | **+0.1033**| ok |
| **OnionQRC (N=15)** | **OnionQRC Panel** | 15 | ring | IQM Garnet Panel | 2.3208 | 1.4033 | 2.2972 | -13.9400 | ok |
| **OnionQRC (N=20)** | **OnionQRC Panel** | 20 | ring | IQM Garnet Panel | 1.7739 | 0.9286 | 1.7428 | -7.7286 | ok |

---

## Evaluation Metrics Explained

Models are evaluated across point forecasting accuracy metrics and formal econometric hypothesis tests:

### 1. Point Forecast Metrics

* **Root Mean Squared Error (RMSE ↓):**

$$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2}$$

  Measures overall prediction error magnitude in log volatility space ($y_t = \ln(\text{RV}_{5,t})$). Lower is better.

* **Quasi-Likelihood Loss (QLIKE ↓):**

$$\text{QLIKE} = \frac{1}{T} \sum_{t=1}^{T} \left( \frac{y_t}{\hat{y}_t} - \ln\left(\frac{y_t}{\hat{y}_t}\right) - 1 \right)$$

  The standard asymmetric loss function in financial volatility forecasting. It heavily penalizes under-predicting volatility spikes, reflecting portfolio risk management constraints. Lower is better.

* **Mean Absolute Error (MAE ↓):**

$$\text{MAE} = \frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|$$

  Average linear magnitude of forecast errors, robust to extreme outlier days. Lower is better.

* **Out-of-Sample Coefficient of Determination ($R^2$ ↑):**

$$R^2 = 1 - \frac{\sum_{t=1}^{T} (y_t - \hat{y}_t)^2}{\sum_{t=1}^{T} (y_t - \bar{y}_{\text{train}})^2}$$

  Proportion of market volatility variance explained by the model relative to a historical mean baseline. Positive values indicate true predictive power beyond naive historical averaging.

### 2. Asymptotic Econometric & Statistical Hypothesis Tests

* **Diebold-Mariano (DM) Loss Differential Test:**

$$\text{DM} = \frac{\bar{d}}{\sqrt{\widehat{\text{V}}(\bar{d})}} \sim \mathcal{N}(0, 1)$$

  Tests whether the loss differential series $d_t = L(e_{1,t}) - L(e_{2,t})$ under QLIKE loss is significantly different from zero using a heteroskedasticity and autocorrelation consistent (HAC / Newey-West) standard error.

* **Mincer-Zarnowitz (MZ) Unbiasedness Regression:**

$$y_t = \alpha + \beta \hat{y}_t + e_t, \quad H_0: (\alpha, \beta) = (0, 1)$$

  Fits linear regression of realized target on forecast and computes a joint Wald test for $H_0: (\alpha, \beta) = (0, 1)$ to verify forecast unbiasedness.

* **Model Confidence Set (MCS):**

$$\widehat{\mathcal{M}}^*_{\alpha = 0.10}$$

  Uses stationary block-bootstrap resampling ($B=200$) at significance $\alpha=0.10$ to determine the superior set of models under QLIKE loss.

---

## Known Limitations and Assumptions

1. **Classically Intractable $N=20$ Simulator:** Full time-series statevector simulation of $N=20$ fully connected reservoirs over 2,500 continuous daily time steps is classically intractable. Therefore, $N=20$ is evaluated via physical QPU hardware validation on **IQM Garnet**.
2. **Archived QPU Execution:** Hardware results rely on pre-archived execution manifests in `artifacts/hardware/` so judges do not require active qBraid hardware credits.
3. **Strict Causal Alignment:** All model features strictly use data available through day $t$ to forecast day $t+1$ realized variance ($RV_{t+1}$). Regime thresholds and scalers are fitted on training data only.

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
    ├── figures/                   # Figures for statistical analysis and model performance
    ├── manifests/                 # summary_table.md, ablations.md, statistical_analysis.md
    └── hardware/                  # Pre-saved QPU job execution files and recurrent results
        ├── panel/                 # 24-circuit pre-saved panel results
        └── recurrent/             # 5-day recurrent QPU job results & evaluation reports
```
