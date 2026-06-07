# Onion Quantum Reservoir for Volatility Forecasting

A hybrid quantum-classical model for **one-day-ahead realized volatility forecasting** with simultaneous **regime classification**, using a multi-scale quantum reservoir computing architecture.

The system encodes financial volatility features into a **transverse-field Ising Hamiltonian** on quantum hardware (simulated via Qiskit), extracts single- and two-qubit observables, and feeds them into a classical ridge regression readout with regime gating. The **Onion architecture** partitions qubits into three bands with differentiated input scalings to capture daily, weekly, and monthly volatility dynamics — analogous to the HAR-RV model but with nonlinear interactions between scales.

## Key Features

- **Onion multi-scale reservoir** — Qubits partitioned into short (fast), mid (moderate), and long (slow) bands with α-scaled input rotations
- **Ising Hamiltonian dynamics** — Fully connected transverse-field Ising model with random-but-fixed couplings; Trotterized time evolution with all-to-all RZZ + RX gates
- **Measurement re-injection feedback** — Long-band qubit measurements fed back as RY rotations at the next time step for extended memory
- **Ridge regression readout** — Closed-form gradient-free training on concatenated quantum observables, classical HAR features, and regime one-hot labels
- **Regime classification** — 3-class volatility regime detector (calm / elevated / crisis) via kernel SVM on reservoir state vectors; labels act as a gating signal for the readout
- **Classical baselines** — HAR-RV, GARCH(1,1), Echo State Network, and LSTM for performance comparison
- **Support for 5–20 qubits** — Scales from N=5 prototyping to N=20 full benchmarking

## Architecture

```
Input Features ──► Angle Encoding ──► Onion Reservoir ──► Observable Extraction ──► Ridge Readout ──► Forecast
                  Ry(arcsin(x))     3 bands (S/M/L)      <σᵢᶻ>, <σᵢᶻσⱼᶻ>          + regime gating      log(RV_{t+1})
                         │                 │
                         │           ┌─────┘
                         ▼           ▼
                   Memory Feedback ◄─┘
                   (long-band re-injection)
```

| Component | Description |
|-----------|-------------|
| **Input encoding** | HAR-RV features (log-RV, daily/weekly/monthly averages, log-return) encoded via RY gates with band-specific rotation scalings |
| **Onion bands** | Short (⌊N/4⌋ qubits, α=1.0), Mid (⌊N/3⌋, α=0.6), Long (remainder, α=0.3) |
| **Hamiltonian** | Fully connected transverse-field Ising: H = −Σ Jᵢⱼ σᵢᶻσⱼᶻ − h Σ σᵢˣ, Jᵢⱼ ∼ N(0,1) fixed |
| **Time evolution** | 4-step Trotterization: alternating RZZ(J·Δt) layers and RX(h·Δt) layers |
| **Feedback** | Long-band ⟨σᵢᶻ⟩ measured → scaled (κ=0.2) → re-injected as RY on long qubits next step |
| **Observables** | N single-qubit ⟨σᵢᶻ⟩ + N(N−1)/2 two-qubit ⟨σᵢᶻσⱼᶻ⟩ = N(N+1)/2 total |
| **Readout** | Ridge regression: ŵ = (XᵀX + λI)⁻¹Xᵀy, λ=1.0 |
| **Regime gating** | 3-class SVM one-hot → appended to feature vector before ridge |

## Repository Structure

```
├── prototype/
│   ├── onion_qrc.py           # OnionQRC (multi-band) and SingleBandQRC implementations
│   ├── readout.py              # RidgeReadout, VolQRCReadout, QuantumKernelClassifier, metrics
│   ├── data_loader.py          # SPX RV data loading, HAR feature engineering, train/val/test split
│   ├── baselines.py            # GARCH(1,1), HAR-RV, ESN, LSTM baselines
│   ├── run_experiment.py       # Main experiment pipeline: runs all models, saves results
│   ├── architecture_diagram.py # Generates architecture, onion allocation, and circuit diagrams
│   ├── requirements.txt
│   └── figures/                # Pre-generated architecture diagrams
├── rv_dataset.csv              # Daily 5-min realized variance for 8 market indices
├── global index etf return/    # S&P 500, DAX, CAC 40, FTSE 100, etc. ETF price data
├── .gitignore
└── README.md
```

## Data

**rv_dataset.csv** contains daily 5-minute realized variance for 8 global market indices from the Oxford-Man Institute's Realized Volatility Library (Heber et al., 2009):

- S&P 500 (.SPX), DAX (.GDAXI), CAC 40 (.FCHI), FTSE 100 (.FTSE)
- OMX Stockholm All Share (.OMXSPI), Nikkei 225 (.N225), KOSPI (.KS11), Hang Seng (.HSI)

Ticker reconstruction via correlation alignment with daily price data yields approximately 2,615 trading days (Feb 2007 – Jun 2017). The `data_loader.py` module falls back to synthetic GARCH(1,1) data if the CSV is not found.

Regimes are labeled by RV quantile thresholds:
- **Calm (0):** RV below 33rd percentile
- **Elevated (1):** RV between 33rd and 66th percentile
- **Crisis (2):** RV above 66th percentile

## Installation

```bash
pip install -r prototype/requirements.txt
```

Requires Python 3.10+ and [Qiskit](https://qiskit.org) ≥ 1.0. The code uses statevector simulation; no quantum hardware access is needed.

## Usage

Run the full experiment pipeline:

```bash
python prototype/run_experiment.py
```

This executes:
1. Loads SPX realized variance data
2. Runs **OnionQRC** at N=5 and N=10
3. Runs **SingleBandQRC** at N=5 and N=10
4. Runs **GARCH(1,1)**, **HAR-RV**, and **ESN** baselines
5. Saves results to `prototype/results/results_{timestamp}.json`
6. Prints a summary table with RMSE, QLIKE, R², regime accuracy, and Mincer-Zarnowitz test results

Generate architecture diagrams:

```bash
python prototype/architecture_diagram.py
```

Run individual module tests:

```bash
python prototype/onion_qrc.py
python prototype/readout.py
python prototype/data_loader.py
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root mean squared error of log(RV) predictions |
| **QLIKE** | Quasi-likelihood loss (robust to volatility proxies) |
| **R²** | Coefficient of determination |
| **MAE** | Mean absolute error |
| **Regime accuracy** | Classification accuracy of the regime detector |
| **Mincer-Zarnowitz** | Test for forecast unbiasedness (intercept=0, slope=1) |

## Baselines

| Model | Description |
|-------|-------------|
| **HAR-RV** | Heterogeneous Autoregressive model (Corsi, 2009) with ridge regression |
| **GARCH(1,1)** | Standard volatility model via `arch` package |
| **ESN** | Echo State Network with 500 reservoir units, spectral radius 0.95 |
| **SingleBandQRC** | Same Ising Hamiltonian but without multi-scale partitioning |
| **OnionQRC** | The full multi-band proposal |

## References

- Heber, G., Lunde, A., Shephard, N., & Sheppard, K. (2009). Oxford-Man Institute's realized library. Oxford-Man Institute.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Report 148.
- Tandon, P., et al. (2025). Onion multi-scale quantum reservoir computing.
- Ahmed, S., et al. (2025). Quantum reservoir computing with measurement feedback.

## License

MIT
