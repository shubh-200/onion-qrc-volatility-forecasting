# Onion Volatility Forecasting

Multi-scale quantum reservoir computing for financial volatility forecasting. Implements the **Onion architecture** — a partitioned quantum reservoir with three bands (short/mid/long) that capture different temporal scales of realized volatility, inspired by the HAR-RV model.

## Architecture

- **Reservoir:** Fully connected transverse-field Ising Hamiltonian with Trotterized time evolution
- **Onion partitioning:** Qubits split into short, mid, and long bands with differentiated input rotation scalings
- **Input encoding:** Angle encoding via RY gates with HAR-RV style features (daily, weekly, monthly averages)
- **Memory:** Measurement re-injection feedback on long-band qubits for extended fading memory
- **Readout:** Ridge regression on concatenated quantum observables + classical HAR features with regime gating
- **Regime classifier:** 3-class volatility regime detection (calm, elevated, crisis) via kernel SVM on reservoir states

## Repository Structure

```
├── prototype/
│   ├── onion_qrc.py           # Onion quantum reservoir implementation
│   ├── readout.py              # Ridge regression readout layer
│   ├── data_loader.py          # Realized variance data loading
│   ├── baselines.py            # Classical baseline models (HAR-RV, GARCH, etc.)
│   ├── run_experiment.py       # Experiment pipeline
│   ├── architecture_diagram.py # Architecture visualization
│   ├── requirements.txt
│   └── figures/                # Generated architecture diagrams
├── rv_dataset.csv              # Daily realized variance data (Oxford-Man)
├── global index etf return/     # Global market index ETF returns
└── README.md
```

## Data

- **rv_dataset.csv:** Daily 5-minute realized variance for 8 market indices (S&P 500, DAX, CAC 40, FTSE 100, OMX Stockholm, Nikkei 225, KOSPI, HANG SENG) from the Oxford-Man Institute's Realized Library
- **global index etf return/:** Corresponding ETF return data from investing.com

## Getting Started

```bash
pip install -r prototype/requirements.txt
python prototype/run_experiment.py
```
