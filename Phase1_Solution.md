---
title: "GIC 2026 — Phase 1 Proposal: VolQRC"
subtitle: "Quantum Reservoir Computing for Financial Volatility Prediction"
---

# COVER PAGE

**Competition:** GIC 2026 Quantum Computing Challenge
**Track:** Financial Volatility Prediction
**File:** TeamName__Phase1_Version1.pdf

---

# Team Qualifications

- **Participant:** Shubham Barge
- **Education:** B.E. Computer Engineering, University of Mumbai
- **Relevant Projects:** Built a hybrid VQE portfolio optimizer in Qiskit benchmarked against IBM CPLEX. Currently integrating an IQP-encoded quantum kernel into a Swin Transformer's ASPP block, providing direct experience with the quantum kernel methods central to the proposed regime classifier.
- **Experience:** Autonomous Systems Intern at AstraYAN (IIT Madras Incubation Cell), designing multimodal data acquisition and processing pipelines for camera and LiDAR sensor fusion — directly analogous to the multi-scale feature engineering architecture proposed in this submission. Two peer-reviewed publications in IEEE and IET proceedings; hands-on experience with qBraid-compatible simulation tools required by the challenge infrastructure.

---

# Steps to Solve the Challenge

1. **Data:** Acquire 10+ years of S&P 500 data; compute 5-minute realized volatility (RV), HAR-style features (daily/weekly/monthly averages), log-returns, and VIX.
2. **Reservoir:** Build a proportionally scalable three-tier Onion QRC (fully connected transverse-field Ising Hamiltonian, random-but-fixed couplings); at N total qubits, allocate ⌊N/4⌋ short-, ⌊N/3⌋ mid-, and remainder long-horizon qubits, each band differentiated by its rotation angles.
3. **Encoding & Memory:** Angle-encode financial features into input qubits; re-inject partial measurement outcomes into memory qubits each step to restore fading memory.
4. **Readout:** Concatenate single- and two-qubit expectation values with HAR-RV regressors; train a ridge-regression readout in closed form.
5. **Regime Module:** Apply a quantum kernel classifier on the same reservoir states to label the volatility regime (calm / elevated / crisis); route regime labels into the readout as a conditioning signal.
6. **Scaling & Noise:** Sweep N ∈ {5, 10, 15, 20} under the same proportional allocation rule; benchmark under depolarizing (p ∈ {0.001, 0.005, 0.01}) and amplitude-damping noise on qBraid, reporting RMSE and QLIKE at each N.
7. **Benchmarks:** Compare against GARCH, EGARCH, HAR-RV, and LSTM via RMSE, QLIKE, and Model Confidence Set; validate expressivity on MNIST with amplitude encoding as a cross-team comparator.

---

# Proposed Solution: VolQRC

VolQRC is a gradient-free, hybrid QRC system for volatility forecasting and regime-change detection. Building on Li et al. (arXiv:2505.13933), who showed a single Ising QRC outperforms GARCH and HAR-RV on realized volatility, VolQRC adds three extensions: (1) a multi-scale Onion reservoir (Tandon et al., arXiv:2505.22837) capturing short-, medium-, and long-horizon dynamics simultaneously; (2) measurement re-injection feedback for long-memory modelling (Ahmed et al., Proc. R. Soc. A 481, 2025); and (3) a quantum kernel regime classifier whose labels gate the ridge readout — a structural capability GARCH-family models lack.

**Technical Approach.** The reservoir Hamiltonian is a fully connected transverse-field Ising model with random-but-fixed couplings, never optimized during training. The Onion architecture scales proportionally: at N qubits, sub-reservoirs receive ⌊N/4⌋, ⌊N/3⌋, and remainder qubits, with rotation angle scaling (α_short, α_mid, α_long) as the band differentiator — preserving the three-frequency decomposition from N = 5 to N = 20. The Ising reservoir generates an exponentially large feature basis in principle; even its partial NISQ realization may capture nonlinear correlations that classical autoregressive models structurally cannot. Noise resilience follows from echo-state contractivity (Ahmed et al., 2025); Shapley attribution (Li et al.) provides regulatory interpretability.

---

# Projected Industry Impact

Regime-aware volatility forecasting reduces Value-at-Risk errors, improves hedging efficiency, and lowers capital requirements under Basel IV's FRTB framework, while the regime-detection module delivers early-warning signals for tail-risk events applicable to stress-testing and systematic trading. Because VolQRC is gradient-free, it scales to QPUs without algorithmic redesign as hardware matures.