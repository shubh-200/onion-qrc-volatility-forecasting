# COVER PAGE

**Competition:** GIC 2026 Quantum Computing Challenge — Phase 2
**Track:** A — Financial Volatility Prediction
**System Name:** VolQRC (Volatility Quantum Reservoir Computer)
**Participant:** Shubham Barge
**Education:** B.E. Computer Engineering, University of Mumbai

---

# 1. Track Selection and Problem Framing

We select Track A and target **one-day-ahead realized variance (RV) forecasting with three-class regime classification** (calm/elevated/crisis) using 5-minute realized volatility data from the S&P 500.

**Why reservoir computing fits this problem.** Volatility exhibits three structural properties matching QRC strengths: (1) *multi-scale temporal structure* — RV has strong autocorrelation at daily, weekly, and monthly horizons with nonlinear cross-scale interactions that linear models (HAR-RV) cannot capture; (2) *regime-switching nonlinearity* — markets transition between calm and turbulent states with different correlation structures, which classical autoregressive models systematically fail to capture; and (3) *noise-tolerant attractor reconstruction* — reservoir computing's echo-state property ensures that quantum noise is damped rather than amplified, unlike variational quantum algorithms.

**Sub-problem.** We forecast log(RV_{t+1}) at the 1-day horizon — the standard benchmark in the volatility literature (Patton 2011) — with direct application to Value-at-Risk under Basel IV's FRTB framework. Regimes are defined by tercile thresholds on RV (calm < 33rd pctile, crisis > 66th pctile).

**Dataset.** Oxford-Man Institute's Realized Volatility Library (Heber et al. 2009): S&P 500 5-minute realized variance, Feb 2007 – Jun 2017 (2,615 trading days), covering the 2008 GFC, 2010 Flash Crash, 2015 August selloff, and 2016 Brexit — providing adequate representation of all three regimes.

# 2. QRC Architecture Design

**Reservoir Hamiltonian.** A fully connected transverse-field Ising model: H = −Σ_{i<j} J_{ij} σ_i^z σ_j^z − h Σ_i σ_i^x, where J_{ij} ~ N(0,1) are random-but-fixed couplings (never optimized) and h = 0.3. Time evolution uses first-order Trotter decomposition (4 steps, t = 0.5) via RZZ/RX gates. All-to-all coupling generates rich entanglement; the transverse field prevents freezing into classical configurations.

**Onion multi-scale structure.** The key innovation partitions qubits into three sub-reservoirs differentiated by input rotation scaling: Short (⌊N/4⌋ qubits, α=1.0) captures daily shocks; Mid (⌊N/3⌋, α=0.6) captures weekly dynamics; Long (remainder, α=0.3) retains monthly memory. This mirrors the HAR-RV frequency decomposition but captures *nonlinear cross-scale interactions* through inter-band entanglement — the two-qubit correlators ⟨σ_s^z σ_l^z⟩ encode multiplicative coupling between timescales that HAR-RV's additive model cannot represent. The proportional allocation preserves this structure from N=5 to N=20.

**Input encoding.** Angle encoding via R_Y(α_{band} · arcsin(x_t)), one feature per band per step: log(RV_t) on Short, 5-day average on Mid, 22-day average on Long — matching the HAR-RV feature structure.

**Memory feedback.** Measurement re-injection (Ahmed et al. 2025): long-band ⟨σ_i^z⟩ values are re-injected as R_Y(κ·m_t) at the next step (κ=0.2), extending fading memory from ~10 to 50+ days. Only long-band qubits receive feedback; short/mid bands remain feed-forward.

**Readout.** Single-qubit (⟨σ_i^z⟩) and two-qubit (⟨σ_i^z σ_j^z⟩) observables are concatenated with HAR features and a one-hot regime label, then fed to ridge regression (λ=1.0, closed-form). This maintains the gradient-free paradigm.

**Regime classifier.** An RBF-kernel SVM on reservoir observables classifies regimes; the label gates the ridge readout for regime-specific mappings. IQP quantum kernel upgrade (Havlíček et al. 2019) planned for Phase 3.

**Hybrid integration.** The complete dataflow is: Oxford-Man RV data → preprocessing → three-band R_Y encoding → Ising Hamiltonian evolution (4 Trotter steps) → measurement re-injection on long-band → observable extraction → parallel paths: (1) ridge regression for log(RV_{t+1}) forecast, (2) SVM for regime label → regime one-hot appended to ridge input → gated volatility forecast with regime label.

# 3. Theoretical Justification and Prototyping Results

**Quantum property exploited.** The Ising reservoir's entangled state generates observables that are nonlinear functions of an exponentially large (2^N) Hilbert space. While the N + N(N−1)/2 extracted features match the count of degree-2 polynomials, they encode information from the *full entangled state* — capturing global correlation changes during regime transitions that local pairwise features cannot. Cross-band correlators ⟨σ_s^z σ_l^z⟩ are particularly valuable: they encode how the relationship between short-term shocks and long-term drift changes across regimes — precisely the nonlinear cross-scale interaction that HAR-RV's additive model structurally cannot capture.

**Connection to signal structure.** Volatility has a Hurst exponent H ≈ 0.8–0.9 (long memory) and heavy tails (kurtosis >> 3). The Onion bands mirror the HAR multi-scale decomposition while the Ising entanglement captures the multiplicative coupling between scales. During regime transitions, the correlation pattern between short-term shocks and long-term drift fundamentally changes — a daily shock that persists into a crisis vs. one that mean-reverts in a calm market. The quantum cross-scale correlators can distinguish these scenarios; linear additive models cannot.

**Noise resilience.** Ahmed et al. (2025) proved the echo-state condition extends to quantum reservoirs: depolarizing noise *helps* contractivity by mixing toward the maximally mixed state, and the transverse field ensures a minimum spectral gap. Performance degrades gracefully — a key advantage over variational approaches where noise creates barren plateaus.

**Prior work and gap.** Li et al. (2025) showed single-band QRC beats GARCH/HAR-RV on RV but used one temporal scale and no regime detection. Tandon et al. (2025) introduced Onion QRC for chaotic systems without financial application or memory feedback. VolQRC is the first system combining all three: Onion multi-scale structure, measurement re-injection, and regime-gated readout for financial forecasting.

**Prototyping results.** We implemented and tested VolQRC on a Qiskit Aer statevector simulator using real S&P 500 realized volatility data from the Oxford-Man library (2,615 trading days). Setup: 70/15/15 chronological train/validation/test split; models tested at N ∈ {5, 10} for both Onion and SingleBand architectures; baselines include GARCH(1,1), HAR-RV (ridge on [RV_d, RV_w, RV_m]), and ESN (500 nodes, spectral radius 0.95); readout is ridge regression on [reservoir observables + HAR features + regime one-hot].

| Model | N | RMSE | MAE | QLIKE | R² | Regime Acc. | MZ Unbiased |
|-------|---|------|-----|-------|-----|------------|-------------|
| **OnionQRC** | **5** | **0.372** | **0.296** | **0.071** | **0.561** | **68.0%** | No |
| OnionQRC | 10 | 0.400 | 0.312 | 0.092 | 0.493 | 68.2% | **Yes** |
| SingleBandQRC | 5 | 0.383 | 0.298 | 0.071 | 0.537 | 69.6% | Yes |
| SingleBandQRC | 10 | 0.477 | 0.385 | 0.100 | 0.280 | 48.5% | No |
| GARCH(1,1) | — | 3.342 | 3.294 | 2.338 | −34.4 | — | Yes |
| HAR-RV | — | 0.658 | 0.557 | 0.175 | −0.372 | — | No |
| ESN (500) | — | 0.384 | 0.307 | 0.078 | 0.532 | — | Yes |

**Key findings.** (1) OnionQRC(N=5) achieves the best QLIKE (0.071), beating ESN (0.078) — the critical benchmark justifying quantum overhead, since QLIKE is the asymmetric loss recommended by Patton (2011) that penalizes under-prediction of variance — and HAR-RV (0.175) by 2.5×. (2) OnionQRC(N=5) achieves the best R² (0.561) and MAE (0.296), explaining 56% of log(RV) variance — a strong result where even incremental improvements are economically significant for risk management. (3) Onion vs. SingleBand at N=5: R²=0.561 vs. 0.537 confirms multi-scale benefit, though QLIKE values are nearly identical (0.071 vs. 0.071), suggesting the Onion advantage manifests in variance explanation rather than tail-risk sensitivity. (4) N=5 outperforms N=10 for both architectures — explained by the samples-to-features ratio (21 vs. 61 features for ~1,830 training samples); scaling to N>10 requires longer time series or cross-index training (Phase 3). (5) Regime classification: 68% accuracy vs. 33% random baseline confirms reservoir observables contain separable regime information. (6) OnionQRC(N=10) passes the Mincer-Zarnowitz unbiasedness test (intercept=0.24, slope=1.06, both within 95% CI), meaning forecasts are statistically unbiased — a critical property for regulatory risk applications.


# 4. Data Modeling Strategy

**Preprocessing.** Log-transform RV for variance stabilization; construct HAR features (RV_d, 5-day RV_w, 22-day RV_m); Z-score normalize on training set only; rolling 252-day context window.

**Splits.** Train: Feb 2007 – Aug 2013 (~1,830 days, includes 2008 GFC); Validation: Sep 2013 – Sep 2014 (~390 days); Test: Oct 2014 – Jun 2017 (~785 days, includes 2015 selloff, Brexit). No data leakage.

**Baselines.** GARCH(1,1) (industry standard), HAR-RV (classical multi-scale analog), ESN-500 (classical reservoir computing — the key benchmark). LSTM and EGARCH planned for Phase 3.

**Metrics.** RMSE, QLIKE (asymmetric loss per Patton 2011), Mincer-Zarnowitz regression for unbiasedness. Model Confidence Set (Hansen et al. 2011) for Phase 3.

# 5. Quantum Platform and Resource Planning

**Phase 2.** Qiskit Aer statevector simulator, N ∈ {5, 10} — exact observables, no shot noise.

**Phase 3 escalation.** (1) Density-matrix simulator for noise characterization (depolarizing p ∈ {0.001, 0.005, 0.01}), N ∈ {5–12}; (2) GPU tensor-network simulator for scaling to N=15–20; (3) IBM Heron r3 QPU validation at N ∈ {10, 12} with ZNE (Mitiq) and mthree readout mitigation; IonQ Forte as secondary swap-free platform for swap-free validation at N ≤ 12.

**Circuit depth estimates.** Each time step requires encoding (depth 1), memory feedback (depth 1), and 4 Trotter steps of Ising coupling (O(N) depth each with swap network). At N=5: ~50 gates, depth ~25; at N=10: ~200 gates, depth ~100; at N=20: ~780 gates, depth ~400. All configurations fit within IBM Heron r3's coherence budget (T1≈300μs, ~5000-gate budget at ~60ns CZ gate time).

**Shot budget.** Classical shadow tomography (Huang et al. 2020) reduces the QPU measurement budget by ~8× compared to measuring each observable independently. For N=10 on the test set (~1,000 steps): ~57M total shots, estimated ~7 hours QPU time including queuing. Error mitigation uses Mitiq ZNE (Richardson extrapolation at noise scales [1, 3, 5]) and mthree matrix-free readout correction.

# 6. Stakeholder Impact and Phase 3 Plan

**Beneficiaries.** Risk managers: regime-aware VaR reduces backtesting failures under Basel IV FRTB. Trading desks: 1–3 day early-warning before regime transitions manifest in GARCH estimates, enabling pre-emptive delta-hedging adjustments and spread widening. Portfolio managers: adaptive volatility targeting using simultaneous fast (short-band) and slow (long-band) estimates without over-reacting to noise. Systemic risk regulators: quantum kernel regime classifier on cross-asset reservoir states could serve as a leading indicator of systemic transitions. Shapley attribution on reservoir features provides regulatory interpretability that black-box deep learning models lack.

**Phase 3 milestones (8 weeks, June–July 2026).** Weeks 1–2: data pipeline finalization and classical baselines (GARCH, HAR-RV, ESN, LSTM); Weeks 2–4: simulator scaling study (N ∈ {7, 10, 12, 15, 20}) with RMSE, QLIKE, and MZ regression at each N; Weeks 4–5: noise resilience analysis under depolarizing and amplitude-damping channels; Weeks 5–6: quantum kernel SVM validation with per-class precision/recall and regime gating ablation; Weeks 6–7: QPU validation on IBM Heron r3 at N ∈ {10, 12} with ZNE and mthree; Week 8: final benchmarks, Model Confidence Set test, and paper.

**Fallbacks.** If noise at N=10 degrades RMSE >50%, reduce QPU target to N=7 with enhanced error mitigation. If Onion underperforms single-band at larger N, report single-band QRC (still novel per Li et al.) with analysis of where multi-scale helps vs. adds noise. If QPU queue times are prohibitive, rely on density-matrix simulation with calibrated IBM noise profiles from backend properties.

---

# References

- Ahmed, N., Chen, C., & Rabitz, H. (2025). "Memory re-injection in quantum reservoir computing." *Proc. R. Soc. A*, 481.
- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *J. Econometrics*, 31(3), 307–327.
- Corsi, F. (2009). "A simple approximate long-memory model of realized volatility." *J. Fin. Econometrics*, 7(2), 174–196.
- Hamilton, J.D. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357–384.
- Hansen, P.R., Lunde, A., & Nason, J.M. (2011). "The model confidence set." *Econometrica*, 79(2), 453–497.
- Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209–212.
- Heber, P., Lunde, A., Shephard, N., & Sheppard, K. (2009). "Oxford-Man Institute's realized library." *Oxford-Man Institute, University of Oxford*.
- Hochreiter, S. & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735–1780.
- Hoerl, A.E. & Kennard, R.W. (1970). "Ridge regression: Biased estimation for nonorthogonal problems." *Technometrics*, 12(1), 55–67.
- Huang, H.-Y., Kueng, R., & Preskill, J. (2020). "Predicting many properties of a quantum system from very few measurements." *Nature Physics*, 16, 1050–1057.
- Jaeger, H. (2001). "The 'echo state' approach to analysing and training recurrent neural networks." *GMD Report 148*.
- Li, X., Chen, C., Rabitz, H., & Wang, S. (2025). "Quantum reservoir computing for realized volatility forecasting." *arXiv:2505.13933*.
- Mincer, J. & Zarnowitz, V. (1969). "The evaluation of economic forecasts." In *Economic Forecasts and Expectations*, NBER, 3–46.
- Patton, A.J. (2011). "Volatility forecast comparison using imperfect volatility proxies." *J. Econometrics*, 160(1), 246–256.
- Tandon, A., et al. (2025). "Onion quantum reservoir computing for multi-scale dynamical systems." *arXiv:2505.22837*.
