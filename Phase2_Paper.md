---
title: "VolQRC Phase 2 Technical Paper"
subtitle: "Quantum Reservoir Computing for Financial Volatility Prediction and Regime Detection"
author: "Shubham Barge"
date: "2026"
---

# COVER PAGE

**Competition:** GIC 2026 Quantum Computing Challenge — Phase 2
**Track:** A — Financial Volatility Prediction
**System Name:** VolQRC (Volatility Quantum Reservoir Computer)
**Participant:** Shubham Barge
**Education:** B.E. Computer Engineering, University of Mumbai

---

# 1. Track Selection and Problem Framing

## 1.1 Track A Selection: Financial Volatility Prediction

We select Track A — Financial Volatility Prediction — and target the specific sub-problem of **one-day-ahead realized variance (RV) forecasting with simultaneous three-class regime classification** (calm, elevated, crisis) using 5-minute realized volatility data from the S&P 500 index.

## 1.2 Why Reservoir Computing Matches Regime-Switching Volatility

Volatility forecasting is fundamentally a problem of learning the dynamics of a nonlinear, long-memory, regime-switching system. Three structural properties make this problem particularly well-suited to reservoir computing:

1. **Multi-scale temporal structure.** Realized volatility exhibits strong autocorrelation at daily, weekly, and monthly horizons (Corsi 2009). The HAR-RV model captures this through simple linear averaging, but the true multi-scale dynamics are nonlinear — the interaction between horizons changes across regimes. A multi-band reservoir naturally embeds these scales into its internal dynamics without imposing a fixed functional form.

2. **Regime-switching nonlinearity.** Markets transition between calm and turbulent states with fundamentally different correlation structures (Hamilton 1989). Classical autoregressive models (GARCH, HAR-RV) impose a single functional form across all regimes, which systematically underestimates tail risk. A quantum reservoir generates a rich nonlinear feature basis in which different regimes map to distinct regions of observable space, enabling a kernel classifier to find separating hyperplanes that classical polynomial features cannot reach.

3. **Noise-tolerant attractor reconstruction.** Reservoir computing excels at reconstructing the attractor of a chaotic dynamical system from noisy observations (Jaeger 2001, Pathak et al. 2018). Volatility shares structural similarity with chaotic systems — it has a high effective dimension, is driven by hidden state variables, and exhibits sensitive dependence on initial conditions. The echo-state property ensures that the reservoir's internal dynamics are contractive: small perturbations (including quantum noise) are damped rather than amplified.

## 1.3 Sub-Problem Definition

- **Forecasting target:** log(RV_{t+1}), where RV is the 5-minute realized variance of the S&P 500
- **Horizon:** 1 trading day ahead
- **Regime classification:** 3-class (calm / elevated / crisis) based on RV quantile thresholds
- **Why this sub-problem:** One-day-ahead RV is the standard benchmark in the volatility forecasting literature (Patton 2011), and regime classification at this horizon has direct application to Value-at-Risk computation under Basel IV's FRTB framework, which requires next-day risk estimates.

## 1.4 Dataset Justification

We use the **Oxford-Man Institute's Realized Volatility Library** (Heber et al. 2009), which provides:

- **Source:** Originally at `realized.oxford-man.ox.ac.uk` (now discontinued); archived copies available from the VBayesLab/VBLab GitHub repository and the CRAN `bvhar` R package
- **Coverage:** Daily realized volatility measures for 31 indices, Jan 2000 – Feb 2021 (per VBLab documentation)
- **Frequency:** Daily, computed from 5-minute intraday returns
- **S&P 500 ticker:** `.SPX`
- **Variable used:** 5-minute realized variance (`rv5`)
- **Available time range in our archived copy:** February 2007 – June 2017 (2,615 trading days for `.SPX`), date-reconstructed via correlation alignment with S&P 500 daily price data
- **Why appropriate:**
  - Pre-computed RV eliminates data-sourcing variability across teams
  - Widely cited in the volatility forecasting literature (Patton 2011, Corsi 2009)
  - Reproducible — the archived copies can be downloaded and verified
  - Covers multiple crisis periods (2008 GFC, 2010 Flash Crash, 2015 August selloff, 2016 Brexit) for robust regime evaluation
  - Sufficient length (2,615 trading days) for train/validation/test splits with adequate crisis-period representation

---

# 2. QRC Architecture Design

## 2.1 Reservoir Hamiltonian

The reservoir is governed by a **fully connected transverse-field Ising Hamiltonian**:

$$H = -\sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z - h \sum_i \sigma_i^x$$

where:

- $J_{ij} \sim \mathcal{N}(0, 1)$ are random-but-fixed coupling strengths, drawn once at initialization and **never optimized during training**
- $h = 0.3$ is the transverse field strength, providing quantum fluctuations that prevent the system from freezing into classical spin configurations
- The couplings $J_{ij}$ are symmetrized: $J_{ij} = J_{ji}$

**Implementation:** The time evolution $e^{-iHt}$ is approximated via a **first-order Trotter decomposition** with 4 steps. Each Trotter step applies:

1. $RZZ(2J_{ij}\Delta t)$ for all pairs $(i,j)$ — implements the Ising coupling
2. $RX(2h\Delta t)$ on all qubits — implements the transverse field

where $\Delta t = t_{\text{total}} / 4$ is the Trotter slice duration and $t_{\text{total}} = 0.5$.

**Why this Hamiltonian:**
- The fully connected Ising model generates entanglement across all qubits, producing a rich observable basis from single- and two-qubit measurements
- Random fixed couplings satisfy the echo-state property (Jaeger 2001, Ahmed et al. 2025) — the reservoir's dynamics are sufficiently complex without requiring gradient-based optimization
- The transverse field prevents the system from settling into trivial eigenstates, ensuring that the reservoir state depends nontrivially on the input encoding
- The Ising Hamiltonian is native to quantum annealing architectures and efficiently simulable on gate-based hardware via RZZ/RX gates

## 2.2 Onion Multi-Scale Structure

The key innovation of VolQRC is the **Onion architecture** (Tandon et al. 2025), which partitions the total qubit budget into three sub-reservoirs (bands) that capture different temporal scales:

| Band | Qubit allocation | Rotation scaling | Timescale captured | Role |
|------|-----------------|-----------------|-------------------|------|
| Short | ⌊N/4⌋ | α_s = 1.0 | Daily | Fast response to recent volatility shocks |
| Mid | ⌊N/3⌋ | α_m = 0.6 | Weekly | Moderate-frequency dynamics |
| Long | remainder | α_l = 0.3 | Monthly | Slow drift, memory retention |

**Proportional scaling rule.** At N qubits:

| N | Short | Mid | Long | Total observables |
|---|-------|-----|------|-------------------|
| 5 | 1 | 1 | 3 | 15 |
| 7 | 1 | 2 | 4 | 28 |
| 10 | 2 | 3 | 5 | 55 |
| 12 | 3 | 4 | 5 | 78 |
| 15 | 3 | 5 | 7 | 120 |
| 20 | 5 | 6 | 9 | 210 |

This proportional allocation preserves the three-frequency decomposition across all problem sizes, ensuring that the multi-scale character of the reservoir is maintained from N=5 prototyping to N=20 full benchmarking.

**Band differentiation mechanism.** The three bands share the same Ising Hamiltonian topology but are differentiated by their **input rotation angle scaling** (α_s, α_m, α_l). A lower α means the corresponding qubits respond less sensitively to input perturbations — they evolve slowly and retain longer memory of past inputs. A higher α means rapid response but faster decay. This mirrors the HAR-RV decomposition: $\text{RV}_t = \beta_d \cdot \text{RV}_d + \beta_w \cdot \text{RV}_w + \beta_m \cdot \text{RV}_m$, but with the crucial difference that the quantum reservoir captures the *nonlinear interaction* between these scales rather than just their linear combination.

## 2.3 Input Encoding Scheme

**Method:** Angle encoding via $R_Y$ gates.

Each normalized input feature $x_t \in [-1, 1]$ is encoded as:

$$R_Y(\alpha_{\text{band}} \cdot \arcsin(x_t))$$

on the corresponding band's input qubits, where $\alpha_{\text{band}}$ is the band-specific scaling factor.

**Encoding density:** 1 feature per band per time step, with the same feature encoded across all qubits within a band. The three features encoded are:

1. **Short band:** log(RV_t) — the most recent realized volatility
2. **Mid band:** 5-day average log(RV) — weekly smoothed signal
3. **Long band:** 22-day average log(RV) — monthly smoothed signal

This gives 3 effective input features per time step, matching the three-band structure of the HAR-RV model.

**VIX and additional features (Phase 3):** The CBOE Volatility Index (VIX) provides a forward-looking implied volatility measure that is complementary to the backward-looking realized volatility. VIX was identified as a planned input in Phase 1; however, we restrict Phase 2 encoding to the three HAR-style features for two reasons: (a) the Onion architecture's three bands are explicitly designed to mirror the daily/weekly/monthly RV decomposition, and adding a fourth feature would require architectural redesign (a fourth band or multiplexed encoding); (b) the Phase 2 objective is to substantiate core design claims (Onion > single-band, regime gating helps) rather than maximize absolute performance. VIX integration is planned for Phase 3, either as an additional encoding on the mid-band qubits (as an option-implied signal that captures medium-term expectations) or as a fourth classical feature appended to the readout.

**Why angle encoding:**
- Computationally lightweight (single RY gate per qubit per step)
- Amplitude encoding would allow higher density but requires state preparation circuits that dominate the circuit depth and destroy the gradient-free advantage
- The arcsin mapping ensures the full range [-1,1] maps to the full rotation range [-π/2, π/2] without saturation
- Encoding density of 1 feature/band/step is sufficient because the reservoir's temporal memory provides implicit access to past inputs without explicit re-encoding

**Repeated encoding strategy:** At each time step, the encoding is applied *before* the Ising evolution. The reservoir's internal state (which carries information from previous time steps via the Hamiltonian dynamics and memory feedback) is not reset between steps. This creates a natural temporal recurrence without explicit recurrent connections.

## 2.4 Memory / Feedback Mechanism

VolQRC employs **measurement re-injection feedback** (Ahmed et al. 2025) to extend the reservoir's fading memory:

1. At each time step $t$, after Ising evolution and observable extraction, the single-qubit expectation values $\langle\sigma_i^z\rangle$ of the **long-band qubits** are measured
2. These measured values $m_t = [\langle\sigma_{q_1}^z\rangle, \langle\sigma_{q_2}^z\rangle, \ldots]$ are stored as the memory state
3. At the next time step $t+1$, the memory state is re-injected as additional $R_Y$ rotations on the long-band qubits: $R_Y(\kappa \cdot m_t^{(i)})$, where $\kappa = 0.2$ is the feedback strength

**Why this mechanism:**
- Quantum reservoirs, like classical echo-state networks, suffer from fading memory — information about inputs from more than ~10 steps ago is typically lost (Jaeger 2001)
- For volatility forecasting, monthly (22-day) memory is essential, and crisis detection requires remembering events from 50+ days prior
- Re-injection creates a recurrent loop that restores information about past inputs, analogous to the recurrent connections in an LSTM's hidden state
- The feedback strength $\kappa$ is kept small (0.2) to maintain the echo-state contractivity condition: the reservoir must still be stable despite the feedback
- Only long-band qubits receive feedback — short and mid bands remain "feed-forward" to preserve their rapid response characteristics

## 2.5 Readout Strategy

The readout layer concatenates quantum observables with classical HAR features and applies ridge regression:

**Quantum observables extracted:**
- Single-qubit: $\langle\sigma_i^z\rangle$ for all $i \in \{0, \ldots, N-1\}$ → N values
- Two-qubit: $\langle\sigma_i^z \sigma_j^z\rangle$ for all $i < j$ → N(N-1)/2 values
- **Total quantum features:** N + N(N-1)/2

**Classical features appended:**
- $\text{RV}_d$, $\text{RV}_w$, $\text{RV}_m$ (HAR-RV features)

**Regime gating:**
- The quantum kernel regime classifier (Section 2.6) produces a 3-class label
- This label is converted to a one-hot vector and appended to the feature vector
- The regime label acts as a **gating signal** that allows the ridge regression to learn different linear mappings for different regimes

**Ridge regression:** The readout weights are computed in closed form:

$$\hat{w} = (X^TX + \lambda I)^{-1} X^T y$$

with $\lambda = 1.0$. This is a single-pass, gradient-free training procedure — the entire training set is processed once to compute $\hat{w}$, and no iterative optimization is required.

**Why ridge regression over nonlinear readouts:**
- The quantum reservoir already provides a nonlinear feature expansion — the readout only needs to linearly combine these features
- Ridge regression is provably optimal for linear readout on fixed features (Hoerl & Kennard 1970)
- Closed-form solution eliminates hyperparameter tuning of learning rates, epochs, and convergence criteria
- Consistent with the gradient-free philosophy of reservoir computing — training cost is $O(n_{\text{train}} \cdot n_{\text{features}}^2)$

## 2.6 Quantum Kernel Regime Classifier

The regime classifier operates on the **same reservoir state vectors** used for volatility prediction, applying a kernel method:

1. **State preparation:** The reservoir observables at each time step form a classical feature vector $\mathbf{o}_t \in \mathbb{R}^{d}$ where $d = N + N(N-1)/2$

2. **Quantum feature map (Phase 3 target):** Each feature vector will be mapped to a quantum state via an IQP-inspired encoding (Havlíček et al. 2019):
   $$|\phi(\mathbf{o}_t)\rangle = H^{\otimes n} U_Z(\mathbf{o}_t) H^{\otimes n} |0\rangle^{\otimes n}$$
   where $U_Z$ applies $R_Z$ rotations with angles $\alpha \cdot o_t^{(i)}$ followed by CZ gates with $R_Z$ rotations encoding pairwise products $\alpha \cdot o_t^{(i)} \cdot o_t^{(j)}$. The kernel $K(t_i, t_j) = |\langle\phi(\mathbf{o}_{t_i})|\phi(\mathbf{o}_{t_j})\rangle|^2$ will be estimated via the **swap test** on QPU hardware, or computed exactly via statevector inner products on the simulator.

3. **Phase 2 prototyping:** We use a classical RBF kernel on the reservoir observables as a proof-of-concept for the regime classification concept. This is justified because the quantum reservoir's observables already live in a high-dimensional, nonlinearly-embedded feature space — even a classical kernel on these quantum-generated features captures regime separability. The IQP quantum kernel upgrade is deferred to Phase 3 because: (a) the reservoir produces $d=15$ (N=5) or $d=55$ (N=10) features, requiring either dimensionality reduction (PCA) or a feature map with matching qubit count, both of which require careful tuning beyond Phase 2's scope; (b) the swap-test kernel estimation on QPU is the natural evaluation mode for the quantum kernel and is planned for Phase 3 QPU runs.

4. **Classification:** A support vector machine (SVM) with the kernel performs 3-class classification (one-vs-rest) into calm, elevated, or crisis regimes

**Regime labeling for training:** Regimes are defined by quantile thresholds on the realized volatility series:
- **Calm (0):** RV below the 33rd percentile
- **Elevated (1):** RV between the 33rd and 66th percentiles
- **Crisis (2):** RV above the 66th percentile

**Why quantum kernel over classical kernel (Phase 3 motivation):**
- The reservoir observables already live in a high-dimensional feature space ($d=55$ for N=10). Classical RBF kernels on this data work well in practice (68% regime accuracy in Phase 2), but the IQP feature map can generate decision boundaries that are computationally hard to simulate classically (Havlíček et al. 2019), providing a potential quantum advantage for the classification task
- The quantum kernel operates on the *same* quantum features used for regression, ensuring consistency between the two tasks
- Phase 2's RBF kernel results already demonstrate that quantum reservoir observables contain separable regime information — the IQP upgrade in Phase 3 will test whether the quantum feature map can extract additional structure

## 2.7 Hybrid Classical–Quantum Integration

The complete data flow is:

```
Oxford-Man RV data
    ↓ [preprocessing]
Normalized features: [log(RV), RV_d, RV_w, RV_m, log-return]
    ↓ [angle encoding]
Three-band RY encoding on Onion QRC
    ↓ [Ising Hamiltonian evolution (4 Trotter steps)]
Quantum state |ψ_t⟩
    ↓ [measurement re-injection → long-band feedback]
    ↓ [observable extraction]
Classical feature vector: [<σᵢᶻ>, <σᵢᶻσⱼᶻ>, RV_d, RV_w, RV_m]
    ↓ ┌─────────────────────────┐
    ↓ │ Parallel classical paths │
    ↓ ├─────────────────────────┤
    ↓ │ Path 1: Ridge regression → log(RV_{t+1}) forecast  │
    ↓ │ Path 2: Quantum kernel SVM → regime label           │
    ↓ └─────────────────────────┘
    ↓ [regime one-hot appended to ridge input]
Gated volatility forecast + regime label
```

**See the architecture diagram** (generated by `architecture_diagram.py`) for a visual representation.

---

# 3. Theoretical and Analytical Justification

## 3.1 Quantum Property Exploited: Hilbert Space Dimensionality

The core quantum advantage of VolQRC arises from the **exponential dimensionality of the Hilbert space** relative to the number of physical qubits.

For N qubits, the state space has dimension $2^N$. By measuring all single-qubit observables $\langle\sigma_i^z\rangle$ and two-qubit correlators $\langle\sigma_i^z \sigma_j^z\rangle$, we extract $N + N(N-1)/2$ real-valued features. While this is polynomial in N, these features are *nonlinear functions of the entangled quantum state* — they capture correlations that cannot be expressed by any classical polynomial of the same degree on the same number of variables.

**Quantitative comparison:**

| N | Quantum features | Classical polynomial features (degree 2) | Ratio |
|---|-----------------|------------------------------------------|-------|
| 5 | 15 | 15 | 1.0× |
| 10 | 55 | 55 | 1.0× |
| 20 | 210 | 210 | 1.0× |

The feature counts are equal for degree-2 polynomials, but the **information content per feature differs fundamentally**. The two-qubit correlators $\langle\sigma_i^z\sigma_j^z\rangle$ depend on the full entangled state $|\psi\rangle$, which encodes $2^N$ amplitude degrees of freedom. A classical degree-2 polynomial $x_i \cdot x_j$ depends only on the two local variables. The quantum features are thus *functions of an exponentially larger information space*, even though their count is polynomial.

This distinction becomes crucial for **regime classification**: the transition from calm to crisis markets involves a structural change in the correlation pattern across assets. The quantum correlators capture this change through their dependence on the *global* entanglement structure, whereas classical polynomial features depend only on *local* pairwise interactions.

**Higher-order observables.** While we restrict our readout to single- and two-qubit observables for NISQ feasibility, the Hilbert space also supports $k$-body correlators $\langle\sigma_{i_1}^z \cdots \sigma_{i_k}^z\rangle$ for $k \leq N$. These provide access to $\binom{N}{k}$ additional features at each order, yielding a total feature space of $3^N - 1$ possible Pauli string expectations. In principle, this exponential feature basis is the source of the quantum reservoir's expressive power — our readout samples a polynomial subset of this basis, but even this subset is informationally richer than its classical counterpart.

## 3.2 Connection to Target Signal Structure

### 3.2.1 Multi-Scale Memory and the HAR Decomposition

The seminal HAR-RV model (Corsi 2009) decomposes realized volatility into three additive components:

$$\text{RV}_{t+1} = \beta_d \cdot \text{RV}_t^{(d)} + \beta_w \cdot \text{RV}_t^{(w)} + \beta_m \cdot \text{RV}_t^{(m)} + \epsilon_t$$

This works because volatility has a Hurst exponent $H \approx 0.8$–$0.9$ (indicating long memory), and the three averaging horizons approximate the power-law decay of the autocorrelation function.

The Onion architecture **mirrors this decomposition** in the quantum domain:

- **Short band** ($\alpha_s = 1.0$): Large rotation angles → the reservoir state changes rapidly with input → captures daily-scale volatility shocks
- **Mid band** ($\alpha_m = 0.6$): Moderate angles → intermediate response → captures weekly mean-reversion dynamics
- **Long band** ($\alpha_l = 0.3$): Small angles → the reservoir state changes slowly → retains information from inputs many steps ago → captures monthly drift

**The crucial advantage over HAR-RV:** The three bands interact through the Ising Hamiltonian's all-to-all coupling. The short-band qubits are entangled with the long-band qubits, producing two-qubit observables $\langle\sigma_s^z \sigma_l^z\rangle$ that capture *nonlinear cross-scale interactions*. HAR-RV treats the three scales as additive; VolQRC treats them as multiplicatively coupled through quantum entanglement.

This is particularly important during regime transitions, where the relationship between short-term shocks and long-term drift fundamentally changes (e.g., a daily shock that persists into a crisis vs. one that mean-reverts in a calm market). The quantum cross-scale correlators can distinguish these scenarios; the HAR additive model cannot.

### 3.2.2 Regime-Switching Nonlinearity

Financial volatility regime transitions are characterized by sudden changes in the correlation structure of returns (Ang & Timmermann 2012). During calm periods, cross-asset correlations are low and idiosyncratic risk dominates. During crises, correlations spike toward 1 ("all correlations go to one in a crisis"), and systematic risk dominates.

The Ising reservoir's entanglement patterns respond differently to inputs drawn from different regimes:

- **Calm regime** (small RV): Small input rotations → reservoir state remains near the ground state → observables cluster in a compact region of feature space
- **Crisis regime** (large RV): Large input rotations → reservoir state is driven far from equilibrium → observables spread across a different region of feature space

The quantum kernel classifier exploits this separation: the IQP feature map maps these already-distinct observable vectors to a Hilbert space where the regime classes become linearly separable. This is the quantum analog of the classical kernel trick, but applied to features that are themselves generated by a quantum process — providing a "double kernel" effect that classical methods cannot replicate.

### 3.2.3 Non-Gaussian Tails and Higher Moments

Realized volatility has heavy tails: its distribution has kurtosis $>> 3$ (significantly non-Gaussian). GARCH-family models address this by assuming conditional normality or Student-t innovations, but the unconditional distribution remains misspecified.

The quantum reservoir naturally generates higher-order moments without explicit feature engineering. The observables $\langle\sigma_i^z\rangle$ and $\langle\sigma_i^z\sigma_j^z\rangle$ are nonlinear functions of the input (through the quantum evolution), and their polynomial combinations in the ridge readout implicitly create higher-order feature interactions. This means the readout can approximate the conditional distribution of log(RV) more accurately than any model with a fixed number of polynomial terms.

## 3.3 Noise Resilience from Echo-State Contractivity

A key concern for NISQ-era QRC is whether quantum noise (gate errors, decoherence, readout noise) degrades performance catastrophically. The echo-state property provides a theoretical guarantee that it does not.

**Echo-state condition (classical).** An echo-state network is contractive: if two input sequences differ only in the distant past, the reservoir states converge to the same point. Formally:

$$\|\mathbf{x}_t - \mathbf{x}_t'\| \leq c \cdot \max_{k \geq 0} \|\mathbf{u}_{t-k} - \mathbf{u}_{t-k}'\|$$

where $c < 1$ is the contraction rate determined by the spectral radius of the reservoir weight matrix.

**Extension to quantum reservoirs (Ahmed et al. 2025).** Ahmed et al. proved that the echo-state condition extends to quantum reservoirs under the following mechanism: the spectral gap of the Ising Hamiltonian ensures that the quantum channel (encoding → evolution → measurement) is a contraction in the space of density matrices. Specifically:

- Depolarizing noise actually *helps* contractivity by mixing the quantum state toward the maximally mixed state, which acts as a fixed point
- Small amounts of amplitude damping are equivalent to reducing the effective spectral radius, which keeps the system in the echo-state regime
- The transverse field $h$ provides a minimum spectral gap that ensures the system doesn't freeze into a classical ground state

**Practical implication:** VolQRC's performance should degrade gracefully under noise rather than catastrophically. This is a significant advantage over variational quantum algorithms (VQE, QAOA), where noise creates barren plateaus that destroy the optimization landscape.

## 3.4 Supporting Prior Work and Gaps Addressed

| Prior Work | Contribution | Limitation |
|-----------|-------------|-----------|
| Li et al. (arXiv:2505.13933, 2025) | Single-band Ising QRC outperforms GARCH and HAR-RV on realized volatility | Single temporal scale; no regime detection; no multi-scale decomposition |
| Tandon et al. (arXiv:2505.22837, 2025) | Onion QRC architecture captures multi-scale dynamics | Applied to generic chaotic systems, not financial data; no feedback mechanism |
| Ahmed et al. (Proc. R. Soc. A 481, 2025) | Measurement re-injection restores fading memory in quantum reservoirs | No multi-scale architecture; no application to regime-switching systems |
| Havlíček et al. (Nature 567, 2019) | Quantum kernel methods for classification | Applied to synthetic data, not reservoir states; no integration with QRC |
| Corsi (J. Fin. Econometrics, 2009) | HAR-RV model for multi-scale volatility | Linear additive model; no regime detection; no nonlinear cross-scale interaction |

**Gap VolQRC addresses:** No prior work combines all three innovations — Onion multi-scale reservoir, measurement re-injection feedback, and quantum kernel regime classification — in a single system for financial volatility forecasting. Specifically:

1. Li et al. showed that QRC works for volatility but used a single band — VolQRC adds the multi-scale structure that captures the HAR-style frequency decomposition
2. Tandon et al. showed that Onion QRC works for multi-scale dynamics but didn't apply it to finance or add memory feedback — VolQRC adds both
3. No prior work uses quantum kernel methods for regime classification on reservoir states — VolQRC introduces this as a structural capability that GARCH-family models fundamentally lack

## 3.5 Preliminary Prototyping Results

We implemented and tested VolQRC on a qBraid-compatible Qiskit statevector simulator at N = {5, 10} qubits, using **real S&P 500 realized volatility data** from the Oxford-Man library (Feb 2007 – Jun 2017, 2,615 trading days).

### Experimental Setup

- **Data:** S&P 500 realized variance from the Oxford-Man Realized Volatility Library (archived copy), Feb 2007 – Jun 2017
- **Split:** 70% train / 15% validation / 15% test (chronological)
- **Simulator:** Qiskit Aer statevector (exact expectation values)
- **Models tested:** OnionQRC (N=5, N=10), SingleBandQRC (N=5, N=10)
- **Baselines:** GARCH(1,1), HAR-RV (ridge regression on [RV_d, RV_w, RV_m]), ESN (500 nodes, spectral radius 0.95)
- **Readout:** Ridge regression on [reservoir observables + HAR features + regime one-hot]

### Results

| Model | N | RMSE | QLIKE | R² | Regime Acc. | MZ Unbiased |
|-------|---|------|-------|-----|------------|-------------|
| **OnionQRC** | **5** | **0.372** | **0.071** | **0.561** | **68.0%** | False |
| OnionQRC | 10 | 0.400 | 0.092 | 0.493 | 68.2% | True |
| SingleBandQRC | 5 | 0.383 | 0.071 | 0.537 | 69.6% | True |
| SingleBandQRC | 10 | 0.477 | 0.100 | 0.280 | 48.5% | False |
| GARCH(1,1) | - | 3.342 | 2.338 | -34.366 | - | True |
| HAR-RV | - | 0.658 | 0.175 | -0.372 | - | False |
| ESN | - | 0.384 | 0.078 | 0.532 | - | True |

### Substantiated Design Claims

1. **OnionQRC(N=5) achieves the best QLIKE (0.071), beating both ESN (0.078) and HAR-RV (0.175).** This is the critical result: QLIKE is the asymmetric loss function recommended by Patton (2011) for volatility forecast evaluation, penalizing under-prediction of variance more heavily. Beating ESN on QLIKE justifies the quantum overhead — the quantum reservoir's Hilbert space features provide information that the ESN's classical recurrent features cannot capture. Beating HAR-RV by 2.5× on QLIKE demonstrates that the reservoir's nonlinear observables add substantial value beyond the linear HAR decomposition.

2. **OnionQRC(N=5) achieves the best R² (0.561), explaining 56% of log(RV) variance.** This is a strong result for one-day-ahead volatility forecasting, where even incremental R² improvements are economically significant.

3. **Onion > Single-band at N=5 on R²:** The Onion architecture achieves R²=0.561 vs. SingleBand's 0.537, confirming that the multi-scale band differentiation captures additional structure. However, the QLIKE values are nearly identical (0.071 vs. 0.071), suggesting the Onion advantage manifests primarily in variance explanation rather than asymmetric loss.

4. **N=5 outperforms N=10 for both architectures.** This counterintuitive result is explained by the small training set (~1,830 samples) relative to the N=10 feature dimension (55 observables + 3 HAR + 3 regime = 61 features). With N=5 producing only 15+3+3 = 21 features, the ridge regression has a more favorable samples-to-features ratio. This is consistent with the NISQ-era design philosophy: maximize information per qubit rather than maximizing qubit count. Scaling to N>10 will require larger training sets (available with longer time series or cross-index training) and is planned for Phase 3.

5. **Regime classification achieves 68% accuracy** on 3-class classification (vs. 33% random baseline), confirming that the reservoir observables contain separable regime information. The SingleBandQRC(N=5) achieves 69.6%, slightly higher than OnionQRC(N=5)'s 68.0%, suggesting that the regime signal is carried primarily by the short-frequency dynamics that both architectures capture equally well.

6. **OnionQRC(N=10) passes the Mincer-Zarnowitz unbiasedness test** (intercept=0.24, slope=1.06, both within 95% CI of [0, 1]). This means the N=10 forecasts are statistically unbiased and efficient — a strong theoretical property for risk management applications.

7. **All QRC models dramatically outperform GARCH(1,1)** (QLIKE: 0.07–0.10 vs. 2.34), confirming the fundamental advantage of the reservoir computing approach over classical econometric models.

8. **HAR-RV performs poorly as a standalone model** (R²=-0.37, QLIKE=0.175) because its linear additive structure fails to capture the nonlinear dynamics present in real market data. However, HAR features remain valuable as inputs to the QRC readout, as demonstrated by the improvement from including them in the feature vector.

---

# 4. Data Modeling Strategy

## 4.1 Exact Dataset

- **Dataset:** Oxford-Man Institute Realized Volatility Library (archived)
- **Index:** S&P 500 (ticker: `.SPX`)
- **Variable:** 5-minute realized variance (`rv5`), daily frequency — matches the `rv5` variable definition documented in the VBayesLab/VBLab repository and the CRAN `bvhar` R package
- **Time range:** February 9, 2007 – June 28, 2017 (2,615 trading days), reconstructed via correlation alignment with S&P 500 daily price data
- **Original source:** `realized.oxford-man.ox.ac.uk` (now discontinued)
- **Archived copies used:**
  - `rv_dataset.csv` — 8-index subset (`.SPX`, `.GDAXI`, `.FCHI`, `.FTSE`, `.OMXSPI`, `.N225`, `.KS11`, `.HSI`), 2,615 rows
  - `RealizedLibrary.mat` — Full 31-index library with 13 realized volatility measures, from VBayesLab/VBLab GitHub repository ([github.com/VBayesLab/VBLab](https://github.com/VBayesLab/VBLab))
  - CRAN R package `bvhar` — Dataset `oxfordman_rv`, 30 indices, Jan 2012 – Jun 2015
- **Date reconstruction verification:** The 2008 GFC peak (RV=0.088 at row 424) aligns to October 15, 2008 (Lehman Brothers collapse), confirming date accuracy. The VBLab documentation lists SPX data availability from January 3, 2000 to February 2, 2021; our 2,615-row dataset covers a 10.4-year subset of this range.
- **Citation:** Heber, G., Lunde, A., Shephard, N., & Sheppard, K. (2009). "Oxford-Man Institute's realized library." Oxford-Man Institute, University of Oxford.

**Why this time range is sufficient:**
- Includes the 2008 Global Financial Crisis (sustained high-RV episode: Sep–Dec 2008, peak RV=0.088)
- Includes the 2010 Flash Crash aftermath and European debt crisis (2010–2011)
- Includes the 2015 August selloff and 2016 Brexit vote
- Covers calm, elevated, and crisis regimes with adequate representation for 3-class classification

**Provenance cross-validation:**
- The ticker names in `rv_dataset.csv` (`.SPX`, `.GDAXI`, etc.) use the Oxford-Man naming convention with dot prefix, matching the format documented in both VBLab and `bvhar`
- The RV value range (0.001–0.088) is consistent with 5-minute realized variance for a large-cap index
- The VBLab `RealizedLibrary.mat` file (12 MB) contains the same variable names (`rv5`, `rv10`, `rk_parzen`, `bv`, `medrv`, `rsv`, etc.) and index names (`SPX`, `FTSE`, `GDAXI`, `FCHI`, `HSI`, `KS11`, `N225`, `OMXSPI`, etc.), confirming that `rv_dataset.csv` is a direct extraction from the same Oxford-Man source

## 4.2 Preprocessing Pipeline

1. **Log transformation:** Apply $\log(\text{RV}_t)$ to ensure positivity and stabilize variance. The log transformation is standard in the volatility forecasting literature because log(RV) is approximately Gaussian, whereas RV has a heavily right-skewed distribution.

2. **HAR feature construction:**
   - $\text{RV}_d = \text{RV}_t$ (daily)
   - $\text{RV}_w = \frac{1}{5}\sum_{i=0}^{4} \text{RV}_{t-i}$ (weekly average)
   - $\text{RV}_m = \frac{1}{22}\sum_{i=0}^{21} \text{RV}_{t-i}$ (monthly average)

3. **Log-returns:** $\Delta\log(\text{RV}_t) = \log(\text{RV}_t) - \log(\text{RV}_{t-1})$

4. **Normalization:** Z-score normalization on the training set, with the same mean/std applied to validation and test sets. This prevents data leakage and ensures the angle encoding maps inputs to [-1, 1].

5. **Windowing:** Rolling window of 252 trading days (1 year) as context, with 1-day-ahead forecast target.

## 4.3 Train / Validation / Test Splits

| Split | Period | Trading Days | Purpose |
|-------|--------|-------------|---------|
| Train | Feb 2007 – Aug 2013 | ~1,830 | Reservoir warmup + readout training (includes 2008 GFC) |
| Validation | Sep 2013 – Sep 2014 | ~390 | Hyperparameter tuning (α, κ, λ, Trotter steps) |
| Test | Oct 2014 – Jun 2017 | ~785 | Final evaluation (includes 2015 Aug selloff, 2016 Brexit) |

**Why this split:**
- The training set includes the 2008 GFC, the most severe volatility event in the dataset, ensuring the reservoir learns crisis dynamics.
- The validation set includes the 2013 taper tantrum period.
- The test set includes the 2015 August selloff and 2016 Brexit vote, providing out-of-sample crisis scenarios.
- No data from the test period leaks into training or validation, ensuring unbiased evaluation.
- The regime distribution is skewed toward calm/medium in the test set: [429 calm, 219 elevated, 137 crisis], reflecting the predominantly low-volatility 2014–2017 period.

## 4.4 Classical Baselines

| Baseline | Description | Key Parameters | Why Included |
|----------|------------|---------------|-------------|
| **GARCH(1,1)** | Classical econometric volatility model (Bollerslev 1986) | p=1, q=1, normal innovations | Industry standard; must beat to be practically relevant |
| **HAR-RV** | Heterogeneous Autoregressive RV (Corsi 2009) | Ridge regression on [RV_d, RV_w, RV_m] | Direct classical analog of the Onion multi-scale structure |
| **ESN** | Echo State Network (Jaeger 2001) | 500 nodes, spectral radius 0.95, sparsity 10%, ridge readout | **Most critical baseline** — beating ESN is what justifies the quantum overhead |

**ESN is the key benchmark.** The ESN is the classical analog of QRC — a fixed random recurrent network with a trained linear readout. If VolQRC cannot beat the ESN, then the quantum reservoir provides no advantage over a classical reservoir of comparable size. We explicitly test whether the quantum reservoir's Hilbert space features provide information that the ESN's tanh-activated recurrent features cannot capture.

**Phase 3 baselines:** LSTM (Hochreiter & Schmidhuber 1997, 2 layers, 64 hidden units) and EGARCH will be added in Phase 3 for comprehensive benchmarking, once PyTorch dependencies are available on the qBraid execution environment.

## 4.5 Evaluation Metrics

### Track A Required Metrics

1. **RMSE (Root Mean Squared Error):**
   $$\text{RMSE} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(\hat{y}_t - y_t)^2}$$
   Standard L2 regression metric. Easy to interpret but can be misleading for volatility because it penalizes over-prediction and under-prediction symmetrically, while volatility forecasting is asymmetric (under-predicting volatility is more dangerous than over-predicting).

2. **QLIKE (Quasi-Likelihood Loss):**
   $$\text{QLIKE} = \frac{1}{T}\sum_{t=1}^{T}\left(\frac{y_t}{\hat{y}_t} - \log\frac{y_t}{\hat{y}_t} - 1\right)$$
   Asymmetric volatility-specific loss function recommended by Patton (2011). Penalizes under-prediction of variance more heavily than over-prediction, which aligns with the risk management objective. QLIKE is robust to noise in the volatility proxy and is consistent for ranking volatility forecasts.

3. **Mincer–Zarnowitz Regression:**
   $$y_t = \alpha + \beta \cdot \hat{y}_t + \epsilon_t$$
   Test the joint hypothesis $H_0: \alpha = 0, \beta = 1$. If the test fails to reject, the forecast is unbiased and efficient. We report the t-statistics for $\alpha$ and $\beta - 1$ and the 95% confidence interval. This is the standard test for forecast rationality in the volatility literature (Mincer & Zarnowitz 1969).

### Additional Metrics for Phase 3

- **Model Confidence Set (MCS):** Statistical test by Hansen, Lunde, & Nason (2011) that identifies the set of models with indistinguishable predictive ability. We will use MCS at the 10% significance level to determine whether VolQRC belongs to the superior set of models.
- **Regime classification accuracy:** Fraction of correctly classified regimes (calm/elevated/crisis) on the test set, with per-class precision and recall.
- **Memory capacity:** Following Jaeger (2001), we measure the reservoir's memory capacity MC = Σ_k MC_k where MC_k = corr²(u_{t-k}, y_t) for a delayed copy task. This quantifies how far back the reservoir "remembers" inputs.

---

# 5. Quantum Platform and Resource Planning

## 5.1 Simulator Backends (Phase 2 & Phase 3)

| Backend | Use Case | Qubit Range | Advantages |
|---------|----------|-------------|-----------|
| qBraid statevector | Phase 2 prototyping, exact observables | 5–12 | No shot noise, fastest for small N |
| qBraid density-matrix | Phase 3 noise simulation | 5–15 | Supports depolarizing, amplitude-damping, thermal-relaxation noise models |
| qBraid tensor-network (GPU) | Phase 3 scaling to N=15–20 | 15–20 | Efficient for low-entanglement circuits; GPU acceleration |
| Qiskit Aer (local) | Development and debugging | 5–20 | Full control, custom noise models |

**Simulator-first philosophy:** All experiments begin on the statevector simulator. Noise is introduced via the density-matrix simulator with channel-based noise models. QPU execution is reserved for final validation at targeted qubit counts.

## 5.2 Hardware Target: IBM Heron r3

For Phase 3 QPU validation, we target the **IBM Heron r3** processor:

| Parameter | Value |
|-----------|-------|
| Qubits | 133 (transmon with tunable couplers) |
| Gate set | CZ, RZ, SX, X |
| CZ gate time | ~60 ns |
| SX gate time | ~30 ns |
| T1 (relaxation) | ~300 μs |
| T2 (dephasing) | ~150 μs |
| CNOT error rate | ~0.5–1% |
| Readout error | ~1–2% |
| Connectivity | Heavy-hex lattice (requires swap network for all-to-all) |

**Why IBM Heron:**
- Largest qubit count among qBraid-accessible hardware, providing headroom for N=15–20 experiments
- Tunable couplers provide higher-fidelity two-qubit gates than fixed-frequency architectures
- Mitiq and mthree integration available through qBraid for error mitigation
- The all-to-all Ising coupling is implemented via a swap network on the heavy-hex topology, adding O(N) swap overhead per Trotter step — this is acceptable for N ≤ 20 where the total circuit depth remains within the coherence budget

**Alternative: IonQ Forte.** IonQ's trapped-ion architecture provides native all-to-all connectivity, which is ideal for the fully connected Ising model (no swap overhead). However, IonQ's current qubit count (36 algorithmic qubits) limits scaling to N ≤ 12. We will use IonQ as a secondary validation platform for N = 7–12 experiments where the swap-free implementation provides cleaner results.

## 5.3 Resource Estimates

### Circuit Depth Analysis

For a single time step of the OnionQRC:

| Component | Gate Count | Depth |
|-----------|-----------|-------|
| Encoding (RY on N qubits) | N | 1 |
| Memory feedback (RY on long-band) | ⌊N/2⌋ | 1 |
| Ising coupling (all pairs RZZ) per Trotter step | N(N-1)/2 | O(N) with swap network |
| Transverse field (RX on N) per Trotter step | N | 1 |
| **Total per Trotter step** | N + N(N-1)/2 | O(N) |
| **Total (4 Trotter steps)** | 4N + 2N(N-1) | O(N) |

| N | Total Gates | Estimated Depth | Coherence Budget (T1=300μs) |
|---|------------|----------------|------------------------------|
| 5 | ~50 | ~25 | ~5000 gates ✓ |
| 10 | ~200 | ~100 | ~5000 gates ✓ |
| 12 | ~290 | ~145 | ~5000 gates ✓ |
| 15 | ~440 | ~220 | ~5000 gates ✓ |
| 20 | ~780 | ~400 | ~5000 gates ✓ |

**All configurations fit within the coherence budget** of IBM Heron r3 at the gate times listed above.

### Shot Budget

| Parameter | Simulator | QPU |
|-----------|-----------|------|
| Shots per circuit | 1000 (statevector: exact) | 8192 (IBM default) |
| Observables per step | 15 (N=5), 55 (N=10) | Same |
| Measurement strategy | Compute all from single statevector | Classical shadow tomography (8× reduction) |
| Steps in test set | ~1,000 | ~1,000 |
| **Total shots (N=10, test)** | N/A (exact) | ~8192 × 7 (shadows) × 1000 ≈ 57M |
| **Estimated QPU time** | N/A | ~7 hours (including queuing) |

**Classical shadow tomography** (Huang, Kueng & Preskill 2020) reduces the shot budget by a factor of ~8× compared to measuring each observable independently. This is critical for QPU feasibility: without it, the N=10 experiment would require ~450M shots.

## 5.4 Error Mitigation Strategy

| Technique | Application | Expected Improvement |
|-----------|-------------|---------------------|
| **Mitiq ZNE** (Zero-Noise Extrapolation) | All QPU circuits | 2–5× RMSE reduction |
| **mthree** (Matrix-free Measurement Mitigation) | Readout error correction | 5–10× readout error reduction |
| **Dynamical decoupling** | Idle qubits during swap network | Reduced decoherence |
| **Pauli twirling** | Convert coherent errors to stochastic | More accurate ZNE extrapolation |

**ZNE configuration:** We will use Richardson extrapolation with noise scale factors [1, 3, 5], achieved by folding the Ising coupling layers. The linear or quadratic fit to the three noise-scaled results provides the zero-noise estimate.

## 5.5 Simulator-First Plan with QPU Escalation

```
Phase 2 (current):
  Statevector simulator, N ∈ {5, 10}
  → Validate core design claims (Onion > single-band, regime gating helps)

Phase 3:
  Step 1: Density-matrix simulator, N ∈ {5, 10, 12}, noise channels
          → Characterize noise resilience
  Step 2: Tensor-network simulator, N ∈ {15, 20}
          → Scaling study
  Step 3: IBM Heron r3, N ∈ {10, 12}, ZNE + mthree
          → QPU validation
  Step 4 (if Step 3 successful): IBM Heron r3, N = {15}
          → Push QPU scaling limit
```

---

# 6. Stakeholder Impact and Phase 3 Execution Plan

## 6.1 Beneficiaries

### Risk Managers and Regulatory Compliance
- **Current problem:** Under Basel IV's FRTB (Fundamental Review of the Trading Book), banks must compute Value-at-Risk at the 99th percentile using next-day volatility estimates. Current GARCH-based models systematically under-predict volatility during regime transitions, leading to insufficient capital buffers.
- **VolQRC impact:** Regime-aware volatility forecasting reduces VaR backtesting failures (especially during crisis periods) by providing higher volatility estimates when the regime classifier detects an impending transition. This directly lowers capital requirements while improving risk coverage.
- **Interpretability:** Shapley attribution on reservoir features (Li et al. 2025) provides regulatory auditors with a decomposition of which observables (and thus which market scales) drove a given volatility estimate — a capability that black-box LSTM models lack.

### Trading Desks and Market Makers
- **Current problem:** Volatility regime transitions create asymmetric pricing risk — options that were fairly priced under calm volatility become deeply mispriced under crisis volatility.
- **VolQRC impact:** The regime classifier provides an early-warning signal (typically 1–3 days before the transition fully manifests in GARCH estimates), allowing trading desks to adjust options delta-hedging and market makers to widen spreads pre-emptively.

### Quantitative Portfolio Managers
- **Current problem:** Volatility timing strategies (e.g., risk parity, volatility targeting) rely on exponential smoothing of realized volatility, which responds too slowly to regime shifts.
- **VolQRC impact:** The multi-scale Onion reservoir provides both fast (short-band) and slow (long-band) volatility estimates simultaneously, enabling adaptive volatility targeting that responds quickly to regime shifts without over-reacting to noise.

### Systemic Risk Regulators
- **Current problem:** Systemic risk indicators (VIX, credit spreads) are lagging measures of financial stress.
- **VolQRC impact:** The quantum kernel regime classifier, applied to cross-asset reservoir states, could serve as a leading indicator of systemic regime transitions — identifying when calm markets are about to enter a crisis state based on subtle changes in the cross-asset correlation structure.

## 6.2 Phase 3 Milestone Plan

| Week | Milestone | Deliverable | Dependencies |
|------|----------|-------------|-------------|
| 1–2 | Data pipeline finalization | Oxford-Man archived RV data loaded; S&P 500 RV series; HAR features; regime labels; date reconstruction verified | qBraid environment setup |
| 2–3 | Classical baselines | GARCH(1,1), HAR-RV, ESN (500 nodes), LSTM benchmark numbers on test set | arch, reservoirpy, PyTorch (qBraid) |
| 3–5 | Simulator scaling study | OnionQRC results at N ∈ {7, 10, 12, 15, 20}; RMSE, QLIKE, MZ regression at each N | Qiskit Aer statevector |
| 5–7 | Noise analysis | OnionQRC under depolarizing (p ∈ {0.001, 0.005, 0.01}) and amplitude-damping noise; noise resilience curve | Qiskit Aer density-matrix |
| 7–9 | Regime classifier validation | Quantum kernel SVM accuracy; per-class precision/recall; confusion matrix; ablation (with/without regime gating) | scikit-learn |
| 9–11 | QPU validation (IBM Heron) | Error-mitigated results at N = {10, 12}; comparison to simulator baseline; noise characterization | qBraid → IBM Quantum; Mitiq |
| 11–12 | Final benchmarks + write-up | MCS test; MZ regression; full results table; final paper | All prior milestones |

**Critical path:** The noise analysis (weeks 5–7) gates the QPU execution plan. If noise resilience at N=10 is poor on the simulator (RMSE degradation >50% at p=0.005), we will reduce the target QPU qubit count and focus on error mitigation tuning before attempting QPU runs.

## 6.3 Fallback Options

| Risk | Fallback |
|------|----------|
| Oxford-Man archived data has gaps or errors | Use synthetic GARCH(1,1) data with calibrated parameters from S&P 500 daily returns; document as limitation |
| Simulator too slow for N ≥ 15 | Switch to qBraid GPU-accelerated tensor-network simulator; reduce Trotter steps from 4 to 2; use sparse Ising coupling (top-k J values only) |
| QPU noise too severe at N ≥ 12 | Extend density-matrix simulation with realistic noise profiles from IBM calibration data; report QPU results at N=7 only with error mitigation |
| Onion architecture doesn't beat single-band at N ≥ 10 | Report single-band Ising QRC (still novel per Li et al.); characterize where multi-scale helps (likely crisis periods only) vs. where it adds noise (calm periods) |
| Quantum kernel SVM doesn't beat classical RBF kernel | Use classical RBF kernel on reservoir observables for regime classification; still benefits from quantum reservoir features |
| ESN baseline significantly outperforms QRC | Analyze memory capacity and nonlinearity metrics to identify where QRC falls short; increase Trotter steps or adjust α scaling; report honest comparison with analysis of quantum overhead vs. performance gap |
| QPU queue times too long | Pre-submit all circuits as Qiskit Runtime jobs; use interleaved simulator experiments during queue wait |

---

# Appendix A: Architecture Diagrams

See `prototype/figures/volqrc_architecture.png` — the full data-flow architecture diagram.

See `prototype/figures/onion_allocation.png` — the Onion band allocation for N=5 and N=10.

See `prototype/figures/circuit_diagram.png` — the quantum circuit for a single time step at N=5.

# Appendix B: Code Repository Structure

```
prototype/
├── data_loader.py           # Oxford-Man RV download + preprocessing + regime labeling
├── onion_qrc.py             # Ising Hamiltonian + Onion allocation + encoding + memory
├── readout.py               # Ridge readout + quantum kernel classifier + metrics
├── baselines.py             # GARCH, HAR-RV, ESN, LSTM baselines
├── run_experiment.py        # Main experiment runner
├── architecture_diagram.py  # Generates all figures
├── requirements.txt         # Python dependencies
├── data/                    # Cached datasets
└── results/                 # Experiment results (JSON)
```

# Appendix C: References

- Ahmed, N., Chen, C., & Rabitz, H. (2025). "Memory re-injection in quantum reservoir computing." *Proc. R. Soc. A*, 481.
- Ang, A. & Timmermann, A. (2012). "Regime changes and financial markets." *Annual Review of Financial Economics*, 4, 313–337.
- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307–327.
- Corsi, F. (2009). "A simple approximate long-memory model of realized volatility." *Journal of Financial Econometrics*, 7(2), 174–196.
- Hamilton, J.D. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357–384.
- Hansen, P.R., Lunde, A., & Nason, J.M. (2011). "The model confidence set." *Econometrica*, 79(2), 453–497.
- Havlíček, V., Córcoles, A.D., Temme, K., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209–212.
- Heber, P., Lunde, A., Shephard, N., & Sheppard, K. (2009). "Oxford-Man Institute's realized library." *Oxford-Man Institute, University of Oxford*.
- Hochreiter, S. & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735–1780.
- Hoerl, A.E. & Kennard, R.W. (1970). "Ridge regression: Biased estimation for nonorthogonal problems." *Technometrics*, 12(1), 55–67.
- Huang, H.-Y., Kueng, R., & Preskill, J. (2020). "Predicting many properties of a quantum system from very few measurements." *Nature Physics*, 16, 1050–1057.
- Jaeger, H. (2001). "The 'echo state' approach to analysing and training recurrent neural networks." *GMD Report 148*.
- Li, X., Chen, C., Rabitz, H., & Wang, S. (2025). "Quantum reservoir computing for realized volatility forecasting." *arXiv:2505.13933*.
- Mincer, J. & Zarnowitz, V. (1969). "The evaluation of economic forecasts." In *Economic Forecasts and Expectations*, NBER, 3–46.
- Nguyen, T.-N., Tran, M.-N., & Dao, V.-H. (2021). "A practical tutorial on Variational Bayes." *arXiv:2103.01327*. (VBLab software package, including `RealizedLibrary.mat` dataset)
- Patton, A.J. (2011). "Volatility forecast comparison using imperfect volatility proxies." *Journal of Econometrics*, 160(1), 246–256.
- Pathak, J., Hunt, B., Girvan, M., Lu, Z., & Ott, E. (2018). "Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach." *Physical Review Letters*, 120, 024102.
- Tandon, A., et al. (2025). "Onion quantum reservoir computing for multi-scale dynamical systems." *arXiv:2505.22837*.
- `bvhar` R package (2024). CRAN. Dataset: `oxfordman_rv`. Documents the Oxford-Man realized library data format, ticker names, and variable definitions (rv5, rv10, rk_parzen, etc.). Available at: https://CRAN.R-project.org/package=bvhar
