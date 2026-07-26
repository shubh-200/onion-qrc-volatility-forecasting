# GIC 2026 Phase 3 Execution Plan

## 1. Executive Recommendation

Phase 3 should **not begin by simply adding `N=15` and `N=20` to the existing experiment loop**. The current prototype demonstrates the overall concept, but several implementation and evaluation issues would invalidate the final comparisons if carried forward.

The recommended strategy is:

1. Correct the causal data pipeline and classical baselines.
2. Make the documented Onion encoding match the implementation.
3. Refactor execution into reproducible simulator and hardware backends.
4. Establish complete simulator results at `N ∈ {5, 10, 15, 20}`.
5. Introduce a hardware-compatible sparse reservoir alongside the fully connected research model.
6. Run a controlled 20-qubit validation on a qBraid-managed QPU.
7. Make claims about predictive utility and higher-order features, but **do not claim computational quantum advantage unless the results support it**.

The main final result should be a complete simulator benchmark. QPU execution should validate that the same reservoir can produce useful observables on real 20-qubit hardware.

---

## 2. What Is Currently Implemented

Based on `prototype/` and the saved results:

| Component | Current status |
|---|---|
| Oxford-Man S&P 500 RV loader | Implemented |
| Daily, weekly, monthly RV features | Implemented |
| Onion qubit allocation | Implemented for arbitrary `N` |
| Fully connected random Ising reservoir | Implemented |
| Four-step Trotter evolution | Implemented |
| Long-band classical measurement feedback | Implemented |
| Exact statevector observables | Implemented for `Z_i` and `Z_iZ_j` |
| Ridge readout | Implemented |
| RBF regime classifier | Implemented |
| IQP-like kernel code | Present but not used by the experiment |
| GARCH, HAR-RV, ESN classes | Implemented |
| LSTM class | Present but not run |
| `N=5` and `N=10` results | Present |
| `N=15` and `N=20` | Not executed |
| Shot-based simulation | Not implemented |
| Noise simulation | Not implemented |
| qBraid QPU execution | Not implemented |
| Hardware transpilation metrics | Not implemented |
| EGARCH, MCS, memory capacity | Not implemented |
| Reproduction-oriented README | Not implemented |

The current `README.md` only describes the upstream financial dataset. It does not reproduce VolQRC, which is a direct Phase 3 organizer requirement.

---

## 3. Issues That Must Be Corrected First

The saved Phase 2 numbers should be treated as **preliminary and not carried into the final submission unchanged**.

### 3.1 Validation Split Is Currently Ignored

`run_experiment.py` computes:

- 70% training
- 15% validation
- 15% test

However, reservoir evaluation starts at `train_end` and continues through `test_end`, combining validation and test into a single 30% evaluation set.

#### Phase 3 correction

Use strict chronological partitions:

- Training: fit scalers, reservoir/readout hyperparameters, and regime thresholds.
- Validation: select hyperparameters and model variants.
- Test: evaluate exactly once after configuration is frozen.

Store the exact dates and row indices in the result manifest.

### 3.2 Regime Thresholds Leak Test Information

`label_regimes()` calculates the 33rd and 66th percentiles over the complete dataset.

#### Phase 3 correction

Calculate both thresholds from training RV only, then apply those fixed thresholds to validation and test.

The regime classifier must predict the regime of `RV(t+1)` using only information available through time `t`.

### 3.3 Regime-Gated Training Uses Oracle Labels

The ridge model is trained with true regime labels but evaluated with predicted labels. This creates a training/inference mismatch and lets target information enter the training readout.

#### Phase 3 correction

Generate training gating features using expanding-window or time-series cross-fitted regime predictions:

1. Train the regime classifier on an earlier training fold.
2. Predict the next fold.
3. Concatenate all out-of-fold predictions.
4. Train the final ridge readout using those predicted labels.
5. Refit the classifier on all training data for validation/test prediction.

Add an explicit ablation:

- No regime signal
- Oracle regime, clearly marked as an upper bound
- Causally predicted regime, used as the actual model

### 3.4 HAR-RV Baseline Is Target-Leaking

The current HAR baseline uses `rv_d[t+1]` to predict `log_rv[t+1]`. Because `rv_d` is the target day's realized variance, this is contemporaneous information rather than a one-day-ahead forecast.

#### Phase 3 correction

For every model, enforce:

```text
Features available through day t → target RV on day t+1
```

Create a unit test that verifies the maximum feature timestamp is strictly earlier than the target timestamp.

### 3.5 GARCH Is Being Fitted to the Wrong Series

The current implementation fits GARCH to differences in `log(RV)`, then compares the resulting conditional variance against realized market variance. These quantities are not equivalent.

#### Phase 3 correction

Fit GARCH and EGARCH to aligned S&P 500 asset returns from `global index etf return/SPX.csv`.

Use rolling or expanding one-day-ahead forecasts. Do not generate one static multi-horizon forecast from the end of the training period.

### 3.6 The Implemented Encoding Differs From the Paper

The paper specifies:

- Short band: daily log RV repeated across short-band qubits
- Mid band: weekly log RV repeated across mid-band qubits
- Long band: monthly log RV repeated across long-band qubits

The implementation instead assigns the five input features sequentially to individual qubits. At `N>5`, many qubits receive no direct input encoding.

#### Phase 3 correction

Implement the documented band encoding:

```text
Every short-band qubit ← bounded daily log RV
Every mid-band qubit   ← bounded weekly log RV
Every long-band qubit  ← bounded monthly log RV
```

Keep price return and optional VIX as separately controlled ablations rather than silently changing the three-band architecture.

### 3.7 The Quantum Memory Is Actually Hybrid Classical Feedback

Each time step starts from a fresh `|0...0⟩` state. Only measured `Z` expectations are carried into the next circuit as classical rotations.

That is a valid hybrid recurrent reservoir, but the paper currently implies that the quantum state itself persists between days.

#### Phase 3 correction

Describe the mechanism precisely:

> Each daily circuit is initialized from zero. Temporal recurrence is provided by classical re-injection of previous long-band measurements.

This representation is also the practical one for QPU execution.

### 3.8 Scaling and Hardware Estimates Are Not Yet Substantiated

For the intended fully connected four-step circuit, the logical resource counts are approximately:

| N | Allocation S/M/L | Observables | Logical `RZZ` | Total logical gates* |
|---:|---:|---:|---:|---:|
| 5 | 1/1/3 | 15 | 40 | 68 |
| 10 | 2/3/5 | 55 | 180 | 235 |
| 15 | 3/5/7 | 120 | 420 | 502 |
| 20 | 5/6/9 | 210 | 760 | 869 |

\*Includes encoding, feedback, `RX`, and `RZZ`, but excludes measurement and native-gate decomposition.

The physical depth cannot be inferred as `O(N)` without actually transpiling against the selected device topology. A fully connected `N=20` circuit could require substantial routing on square or sparse superconducting hardware.

### 3.9 Classical Shadows Are Unnecessary for Current Observables

All current observables are products of `Z` operators. Therefore:

- `Z_i`
- `Z_iZ_j`
- Selected `Z_iZ_jZ_k`

can all be calculated from the same computational-basis bitstrings.

No separate measurement circuit or classical shadow is needed unless `X` or `Y` observables are introduced. This reduces the QPU shot requirement dramatically relative to the Phase 2 estimate.

---

## 4. Target Phase 3 Architecture

Two reservoir topologies should be evaluated.

### 4.1 FC-OnionQRC: Research Scaling Model

This preserves the Phase 1/2 proposal:

- Fully connected random `RZZ` coupling
- Four Trotter steps
- `N ∈ {5, 10, 15, 20}`
- Exact statevector simulation
- Primary purpose: architecture and scaling analysis

This model does not have to be the primary hardware implementation if transpilation makes it infeasible.

### 4.2 HW-OnionQRC: Hardware-Compatible Model

Create a bounded-degree Ising graph matching or embeddable in the selected QPU topology:

- Ring, grid, or device coupling subgraph
- Fixed nested topology across `N=5,10,15,20`
- Random fixed weights only on valid graph edges
- Same Onion encoding and feedback as FC-OnionQRC
- Same model simulated before QPU submission

This avoids presenting simulator and hardware results from structurally unrelated circuits.

### 4.3 Input Definition

Use the standard causal HAR signals:

```text
daily_t   = log(RV_t)
weekly_t  = log(mean(RV_{t-4:t}))
monthly_t = log(mean(RV_{t-21:t}))
target_t  = log(RV_{t+1})
```

Fit scaling parameters on training only. Instead of hard clipping large z-scores at `±1`, compare on validation:

- `clip(z / c, -1, 1)`
- `tanh(z / c)`

A smooth `tanh` transform is preferable because it avoids mapping all crisis observations to the same angle.

### 4.4 Observable Sets

Run three readout ablations using the same circuits:

1. Singles only: `Z_i`
2. Singles and pairs: `Z_i`, `Z_iZ_j`
3. Singles, pairs, and a fixed budget of selected triples

Selected `ZZZ` correlations can demonstrate higher-order pattern extraction without extra QPU measurement settings. Limit the triple feature count to avoid overfitting.

---

## 5. Classical Benchmark Protocol

All models must use identical chronological splits and targets.

### Required Baselines

| Baseline | Phase 3 implementation |
|---|---|
| Persistence | Predict `log(RV_t)` for `t+1` |
| HAR-RV | Properly lagged daily/weekly/monthly log RV |
| GARCH(1,1) | Rolling one-step forecast from SPX returns |
| EGARCH(1,1) | Rolling one-step forecast from SPX returns |
| ESN-210 | Comparable to the 210 `N=20` quantum observables |
| ESN-500 | Strong classical reservoir baseline |
| LSTM | Fixed architecture, validation early stopping |
| RBF-SVM | Classical regime-classifier baseline |

Add a classical random-feature ridge model with 210 features. This helps determine whether improvements come from the quantum reservoir specifically or merely from adding a random nonlinear expansion.

All random models should use at least three fixed seeds and report mean and standard deviation.

---

## 6. Hyperparameter Protocol

Do not tune independently on the test set or separately optimize every qubit count.

### Stage A: Architecture Selection

Use only training and validation at `N=5` and `N=10` to select:

- Total evolution time
- Trotter steps from `{1, 2, 4}`
- Feedback strength
- Transverse field
- Ridge regularization
- Input bounding scale
- Fully connected versus sparse topology

### Stage B: Scaling

Freeze those settings and evaluate:

```text
N = 5, 10, 15, 20
```

This makes any performance trend attributable to qubit scaling rather than increasing hyperparameter search effort.

### Stage C: Reservoir Seed Robustness

Use fixed seeds such as:

```text
42, 123, 2026
```

Report each seed and aggregate statistics. Avoid making the final claim from a single favorable random Hamiltonian.

---

## 7. Simulator Execution Plan

### 7.1 Correctness Run

Before full execution:

- `N=3` or `N=5`
- 50–100 time steps
- Verify observables against `SparsePauliOp` expectations
- Verify count-derived and statevector-derived `Z/ZZ` values agree
- Verify feedback reset and sequence behavior
- Verify deterministic output from a fixed seed

### 7.2 Performance Feasibility Run

For each `N`:

1. Run 100 time steps.
2. Record:
   - Circuit construction time
   - Transpilation time
   - Execution time per step
   - Peak memory
3. Extrapolate the full-run cost.
4. Select CPU, GPU statevector, or tensor-network method accordingly.

At `N=20`, use a parameterized circuit transpiled once where possible. Avoid rebuilding and retranspiling the complete reservoir for every day.

### 7.3 Primary Noiseless Matrix

Run:

```text
Topologies: FC-Onion, HW-Onion
Qubits:     5, 10, 15, 20
Seeds:      42, 123, 2026
Backend:    exact statevector
```

Cache the reservoir feature arrays. Readout and regime ablations can then be run without repeating quantum simulation.

### 7.4 Noise Matrix

Use:

- Depolarizing noise: `p ∈ {0.001, 0.005, 0.01}`
- Amplitude damping: canonical sweep plus device-derived `γ = 1 - exp(-t_gate/T1)`
- Finite shots: `{256, 512, 1024, 4096}` on a smaller shot-convergence subset

Exact density-matrix simulation should be restricted to small systems. A 20-qubit density matrix is not practical. For `N=15–20`, use:

- Shot-based noisy Aer simulation
- MPS/tensor-network simulation where suitable
- A fixed test interval rather than the complete training sequence if necessary

The main noise evaluation can use the test period with a documented warm-start memory state from the noiseless training sequence.

---

## 8. Quantum Regime-Kernel Plan

The current experiment uses an RBF kernel, not the proposed quantum kernel.

### Required Phase 3 Implementation

1. Standardize reservoir observables using training statistics.
2. Reduce dimension using training-only PCA.
3. Evaluate IQP feature maps at manageable dimensions such as:
   - 4 qubits
   - 8 qubits
   - 12 qubits if feasible
4. Use a compute-uncompute fidelity circuit rather than a swap test where possible.
5. Compare on the exact same samples:
   - Linear SVM
   - RBF SVM
   - IQP quantum-kernel SVM
6. Report:
   - Balanced accuracy
   - Macro F1
   - Per-class precision and recall
   - Confusion matrix
   - Kernel alignment
   - Runtime and kernel circuit count

A full QPU kernel matrix is not a sensible use of 10,000 credits. The complete quantum-kernel comparison should be simulator-based. Hardware kernel execution should be an optional small demonstration only after the QRC hardware budget is secured.

---

## 9. qBraid Hardware Strategy

### 9.1 Change the Primary Hardware Target

The Phase 2 plan targets IBM Heron through qBraid credits. Current qBraid documentation states that IBM hardware requires separate IBM credentials and is not provided through qBraid-managed access.

Also, current Heron r3 processors have 156 qubits, not the 133 stated in the Phase 2 paper.

Recommended priority:

1. **IQM Garnet** — exactly 20 qubits, CZ-based superconducting hardware.
2. **Rigetti Cepheus-1-108Q through AWS** — fallback with sufficient qubits.
3. IBM Heron — only if separate IBM access is already available.
4. IonQ Forte — technically attractive due to connectivity but too expensive for a meaningful shot panel under the current credit budget.

Sources:

- [qBraid on-demand pricing](https://docs.qbraid.com/v2/home/pricing)
- [qBraid quantum jobs](https://docs.qbraid.com/v2/lab/user-guide/quantum-jobs)
- [IBM processor types](https://quantum.cloud.ibm.com/docs/en/guides/processor-types)

Before committing credits, query `provider.get_devices()` and save the live metadata and pricing response.

### 9.2 Hardware Qualification Gates

For each candidate device:

1. Retrieve current:
   - Qubit count
   - Coupling map
   - Native gate set
   - Calibration data
   - Queue depth
   - Maximum shots
   - Live price
2. Select the best connected 20-qubit subgraph.
3. Transpile FC-Onion and HW-Onion at optimization levels 1 and 3.
4. Record:
   - Logical and transpiled depth
   - Two-qubit gate count
   - Two-qubit depth
   - SWAP count
   - Selected physical qubits
   - Estimated circuit error
5. Use HW-Onion for production if FC-Onion routing is excessive.
6. Reduce Trotter steps `4 → 2 → 1` before reducing below 20 qubits.

The production circuit must be frozen before hardware results are viewed.

### 9.3 QPU Evaluation Design

#### A. Scaling smoke tests

Run three representative observations at each:

```text
N = 5, 10, 15, 20
```

Use calm, elevated, and crisis examples. This confirms that the implementation genuinely reaches 20 physical qubits.

#### B. Balanced 20-qubit hardware panel

Pre-register 24 test observations:

- 8 calm
- 8 elevated
- 8 crisis

The dates must be selected programmatically before seeing QPU results.

Use simulator-derived prior memory for these independent circuits. Label this as **teacher-forced memory hardware validation**, intended to isolate hardware distortion.

#### C. End-to-end recurrent sequence

Run one contiguous 8-day sequence in which each next circuit uses the previous hardware measurement as its feedback state.

This is the genuine hardware recurrence demonstration. It is short because each circuit depends on completion of the preceding task.

#### D. Error mitigation subset

On four preselected observations, compare:

- Raw counts
- Local readout mitigation
- ZNE scale factors `1, 3, 5`

Do not promise that mitigation will improve the results. Report whether it did.

---

## 10. Proposed 10,000-Credit Budget

Using the currently listed IQM Garnet price of approximately:

```text
30 credits/task + 0.145 credits/shot
```

the following budget remains under 10,000 credits:

| Activity | Tasks | Shots/task | Estimated credits |
|---|---:|---:|---:|
| Device qualification | 8 | 128 | 388 |
| Scaling pilots at 5/10/15/20 | 12 | 256 | 805 |
| N=20 balanced panel | 24 | 1024 | 4,284 |
| Eight-day recurrent run | 8 | 512 | 834 |
| Extra ZNE circuits | 8 | 1024 | 1,428 |
| Readout calibration | 2 | 2048 | 654 |
| **Planned total** | 62 | — | **8,393** |
| **Reserve** | — | — | **1,607** |

This estimate must be regenerated from live device pricing before submission. Set code-level spending caps so a configuration error cannot consume the complete balance.

All `Z` and `ZZ` observables should be reconstructed from each circuit's bitstrings, so the number of observables does not multiply the shot cost.

---

## 11. Evaluation and Statistical Analysis

### Forecast Metrics

For every model:

- RMSE on log RV
- MAE on log RV
- QLIKE on RV scale
- R²
- Mincer-Zarnowitz intercept and slope
- Joint Mincer-Zarnowitz test using robust/HAC covariance
- Wall-clock training and inference time

### Statistical Comparisons

- Block-bootstrap 95% confidence intervals
- Diebold-Mariano tests on QLIKE loss differentials
- Model Confidence Set at 10%
- Mean and standard deviation across reservoir seeds

### Regime Metrics

- Accuracy
- Balanced accuracy
- Macro F1
- Per-class precision and recall
- Confusion matrix
- Transition-day accuracy

### Hardware Metrics

- Observable MAE versus ideal simulation
- Observable MAE versus device-noise simulation
- Prediction RMSE and QLIKE on the fixed hardware panel
- Regime balanced accuracy on the panel
- Raw versus mitigated difference
- Shots, queue time, execution time, and credits consumed
- Transpiled depth and two-qubit gate count

Do not merge the 24-date QPU panel metrics with the complete simulator test metrics. Report them in separate tables.

---

## 12. How to Present Quantum Advantage

At 20 qubits, the complete system remains classically simulable. Therefore, the final paper should not claim a proven asymptotic or computational advantage.

Use the following claim hierarchy.

### Strong Outcome

If VolQRC beats ESN, HAR-RV, and LSTM on test QLIKE with statistical support:

> VolQRC provides evidence of predictive quantum utility through quantum-generated nonlinear observables.

### Mixed Outcome

If VolQRC performs similarly but shows stronger crisis or transition behavior:

> VolQRC provides regime-conditional utility and higher-order feature sensitivity, but no broad performance advantage.

### Negative Outcome

If the classical reservoir wins:

> The experiment demonstrates successful 20-qubit execution but does not establish predictive advantage at this scale.

The quantum-specific evidence should come from:

1. Quantum-observable versus HAR-only ablation.
2. Singles versus pair/triple correlations.
3. Onion versus single-band reservoir.
4. Quantum reservoir versus matched 210-feature classical reservoir.
5. IQP kernel versus RBF kernel.
6. Crisis and transition-period analysis.
7. Hardware versus simulator observable preservation.

Runtime results must be reported even if the classical models are faster.

---

## 13. Reproducibility Work

The organizer says judges will rerun the submitted code exactly. The final repository should provide:

```text
src/volqrc/
    data.py
    circuits.py
    observables.py
    readout.py
    kernels.py
    baselines.py
    metrics.py
    backends/
        statevector.py
        noisy_aer.py
        qbraid_qpu.py

configs/
    phase3.yaml
    qpu_iqm.yaml
    qpu_rigetti.yaml

scripts/
    prepare_data.py
    run_baselines.py
    run_scaling.py
    run_noise.py
    submit_qpu.py
    retrieve_qpu.py
    build_report.py

tests/
artifacts/
    manifests/
    simulator/
    hardware/
    figures/
```

### Required Tests

- Onion allocation sums to `N`
- Every qubit receives the intended band input
- No scaler sees validation/test data
- No feature timestamp reaches or exceeds its target timestamp
- Regime thresholds are training-only
- Statevector and count observables agree
- Kernel matrices are symmetric and approximately positive semidefinite
- Fixed seeds reproduce identical results
- Result manifests contain all required resource numbers

### Hardware Artifacts to Preserve

- Provider and backend ID
- Job IDs
- Submission and completion timestamps
- Raw counts
- Circuit source or QASM
- Transpiled circuit
- Physical-qubit mapping
- Backend calibration snapshot
- Shot count
- Credit estimate and actual cost
- Software versions
- Git revision used for submission

The default judge workflow should use a simulator and should not require a secret. QPU submission should require `QBRAID_API_KEY` through the environment, never in the repository.

---

## 14. README and Final Paper Structure

### README Order

1. One-paragraph project summary
2. Final headline results table
3. Repository structure
4. Exact environment creation commands
5. One-command smoke test
6. Data provenance and checksum
7. Classical baseline reproduction
8. `N=5,10,15,20` simulator reproduction
9. Noise reproduction
10. Hardware result retrieval
11. Optional new QPU submission
12. Expected runtimes and hardware requirements
13. Limitations
14. Citation

### Final Paper Result Tables

1. Full classical and quantum forecast comparison
2. Scaling across `N=5,10,15,20`
3. Onion versus single-band ablation
4. Regime-gating ablation
5. Observable-order ablation
6. Noise-resilience table
7. Quantum versus classical regime kernels
8. QPU resource and execution table
9. Raw versus mitigated QPU results
10. Credit and wall-clock budget

---

## 15. Execution Milestones

| Milestone | Work | Exit criterion |
|---|---|---|
| M0: Correctness | Fix alignments, leakage, encoding, splits | Causal unit tests pass |
| M1: Reproducibility | Configs, CLI, manifests, environment lock | Clean environment runs smoke test |
| M2: Baselines | Persistence, HAR, GARCH, EGARCH, ESN, LSTM | All share identical test targets |
| M3: QRC scaling | `N=5,10,15,20`, FC and sparse | Complete noiseless results saved |
| M4: Noise | Shot convergence and noise sweeps | Noise degradation curves generated |
| M5: Regime module | Cross-fitted gating and IQP comparison | No oracle labels in final pipeline |
| M6: QPU qualification | Live device discovery and transpilation | Frozen 20-qubit circuit selected |
| M7: QPU execution | Pilots, balanced panel, recurrence, mitigation | Raw jobs and metadata archived |
| M8: Statistics | CIs, DM, MCS, MZ, regime analysis | Claims tied to statistical results |
| M9: Submission | README, figures, final paper | Fresh judge-style reproduction passes |

A realistic implementation and execution estimate is approximately **15–20 focused working days**, plus QPU queue time.

---

## 16. Risks and Fallbacks

| Risk | Primary mitigation | Fallback |
|---|---|---|
| Fully connected `N=20` simulation is too slow | Benchmark 100 steps and use GPU statevector | Use sparse hardware topology for full sequence and FC topology on a fixed subset |
| Fully connected QPU transpilation is too deep | Use HW-Onion on a native coupling subgraph | Reduce Trotter steps while preserving 20 qubits |
| QPU queue or outage | Qualify IQM and Rigetti before freezing backend | Execute on the available qualified device and report the change |
| Live prices exceed budget | Regenerate cost manifest and enforce spending cap | Reduce mitigation subset before reducing the main N=20 panel |
| Noise destroys pair correlations | Readout mitigation and shot convergence analysis | Report single-observable hardware readout and full noisy-simulator comparison |
| Onion model does not beat single-band | Report the result honestly and identify regime-specific behavior | Retain single-band as a benchmark, not as an unreported replacement |
| ESN or LSTM wins | Analyze matched feature count, memory capacity, and crisis periods | State that no predictive quantum advantage was observed |
| IQP kernel does not beat RBF | Report kernel alignment and accuracy honestly | Keep RBF gating over quantum reservoir features |
| VIX cannot be reproduced reliably | Keep bundled Oxford-Man data as primary dataset | Treat VIX as an optional ablation, not a required input |

---

## 17. Definition of Phase 3 Completion

Phase 3 is complete only when all of the following are true:

- A causally valid baseline suite has been executed.
- OnionQRC has been evaluated at `5`, `10`, `15`, and `20` qubits.
- Exact gate counts, transpiled depth, shots, runtime, and backend are reported.
- At least one real QPU run uses 20 physical qubits.
- Hardware results are compared with the matching simulator circuit.
- Noise and error-mitigation results are included.
- The regime classifier has no target leakage.
- Quantum, classical, and hybrid ablations are reported.
- The README reproduces a smoke test and the principal simulator table.
- Raw hardware counts and qBraid job metadata are preserved.
- The final paper explicitly states whether predictive quantum advantage was or was not observed.
