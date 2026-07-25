"""Phase 3 Causal Alignment & System Verification Unit Tests.

Enforces all 9 mandatory verification rules specified in Phase3_Execution_Plan.md §13.1:
1. Onion allocation sums to N for arbitrary N
2. Every qubit receives the intended band input
3. No scaler sees validation/test data
4. No feature timestamp reaches or exceeds its target timestamp
5. Regime thresholds are training-only
6. Statevector and count observables agree
7. Kernel matrices are symmetric and approximately positive semidefinite
8. Fixed seeds reproduce identical results
9. Result manifests contain all required resource numbers
"""

from math import comb
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gic.prototype.onion_qrc import OnionQRC, allocate_onion
from gic.prototype.data_loader import _add_har_features, make_windows, split_data
from gic.prototype.readout import IQPQuantumKernel, QuantumKernelClassifier


# -----------------------------------------------------------------------------
# Rule 1: Onion allocation sums to N
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("n_qubits", [5, 10, 15, 20, 24])
def test_onion_allocation_sums_to_n(n_qubits: int):
    """Verify that onion qubit partitioning sums to N and allocates across all 3 bands."""
    alloc = allocate_onion(n_qubits)
    assert alloc.total == n_qubits
    assert alloc.short_n + alloc.mid_n + alloc.long_n == n_qubits
    assigned = alloc.short_qubits + alloc.mid_qubits + alloc.long_qubits
    assert sorted(assigned) == list(range(n_qubits))
    assert min(alloc.short_n, alloc.mid_n, alloc.long_n) >= 1


# -----------------------------------------------------------------------------
# Rule 2: Every qubit receives the intended band input
# -----------------------------------------------------------------------------
def test_every_qubit_receives_intended_band_input():
    """Verify each qubit in short/mid/long bands receives its respective HAR feature."""
    pytest.importorskip("qiskit")
    qrc = OnionQRC(
        n_qubits=10,
        h_field=0.0,
        trotter_steps=1,
        alpha_short=1.0,
        alpha_mid=1.0,
        alpha_long=1.0,
    )
    # Phase 3 layout input: [log_rv_d, log_rv_w, log_rv_m] -> [1.0, 0.25, -1.0]
    features = np.array([1.0, 0.25, -1.0])
    circuit = qrc.build_circuit(features)

    ry_angles = {}
    for inst in circuit.data:
        if inst.operation.name == "ry":
            q_idx = circuit.find_bit(inst.qubits[0]).index
            ry_angles[q_idx] = float(inst.operation.params[0])

    assert set(ry_angles.keys()) == set(range(10))
    for q in qrc.alloc.short_qubits:
        assert ry_angles[q] == pytest.approx(np.pi / 2)
    for q in qrc.alloc.mid_qubits:
        assert ry_angles[q] == pytest.approx(np.arcsin(0.25))
    for q in qrc.alloc.long_qubits:
        assert ry_angles[q] == pytest.approx(-np.pi / 2)


# -----------------------------------------------------------------------------
# Rule 3: No scaler sees validation or test data
# -----------------------------------------------------------------------------
def test_no_scaler_sees_val_or_test_data():
    """Verify scaler parameters are fitted strictly on the training set."""
    dates = pd.date_range("2021-01-01", periods=100, freq="B", name="date")
    # Non-stationary synthetic series so train vs full means differ
    rv = np.linspace(0.01, 0.10, 100)
    df = _add_har_features(pd.DataFrame({"rv": rv}, index=dates)).dropna()

    split = split_data(
        df,
        context=1,
        train_fraction=0.60,
        val_fraction=0.20,
        features=["log_rv_d"],
    )

    train_mean = split["scaler"].mean_[0]
    n_train = len(split["train_idx"])
    expected_train_mean = df["log_rv_d"].iloc[:n_train].mean()
    assert train_mean == pytest.approx(expected_train_mean)

    # Full data mean must be different from train-only mean
    full_mean = df["log_rv_d"].mean()
    assert train_mean != pytest.approx(full_mean)


# -----------------------------------------------------------------------------
# Rule 4: No feature timestamp reaches or exceeds its target timestamp
# -----------------------------------------------------------------------------
def test_no_feature_timestamp_reaches_target_timestamp():
    """Verify for every window that max feature date < target date (strictly causal)."""
    dates = pd.date_range("2021-01-01", periods=60, freq="B", name="date")
    rv = np.linspace(0.01, 0.05, 60)
    df = _add_har_features(pd.DataFrame({"rv": rv}, index=dates)).dropna()

    windows = make_windows(
        df,
        context=5,
        horizon=1,
        features=["log_rv_d", "log_rv_w", "log_rv_m"],
        return_metadata=True,
    )

    feat_dates = windows["feature_index"]
    target_dates = windows["target_index"]

    assert len(feat_dates) == len(target_dates)
    for i in range(len(feat_dates)):
        assert feat_dates[i] < target_dates[i], (
            f"Sample {i} causal violation: feature timestamp {feat_dates[i]} >= target {target_dates[i]}"
        )


# -----------------------------------------------------------------------------
# Rule 5: Regime thresholds are training-only
# -----------------------------------------------------------------------------
def test_regime_thresholds_are_training_only():
    """Verify regime quantile thresholds are calculated strictly from training target RV."""
    dates = pd.date_range("2021-01-01", periods=120, freq="B", name="date")
    rv = np.linspace(0.005, 0.08, 120)
    df = _add_har_features(pd.DataFrame({"rv": rv}, index=dates)).dropna()

    split = split_data(
        df,
        context=1,
        train_fraction=0.60,
        val_fraction=0.20,
        features=["log_rv_d"],
    )

    t_train = split["regime_thresholds"]
    n_train = len(split["train_idx"])
    expected_thresholds = np.quantile(df["rv"].iloc[1:n_train + 1].to_numpy(), [0.33, 0.66])
    np.testing.assert_allclose(t_train, expected_thresholds)

    # Whole dataset thresholds must differ
    t_full = np.quantile(df["rv"].to_numpy(), [0.33, 0.66])
    assert not np.allclose(t_train, t_full)


# -----------------------------------------------------------------------------
# Rule 6: Statevector and count observables agree
# -----------------------------------------------------------------------------
def test_statevector_and_count_observables_agree():
    """Verify observables extracted from statevector match expectation values from counts."""
    qiskit = pytest.importorskip("qiskit")
    Statevector = qiskit.quantum_info.Statevector

    qrc = OnionQRC(
        n_qubits=5,
        topology="ring",
        trotter_steps=2,
        observable_order=2,
    )
    circuit = qrc.build_circuit([0.1, -0.2, 0.3])
    sv = Statevector.from_instruction(circuit)

    obs_sv = qrc.observables_from_statevector(sv)
    obs_counts = qrc.observables_from_counts(sv.probabilities_dict())

    np.testing.assert_allclose(obs_sv, obs_counts, atol=1e-12)


# -----------------------------------------------------------------------------
# Rule 7: Kernel matrices are symmetric and positive semi-definite
# -----------------------------------------------------------------------------
def test_kernel_matrices_are_symmetric_and_psd():
    """Verify IQP and RBF Gram matrices are symmetric and PSD."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 4))

    # Test IQP Quantum Kernel
    iqp_kernel = IQPQuantumKernel(n_qubits=4, scale=0.5)
    K_iqp = iqp_kernel.compute_kernel_matrix(X)

    np.testing.assert_allclose(K_iqp, K_iqp.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(K_iqp), 1.0, atol=1e-12)
    min_eig_iqp = np.linalg.eigvalsh(K_iqp).min()
    assert min_eig_iqp >= -1e-10

    # Test QuantumKernelClassifier kernel computation
    classifier = QuantumKernelClassifier(gamma=0.5)
    K_rbf = classifier._compute_kernel(X)
    np.testing.assert_allclose(K_rbf, K_rbf.T, atol=1e-12)
    min_eig_rbf = np.linalg.eigvalsh(K_rbf).min()
    assert min_eig_rbf >= -1e-10


# -----------------------------------------------------------------------------
# Rule 8: Fixed seeds reproduce identical results
# -----------------------------------------------------------------------------
def test_fixed_seeds_reproduce_identical_results():
    """Verify fixed RNG seed produces identical circuit parameters and expectation outputs."""
    qrc1 = OnionQRC(n_qubits=10, seed=2026, topology="ring")
    qrc2 = OnionQRC(n_qubits=10, seed=2026, topology="ring")

    np.testing.assert_array_equal(qrc1.ising.J, qrc2.ising.J)
    assert qrc1.ising.h == qrc2.ising.h

    pytest.importorskip("qiskit")
    obs1 = qrc1.step([0.2, -0.1, 0.4])
    qrc1_fresh = OnionQRC(n_qubits=10, seed=2026, topology="ring")
    obs2 = qrc1_fresh.step([0.2, -0.1, 0.4])
    np.testing.assert_allclose(obs1, obs2, atol=1e-12)


# -----------------------------------------------------------------------------
# Rule 9: Result manifests contain all required resource & performance metrics
# -----------------------------------------------------------------------------
def test_result_manifests_contain_required_resource_numbers():
    """Verify saved manifest files exist and contain required resource/performance fields."""
    manifest_dir = Path(__file__).resolve().parents[1] / "artifacts" / "manifests"
    assert manifest_dir.exists()

    summary_md = manifest_dir / "summary_table.md"
    assert summary_md.exists()
    content = summary_md.read_text(encoding="utf-8")
    assert "Test RMSE" in content
    assert "Test QLIKE" in content
    assert "Test MAE" in content
    assert "Test R2" in content
    assert "Runtime_s" in content

    ablations_json = manifest_dir / "ablations.json"
    assert ablations_json.exists()
    with ablations_json.open("r", encoding="utf-8") as f:
        ablations_data = json.load(f)
    assert "observable_order_ablation" in ablations_data
    assert "regime_gating_ablation" in ablations_data
    assert "regime_kernel_ablation" in ablations_data
