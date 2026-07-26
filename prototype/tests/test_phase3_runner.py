import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prototype.data_loader import _add_har_features
from prototype import qbraid_hardware as hardware
from prototype import run_phase3 as phase3


def _frame(n=120):
    dates = pd.date_range("2022-01-03", periods=n, freq="B", name="date")
    x = np.linspace(0.0, 8.0, n)
    rv = 0.01 + 0.002 * np.sin(x) + np.linspace(0.0, 0.01, n)
    returns = 0.01 * np.cos(x)
    return _add_har_features(
        pd.DataFrame({"rv": rv, "price_return": returns}, index=dates)
    ).dropna()


def test_prepare_uses_context_one_strict_splits_and_bounds_only_har():
    data = phase3.prepare_phase3_data(_frame(), tanh_scale=2.5)

    assert data["features"] == ("log_rv_d", "log_rv_w", "log_rv_m", "price_return")
    for name in phase3.SPLIT_NAMES:
        assert data[f"X_{name}"].ndim == 2
        np.testing.assert_allclose(
            data[f"X_{name}"][:, :3],
            np.tanh(data[f"X_raw_{name}"][:, :3] / 2.5),
        )
        np.testing.assert_allclose(
            data[f"X_{name}"][:, 3], data[f"X_raw_{name}"][:, 3]
        )
    assert data["target_index_train"][-1] < data["target_index_val"][0]
    assert data["target_index_val"][-1] < data["target_index_test"][0]


def test_quick_mode_uses_chronological_tail_before_split():
    frame = _frame(180)
    data = phase3.prepare_phase3_data(frame, quick=True, quick_rows=60)

    assert data["frame"].index[0] == frame.tail(60).index[0]
    assert data["frame"].index[-1] == frame.index[-1]
    assert sum(len(data[f"y_{name}"]) for name in phase3.SPLIT_NAMES) == 59


def test_persistence_recovers_unscaled_daily_har_feature(monkeypatch):
    data = phase3.prepare_phase3_data(_frame())
    monkeypatch.setattr(phase3, "_run_ridge_family", lambda *a, **k: {"model": "stub", "status": "ok"})
    monkeypatch.setattr(phase3, "_run_esn", lambda *a, **k: {"model": "stub", "status": "ok"})
    monkeypatch.setattr(phase3, "_run_arch_models", lambda *a, **k: [])
    monkeypatch.setattr(phase3, "_run_lstm", lambda *a, **k: {"model": "LSTM", "status": "skipped"})

    results = phase3.run_baselines(data, quick=True)
    persistence = results[0]
    assert persistence["model"] == "Persistence"
    assert persistence["status"] == "ok"
    assert persistence["validation_metrics"]["n_obs"] == len(data["y_val"])
    assert persistence["test_metrics"]["n_obs"] == len(data["y_test"])


def test_parser_allows_individual_n20_and_all_modes():
    parser = phase3.build_parser()
    args = parser.parse_args(["simulator", "--n-qubits", "20", "--quick"])
    assert args.n_qubits == [20]
    assert args.quick is True
    for mode in ("baselines", "simulator", "all"):
        assert parser.parse_args([mode]).mode == mode


def test_reservoir_features_are_cached_without_qiskit(monkeypatch, tmp_path):
    data = phase3.prepare_phase3_data(_frame(80), quick=True, quick_rows=50)

    class FakeQRC:
        calls = 0

        def __init__(self, n_qubits, **kwargs):
            self.n_observables = n_qubits

        def reset_memory(self):
            pass

        def step(self, row):
            FakeQRC.calls += 1
            return np.full(self.n_observables, row[0])

    monkeypatch.setattr(phase3, "OnionQRC", FakeQRC)
    first, first_info = phase3.reservoir_features(
        data, n_qubits=5, topology="ring", seed=42, cache_dir=tmp_path
    )
    call_count = FakeQRC.calls
    second, second_info = phase3.reservoir_features(
        data, n_qubits=5, topology="ring", seed=42, cache_dir=tmp_path
    )

    assert first_info["cache"] == "miss"
    assert second_info["cache"] == "hit"
    assert FakeQRC.calls == call_count
    np.testing.assert_allclose(first, second)


def test_json_and_csv_artifacts_include_metrics_and_metadata(tmp_path):
    result = phase3._model_result(
        "tiny", "ok", 0.125,
        validation={"rmse": 1.0}, test={"rmse": 2.0},
        backend="test_backend",
    )
    json_path, csv_path = phase3.write_artifacts(
        [result], tmp_path, {"versions": {"python": "test"}}
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["backend"] == "test_backend"
    assert payload["metadata"]["versions"]["python"] == "test"
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "val_rmse" in csv_text
    assert "test_rmse" in csv_text


def test_balanced_panel_is_deterministic_equal_and_chronological():
    dates = pd.date_range("2024-01-01", periods=18, freq="B")
    regimes = np.repeat([0, 1, 2], 6)
    first = hardware.select_balanced_panel(dates, regimes, 3, seed=7)
    second = hardware.select_balanced_panel(dates, regimes, 3, seed=7)

    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first) > 0)
    assert np.bincount(regimes[first], minlength=3).tolist() == [3, 3, 3]


def test_spend_cap_blocks_excess_and_submission_is_opt_in(monkeypatch):
    estimate = hardware.estimate_cost(6, 1000, cost_per_shot_usd=0.002)
    assert estimate.estimated_total_usd == pytest.approx(12.0)
    hardware.enforce_spend_cap(estimate, 12.0)
    with pytest.raises(RuntimeError, match="nothing was submitted"):
        hardware.enforce_spend_cap(estimate, 11.99)

    assert hardware.submit_circuits([object()], device_id="device", shots=10) == []
    monkeypatch.delenv(hardware.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=hardware.API_KEY_ENV):
        hardware.submit_circuits(
            [object()], device_id="device", shots=10, submit=True
        )


def test_device_discovery_is_offline_by_default():
    assert hardware.discover_devices() == []


def test_prepare_circuit_optional_qiskit_case(tmp_path):
    pytest.importorskip("qiskit")
    manifest, circuits = hardware.prepare_circuits(
        np.asarray([[0.1, -0.2, 0.3, 0.0]]),
        pd.DatetimeIndex(["2025-01-02"]),
        n_qubits=[5],
        topology="ring",
        seeds=[42],
        output_dir=tmp_path,
        optimization_level=0,
    )
    assert len(manifest) == len(circuits) == 1
    assert Path(manifest[0]["file"]).exists()
    assert manifest[0]["n_qubits"] == 5


def test_fixed_seed_reservoir_is_deterministic(tmp_path):
    """Running reservoir_features twice with the same mock QRC produces identical output."""
    data = phase3.prepare_phase3_data(_frame(80), quick=True, quick_rows=50)

    class CountingQRC:
        call_count = 0

        def __init__(self, n_qubits, **kwargs):
            # Use a fixed observable count: N + C(N,2) = 5 + 10 = 15 for N=5
            self.n_observables = n_qubits + n_qubits * (n_qubits - 1) // 2
            self._n_qubits = n_qubits

        def reset_memory(self):
            pass

        def step(self, row):
            CountingQRC.call_count += 1
            # Deterministic: hash of row drives output (same input → same output)
            rng = np.random.default_rng(int(abs(row[0]) * 1e6) % (2**31))
            return rng.standard_normal(self.n_observables)

    import gic.prototype.run_phase3 as _p3
    original_cls = _p3.OnionQRC
    _p3.OnionQRC = CountingQRC
    try:
        first, first_info = phase3.reservoir_features(
            data, n_qubits=5, topology="ring", seed=42, cache_dir=tmp_path / "a"
        )
        second, second_info = phase3.reservoir_features(
            data, n_qubits=5, topology="ring", seed=42, cache_dir=tmp_path / "a"
        )
    finally:
        _p3.OnionQRC = original_cls

    assert first_info["cache"] == "miss"
    assert second_info["cache"] == "hit"
    np.testing.assert_array_equal(first, second)


def test_result_manifest_contains_required_resource_keys(tmp_path):
    """JSON artifact must include n_qubits, topology, seed, backend, and resource_counts."""
    required_gate_keys = {"n_qubits", "topology", "interaction_edges", "trotter_steps",
                          "encoding_ry", "rzz", "two_qubit_gates", "n_observables"}
    result = phase3._model_result(
        "OnionQRC", "ok", 1.23,
        validation={"rmse": 0.5, "qlike": 0.1, "mae": 0.3, "r2": 0.8, "n_obs": 10},
        test={"rmse": 0.6, "qlike": 0.12, "mae": 0.35, "r2": 0.75, "n_obs": 5},
        n_qubits=10,
        topology="ring",
        seed=42,
        backend="qiskit_statevector_noiseless",
        noise_model="none",
        resource_counts={k: 0 for k in required_gate_keys},
    )
    json_path, _ = phase3.write_artifacts([result], tmp_path, {"test": True})
    import json
    payload = json.loads(json_path.read_text())
    rec = payload["results"][0]

    assert rec["n_qubits"] == 10
    assert rec["topology"] == "ring"
    assert rec["seed"] == 42
    assert rec["backend"] == "qiskit_statevector_noiseless"
    assert "resource_counts" in rec
    for key in required_gate_keys:
        assert key in rec["resource_counts"], f"Missing resource key: {key}"
