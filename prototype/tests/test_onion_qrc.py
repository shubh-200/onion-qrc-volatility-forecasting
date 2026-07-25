from math import comb
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onion_qrc import OnionQRC, allocate_onion


@pytest.mark.parametrize("n_qubits", [5, 10, 15, 20])
def test_allocations_cover_every_qubit(n_qubits):
    allocation = allocate_onion(n_qubits)
    assigned = (
        allocation.short_qubits
        + allocation.mid_qubits
        + allocation.long_qubits
    )

    assert allocation.total == n_qubits
    assert allocation.short_n + allocation.mid_n + allocation.long_n == n_qubits
    assert assigned == list(range(n_qubits))
    assert min(allocation.short_n, allocation.mid_n, allocation.long_n) >= 1


def test_observable_orders_and_selected_third_order_terms_are_deterministic():
    first_order = OnionQRC(5, observable_order=1)
    second_order = OnionQRC(5, observable_order=2)
    with_third_order = OnionQRC(5, observable_order=2, n_third_order=3)

    assert first_order.n_observables == 5
    assert second_order.n_observables == 5 + comb(5, 2)
    assert with_third_order.n_observables == 5 + comb(5, 2) + 3
    assert with_third_order.third_order_terms == ((0, 1, 2), (0, 1, 3), (0, 1, 4))
    assert OnionQRC(5, observable_order=2, n_third_order=3).observable_terms == (
        with_third_order.observable_terms
    )


@pytest.mark.parametrize(
    ("topology", "edge_count"),
    [("fully_connected", comb(10, 2)), ("ring", 10)],
)
def test_resource_estimates_count_logical_gates(topology, edge_count):
    qrc = OnionQRC(10, topology=topology, trotter_steps=3)
    estimate = qrc.estimate_resources(include_feedback=True, measure=True)

    assert len(qrc.interaction_edges) == edge_count
    assert estimate["encoding_ry"] == 10
    assert estimate["feedback_ry"] == qrc.alloc.long_n
    assert estimate["rx"] == 30
    assert estimate["rzz"] == edge_count * 3
    assert estimate["two_qubit_gates"] == edge_count * 3
    assert estimate["single_qubit_gates"] == 10 + qrc.alloc.long_n + 30
    assert estimate["measurements"] == 10
    assert estimate["logical_gates"] == (
        estimate["single_qubit_gates"] + estimate["two_qubit_gates"]
    )
    assert estimate["total_operations"] == estimate["logical_gates"] + 10


def test_band_encoding_repeats_bounded_har_feature_on_every_qubit():
    pytest.importorskip("qiskit")
    qrc = OnionQRC(
        10,
        h_field=0.0,
        trotter_steps=1,
        alpha_short=1.0,
        alpha_mid=1.0,
        alpha_long=1.0,
    )
    # Canonical layout: log_rv, rv_d, rv_w, rv_m, log_return.
    circuit = qrc.build_circuit(np.array([99.0, 2.0, 0.25, -2.0, -99.0]))
    encoded = {}
    for instruction in circuit.data:
        if instruction.operation.name == "ry":
            qubit = circuit.find_bit(instruction.qubits[0]).index
            encoded[qubit] = float(instruction.operation.params[0])

    assert set(encoded) == set(range(10))
    for qubit in qrc.alloc.short_qubits:
        assert encoded[qubit] == pytest.approx(np.pi / 2)
    for qubit in qrc.alloc.mid_qubits:
        assert encoded[qubit] == pytest.approx(np.arcsin(0.25))
    for qubit in qrc.alloc.long_qubits:
        assert encoded[qubit] == pytest.approx(-np.pi / 2)


def test_statevector_and_count_observables_agree():
    qiskit = pytest.importorskip("qiskit")
    Statevector = qiskit.quantum_info.Statevector
    qrc = OnionQRC(
        5,
        topology="ring",
        trotter_steps=2,
        observable_order=2,
        n_third_order=2,
    )
    state = Statevector.from_instruction(qrc.build_circuit([0.2, -0.3, 0.4]))
    from_state = qrc.observables_from_statevector(state)
    # Probability weights exercise the same public count path without sampling noise.
    from_counts = qrc.observables_from_counts(state.probabilities_dict())

    np.testing.assert_allclose(from_counts, from_state, atol=1e-12)


def test_measurement_option_and_actual_ring_gate_counts():
    pytest.importorskip("qiskit")
    qrc = OnionQRC(5, topology="ring", trotter_steps=2)
    circuit = qrc.build_circuit([0.1, 0.2, 0.3], measure=True)
    operations = circuit.count_ops()

    assert circuit.num_clbits == 5
    assert operations["ry"] == 5
    assert operations["rx"] == 10
    assert operations["rzz"] == 10
    assert operations["measure"] == 5


def test_feedback_uses_fresh_circuits():
    pytest.importorskip("qiskit")
    qrc = OnionQRC(5, topology="ring", trotter_steps=1)
    qrc.step([0.1, 0.2, 0.3])
    first = qrc.build_circuit([0.1, 0.2, 0.3])
    second = qrc.build_circuit([0.1, 0.2, 0.3])

    assert first is not second
    expected_ry = qrc.n_qubits + qrc.alloc.long_n
    assert first.count_ops()["ry"] == expected_ry
    assert second.count_ops()["ry"] == expected_ry


def test_fixed_seed_produces_identical_ising_params():
    """Two OnionQRC instances built with the same seed must share identical Ising J matrices."""
    qrc_a = OnionQRC(10, seed=2026, topology="ring")
    qrc_b = OnionQRC(10, seed=2026, topology="ring")
    np.testing.assert_array_equal(qrc_a.ising.J, qrc_b.ising.J)
    assert qrc_a.ising.h == qrc_b.ising.h

    # Different seeds must produce different J matrices
    qrc_c = OnionQRC(10, seed=42, topology="ring")
    assert not np.array_equal(qrc_a.ising.J, qrc_c.ising.J)

    # Statevector step is deterministic for same seed + same input
    pytest.importorskip("qiskit")
    obs_a = qrc_a.step([0.3, -0.1, 0.5])
    qrc_a2 = OnionQRC(10, seed=2026, topology="ring")
    obs_a2 = qrc_a2.step([0.3, -0.1, 0.5])
    np.testing.assert_allclose(obs_a, obs_a2, atol=1e-12)
