import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import importlib.util

import numpy as np
import pytest

from prototype import run_noise as noise


def test_case_generation_is_conservative_and_full_matrix_is_explicit():
    default_cases = noise.generate_cases()
    assert default_cases == [
        noise.NoiseCase(5, 42, 256, "ring", "noiseless_shots", None)
    ]

    selected = noise.generate_cases(
        n_qubits=[5, 10],
        seeds=[7],
        shots=[128],
        depolarizing_p=[0.01],
        amplitude_damping_gamma=[0.02],
        max_cases=5,
    )
    assert [(case.n_qubits, case.noise_kind) for case in selected] == [
        (5, "noiseless_shots"),
        (5, "depolarizing"),
        (5, "amplitude_damping"),
        (10, "noiseless_shots"),
        (10, "depolarizing"),
    ]

    full = noise.generate_cases(full_matrix=True)
    expected = (
        len(noise.FULL_N_QUBITS)
        * len(noise.FULL_SEEDS)
        * len(noise.FULL_SHOTS)
        * (1 + len(noise.FULL_DEPOLARIZING) + len(noise.FULL_AMPLITUDE_DAMPING))
    )
    assert len(full) == expected
    assert {case.n_qubits for case in full} == set(noise.FULL_N_QUBITS)

    with pytest.raises(ValueError, match="--full-matrix"):
        noise.generate_cases(
            n_qubits=noise.FULL_N_QUBITS,
            depolarizing_p=noise.FULL_DEPOLARIZING,
            amplitude_damping_gamma=noise.FULL_AMPLITUDE_DAMPING,
            max_cases=1,
        )


def test_shot_sequence_updates_feedback_from_one_counts_mapping_per_step():
    class FakeCircuit:
        def depth(self):
            return 3

        def count_ops(self):
            return {"measure": 2}

    class FakeQRC:
        n_observables = 3

        def __init__(self):
            self.memory_updates = []
            self.count_calls = []
            self.build_memory_lengths = []

        def reset_memory(self):
            self.memory_updates.clear()

        def build_circuit(self, row, measure=False):
            assert measure is True
            self.build_memory_lengths.append(len(self.memory_updates))
            return FakeCircuit()

        def observables_from_counts(self, counts):
            self.count_calls.append(counts)
            zeros = counts.get("00", 0)
            ones = counts.get("11", 0)
            total = zeros + ones
            return np.array([(zeros - ones) / total, 0.25, -0.5])

        def update_memory_from_observables(self, observables):
            self.memory_updates.append(np.asarray(observables).copy())

    class FakeResult:
        def __init__(self, counts):
            self.counts = counts

        def get_counts(self, circuit):
            return self.counts

    class FakeJob:
        def __init__(self, counts):
            self.counts = counts

        def result(self):
            return FakeResult(self.counts)

    class FakeBackend:
        def __init__(self):
            self.calls = 0

        def run(self, circuit, shots, seed_simulator):
            self.calls += 1
            assert shots == 10
            return FakeJob({"00": 7, "11": 3})

    transpile_calls = []

    def fake_transpile(circuit, **kwargs):
        transpile_calls.append(kwargs)
        return circuit

    qrc = FakeQRC()
    output, info = noise.run_shot_sequence(
        qrc,
        np.zeros((2, 4)),
        backend=FakeBackend(),
        transpile_fn=fake_transpile,
        shots=10,
        simulation_seed=9,
        basis_gates=["rx", "ry", "rzz"],
    )

    np.testing.assert_allclose(output, [[0.4, 0.25, -0.5], [0.4, 0.25, -0.5]])
    assert qrc.build_memory_lengths == [0, 1]
    assert len(qrc.count_calls) == len(qrc.memory_updates) == 2
    np.testing.assert_allclose(qrc.memory_updates, output)
    assert len(transpile_calls) == 2
    assert info["execution_resources"]["total_shots"] == 20


def test_parser_defaults_do_not_require_aer_or_select_full_matrix():
    args = noise.build_parser().parse_args([])
    assert args.n_qubits is None
    assert args.seeds is None
    assert args.shots is None
    assert args.full_matrix is False
    assert args.max_cases is None
    assert args.include_noiseless is True
    assert args.depolarizing_p is None
    assert args.amplitude_damping_gamma is None
    assert args.quick is False
    assert args.quick_rows == 90
    assert noise.generate_cases(
        n_qubits=args.n_qubits,
        seeds=args.seeds,
        shots=args.shots,
        topology=args.topology,
        depolarizing_p=args.depolarizing_p,
        amplitude_damping_gamma=args.amplitude_damping_gamma,
        include_noiseless=args.include_noiseless,
        full_matrix=args.full_matrix,
        max_cases=args.max_cases,
    ) == [noise.NoiseCase(5, 42, 256, "ring", "noiseless_shots", None)]


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None
    or importlib.util.find_spec("qiskit_aer") is None,
    reason="optional Qiskit Aer is not installed",
)
def test_actual_aer_single_step_returns_finite_shot_observables():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qrc = noise.OnionQRC(3, topology="ring", seed=4, trotter_steps=1)
    output, info = noise.run_shot_sequence(
        qrc,
        np.asarray([[0.1, -0.2, 0.3, 0.0]]),
        backend=AerSimulator(seed_simulator=4),
        transpile_fn=transpile,
        shots=32,
        simulation_seed=4,
        basis_gates=noise.DEFAULT_BASIS_GATES,
        optimization_level=0,
    )

    assert output.shape == (1, qrc.n_observables)
    assert np.isfinite(output).all()
    assert np.all(np.abs(output) <= 1.0)
    assert info["execution_resources"]["total_shots"] == 32
