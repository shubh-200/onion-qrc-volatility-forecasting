# Multi-QPU Recurrent Execution Cross-Hardware Comparison Report

Evaluation of 5-day sequential recurrent quantum reservoir computing across two distinct physical QPU architectures via qBraid.

| QPU Device Target | Architecture | Active Qubits | Shots/Step | Evaluated Steps | Test RMSE ↓ | Test QLIKE ↓ | Test MAE ↓ | Test R² ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **IQM Garnet (20Q)** | CZ Star | 15 | 512 | 5 | 0.1009 | 0.0053 | 0.0762 | **+0.1523** |
| **Rigetti Cepheus-1 (108Q)** | 8-Qubit Lattice | 15 | 512 | 5 | 0.1037 | 0.0056 | 0.0790 | **+0.1033** |
