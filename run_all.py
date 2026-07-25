#!/usr/bin/env python3
"""run_all.py — Master reproduction entrypoint for VolQRC Phase 3.

Executes the full end-to-end VolQRC benchmark suite:
1. Runs automated unit tests & causal verification rules (pytest)
2. Trains and evaluates classical baselines (Persistence, HAR-Ridge, ESN-210/500, GARCH, EGARCH, LSTM)
3. Runs quantum simulator scaling for N in {5, 10, 15} (ring) and N in {5, 10} (fully-connected)
4. Runs Phase 3 ablation studies (Observable-Order, Regime-Gating, Quantum Regime Kernel)
5. Evaluates physical 20-qubit hardware QPU results (from archived qBraid IQM Garnet artifacts)
6. Computes statistical diagnostics (Diebold-Mariano, Mincer-Zarnowitz, Seed Aggregation, MCS)
7. Rebuilds summary markdown & CSV manifests (summary_table.md, summary_table.csv)

Usage:
    python run_all.py [--quick]
"""

from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n==================================================================")
    print(f"  STEP: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"==================================================================")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(_HERE))
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"[OK] {name} finished successfully in {elapsed:.2f}s")
        return True
    else:
        print(f"[FAIL] {name} failed with exit code {res.returncode} ({elapsed:.2f}s)")
        return False


def main() -> int:
    t_start = time.time()
    print("==================================================================")
    print("  VolQRC Phase 3 Master End-to-End Reproduction Runner")
    print("==================================================================")

    steps = [
        ("Automated Unit Tests & Causal Verification", [sys.executable, "-m", "pytest"]),
        ("Classical Baselines (HAR, ESN, GARCH, EGARCH, LSTM)", [sys.executable, "scripts/run_baselines.py"]),
        ("Quantum Simulator Scaling (Ring N=5, 10, 15)", [sys.executable, "scripts/run_scaling.py", "--n-qubits", "5", "10", "15", "--topology", "ring"]),
        ("Quantum Simulator Scaling (Fully-Connected N=5, 10)", [sys.executable, "scripts/run_scaling.py", "--n-qubits", "5", "10", "--topology", "fully_connected"]),
        ("Phase 3 Ablation Studies (Observables, Gating, IQP Kernel)", [sys.executable, "scripts/run_ablations.py"]),
        ("Hardware QPU Evaluation (IQM Garnet N=15/20 Artifacts)", [sys.executable, "scripts/evaluate_hardware_results.py"]),
        ("Statistical Diagnostics (DM, MZ, Bootstraps, MCS)", [sys.executable, "scripts/compute_statistics.py"]),
        ("Build Deliverables & Summary Tables", [sys.executable, "scripts/build_report.py"]),
    ]

    failed = []
    for name, cmd in steps:
        success = _run_step(name, cmd)
        if not success:
            failed.append(name)

    total_time = time.time() - t_start
    print("\n==================================================================")
    if not failed:
        print(f"[SUCCESS] Full VolQRC Phase 3 pipeline executed cleanly in {total_time:.2f}s!")
        print("   Manifests updated in: artifacts/manifests/summary_table.md")
        return 0
    else:
        print(f"[WARNING] COMPLETED WITH WARNINGS: {len(failed)} steps failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
