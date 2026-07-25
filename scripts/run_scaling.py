#!/usr/bin/env python3
"""run_scaling.py — Run noiseless statevector OnionQRC for N=5,10,15,20.

Runs ring topology first (default), then optionally fully_connected for N=5,10.

Usage:
    python scripts/run_scaling.py [--topology ring|fully_connected]
                                   [--n-qubits 5 10 15 20]
                                   [--seeds 42 123 2026]
                                   [--quick]
                                   [--output-dir artifacts/simulator]
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from prototype.run_phase3 import main as _main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(a in ("baselines", "simulator", "all") for a in args):
        args = ["simulator"] + args
    sys.argv = [sys.argv[0]] + args
    raise SystemExit(_main())
