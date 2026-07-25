#!/usr/bin/env python3
"""run_baselines.py — Run all classical baselines and save results.

Usage:
    python scripts/run_baselines.py [--quick] [--output-dir artifacts/simulator]
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from prototype.run_phase3 import main as _main  # noqa: E402

if __name__ == "__main__":
    # Forward all args but force mode=baselines
    args = sys.argv[1:]
    if not any(a in ("baselines", "simulator", "all") for a in args):
        args = ["baselines"] + args
    sys.argv = [sys.argv[0]] + args
    raise SystemExit(_main())
