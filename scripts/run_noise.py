#!/usr/bin/env python3
"""run_noise.py — Run shot-based noise simulation sweep.

Usage:
    python scripts/run_noise.py [--full-matrix] [--max-cases N]
                                 [--topology ring] [--quick]
                                 [--output-dir artifacts/simulator]
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from prototype.run_noise import main as _main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_main())
