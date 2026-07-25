#!/usr/bin/env python3
"""submit_qpu.py — Prepare and (optionally) submit circuits to a QPU via qBraid.

This script covers all four execution modes defined in Phase 3 §9.3:

Modes:
  smoke     — N=5,10,15,20 scaling smoke test (3 obs per config, 256 shots)
  panel     — Balanced 24-circuit panel (8 per regime, 1024 shots) [DEFAULT]
  recurrent — 8-day chronological sequence (512 shots)
  zne       — ZNE mitigation subset (4 obs × 3 scale factors, 1024 shots)
  calibrate — Readout calibration (2 tasks, 2048 shots)

Set QBRAID_API_KEY in your environment before using --submit.

Usage (offline, no credits spent):
    python scripts/submit_qpu.py --mode panel --n-qubits 20

Usage (live submission):
    python scripts/submit_qpu.py --mode panel --n-qubits 20 \\
        --device-id iqm_garnet --submit

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from prototype.qbraid_hardware import main as _hw_main  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["smoke", "panel", "recurrent", "zne", "calibrate"],
                   default="panel",
                   help="Which execution panel to prepare (default: panel)")
    p.add_argument("--n-qubits", type=int, nargs="+", default=[20])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--topology", default="ring")
    p.add_argument("--device-id", default=None,
                   help="qBraid device ID (e.g. iqm_garnet). Required for --submit.")
    p.add_argument("--submit", action="store_true",
                   help="Actually submit circuits (requires QBRAID_API_KEY env var)")
    p.add_argument("--live-discovery", action="store_true",
                   help="Query qBraid for available devices before preparing circuits")
    p.add_argument("--spend-cap-usd", type=float, default=25.0,
                   help="Hard spend cap in USD — abort if estimated cost exceeds this")
    p.add_argument("--shots", type=int, default=None,
                   help="Override default shots for the selected mode")
    p.add_argument("--per-regime", type=int, default=8,
                   help="Observations per regime for the 'panel' mode (default: 8)")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/hardware"))
    return p


_MODE_SHOTS = {
    "smoke": 256,
    "panel": 1024,
    "recurrent": 512,
    "zne": 1024,
    "calibrate": 2048,
}

_MODE_N_QUBITS = {
    "smoke": [5, 10, 15, 20],
    "panel": [20],
    "recurrent": [20],
    "zne": [20],
    "calibrate": [20],
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    shots = args.shots or _MODE_SHOTS[args.mode]
    n_qubits = args.n_qubits or _MODE_N_QUBITS[args.mode]
    per_regime = args.per_regime if args.mode == "panel" else 3

    hw_argv = [
        "--n-qubits", *map(str, n_qubits),
        "--seeds", *map(str, args.seeds),
        "--topology", args.topology,
        "--shots", str(shots),
        "--per-regime", str(per_regime),
        "--spend-cap-usd", str(args.spend_cap_usd),
        "--output-dir", str(args.output_dir / args.mode),
    ]
    if args.live_discovery:
        hw_argv.append("--live-discovery")
    if args.device_id:
        hw_argv += ["--device-id", args.device_id]
    if args.submit:
        hw_argv.append("--submit")

    print(f"[submit_qpu] mode={args.mode}  shots={shots}  n_qubits={n_qubits}")
    return _hw_main(hw_argv)


if __name__ == "__main__":
    raise SystemExit(main())
