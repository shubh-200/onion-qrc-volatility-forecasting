#!/usr/bin/env python3
"""prepare_data.py — Validate, checksum, and document the SPX RV dataset.

Usage:
    python scripts/prepare_data.py [--output-dir artifacts/manifests]
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing the package.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from prototype.data_loader import DATA_SOURCE_INFO, load_spx_rv  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/manifests"))
    args = parser.parse_args()

    print("[prepare_data] loading SPX RV data")
    df = load_spx_rv(allow_synthetic=False)
    print(f"  rows={len(df)}  date_range={df.index.min().date()} to {df.index.max().date()}")

    rv_path = Path("rv_dataset.csv")
    price_path = Path("global index etf return/SPX.csv")
    checksum_rv = _sha256(rv_path) if rv_path.exists() else None
    checksum_price = _sha256(price_path) if price_path.exists() else None

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "date_start": str(df.index.min().date()),
        "date_end": str(df.index.max().date()),
        "columns": list(df.columns),
        "checksums": {
            "rv_dataset.csv": checksum_rv,
            "SPX.csv": checksum_price,
        },
        "source_info": DATA_SOURCE_INFO,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[prepare_data] manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
