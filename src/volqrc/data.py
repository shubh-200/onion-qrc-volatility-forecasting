"""Data loading and preprocessing — thin re-export from gic.prototype.data_loader."""

from gic.prototype.data_loader import (  # noqa: F401
    DEFAULT_FEATURES,
    HAR_FEATURES,
    DATA_SOURCE_INFO,
    load_spx_rv,
    make_windows,
    split_data,
    label_regimes,
)

__all__ = [
    "DEFAULT_FEATURES",
    "HAR_FEATURES",
    "DATA_SOURCE_INFO",
    "load_spx_rv",
    "make_windows",
    "split_data",
    "label_regimes",
]
