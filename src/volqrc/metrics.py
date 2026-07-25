"""Forecast and regime metrics — re-export from gic.prototype.readout."""

from gic.prototype.readout import (  # noqa: F401
    compute_metrics,
    compute_regime_metrics,
    mincer_zarnowitz,
    diebold_mariano,
    diebold_mariano_test,
    block_bootstrap_confidence_interval,
    block_bootstrap_ci,
    model_confidence_set,
)

__all__ = [
    "compute_metrics",
    "compute_regime_metrics",
    "mincer_zarnowitz",
    "diebold_mariano",
    "diebold_mariano_test",
    "block_bootstrap_confidence_interval",
    "block_bootstrap_ci",
    "model_confidence_set",
]
