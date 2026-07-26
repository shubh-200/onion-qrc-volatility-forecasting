"""Readout and statistical utilities — re-export from gic.prototype.readout."""

from prototype.readout import (  # noqa: F401
    RidgeReadout,
    IQPQuantumKernel,
    QuantumKernelClassifier,
    VolQRCReadout,
    compute_regime_metrics,
    compute_metrics,
    mincer_zarnowitz,
    diebold_mariano,
    diebold_mariano_test,
    block_bootstrap_confidence_interval,
    block_bootstrap_ci,
    model_confidence_set,
)

__all__ = [
    "RidgeReadout",
    "IQPQuantumKernel",
    "QuantumKernelClassifier",
    "VolQRCReadout",
    "compute_regime_metrics",
    "compute_metrics",
    "mincer_zarnowitz",
    "diebold_mariano",
    "diebold_mariano_test",
    "block_bootstrap_confidence_interval",
    "block_bootstrap_ci",
    "model_confidence_set",
]
