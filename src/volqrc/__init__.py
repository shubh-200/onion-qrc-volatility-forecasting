"""VolQRC — Volatility Quantum Reservoir Computing.

Public API re-exports.  All heavy lifting is in ``prototype``; this
package provides the clean import surface required by the Phase 3 submission
layout and judge reproducibility instructions.
"""

from prototype.data_loader import (  # noqa: F401
    load_spx_rv,
    _add_har_features,
    make_windows,
    split_data,
    DATA_SOURCE_INFO,
    DEFAULT_FEATURES,
    HAR_FEATURES,
)
from prototype.onion_qrc import (  # noqa: F401
    OnionQRC,
    SingleBandQRC,
    OnionAllocation,
    allocate_onion,
)
from prototype.readout import (  # noqa: F401
    RidgeReadout,
    IQPQuantumKernel,
    QuantumKernelClassifier,
    compute_metrics,
    compute_regime_metrics,
    diebold_mariano,
    mincer_zarnowitz,
    model_confidence_set,
)
from prototype.baselines import (  # noqa: F401
    GARCHBaseline,
    EGARCHBaseline,
    HARRVBaseline,
    PersistenceBaseline,
    ESNBaseline,
    LSTMBaseline,
    RandomFeatureRidgeBaseline,
)

__version__ = "0.3.0"
__all__ = [
    # data
    "DEFAULT_FEATURES", "HAR_FEATURES", "load_spx_rv", "make_windows",
    "split_data", "label_regimes", "DATA_SOURCE_INFO",
    # circuits
    "OnionQRC", "SingleBandQRC", "allocate_onion", "random_ising_params",
    # readout / stats
    "VolQRCReadout", "RidgeReadout", "IQPQuantumKernel", "QuantumKernelClassifier",
    "compute_metrics", "compute_regime_metrics", "mincer_zarnowitz",
    "diebold_mariano", "block_bootstrap_confidence_interval", "model_confidence_set",
    # baselines
    "GARCHBaseline", "EGARCHBaseline", "HARRVBaseline", "PersistenceBaseline",
    "ESNBaseline", "LSTMBaseline", "RandomFeatureRidgeBaseline",
]
