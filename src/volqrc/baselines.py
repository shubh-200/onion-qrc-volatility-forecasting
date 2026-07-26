"""Classical baselines — re-export from gic.prototype.baselines."""

from prototype.baselines import (  # noqa: F401
    GARCHBaseline,
    EGARCHBaseline,
    HARRVBaseline,
    PersistenceBaseline,
    ESNBaseline,
    LSTMBaseline,
    RandomFeatureRidgeBaseline,
    RandomFeatureBaseline,
)

__all__ = [
    "GARCHBaseline",
    "EGARCHBaseline",
    "HARRVBaseline",
    "PersistenceBaseline",
    "ESNBaseline",
    "LSTMBaseline",
    "RandomFeatureRidgeBaseline",
    "RandomFeatureBaseline",
]
