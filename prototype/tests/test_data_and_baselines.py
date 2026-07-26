import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from prototype.baselines import HARRVBaseline
from prototype.data_loader import _add_har_features, make_windows, split_data


def _rv_frame(n=100):
    dates = pd.date_range("2020-01-01", periods=n, freq="B", name="date")
    rv = np.linspace(0.01, 0.05, n)
    returns = np.linspace(-0.01, 0.01, n)
    return _add_har_features(
        pd.DataFrame({"rv": rv, "price_return": returns}, index=dates)
    ).dropna()


def test_windows_end_at_feature_timestamp_and_target_next_timestamp():
    df = _rv_frame(40)
    windows = make_windows(
        df,
        context=3,
        horizon=1,
        features=["log_rv_d"],
        return_metadata=True,
    )

    np.testing.assert_allclose(
        windows["X"][0, :, 0], df["log_rv_d"].iloc[:3].to_numpy()
    )
    assert windows["feature_index"][0] == df.index[2]
    assert windows["target_index"][0] == df.index[3]
    assert windows["feature_index"][-1] < windows["target_index"][-1]
    np.testing.assert_allclose(windows["y"][0], df["log_rv"].iloc[3])


def test_split_fits_scaler_and_regime_thresholds_on_train_only():
    df = _rv_frame(120)
    split = split_data(
        df,
        context=1,
        train_fraction=0.60,
        val_fraction=0.20,
        features=["log_rv_d"],
    )

    n_train = len(split["train_idx"])
    raw_feature_train = df["log_rv_d"].iloc[:n_train].to_numpy()
    np.testing.assert_allclose(split["scaler"].mean_, [raw_feature_train.mean()])

    # context=1 and horizon=1 means training targets occupy rows 1..n_train.
    expected_thresholds = np.quantile(
        df["rv"].iloc[1:n_train + 1].to_numpy(), [0.33, 0.66]
    )
    np.testing.assert_allclose(split["regime_thresholds"], expected_thresholds)

    assert split["target_index_train"][-1] < split["target_index_val"][0]
    assert split["target_index_val"][-1] < split["target_index_test"][0]


def test_har_features_are_causal_and_target_is_next_day():
    rv = pd.Series(
        np.linspace(0.01, 0.06, 45),
        index=pd.date_range("2021-01-01", periods=45, freq="B"),
        name="rv",
    )
    changed = rv.copy()
    changed.iloc[30:] *= 1000

    original_features = _add_har_features(pd.DataFrame({"rv": rv}))
    changed_features = _add_har_features(pd.DataFrame({"rv": changed}))
    columns = ["log_rv_d", "log_rv_w", "log_rv_m"]
    pd.testing.assert_frame_equal(
        original_features.loc[rv.index[:30], columns],
        changed_features.loc[rv.index[:30], columns],
    )

    X, y, feature_index, target_index = HARRVBaseline.make_design(
        rv, return_index=True
    )
    assert feature_index[0] == rv.index[21]
    assert target_index[0] == rv.index[22]
    np.testing.assert_allclose(X.iloc[0, 0], np.log(rv.iloc[21]))
    np.testing.assert_allclose(y.iloc[0], np.log(rv.iloc[22]))


def test_no_feature_timestamp_reaches_its_target_timestamp():
    """Every window's last feature date must be strictly before the target date."""
    df = _rv_frame(80)
    windows = make_windows(df, context=5, horizon=1, features=["log_rv_d", "log_rv_w", "log_rv_m"],
                           return_metadata=True)
    # For every sample, max feature date < target date
    feature_idx = windows["feature_index"]  # date at position context-1 (last feature day)
    target_idx = windows["target_index"]
    for i in range(len(feature_idx)):
        assert feature_idx[i] < target_idx[i], (
            f"Sample {i}: feature date {feature_idx[i]} >= target date {target_idx[i]}"
        )


def test_scaler_does_not_see_val_or_test_data():
    """A scaler fit on all data must differ from the training-only scaler."""
    df = _rv_frame(120)
    split = split_data(df, context=1, train_fraction=0.60, val_fraction=0.20,
                       features=["log_rv_d"])
    # The training-only scaler stored inside split
    train_only_mean = split["scaler"].mean_[0]

    # A scaler fit on all rows (leaky). split arrays have shape (n, n_features) at context=1.
    n_feat = split["X_train"].shape[-1]
    all_data = np.vstack([
        split[f"X_{name}"].reshape(-1, n_feat) for name in ("train", "val", "test")
    ])
    leaky_mean = StandardScaler().fit(all_data).mean_[0]

    # If leaky and training-only means are identical, the scaler saw more data than training.
    # They should NOT be identical because val/test rows have different distribution.
    assert train_only_mean != leaky_mean, (
        "Training-only scaler mean equals leaky full-data mean; scaler may have seen val/test."
    )
    # Also assert chronological ordering is preserved.
    assert split["target_index_train"][-1] < split["target_index_val"][0]
    assert split["target_index_val"][-1] < split["target_index_test"][0]


def test_regime_thresholds_are_training_only():
    """Regime thresholds must differ from whole-dataset quantiles, proving training-only fit."""
    df = _rv_frame(150)
    split = split_data(df, context=1, train_fraction=0.60, val_fraction=0.20,
                       features=["log_rv_d"])
    q1_train, q2_train = split["regime_thresholds"]

    # Compute quantiles over the full target RV (training + val + test)
    from prototype.data_loader import make_windows
    meta = make_windows(df, context=1, horizon=1, features=["log_rv_d"], return_metadata=True)
    all_rv = df["rv"].iloc[meta["target_positions"]].to_numpy()
    q1_all, q2_all = np.quantile(all_rv, [0.33, 0.66])

    # With linearly increasing RV the training subset quantiles must differ from whole-dataset.
    assert q1_train != q1_all or q2_train != q2_all, (
        "Regime thresholds appear identical to whole-dataset quantiles; "
        "they may not be fit on training data only."
    )
