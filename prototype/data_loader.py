from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).parent.parent
RV_DATASET_PATH = BASE_DIR / "rv_dataset.csv"
PRICE_DATASET_PATH = BASE_DIR / "global index etf return" / "SPX.csv"
REALIZED_LIBRARY_MAT_PATH = BASE_DIR / "RealizedLibrary.mat"
SPX_TICKER = ".SPX"
DATE_OFFSET = 73
HAR_FEATURES = ("log_rv_d", "log_rv_w", "log_rv_m")
DEFAULT_FEATURES = HAR_FEATURES + ("price_return",)


DATA_SOURCE_INFO = {
    "name": "Oxford-Man Institute Realized Volatility Library",
    "original_url": "https://realized.oxford-man.ox.ac.uk/",
    "status": "Discontinued (source no longer available)",
    "archived_sources": [
        "rv_dataset.csv (8-index subset, 2615 trading days, .SPX/.GDAXI/.FCHI/.FTSE/.OMXSPI/.N225/.KS11/.HSI)",
        "RealizedLibrary.mat (VBayesLab/VBLab GitHub repo, 31 indices, 13 RV measures, Jan 2000 – Feb 2021)",
        "CRAN R package 'bvhar' dataset 'oxfordman_rv' (30 indices, 905 days, Jan 2012 – Jun 2015)",
    ],
    "variable": "rv5 (5-minute Realized Variance)",
    "date_range_reconstructed": "Feb 9, 2007 – Jun 28, 2017 (via correlation alignment with S&P 500 daily prices, offset=73)",
    "date_range_vblab_docs": "SPX: Jan 3, 2000 – Feb 2, 2021 (from VBayesLab/VBLab documentation)",
    "alignment_verified": "Row 424 peak RV=0.088 aligns to Oct 15, 2008 (Lehman collapse / GFC peak)",
    "citation": 'Heber, G., Lunde, A., Shephard, N., & Sheppard, K. (2009). "Oxford-Man Institute\'s realized library." Oxford-Man Institute, University of Oxford.',
}


def _read_prices() -> pd.DataFrame:
    if not PRICE_DATASET_PATH.exists():
        raise FileNotFoundError(f"S&P 500 price data not found: {PRICE_DATASET_PATH}")

    prices = pd.read_csv(PRICE_DATASET_PATH)
    required = {"Date", "Price"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")

    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="raise")
    prices = prices.sort_values("Date").drop_duplicates("Date", keep="last")
    prices["Price_num"] = pd.to_numeric(
        prices["Price"].astype(str).str.replace(",", "", regex=False),
        errors="raise",
    )
    prices["price_return"] = prices["Price_num"].pct_change()
    return prices.set_index("Date")


def _reconstruct_dates(
    n_rv: int, allow_approximate_dates: bool = False
) -> pd.DatetimeIndex:
    try:
        prices = _read_prices()
        dates = prices.index[DATE_OFFSET:DATE_OFFSET + n_rv]
        if len(dates) != n_rv:
            raise ValueError(
                f"Need {n_rv} aligned dates but only found {len(dates)} after "
                f"offset {DATE_OFFSET}"
            )
        return pd.DatetimeIndex(dates, name="date")
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        if not allow_approximate_dates:
            raise
        return pd.date_range("2007-02-09", periods=n_rv, freq="B", name="date")


def _add_har_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal HAR-RV signals available at the close of each row's date."""
    result = df.copy()
    rv = result["rv"].astype(float)
    if (rv <= 0).any():
        raise ValueError("Realized variance must be strictly positive")

    result["log_rv"] = np.log(rv)

    # Keep level columns for callers of the original prototype API.
    result["rv_d"] = rv
    result["rv_w"] = rv.rolling(5, min_periods=5).mean()
    result["rv_m"] = rv.rolling(22, min_periods=22).mean()

    # Standard log-HAR regressors: log RV_t, log(mean RV_{t-4:t}), and
    # log(mean RV_{t-21:t}). All windows end at t and therefore use no future RV.
    result["log_rv_d"] = result["log_rv"]
    result["log_rv_w"] = np.log(result["rv_w"])
    result["log_rv_m"] = np.log(result["rv_m"])
    result["log_return"] = result["log_rv"].diff()
    return result


def load_spx_rv(
    allow_synthetic: bool = False,
    *,
    allow_approximate_dates: bool = False,
    synthetic_days: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load date-aligned S&P 500 RV and returns.

    Production/final mode is the default: missing source or alignment data raises
    an exception. Synthetic data is available only through ``allow_synthetic``.
    """
    if not RV_DATASET_PATH.exists():
        if allow_synthetic:
            return _synthetic_rv(n_days=synthetic_days, seed=seed)
        raise FileNotFoundError(
            f"Realized-volatility data not found: {RV_DATASET_PATH}. "
            "Pass allow_synthetic=True only for explicit simulation runs."
        )

    raw = pd.read_csv(RV_DATASET_PATH, index_col=0)
    if SPX_TICKER not in raw.columns:
        raise ValueError(f"RV data does not contain required ticker {SPX_TICKER!r}")

    spx_rv = pd.to_numeric(raw[SPX_TICKER], errors="raise").to_numpy()
    dates = _reconstruct_dates(
        len(spx_rv), allow_approximate_dates=allow_approximate_dates
    )
    result = pd.DataFrame({"rv": spx_rv}, index=dates)

    try:
        price_returns = _read_prices()["price_return"].reindex(dates)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        if not allow_approximate_dates:
            raise
        price_returns = pd.Series(np.nan, index=dates, name="price_return")

    if price_returns.isna().any():
        if not allow_approximate_dates:
            missing = price_returns.index[price_returns.isna()]
            raise ValueError(
                f"Asset returns are not aligned for {len(missing)} RV dates; "
                f"first missing date is {missing[0]}"
            )
        price_returns = price_returns.fillna(0.0)

    result["price_return"] = price_returns.to_numpy()
    result = _add_har_features(result)
    return result.dropna().sort_index()


def _synthetic_rv(n_days: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(n_days)
    returns = np.zeros(n_days)
    sigma2[0] = 0.04
    omega, alpha, beta = 0.0001, 0.1, 0.85
    for t in range(1, n_days):
        returns[t - 1] = np.sqrt(sigma2[t - 1]) * rng.standard_normal()
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    returns[-1] = np.sqrt(sigma2[-1]) * rng.standard_normal()

    dates = pd.date_range("2000-01-03", periods=n_days, freq="B", name="date")
    df = pd.DataFrame(
        {"rv": sigma2.clip(1e-12), "price_return": returns}, index=dates
    )
    return _add_har_features(df).dropna()


def _validate_chronology(df: pd.DataFrame) -> None:
    if not df.index.is_monotonic_increasing:
        raise ValueError("Data must be sorted in strictly chronological order")
    if not df.index.is_unique:
        raise ValueError("Data index must contain unique timestamps")


def _window_arrays(
    df: pd.DataFrame,
    context: int,
    horizon: int,
    features: Sequence[str],
):
    if context < 1 or horizon < 1:
        raise ValueError("context and horizon must both be positive")
    _validate_chronology(df)

    missing = set(features).difference(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if "log_rv" not in df:
        raise ValueError("Data must contain a log_rv target column")

    X, y, feature_positions, target_positions = [], [], [], []
    for feature_t in range(context - 1, len(df) - horizon):
        target_t = feature_t + horizon
        X.append(df.loc[:, features].iloc[feature_t - context + 1:feature_t + 1].to_numpy())
        y.append(df["log_rv"].iloc[target_t])
        feature_positions.append(feature_t)
        target_positions.append(target_t)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(feature_positions, dtype=int),
        np.asarray(target_positions, dtype=int),
    )


def make_windows(
    df: pd.DataFrame,
    context: int = 252,
    horizon: int = 1,
    features: Optional[Sequence[str]] = None,
    *,
    return_metadata: bool = False,
):
    """Build windows ending at t with targets at t + horizon.

    The default return remains ``(X, y)``. With ``return_metadata=True``, a
    dictionary also exposes the feature and target timestamps.
    """
    feature_names = tuple(features) if features is not None else DEFAULT_FEATURES
    X, y, feature_pos, target_pos = _window_arrays(
        df, context, horizon, feature_names
    )
    if not return_metadata:
        return X, y
    return {
        "X": X,
        "y": y,
        "feature_index": df.index.take(feature_pos),
        "target_index": df.index.take(target_pos),
        "feature_positions": feature_pos,
        "target_positions": target_pos,
        "features": feature_names,
    }


def split_data(
    df: pd.DataFrame,
    context: int = 252,
    horizon: int = 1,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    features: Optional[Sequence[str]] = None,
    q_low: float = 0.33,
    q_high: float = 0.66,
):
    """Create strict chronological splits with train-only fitted transforms."""
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("Invalid train/validation fractions")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than one")

    metadata = make_windows(
        df,
        context=context,
        horizon=horizon,
        features=features,
        return_metadata=True,
    )
    X_raw, y = metadata["X"], metadata["y"]
    n = len(X_raw)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    if train_end == 0 or val_end == train_end or val_end == n:
        raise ValueError("Each chronological split must contain at least one sample")

    scaler = StandardScaler()
    n_features = X_raw.shape[-1]
    scaler.fit(X_raw[:train_end].reshape(-1, n_features))
    X = scaler.transform(X_raw.reshape(-1, n_features)).reshape(X_raw.shape)

    y_mean = float(y[:train_end].mean())
    y_std = float(y[:train_end].std())

    target_positions = metadata["target_positions"]
    target_rv = df["rv"].iloc[target_positions].to_numpy()
    q1, q2 = np.quantile(target_rv[:train_end], [q_low, q_high])
    regime_windowed = _labels_from_values(target_rv, q1, q2)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    slices = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }

    result = {
        "y_mean": y_mean,
        "y_std": y_std,
        "scaler": scaler,
        "regime_thresholds": (float(q1), float(q2)),
        "features": metadata["features"],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        # Alternative names are useful to integrations that reserve *_idx for dates.
        "idx_train": train_idx,
        "idx_val": val_idx,
        "idx_test": test_idx,
    }
    for name, split_slice in slices.items():
        result[f"X_{name}"] = X[split_slice]
        result[f"y_{name}"] = y[split_slice]
        result[f"regime_{name}"] = regime_windowed[split_slice]
        result[f"feature_index_{name}"] = metadata["feature_index"][split_slice]
        result[f"target_index_{name}"] = metadata["target_index"][split_slice]
    return result


def _labels_from_values(values: np.ndarray, q1: float, q2: float) -> np.ndarray:
    labels = np.zeros(len(values), dtype=int)
    labels[values >= q2] = 2
    labels[(values >= q1) & (values < q2)] = 1
    return labels


def label_regimes(
    df: pd.DataFrame,
    q_low: float = 0.33,
    q_high: float = 0.66,
    *,
    thresholds: Optional[Tuple[float, float]] = None,
    fit_indices: Optional[Sequence[int]] = None,
    return_thresholds: bool = False,
):
    """Label RV regimes, optionally fitting quantiles on selected training rows."""
    values = df["rv"].to_numpy()
    if thresholds is None:
        fit_values = values if fit_indices is None else values[np.asarray(fit_indices)]
        q1, q2 = np.quantile(fit_values, [q_low, q_high])
    else:
        q1, q2 = thresholds
    if q1 > q2:
        raise ValueError("Low regime threshold cannot exceed high threshold")
    labels = _labels_from_values(values, float(q1), float(q2))
    if return_thresholds:
        return labels, (float(q1), float(q2))
    return labels


if __name__ == "__main__":
    df = load_spx_rv()
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(df.describe())
    labels = label_regimes(df)
    print(f"Regime distribution: {np.bincount(labels)}")
