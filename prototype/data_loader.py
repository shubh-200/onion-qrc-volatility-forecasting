import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).parent.parent
RV_DATASET_PATH = BASE_DIR / "rv_dataset.csv"
PRICE_DATASET_PATH = BASE_DIR / "global index etf return" / "SPX.csv"
REALIZED_LIBRARY_MAT_PATH = BASE_DIR / "RealizedLibrary.mat"
SPX_TICKER = ".SPX"
DATE_OFFSET = 73


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


def _reconstruct_dates(n_rv: int) -> pd.DatetimeIndex:
    try:
        prices = pd.read_csv(PRICE_DATASET_PATH)
        prices["Date"] = pd.to_datetime(prices["Date"], format="%m/%d/%Y")
        prices = prices.sort_values("Date").reset_index(drop=True)
        dates = prices["Date"].iloc[DATE_OFFSET:DATE_OFFSET + n_rv]
        return pd.DatetimeIndex(dates)
    except Exception:
        return pd.date_range("2007-02-09", periods=n_rv, freq="B")


def load_spx_rv() -> pd.DataFrame:
    if RV_DATASET_PATH.exists():
        raw = pd.read_csv(RV_DATASET_PATH, index_col=0)
        spx_rv = raw[SPX_TICKER].values
        n_rv = len(spx_rv)
        dates = _reconstruct_dates(n_rv)

        result = pd.DataFrame(index=dates)
        result["rv"] = spx_rv
        result["log_rv"] = np.log(result["rv"].clip(lower=1e-12))
        result = _add_har_features(result)

        try:
            prices = pd.read_csv(PRICE_DATASET_PATH)
            prices["Date"] = pd.to_datetime(prices["Date"], format="%m/%d/%Y")
            prices = prices.sort_values("Date").reset_index(drop=True)
            prices["Price_num"] = prices["Price"].str.replace(",", "").astype(float)
            prices["Return"] = prices["Price_num"].pct_change()
            returns = prices["Return"].iloc[DATE_OFFSET:DATE_OFFSET + n_rv].values
            result["price_return"] = np.nan_to_num(returns, nan=0.0)
        except Exception:
            result["price_return"] = 0.0

        return result.dropna().reset_index(drop=True)

    print("rv_dataset.csv not found; using synthetic GARCH data")
    return _synthetic_rv()


def _add_har_features(df: pd.DataFrame) -> pd.DataFrame:
    rv = df["rv"]
    df["rv_d"] = rv.rolling(1).mean()
    df["rv_w"] = rv.rolling(5, min_periods=1).mean()
    df["rv_m"] = rv.rolling(22, min_periods=1).mean()
    df["log_return"] = np.log(rv / rv.shift(1)).replace([np.inf, -np.inf], 0).fillna(0)
    return df


def _synthetic_rv(n_days: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(n_days)
    sigma2[0] = 0.04
    omega, alpha, beta = 0.0001, 0.1, 0.85
    for t in range(1, n_days):
        sigma2[t] = omega + alpha * sigma2[t - 1] * rng.standard_normal() ** 2 + beta * sigma2[t - 1]
    rv = np.sqrt(sigma2) * np.abs(rng.standard_normal(n_days))
    df = pd.DataFrame({"rv": rv})
    df["log_rv"] = np.log(df["rv"].clip(lower=1e-12))
    df = _add_har_features(df)
    return df.dropna().reset_index(drop=True)


def make_windows(df: pd.DataFrame, context: int = 252, horizon: int = 1):
    features = ["log_rv", "rv_d", "rv_w", "rv_m", "log_return"]
    X, y = [], []
    for i in range(context, len(df) - horizon):
        X.append(df[features].iloc[i - context:i].values)
        y.append(df["log_rv"].iloc[i + horizon])
    X = np.array(X)
    y = np.array(y)
    return X, y


def split_data(df: pd.DataFrame, context: int = 252, horizon: int = 1):
    X, y = make_windows(df, context, horizon)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    scaler = StandardScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_flat[:train_end] = scaler.fit_transform(X_flat[:train_end])
    X_flat[train_end:] = scaler.transform(X_flat[train_end:])
    X = X_flat.reshape(X.shape)

    y_mean, y_std = y[:train_end].mean(), y[:train_end].std()

    regime_labels = label_regimes(df)
    regime_windowed = regime_labels[context + horizon:context + horizon + n]

    return {
        "X_train": X[:train_end], "y_train": y[:train_end],
        "X_val": X[train_end:val_end], "y_val": y[train_end:val_end],
        "X_test": X[val_end:], "y_test": y[val_end:],
        "y_mean": y_mean, "y_std": y_std,
        "scaler": scaler,
        "regime_train": regime_windowed[:train_end],
        "regime_val": regime_windowed[train_end:val_end],
        "regime_test": regime_windowed[val_end:],
    }


def label_regimes(df: pd.DataFrame, q_low: float = 0.33, q_high: float = 0.66):
    q1 = df["rv"].quantile(q_low)
    q2 = df["rv"].quantile(q_high)
    labels = np.zeros(len(df), dtype=int)
    labels[df["rv"] >= q2] = 2
    labels[(df["rv"] >= q1) & (df["rv"] < q2)] = 1
    return labels


if __name__ == "__main__":
    df = load_spx_rv()
    print(f"Loaded {len(df)} rows")
    if hasattr(df.index, "min") and not isinstance(df.index, pd.RangeIndex):
        try:
            print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        except Exception:
            pass
    print(df.describe())
    labels = label_regimes(df)
    print(f"Regime distribution: {np.bincount(labels)}")
