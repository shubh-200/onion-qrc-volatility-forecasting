import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class GARCHBaseline:
    """GARCH model for asset returns, not changes in realized volatility."""

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        *,
        mean: str = "Constant",
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self._model = None
        self._fitted = None

    @property
    def vol(self) -> str:
        return "GARCH"

    @property
    def o(self) -> int:
        return 0

    @staticmethod
    def _validated_returns(returns) -> np.ndarray:
        values = np.asarray(returns, dtype=float).reshape(-1)
        if len(values) < 2:
            raise ValueError("At least two aligned asset returns are required")
        if not np.isfinite(values).all():
            raise ValueError("Asset returns must be finite and date-aligned")
        return values

    def _make_model(self, returns: np.ndarray):
        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError(
                "GARCH and EGARCH baselines require the optional 'arch' package"
            ) from exc
        return arch_model(
            returns * 100.0,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist,
            rescale=False,
        )

    def fit(self, returns):
        values = self._validated_returns(returns)
        self._model = self._make_model(values)
        self._fitted = self._model.fit(disp="off")
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("Model not fitted")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        fc = self._fitted.forecast(horizon=horizon, reindex=False)
        return fc.variance.values[-1, :] / 10000.0

    def rolling_forecast(
        self,
        returns,
        window: int = 252,
        *,
        mode: str = "expanding",
        start=None,
        expanding=None,
    ):
        """Issue aligned one-step forecasts using information strictly before t.

        ``mode='rolling'`` uses the previous ``window`` returns; expanding mode
        uses all returns before t. A pandas Series input yields a Series indexed
        by the forecast target timestamps.
        """
        values = self._validated_returns(returns)
        if expanding is not None:
            mode = "expanding" if expanding else "rolling"
        if mode not in {"rolling", "expanding"}:
            raise ValueError("mode must be 'rolling' or 'expanding'")
        if window < 2:
            raise ValueError("window must be at least two")
        forecast_start = window if start is None else start
        if forecast_start < 2 or forecast_start >= len(values):
            raise ValueError("start must leave training data and forecast targets")

        forecasts = np.empty(len(values) - forecast_start)
        for output_i, target_t in enumerate(range(forecast_start, len(values))):
            history_start = max(0, target_t - window) if mode == "rolling" else 0
            fitted = self._make_model(values[history_start:target_t]).fit(disp="off")
            fc = fitted.forecast(horizon=1, reindex=False)
            forecasts[output_i] = fc.variance.values[-1, 0] / 10000.0

        if isinstance(returns, pd.Series):
            return pd.Series(
                forecasts,
                index=returns.index[forecast_start:],
                name=f"{self.vol.lower()}_variance_forecast",
            )
        return forecasts


class EGARCHBaseline(GARCHBaseline):
    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = "normal",
        *,
        mean: str = "Constant",
    ):
        super().__init__(p=p, q=q, dist=dist, mean=mean)
        self._o = o

    @property
    def vol(self) -> str:
        return "EGARCH"

    @property
    def o(self) -> int:
        return self._o


class PersistenceBaseline:
    """One-step log-RV forecast equal to the latest observed log RV."""

    def __init__(self):
        self.last_value = None

    def fit(self, log_rv, y=None):
        values = np.asarray(y if y is not None else log_rv, dtype=float).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Persistence fit values must be non-empty and finite")
        self.last_value = float(values[-1])
        return self

    def predict(self, log_rv=None) -> np.ndarray:
        if log_rv is not None:
            values = np.asarray(log_rv, dtype=float)
            return values.copy()
        if self.last_value is None:
            raise ValueError("Model not fitted")
        return np.asarray([self.last_value])

    def forecast(self, horizon: int = 1) -> np.ndarray:
        if self.last_value is None:
            raise ValueError("Model not fitted")
        return np.full(horizon, self.last_value)


class HARRVBaseline:
    """Ridge HAR model using log daily, weekly, and monthly RV at t."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.feature_index_ = None
        self.target_index_ = None

    @staticmethod
    def make_design(rv, *, return_index: bool = False):
        """Build causal log-HAR X_t and log(RV_{t+1}) targets."""
        if isinstance(rv, pd.Series):
            series = rv.astype(float).copy()
        else:
            series = pd.Series(np.asarray(rv, dtype=float).reshape(-1))
        if (series <= 0).any() or not np.isfinite(series).all():
            raise ValueError("Realized variance must be positive and finite")

        frame = pd.DataFrame(index=series.index)
        frame["log_rv_d"] = np.log(series)
        frame["log_rv_w"] = np.log(series.rolling(5, min_periods=5).mean())
        frame["log_rv_m"] = np.log(series.rolling(22, min_periods=22).mean())
        frame["target"] = np.log(series.shift(-1))
        frame["target_index"] = pd.Series(series.index, index=series.index).shift(-1)
        design = frame.dropna()
        X = design[["log_rv_d", "log_rv_w", "log_rv_m"]]
        y = design["target"]
        if return_index:
            return X, y, X.index, pd.Index(design["target_index"])
        return X, y

    def fit(
        self,
        rv_daily,
        rv_weekly=None,
        rv_monthly=None,
        y=None,
    ):
        # A single RV series selects the safe, automatically aligned API.
        if rv_weekly is None and rv_monthly is None and y is None:
            X, target, feature_index, target_index = self.make_design(
                rv_daily, return_index=True
            )
            self.feature_index_ = feature_index
            self.target_index_ = target_index
            self.model.fit(X.to_numpy(), target.to_numpy())
            return self
        if rv_weekly is None or rv_monthly is None or y is None:
            raise ValueError("Provide either one RV series or all three HAR signals and y")
        X = np.column_stack([rv_daily, rv_weekly, rv_monthly])
        self.model.fit(X, y)
        return self

    def predict(
        self,
        rv_daily,
        rv_weekly=None,
        rv_monthly=None,
    ) -> np.ndarray:
        if rv_weekly is None and rv_monthly is None:
            X, _, _, target_index = self.make_design(rv_daily, return_index=True)
            prediction = self.model.predict(X.to_numpy())
            if isinstance(rv_daily, pd.Series):
                return pd.Series(
                    prediction,
                    index=target_index,
                    name="har_log_rv_forecast",
                )
            return prediction
        if rv_weekly is None or rv_monthly is None:
            raise ValueError("Both weekly and monthly HAR signals are required")
        X = np.column_stack([rv_daily, rv_weekly, rv_monthly])
        return self.model.predict(X)


class RandomFeatureRidgeBaseline:
    """Deterministic random nonlinear features with a ridge readout."""

    def __init__(
        self,
        n_features: int = 500,
        alpha: float = 1.0,
        seed: int = 42,
        activation: str = "tanh",
    ):
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if activation not in {"tanh", "relu"}:
            raise ValueError("activation must be 'tanh' or 'relu'")
        self.n_features = n_features
        self.alpha = alpha
        self.seed = seed
        self.activation = activation
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.weights_ = None
        self.bias_ = None

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("X must be a finite one- or two-dimensional array")
        return values

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Random features have not been initialized")
        projected = X @ self.weights_ + self.bias_
        if self.activation == "relu":
            return np.maximum(projected, 0.0)
        return np.tanh(projected)

    def fit(self, X, y):
        values = self._as_2d(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(values) != len(target) or not np.isfinite(target).all():
            raise ValueError("X and y must have matching finite rows")
        scaled = self.scaler.fit_transform(values)
        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(
            scale=1.0 / np.sqrt(values.shape[1]),
            size=(values.shape[1], self.n_features),
        )
        self.bias_ = rng.uniform(-np.pi, np.pi, size=self.n_features)
        self.model.fit(self._transform(scaled), target)
        return self

    def predict(self, X) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Model not fitted")
        values = self.scaler.transform(self._as_2d(X))
        return self.model.predict(self._transform(values))


# Concise alias for integrations that use the model family rather than readout name.
RandomFeatureBaseline = RandomFeatureRidgeBaseline


class ESNBaseline:
    def __init__(self, n_reservoir: int = 500, spectral_radius: float = 0.95,
                 sparsity: float = 0.1, ridge_alpha: float = 1e-4, seed: int = 42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.ridge_alpha = ridge_alpha
        self.rng = np.random.default_rng(seed)
        self._W = None
        self._W_in = None
        self._W_out = None
        self._state = None

    def _init_weights(self, n_input: int):
        W = self.rng.standard_normal((self.n_reservoir, self.n_reservoir)) * 0.1
        mask = self.rng.random((self.n_reservoir, self.n_reservoir)) > self.sparsity
        W[mask] = 0
        max_eig = np.max(np.abs(np.linalg.eigvals(W)))
        if max_eig > 0:
            W = W / max_eig * self.spectral_radius
        self._W = W
        self._W_in = self.rng.standard_normal((self.n_reservoir, n_input)) * 0.5

    def _update(self, state: np.ndarray, inp: np.ndarray) -> np.ndarray:
        preactivation = self._W @ state + self._W_in @ inp
        return np.tanh(preactivation)

    def fit(self, X: np.ndarray, y: np.ndarray, washout: int = 50):
        n_input = X.shape[1] if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._init_weights(n_input)
        n_steps = len(X)
        states = np.zeros((n_steps, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(n_steps):
            state = self._update(state, X[t])
            states[t] = state
        states = states[washout:]
        y_wash = y[washout:]
        X_feat = np.column_stack([states, np.ones(len(states))])
        A = X_feat.T @ X_feat + self.ridge_alpha * np.eye(X_feat.shape[1])
        b = X_feat.T @ y_wash
        self._W_out = np.linalg.solve(A, b)
        self._state = state
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._W_out is None:
            raise ValueError("Model not fitted")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_steps = len(X)
        states = np.zeros((n_steps, self.n_reservoir))
        state = self._state.copy() if self._state is not None else np.zeros(self.n_reservoir)
        for t in range(n_steps):
            state = self._update(state, X[t])
            states[t] = state
        X_feat = np.column_stack([states, np.ones(len(states))])
        return X_feat @ self._W_out


class LSTMBaseline:
    def __init__(self, hidden_size: int = 64, n_layers: int = 2,
                 epochs: int = 50, lr: float = 1e-3, seq_len: int = 252,
                 seed: int = 42, patience: int = 8):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.seq_len = seq_len
        self.seed = seed
        self.patience = patience
        self._model = None
        self._fitted = False
        self.epochs_trained_ = 0

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return values.reshape(-1, 1) if values.ndim == 1 else values

    def _sequences(self, X: np.ndarray, y: np.ndarray):
        sequences = [X[i - self.seq_len:i] for i in range(self.seq_len, len(X))]
        targets = np.asarray(y, dtype=float)[self.seq_len:]
        if not sequences:
            raise ValueError("LSTM sequence length must be shorter than the data")
        return np.asarray(sequences), targets

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            print("PyTorch not available; LSTM baseline skipped")
            return self

        torch.manual_seed(self.seed)
        X = self._as_2d(X)

        class _LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, n_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self._model = _LSTM(X.shape[1], self.hidden_size, self.n_layers)
        train_x, train_y = self._sequences(X, y)
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)

        val_tensors = None
        if X_val is not None and y_val is not None:
            X_val = self._as_2d(X_val)
            combined = np.vstack([X[-self.seq_len:], X_val])
            val_x = np.asarray([
                combined[i:i + self.seq_len] for i in range(len(X_val))
            ])
            val_tensors = (
                torch.tensor(val_x, dtype=torch.float32),
                torch.tensor(np.asarray(y_val), dtype=torch.float32).unsqueeze(1),
            )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        best_loss = np.inf
        best_state = None
        stale_epochs = 0

        for epoch in range(self.epochs):
            self._model.train()
            optimizer.zero_grad()
            loss = criterion(self._model(train_x), train_y)
            loss.backward()
            optimizer.step()
            self.epochs_trained_ = epoch + 1

            if val_tensors is None:
                continue
            self._model.eval()
            with torch.no_grad():
                val_loss = float(criterion(self._model(val_tensors[0]), val_tensors[1]))
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = {
                    name: value.detach().clone()
                    for name, value in self._model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            return np.full(len(X), np.nan)
        try:
            import torch
        except ImportError:
            return np.full(len(X), np.nan)

        X = self._as_2d(X)
        preds = np.full(len(X), np.nan)
        self._model.eval()
        with torch.no_grad():
            for i in range(self.seq_len, len(X)):
                seq = torch.tensor(
                    X[i - self.seq_len:i], dtype=torch.float32
                ).unsqueeze(0)
                preds[i] = self._model(seq).item()
        return preds


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    sigma2 = np.zeros(n)
    sigma2[0] = 0.04
    for t in range(1, n):
        sigma2[t] = 0.0001 + 0.1 * sigma2[t - 1] + 0.85 * sigma2[t - 1]
    rv = np.sqrt(sigma2) * np.abs(np.random.randn(n))
    returns = np.diff(np.log(np.cumprod(1 + rv / 100)))

    garch = GARCHBaseline()
    garch.fit(returns)
    fc = garch.forecast(5)
    print(f"GARCH 5-day forecast: {fc}")

    har = HARRVBaseline()
    rv_d = rv[:-1]
    rv_w = np.convolve(rv, np.ones(5) / 5, mode="same")[:-1]
    rv_m = np.convolve(rv, np.ones(22) / 22, mode="same")[:-1]
    y = rv[1:]
    har.fit(rv_d[:-1], rv_w[:-1], rv_m[:-1], y[:-1])
    print(f"HAR-RV sample pred: {har.predict(rv_d[-1:], rv_w[-1:], rv_m[-1:])}")

    esn = ESNBaseline(n_reservoir=100)
    X_esn = rv[:-1].reshape(-1, 1)
    y_esn = rv[1:]
    esn.fit(X_esn, y_esn, washout=20)
    preds = esn.predict(X_esn[-50:])
    print(f"ESN RMSE on last 50: {np.sqrt(np.mean((preds - y_esn[-50:]) ** 2)):.6f}")
