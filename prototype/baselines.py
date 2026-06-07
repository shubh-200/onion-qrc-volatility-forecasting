import numpy as np
from arch import arch_model
from sklearn.linear_model import Ridge


class GARCHBaseline:
    def __init__(self, p: int = 1, q: int = 1, dist: str = "normal"):
        self.p = p
        self.q = q
        self.dist = dist
        self._model = None
        self._fitted = None

    def fit(self, returns: np.ndarray):
        self._model = arch_model(returns * 100, vol="Garch", p=self.p, q=self.q, dist=self.dist)
        self._fitted = self._model.fit(disp="off")
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("Model not fitted")
        fc = self._fitted.forecast(horizon=horizon)
        var_forecast = fc.variance.values[-1, :] / 10000
        return var_forecast

    def rolling_forecast(self, returns: np.ndarray, window: int = 252) -> np.ndarray:
        n = len(returns)
        forecasts = []
        for t in range(window, n):
            ret_window = returns[:t]
            try:
                model = arch_model(ret_window * 100, vol="Garch", p=self.p, q=self.q, dist=self.dist)
                fitted = model.fit(disp="off")
                fc = fitted.forecast(horizon=1)
                var_fc = fc.variance.values[-1, 0] / 10000
                forecasts.append(var_fc)
            except Exception:
                forecasts.append(np.nan)
        return np.array(forecasts)


class HARRVBaseline:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, rv_daily: np.ndarray, rv_weekly: np.ndarray, rv_monthly: np.ndarray, y: np.ndarray):
        X = np.column_stack([rv_daily, rv_weekly, rv_monthly])
        self.model.fit(X, y)
        return self

    def predict(self, rv_daily: np.ndarray, rv_weekly: np.ndarray, rv_monthly: np.ndarray) -> np.ndarray:
        X = np.column_stack([rv_daily, rv_weekly, rv_monthly])
        return self.model.predict(X)


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
                 epochs: int = 50, lr: float = 1e-3, seq_len: int = 252):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.seq_len = seq_len
        self._model = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            print("PyTorch not available; LSTM baseline skipped")
            return self

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        class _LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, n_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        n_input = X.shape[1]
        self._model = _LSTM(n_input, self.hidden_size, self.n_layers)

        sequences = []
        targets = []
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len:i])
            targets.append(y[i])
        sequences = torch.FloatTensor(np.array(sequences))
        targets = torch.FloatTensor(np.array(targets)).unsqueeze(1)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self._model(sequences)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

        self._fitted = True
        self._X_train = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            return np.full(len(X), np.nan)
        try:
            import torch
        except ImportError:
            return np.full(len(X), np.nan)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._model.eval()
        n = len(X)
        preds = np.zeros(n)
        with torch.no_grad():
            for i in range(self.seq_len, n):
                seq = torch.FloatTensor(X[i - self.seq_len:i]).unsqueeze(0)
                preds[i] = self._model(seq).item()
        preds[:self.seq_len] = np.nan
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
