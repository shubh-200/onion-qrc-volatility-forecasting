import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform


class RidgeReadout:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @staticmethod
    def closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
        n_features = X.shape[1]
        A = X.T @ X + alpha * np.eye(n_features)
        b = X.T @ y
        w = np.linalg.solve(A, b)
        return w


class IQPQuantumKernel:
    def __init__(self, n_qubits: int = None, scale: float = 1.0):
        self.n_qubits = n_qubits
        self.scale = scale
        self._feature_vectors = {}

    def _feature_map(self, x: np.ndarray) -> np.ndarray:
        import qiskit
        from qiskit.quantum_info import Statevector
        from qiskit.circuit import QuantumCircuit

        d = len(x)
        n = self.n_qubits if self.n_qubits is not None else d
        if n > d:
            x_padded = np.zeros(n)
            x_padded[:d] = x
            x = x_padded
        elif n < d:
            x = x[:n]

        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        for i in range(n):
            qc.rz(self.scale * x[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cz(i, j)
                qc.rz(self.scale * x[i] * x[j], j)
        for i in range(n):
            qc.h(i)

        sv = Statevector.from_instruction(qc)
        return sv.data

    def compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        if Y is None:
            Y = X

        X_vecs = np.array([self._feature_map(x) for x in X])

        if Y is X:
            K = np.abs(X_vecs @ X_vecs.conj().T) ** 2
        else:
            Y_vecs = np.array([self._feature_map(y) for y in Y])
            K = np.abs(X_vecs @ Y_vecs.conj().T) ** 2

        K = np.clip(K, 0, 1)
        K = (K + 1e-10) / (1 + 1e-10)
        return np.real(K)


class QuantumKernelClassifier:
    def __init__(self, C: float = 1.0, gamma: float = None,
                 use_quantum_kernel: bool = False, n_qubits: int = None,
                 scale: float = 1.0):
        self.C = C
        self.gamma = gamma
        self.use_quantum_kernel = use_quantum_kernel
        self.n_qubits = n_qubits
        self.scale = scale
        self.model = None
        self._support = None
        self._le = LabelEncoder()
        self._iqp = None

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        if self.use_quantum_kernel:
            if self._iqp is None:
                n_q = self.n_qubits if self.n_qubits else min(X.shape[1], 10)
                self._iqp = IQPQuantumKernel(n_qubits=n_q, scale=self.scale)
            return self._iqp.compute_kernel_matrix(X, Y)

        if Y is None:
            Y = X
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        dists = squareform(pdist(np.vstack([X, Y]), metric="sqeuclidean"))
        n_x = len(X)
        K_full = np.exp(-self.gamma * dists)
        K = K_full[:n_x, :n_x] if Y is X else K_full[:n_x, n_x:]
        K = np.clip(K, 0, 1)
        return (K + 1e-10) / (1 + 1e-10)

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_enc = self._le.fit_transform(y)
        self._support = X.copy()
        K = self._compute_kernel(X)
        self.model = SVC(kernel="precomputed", C=self.C, decision_function_shape="ovr")
        self.model.fit(K, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K = self._compute_kernel(X, self._support)
        y_enc = self.model.predict(K)
        return self._le.inverse_transform(y_enc)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_enc = self._le.transform(y)
        K = self._compute_kernel(X, self._support)
        return self.model.score(K, y_enc)


class VolQRCReadout:
    def __init__(self, ridge_alpha: float = 1.0, kernel_C: float = 1.0,
                 kernel_gamma: float = None, use_regime: bool = True,
                 use_quantum_kernel: bool = False, n_qubits: int = None,
                 kernel_scale: float = 1.0):
        self.ridge = RidgeReadout(alpha=ridge_alpha)
        self.regime_clf = QuantumKernelClassifier(
            C=kernel_C, gamma=kernel_gamma,
            use_quantum_kernel=use_quantum_kernel,
            n_qubits=n_qubits, scale=kernel_scale,
        )
        self.use_regime = use_regime
        self._regime_fitted = False

    def fit(self, X_reservoir: np.ndarray, y_vol: np.ndarray,
            y_regime: np.ndarray = None, X_classical: np.ndarray = None):
        if self.use_regime and y_regime is not None:
            self.regime_clf.fit(X_reservoir, y_regime)
            self._regime_fitted = True
            regime_preds = y_regime.astype(int)
        else:
            regime_preds = None
        features = self._combine(X_reservoir, X_classical, regime_preds)
        self.ridge.fit(features, y_vol)
        return self

    def predict(self, X_reservoir: np.ndarray, X_classical: np.ndarray = None) -> np.ndarray:
        regime_labels = None
        if self.use_regime and self._regime_fitted:
            regime_labels = self.regime_clf.predict(X_reservoir)
        features = self._combine(X_reservoir, X_classical, regime_labels)
        return self.ridge.predict(features)

    def predict_regime(self, X_reservoir: np.ndarray) -> np.ndarray:
        if not self._regime_fitted:
            raise ValueError("Regime classifier not fitted")
        return self.regime_clf.predict(X_reservoir)

    def _combine(self, X_reservoir: np.ndarray, X_classical: np.ndarray = None,
                 regime_labels: np.ndarray = None) -> np.ndarray:
        parts = [X_reservoir]
        if X_classical is not None:
            parts.append(X_classical)
        if regime_labels is not None:
            n_classes = 3
            one_hot = np.zeros((len(regime_labels), n_classes))
            for i, r in enumerate(regime_labels):
                r_int = int(r)
                if 0 <= r_int < n_classes:
                    one_hot[i, r_int] = 1.0
            parts.append(one_hot)
        return np.hstack(parts)

    def regime_accuracy(self, X_reservoir: np.ndarray, y_regime: np.ndarray) -> float:
        if not self._regime_fitted:
            return 0.0
        preds = self.regime_clf.predict(X_reservoir)
        return np.mean(preds == y_regime)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_log_rv: bool = True) -> dict:
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(residuals))

    qlike = np.nan
    if is_log_rv:
        rv_true = np.exp(y_true)
        rv_pred = np.exp(y_pred)
        qlike = np.mean(rv_true / rv_pred - np.log(rv_true / rv_pred) - 1)
    else:
        eps = 1e-12
        rv_true = np.clip(y_true, eps, None)
        rv_pred = np.clip(y_pred, eps, None)
        qlike = np.mean(rv_true / rv_pred - np.log(rv_true / rv_pred) - 1)

    return {"rmse": rmse, "r2": r2, "qlike": qlike, "mae": mae}


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from numpy.linalg import lstsq
    X_mz = np.column_stack([np.ones(len(y_pred)), y_pred])
    beta, _, _, _ = lstsq(X_mz, y_true, rcond=None)
    intercept, slope = beta[0], beta[1]
    residuals = y_true - X_mz @ beta
    n = len(y_true)
    k = 2
    sigma2 = np.sum(residuals ** 2) / (n - k)
    var_cov = sigma2 * np.linalg.inv(X_mz.T @ X_mz)
    se = np.sqrt(np.diag(var_cov))
    t_intercept = intercept / se[0] if se[0] > 0 else 0
    t_slope = (slope - 1) / se[1] if se[1] > 0 else 0
    return {
        "intercept": intercept, "slope": slope,
        "t_intercept": t_intercept, "t_slope": t_slope,
        "unbiased": abs(t_intercept) < 1.96 and abs(t_slope) < 1.96,
    }


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(200, 15)
    y = X @ np.random.randn(15) + np.random.randn(200) * 0.1
    regimes = np.random.choice([0, 1, 2], 200)

    readout = VolQRCReadout(ridge_alpha=1.0, use_regime=True)
    readout.fit(X[:150], y[:150], regimes[:150])
    preds = readout.predict(X[150:])
    metrics = compute_metrics(y[150:], preds)
    print(f"RMSE: {metrics['rmse']:.4f}, QLIKE: {metrics['qlike']:.4f}")

    regime_acc = readout.regime_accuracy(X[150:], regimes[150:])
    print(f"Regime accuracy: {regime_acc:.4f}")

    mz = mincer_zarnowitz(y[150:], preds)
    print(f"Mincer-Zarnowitz: intercept={mz['intercept']:.4f}, slope={mz['slope']:.4f}, unbiased={mz['unbiased']}")

    readout_no_regime = VolQRCReadout(ridge_alpha=1.0, use_regime=False)
    readout_no_regime.fit(X[:150], y[:150])
    preds_no_regime = readout_no_regime.predict(X[150:])
    metrics_no_regime = compute_metrics(y[150:], preds_no_regime)
    print(f"No-regime RMSE: {metrics_no_regime['rmse']:.4f}, QLIKE: {metrics_no_regime['qlike']:.4f}")
