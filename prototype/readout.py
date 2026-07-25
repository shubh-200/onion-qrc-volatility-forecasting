import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


_EPS = 1e-12


def _paired_finite(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        raise ValueError("No finite paired observations")
    return y_true[mask], y_pred[mask]


def _hac_meat(scores: np.ndarray, max_lags: int) -> np.ndarray:
    """Newey-West meat matrix for rows of estimating-equation scores."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    max_lags = max(0, min(int(max_lags), n - 1))
    meat = scores.T @ scores
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    return meat


def _default_hac_lags(n: int) -> int:
    return max(0, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


class RidgeReadout:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Ridge readout not fitted")
        return self.model.predict(X)

    @staticmethod
    def closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
        n_features = X.shape[1]
        A = X.T @ X + alpha * np.eye(n_features)
        b = X.T @ y
        return np.linalg.solve(A, b)


class IQPQuantumKernel:
    """Exact IQP-style fidelity kernel using data-dependent one- and two-body phases."""

    def __init__(self, n_qubits: int = None, scale: float = 1.0):
        self.n_qubits = n_qubits
        self.scale = scale

    def _feature_map(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        n = self.n_qubits if self.n_qubits is not None else len(x)
        if n < 1:
            raise ValueError("n_qubits must be positive")
        if n > len(x):
            x = np.pad(x, (0, n - len(x)))
        else:
            x = x[:n]

        # D(x)|+> with D(x) = exp(i sum x_i Z_i + i sum x_i x_j Z_i Z_j).
        # Unlike the old CZ construction, every pair phase depends on the data.
        basis = np.arange(1 << n, dtype=np.uint64)[:, None]
        bits = (basis >> np.arange(n, dtype=np.uint64)) & 1
        signs = 1.0 - 2.0 * bits.astype(float)
        scaled = self.scale * x
        phase = signs @ scaled
        for i in range(n):
            for j in range(i + 1, n):
                phase += self.scale * x[i] * x[j] * signs[:, i] * signs[:, j]
        return np.exp(1j * phase) / np.sqrt(1 << n)

    @staticmethod
    def _stabilize_gram(K: np.ndarray) -> np.ndarray:
        K = np.asarray(np.real_if_close(K), dtype=float)
        K = np.nan_to_num((K + K.T) / 2.0, nan=0.0, posinf=1.0, neginf=0.0)
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        K = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        diagonal = np.sqrt(np.maximum(np.diag(K), _EPS))
        K = K / np.outer(diagonal, diagonal)
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
        return K

    def compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self_kernel = Y is None or Y is X
        Y = X if Y is None else np.atleast_2d(np.asarray(Y, dtype=float))
        X_vecs = np.asarray([self._feature_map(x) for x in X])
        Y_vecs = X_vecs if self_kernel else np.asarray([self._feature_map(y) for y in Y])
        K = np.abs(X_vecs @ Y_vecs.conj().T) ** 2
        if self_kernel:
            return self._stabilize_gram(K)
        return np.clip(np.real(K), 0.0, 1.0)


class QuantumKernelClassifier:
    def __init__(self, C: float = 1.0, gamma: float = None,
                 use_quantum_kernel: bool = False, n_qubits: int = None,
                 scale: float = 1.0, pca_components: int = None):
        self.C = C
        self.gamma = gamma
        self.use_quantum_kernel = use_quantum_kernel
        self.n_qubits = n_qubits
        self.scale = scale
        self.pca_components = pca_components
        self.model = None
        self._support = None
        self._le = LabelEncoder()
        self._iqp = None
        self._scaler = None
        self._pca = None

    @staticmethod
    def _stabilize_gram(K: np.ndarray) -> np.ndarray:
        return IQPQuantumKernel._stabilize_gram(K)

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self_kernel = Y is None or Y is X
        Y = X if Y is None else np.atleast_2d(np.asarray(Y, dtype=float))
        if self.use_quantum_kernel:
            if self._iqp is None:
                n_q = self.n_qubits if self.n_qubits is not None else min(X.shape[1], 10)
                self._iqp = IQPQuantumKernel(n_qubits=n_q, scale=self.scale)
            return self._iqp.compute_kernel_matrix(X, None if self_kernel else Y)

        gamma = self.gamma if self.gamma is not None else 1.0 / max(X.shape[1], 1)
        K = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))
        return self._stabilize_gram(K) if self_kernel else np.clip(K, 0.0, 1.0)

    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.pca_components is None:
            return X
        max_components = min(len(X), X.shape[1])
        n_components = min(int(self.pca_components), max_components)
        if n_components < 1:
            raise ValueError("pca_components must be positive")
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=n_components, svd_solver="full")
        return self._pca.fit_transform(self._scaler.fit_transform(X))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._pca is None:
            return X
        return self._pca.transform(self._scaler.transform(X))

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y).reshape(-1)
        if len(np.unique(y)) < 2:
            raise ValueError("Regime classifier requires at least two classes")
        X_fit = self._fit_transform(X)
        y_enc = self._le.fit_transform(y)
        self._support = X_fit.copy()
        K = self._compute_kernel(X_fit)
        self.model = SVC(kernel="precomputed", C=self.C, decision_function_shape="ovr")
        self.model.fit(K, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Regime classifier not fitted")
        K = self._compute_kernel(self._transform(X), self._support)
        return self._le.inverse_transform(self.model.predict(K))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class VolQRCReadout:
    def __init__(self, ridge_alpha: float = 1.0, kernel_C: float = 1.0,
                 kernel_gamma: float = None, use_regime: bool = True,
                 use_quantum_kernel: bool = False, n_qubits: int = None,
                 kernel_scale: float = 1.0, gating_mode: str = "cross_fitted",
                 regime_cv_splits: int = 5, classifier_pca_components: int = None):
        if gating_mode not in {"cross_fitted", "oracle"}:
            raise ValueError("gating_mode must be 'cross_fitted' or 'oracle'")
        self.ridge = RidgeReadout(alpha=ridge_alpha)
        self._classifier_kwargs = dict(
            C=kernel_C, gamma=kernel_gamma,
            use_quantum_kernel=use_quantum_kernel,
            n_qubits=n_qubits, scale=kernel_scale,
            pca_components=classifier_pca_components,
        )
        self.regime_clf = QuantumKernelClassifier(**self._classifier_kwargs)
        self.use_regime = use_regime
        self.gating_mode = gating_mode
        self.regime_cv_splits = max(2, int(regime_cv_splits))
        self._regime_fitted = False
        self._regime_classes = None
        self.training_regime_predictions_ = None

    def _cross_fitted_regimes(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Expanding-window predictions; every prediction uses strictly earlier labels."""
        n = len(y)
        predictions = np.empty(n, dtype=object)
        predictions[:] = None
        initial = max(2, n // (self.regime_cv_splits + 1))
        boundaries = np.linspace(initial, n, self.regime_cv_splits + 1, dtype=int)
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop <= start:
                continue
            past_y = y[:start]
            classes = np.unique(past_y)
            if len(classes) == 1:
                predictions[start:stop] = classes[0]
                continue
            classifier = QuantumKernelClassifier(**self._classifier_kwargs)
            classifier.fit(X[:start], past_y)
            predictions[start:stop] = classifier.predict(X[start:stop])
        return predictions

    def fit(self, X_reservoir: np.ndarray, y_vol: np.ndarray,
            y_regime: np.ndarray = None, X_classical: np.ndarray = None):
        X_reservoir = np.asarray(X_reservoir, dtype=float)
        y_vol = np.asarray(y_vol, dtype=float).reshape(-1)
        if len(X_reservoir) != len(y_vol):
            raise ValueError("X_reservoir and y_vol must have the same length")

        regime_preds = None
        if self.use_regime and y_regime is not None:
            y_regime = np.asarray(y_regime).reshape(-1)
            if len(y_regime) != len(y_vol):
                raise ValueError("y_regime and y_vol must have the same length")
            self._regime_classes = np.unique(y_regime)
            if len(self._regime_classes) >= 2:
                if self.gating_mode == "oracle":
                    regime_preds = y_regime.copy()
                else:
                    regime_preds = self._cross_fitted_regimes(X_reservoir, y_regime)
                self.training_regime_predictions_ = regime_preds.copy()
                self.regime_clf.fit(X_reservoir, y_regime)
                self._regime_fitted = True

        features = self._combine(X_reservoir, X_classical, regime_preds)
        self.ridge.fit(features, y_vol)
        return self

    def predict(self, X_reservoir: np.ndarray, X_classical: np.ndarray = None) -> np.ndarray:
        regime_labels = None
        if self.use_regime and self._regime_fitted:
            regime_labels = self.regime_clf.predict(X_reservoir)
        return self.ridge.predict(self._combine(X_reservoir, X_classical, regime_labels))

    def predict_regime(self, X_reservoir: np.ndarray) -> np.ndarray:
        if not self._regime_fitted:
            raise ValueError("Regime classifier not fitted")
        return self.regime_clf.predict(X_reservoir)

    def _combine(self, X_reservoir: np.ndarray, X_classical: np.ndarray = None,
                 regime_labels: np.ndarray = None) -> np.ndarray:
        parts = [np.asarray(X_reservoir, dtype=float)]
        if X_classical is not None:
            parts.append(np.asarray(X_classical, dtype=float))
        if regime_labels is not None and self._regime_classes is not None:
            labels = np.asarray(regime_labels, dtype=object)
            one_hot = np.zeros((len(labels), len(self._regime_classes)))
            class_to_index = {label: i for i, label in enumerate(self._regime_classes)}
            for row, label in enumerate(labels):
                if label is not None and label in class_to_index:
                    one_hot[row, class_to_index[label]] = 1.0
            parts.append(one_hot)
        return np.hstack(parts)

    def regime_accuracy(self, X_reservoir: np.ndarray, y_regime: np.ndarray) -> float:
        if not self._regime_fitted:
            return 0.0
        return float(accuracy_score(y_regime, self.regime_clf.predict(X_reservoir)))

    def regime_metrics(self, X_reservoir: np.ndarray, y_regime: np.ndarray) -> dict:
        if not self._regime_fitted:
            raise ValueError("Regime classifier not fitted")
        return compute_regime_metrics(y_regime, self.regime_clf.predict(X_reservoir))


def compute_regime_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           labels: np.ndarray = None) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Regime arrays must be non-empty and have the same length")
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "labels": labels,
        "support": support,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def _forecast_losses(y_true: np.ndarray, y_pred: np.ndarray,
                     loss: str, is_log_rv: bool) -> np.ndarray:
    if loss in {"squared", "mse"}:
        return (y_true - y_pred) ** 2
    if loss in {"absolute", "mae"}:
        return np.abs(y_true - y_pred)
    if loss != "qlike":
        raise ValueError("loss must be 'qlike', 'squared', or 'absolute'")
    if is_log_rv:
        log_ratio = np.clip(y_true - y_pred, -50.0, 50.0)
        return np.exp(log_ratio) - log_ratio - 1.0
    actual = np.maximum(y_true, _EPS)
    forecast = np.maximum(y_pred, _EPS)
    ratio = np.clip(actual / forecast, _EPS, np.exp(50.0))
    return ratio - np.log(ratio) - 1.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_log_rv: bool = True) -> dict:
    y_true, y_pred = _paired_finite(y_true, y_pred)
    residuals = y_true - y_pred
    ss_res = float(residuals @ residuals)
    centered = y_true - np.mean(y_true)
    ss_tot = float(centered @ centered)
    return {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else 0.0,
        "qlike": float(np.mean(_forecast_losses(y_true, y_pred, "qlike", is_log_rv))),
        "mae": float(np.mean(np.abs(residuals))),
        "n_obs": int(len(y_true)),
    }


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray,
                      hac_lags: int = None, alpha: float = 0.05) -> dict:
    y_true, y_pred = _paired_finite(y_true, y_pred)
    n = len(y_true)
    if n < 3:
        raise ValueError("Mincer-Zarnowitz test requires at least three observations")
    X_mz = np.column_stack([np.ones(n), y_pred])
    beta = np.linalg.lstsq(X_mz, y_true, rcond=None)[0]
    residuals = y_true - X_mz @ beta
    lags = _default_hac_lags(n) if hac_lags is None else max(0, int(hac_lags))
    bread = np.linalg.pinv(X_mz.T @ X_mz)
    covariance = bread @ _hac_meat(X_mz * residuals[:, None], lags) @ bread
    covariance = (covariance + covariance.T) / 2.0
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    restrictions = beta - np.array([0.0, 1.0])
    joint_stat = float(restrictions @ np.linalg.pinv(covariance) @ restrictions)
    joint_pvalue = float(chi2.sf(max(joint_stat, 0.0), df=2))
    t_intercept = float(beta[0] / se[0]) if se[0] > _EPS else np.nan
    t_slope = float((beta[1] - 1.0) / se[1]) if se[1] > _EPS else np.nan
    return {
        "intercept": float(beta[0]), "slope": float(beta[1]),
        "t_intercept": t_intercept, "t_slope": t_slope,
        "p_intercept": float(2 * norm.sf(abs(t_intercept))) if np.isfinite(t_intercept) else np.nan,
        "p_slope": float(2 * norm.sf(abs(t_slope))) if np.isfinite(t_slope) else np.nan,
        "joint_stat": joint_stat, "joint_pvalue": joint_pvalue,
        "hac_lags": lags, "unbiased": bool(joint_pvalue >= alpha),
    }


def diebold_mariano(y_true: np.ndarray, forecast_a: np.ndarray, forecast_b: np.ndarray,
                     loss: str = "qlike", is_log_rv: bool = True,
                     hac_lags: int = None, alternative: str = "two-sided") -> dict:
    y_true, forecast_a = _paired_finite(y_true, forecast_a)
    forecast_b = np.asarray(forecast_b, dtype=float).reshape(-1)
    if len(forecast_b) != len(y_true) or not np.all(np.isfinite(forecast_b)):
        raise ValueError("All forecast arrays must contain aligned finite observations")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    differential = (_forecast_losses(y_true, forecast_a, loss, is_log_rv)
                    - _forecast_losses(y_true, forecast_b, loss, is_log_rv))
    n = len(differential)
    lags = _default_hac_lags(n) if hac_lags is None else min(max(0, int(hac_lags)), n - 1)
    centered = differential - np.mean(differential)
    long_run_variance = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2 * weight * float(centered[lag:] @ centered[:-lag] / n)
    standard_error = np.sqrt(max(long_run_variance, 0.0) / n)
    statistic = float(np.mean(differential) / standard_error) if standard_error > _EPS else 0.0
    if alternative == "two-sided":
        pvalue = float(2 * norm.sf(abs(statistic)))
    elif alternative == "less":
        pvalue = float(norm.cdf(statistic))
    else:
        pvalue = float(norm.sf(statistic))
    return {
        "statistic": statistic, "pvalue": pvalue,
        "mean_loss_difference": float(np.mean(differential)),
        "hac_lags": lags, "n_obs": n,
    }


def block_bootstrap_confidence_interval(
        y_true: np.ndarray, y_pred: np.ndarray, metric: str = "rmse",
        block_size: int = None, n_bootstrap: int = 1000,
        confidence: float = 0.95, random_state=None,
        is_log_rv: bool = True) -> dict:
    y_true, y_pred = _paired_finite(y_true, y_pred)
    n = len(y_true)
    if n_bootstrap < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("n_bootstrap must be positive and confidence must be in (0, 1)")
    block_size = max(1, int(round(n ** (1.0 / 3.0)))) if block_size is None else int(block_size)
    if not 1 <= block_size <= n:
        raise ValueError("block_size must be between 1 and the sample length")
    if metric not in {"rmse", "mae", "r2", "qlike"}:
        raise ValueError("metric must be one of rmse, mae, r2, or qlike")
    rng = np.random.default_rng(random_state)
    draws = np.empty(int(n_bootstrap))
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)
    for draw in range(int(n_bootstrap)):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets) % n).reshape(-1)[:n]
        draws[draw] = compute_metrics(y_true[indices], y_pred[indices], is_log_rv)[metric]
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": compute_metrics(y_true, y_pred, is_log_rv)[metric],
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
        "confidence": float(confidence), "block_size": block_size,
        "n_bootstrap": int(n_bootstrap),
    }


# Concise aliases for downstream notebooks.
diebold_mariano_test = diebold_mariano
block_bootstrap_ci = block_bootstrap_confidence_interval


def model_confidence_set(
    losses: dict,
    alpha: float = 0.10,
    n_bootstrap: int = 1000,
    block_size: int = None,
    random_state=None,
) -> dict:
    """Block-bootstrap Model Confidence Set (Hansen, Lunde & Nason, 2011).

    Parameters
    ----------
    losses:
        ``{model_name: np.ndarray}`` of per-observation loss values.  All
        arrays must be the same length and contain finite values.
    alpha:
        Significance level for exclusion.  Default 0.10 as required by §11.
    n_bootstrap:
        Number of bootstrap replications.
    block_size:
        Moving-block size.  Defaults to ``ceil(n^(1/3))``.
    random_state:
        Seed or ``numpy.random.Generator`` for reproducibility.

    Returns
    -------
    dict with keys:

    - ``"included"``  – ``{model: bool}`` whether each model is in the MCS.
    - ``"pvalues"``   – ``{model: float}`` MCS p-value for each model.
    - ``"ranking"``   – list of model names ordered by mean loss (best first).
    - ``"alpha"``     – the significance level used.
    - ``"n_obs"``     – number of observations.
    - ``"n_bootstrap"`` – number of bootstrap replications.
    """
    model_names = list(losses)
    if len(model_names) < 2:
        raise ValueError("MCS requires at least two models")
    arrays = {name: np.asarray(val, dtype=float).reshape(-1) for name, val in losses.items()}
    n = len(next(iter(arrays.values())))
    for name, arr in arrays.items():
        if arr.shape != (n,):
            raise ValueError(f"Loss array for '{name}' has wrong length")
        if not np.isfinite(arr).all():
            raise ValueError(f"Loss array for '{name}' contains non-finite values")

    bs = max(1, int(np.ceil(n ** (1.0 / 3.0)))) if block_size is None else int(block_size)
    rng = np.random.default_rng(random_state)

    # Candidate set starts with all models.
    remaining = list(model_names)
    pvalues: dict[str, float] = {name: 0.0 for name in model_names}

    while len(remaining) > 1:
        # Pairwise loss differentials relative to the best model.
        mean_losses = {name: float(arrays[name].mean()) for name in remaining}
        best = min(mean_losses, key=mean_losses.__getitem__)

        # Range statistic: max over all (i,j) pairs of standardised mean diff.
        t_stats: dict[str, float] = {}
        for name in remaining:
            if name == best:
                continue
            d = arrays[name] - arrays[best]
            d_bar = float(d.mean())
            # Bootstrap variance of d_bar.
            boots = np.empty(n_bootstrap)
            n_blocks = int(np.ceil(n / bs))
            offsets = np.arange(bs)
            for b in range(n_bootstrap):
                starts = rng.integers(0, n, size=n_blocks)
                idx = ((starts[:, None] + offsets) % n).reshape(-1)[:n]
                boots[b] = d[idx].mean()
            variance = float(np.var(boots, ddof=1))
            se = np.sqrt(max(variance, 1e-30))
            t_stats[name] = abs(d_bar) / se

        worst = max(t_stats, key=t_stats.__getitem__)
        t_max = t_stats[worst]

        # Bootstrap distribution of the range statistic under H0.
        boot_t_max = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            n_blocks = int(np.ceil(n / bs))
            offsets = np.arange(bs)
            starts = rng.integers(0, n, size=n_blocks)
            idx = ((starts[:, None] + offsets) % n).reshape(-1)[:n]
            boot_vals = []
            for name in remaining:
                if name == best:
                    continue
                d = arrays[name] - arrays[best]
                d_b = d[idx].mean() - d.mean()
                boots_inner = np.empty(n_bootstrap)
                for bb in range(n_bootstrap):
                    s2 = rng.integers(0, n, size=n_blocks)
                    idx2 = ((s2[:, None] + offsets) % n).reshape(-1)[:n]
                    boots_inner[bb] = d[idx2].mean()
                se_inner = np.sqrt(max(np.var(boots_inner, ddof=1), 1e-30))
                boot_vals.append(abs(d_b) / se_inner)
            boot_t_max[b] = max(boot_vals) if boot_vals else 0.0

        pval = float(np.mean(boot_t_max >= t_max))
        pvalues[worst] = pval

        if pval < alpha:
            remaining.remove(worst)
        else:
            # All remaining models are in the MCS at this alpha level.
            for name in remaining:
                pvalues[name] = max(pvalues[name], pval)
            break

    included = {name: (name in remaining) for name in model_names}
    ranking = sorted(model_names, key=lambda m: float(arrays[m].mean()))
    return {
        "included": included,
        "pvalues": pvalues,
        "ranking": ranking,
        "alpha": float(alpha),
        "n_obs": n,
        "n_bootstrap": int(n_bootstrap),
    }
