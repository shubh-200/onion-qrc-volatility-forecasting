import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

from data_loader import load_spx_rv, split_data, label_regimes
from onion_qrc import OnionQRC, SingleBandQRC
from readout import VolQRCReadout, compute_metrics, mincer_zarnowitz
from baselines import GARCHBaseline, HARRVBaseline, ESNBaseline


RESULTS_DIR = Path(__file__).parent / "results"


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _result_dict(model, n_qubits, metrics, mz, regime_accuracy=None):
    d = {
        "model": model,
        "n_qubits": n_qubits,
        "rmse": _to_native(metrics["rmse"]),
        "qlike": _to_native(metrics["qlike"]),
        "r2": _to_native(metrics["r2"]),
        "mae": _to_native(metrics["mae"]),
        "mz_intercept": _to_native(mz["intercept"]),
        "mz_slope": _to_native(mz["slope"]),
        "mz_unbiased": _to_native(mz["unbiased"]),
    }
    if regime_accuracy is not None:
        d["regime_accuracy"] = _to_native(regime_accuracy)
    return d


def run_qrc_experiment(qrc, n_qubits, features_seq, y_seq, regime_seq,
                       har_seq, train_end, test_end,
                       warmup: int = 100, model_name: str = "QRC") -> dict:
    print(f"\n{'='*60}")
    print(f"Running {model_name} (N={n_qubits})")
    print(f"{'='*60}")

    n_total = len(features_seq)
    n_train = train_end
    n_test = test_end - train_end

    print(f"  Processing {n_total} total steps through reservoir...")
    t0 = time.time()
    all_obs = np.zeros((n_total, qrc.n_observables))
    qrc.reset_memory()
    for t in range(n_total):
        all_obs[t] = qrc.step(features_seq[t])
        if (t + 1) % 1000 == 0:
            print(f"    Step {t+1}/{n_total} ({time.time()-t0:.1f}s)")
    print(f"  Reservoir processing done in {time.time()-t0:.1f}s")

    train_obs = all_obs[warmup:n_train]
    y_train = y_seq[warmup:n_train]
    r_train = regime_seq[warmup:n_train]
    har_train = har_seq[warmup:n_train]

    test_obs = all_obs[n_train:n_test + n_train]
    y_test = y_seq[n_train:n_test + n_train]
    r_test = regime_seq[n_train:n_test + n_train]
    har_test = har_seq[n_train:n_test + n_train]

    n_classes = len(np.unique(r_train))
    use_regime = n_classes >= 2
    print(f"  Training readout (regime_classes={n_classes}, use_regime={use_regime})...")

    readout = VolQRCReadout(ridge_alpha=1.0, use_regime=use_regime)
    if use_regime:
        readout.fit(train_obs, y_train, r_train, har_train)
    else:
        readout.fit(train_obs, y_train, X_classical=har_train)

    preds = readout.predict(test_obs, har_test)
    metrics = compute_metrics(y_test, preds, is_log_rv=True)
    mz = mincer_zarnowitz(y_test, preds)

    regime_acc = None
    if use_regime:
        try:
            regime_acc = readout.regime_accuracy(test_obs, r_test)
        except Exception:
            regime_acc = 0.0

    print(f"  Results: RMSE={metrics['rmse']:.6f}, QLIKE={metrics['qlike']:.6f}, "
          f"R²={metrics['r2']:.6f}, RegimeAcc={regime_acc}")
    print(f"  Mincer-Zarnowitz: intercept={mz['intercept']:.4f}, slope={mz['slope']:.4f}, unbiased={mz['unbiased']}")

    return _result_dict(model_name, n_qubits, metrics, mz, regime_acc)


def run_baselines(y_seq, log_rv, rv, df_raw, train_end_idx, test_end_idx) -> list:
    print(f"\n{'='*60}")
    print(f"Running Classical Baselines")
    print(f"{'='*60}")

    n_total = len(y_seq)
    train_end = train_end_idx
    results = []

    returns = np.diff(log_rv)
    returns = np.clip(returns, -10, 10)

    print("  Fitting GARCH(1,1)...")
    try:
        garch = GARCHBaseline()
        garch.fit(returns[:train_end])
        n_test = test_end_idx - train_end
        garch_fc = garch.forecast(horizon=n_test)
        if len(garch_fc) >= n_test:
            garch_pred = np.log(garch_fc[:n_test].clip(1e-12))
            y_test = y_seq[train_end:train_end + n_test]
            garch_metrics = compute_metrics(y_test, garch_pred, is_log_rv=True)
            garch_mz = mincer_zarnowitz(y_test, garch_pred)
            results.append(_result_dict("GARCH", "-", garch_metrics, garch_mz))
            print(f"    GARCH: RMSE={garch_metrics['rmse']:.6f}, QLIKE={garch_metrics['qlike']:.6f}")
    except Exception as e:
        print(f"    GARCH failed: {e}")

    print("  Fitting HAR-RV...")
    try:
        rv_d = df_raw["rv_d"].values
        rv_w = df_raw["rv_w"].values
        rv_m = df_raw["rv_m"].values
        y_har = log_rv

        har = HARRVBaseline()
        har.fit(rv_d[1:train_end], rv_w[1:train_end], rv_m[1:train_end], y_har[1:train_end])
        n_test = test_end_idx - train_end
        har_pred = har.predict(rv_d[train_end + 1:train_end + 1 + n_test],
                                rv_w[train_end + 1:train_end + 1 + n_test],
                                rv_m[train_end + 1:train_end + 1 + n_test])
        y_test = y_seq[train_end:train_end + n_test]
        n_har = min(len(har_pred), n_test)
        har_metrics = compute_metrics(y_test[:n_har], har_pred[:n_har], is_log_rv=True)
        har_mz = mincer_zarnowitz(y_test[:n_har], har_pred[:n_har])
        results.append(_result_dict("HAR-RV", "-", har_metrics, har_mz))
        print(f"    HAR-RV: RMSE={har_metrics['rmse']:.6f}, QLIKE={har_metrics['qlike']:.6f}")
    except Exception as e:
        print(f"    HAR-RV failed: {e}")

    print("  Fitting ESN...")
    try:
        esn = ESNBaseline(n_reservoir=500, spectral_radius=0.95, sparsity=0.1, seed=42)
        X_esn = log_rv[:-1].reshape(-1, 1)
        y_esn = log_rv[1:]
        esn_train = train_end - 1
        esn.fit(X_esn[:esn_train], y_esn[:esn_train], washout=50)

        n_test = test_end_idx - train_end
        X_esn_test = log_rv[train_end - 1:train_end - 1 + n_test + 50].reshape(-1, 1)
        esn_pred = esn.predict(X_esn_test)
        y_test = y_seq[train_end:train_end + n_test]
        n_esn = min(len(esn_pred), n_test)
        esn_metrics = compute_metrics(y_test[:n_esn], esn_pred[:n_esn], is_log_rv=True)
        esn_mz = mincer_zarnowitz(y_test[:n_esn], esn_pred[:n_esn])
        results.append(_result_dict("ESN", "-", esn_metrics, esn_mz))
        print(f"    ESN: RMSE={esn_metrics['rmse']:.6f}, QLIKE={esn_metrics['qlike']:.6f}")
    except Exception as e:
        print(f"    ESN failed: {e}")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_spx_rv()
    n_total = len(df)
    print(f"  {n_total} rows loaded")

    log_rv = df["log_rv"].values
    rv = df["rv"].values

    features = df[["log_rv", "rv_d", "rv_w", "rv_m", "log_return"]].values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    n_train_raw = int(n_total * 0.70)
    features[:n_train_raw] = scaler.fit_transform(features[:n_train_raw])
    features[n_train_raw:] = scaler.transform(features[n_train_raw:])

    har_seq = df[["rv_d", "rv_w", "rv_m"]].values[:-1]
    har_scaler = StandardScaler()
    har_seq[:n_train_raw] = har_scaler.fit_transform(har_seq[:n_train_raw])
    har_seq[n_train_raw:] = har_scaler.transform(har_seq[n_train_raw:])

    y = log_rv[1:]
    features_seq = features[:-1]
    regime_all = label_regimes(df)
    regime_seq = regime_all[1:]

    train_end = int(len(y) * 0.70)
    val_end = int(len(y) * 0.85)
    test_end = len(y)

    print(f"  Train: {train_end}, Val: {val_end}, Test: {test_end}")
    print(f"  Regime distribution (train): {np.bincount(regime_seq[:train_end])}")
    print(f"  Regime distribution (test): {np.bincount(regime_seq[train_end:test_end])}")

    all_results = []

    for n in [5, 10]:
        try:
            qrc = OnionQRC(n, seed=42)
            print(f"  Onion allocation: {qrc.band_info()}")
            res = run_qrc_experiment(qrc, n, features_seq, y, regime_seq,
                                      har_seq, train_end, test_end, warmup=100,
                                      model_name="OnionQRC")
            all_results.append(res)
        except Exception as e:
            print(f"  OnionQRC N={n} failed: {e}")
            import traceback
            traceback.print_exc()

    for n in [5, 10]:
        try:
            qrc = SingleBandQRC(n, seed=42)
            res = run_qrc_experiment(qrc, n, features_seq, y, regime_seq,
                                      har_seq, train_end, test_end, warmup=100,
                                      model_name="SingleBandQRC")
            all_results.append(res)
        except Exception as e:
            print(f"  SingleBandQRC N={n} failed: {e}")
            import traceback
            traceback.print_exc()

    baseline_results = run_baselines(y, log_rv, rv, df, train_end, test_end)
    all_results.extend(baseline_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_to_native)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'N':>4} {'RMSE':>12} {'QLIKE':>12} {'R²':>10} {'RegAcc':>8} {'MZ-Unb':>8}")
    print("-" * 80)
    for r in all_results:
        n = r.get("n_qubits", "-")
        ra = r.get("regime_accuracy", None)
        ra_str = f"{ra:.4f}" if ra is not None else "-"
        mz = r.get("mz_unbiased", "-")
        print(f"{r['model']:<20} {str(n):>4} {r['rmse']:>12.6f} {r['qlike']:>12.6f} {r['r2']:>10.6f} {ra_str:>8} {str(mz):>8}")

    print("\nKey comparisons:")
    onion_qlike = onion_rmse = None
    for r in all_results:
        if r["model"] == "OnionQRC" and r.get("n_qubits") == 10:
            onion_qlike = r["qlike"]
            onion_rmse = r["rmse"]
            break

    esn_qlike = esn_rmse = None
    for r in all_results:
        if r["model"] == "ESN":
            esn_qlike = r["qlike"]
            esn_rmse = r["rmse"]
            break

    har_qlike = har_rmse = None
    for r in all_results:
        if r["model"] == "HAR-RV":
            har_qlike = r["qlike"]
            har_rmse = r["rmse"]
            break

    if onion_qlike is not None and esn_qlike is not None:
        print(f"  OnionQRC(N=10) vs ESN: QLIKE {onion_qlike:.4f} vs {esn_qlike:.4f} "
              f"({'QRC wins' if onion_qlike < esn_qlike else 'ESN wins'})")
    if onion_qlike is not None and har_qlike is not None:
        print(f"  OnionQRC(N=10) vs HAR-RV: QLIKE {onion_qlike:.4f} vs {har_qlike:.4f} "
              f"({'QRC wins' if onion_qlike < har_qlike else 'HAR wins'})")


if __name__ == "__main__":
    main()
