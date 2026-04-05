import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data_final/master_daily.csv"
TARGET = "vol_rv5"

# Expanding window settings
INITIAL_TRAIN_YEARS = 5
TRADING_DAYS = 252
INITIAL_TRAIN_LEN = INITIAL_TRAIN_YEARS * TRADING_DAYS

TEST_HORIZON = 1
MIN_TRAIN_OBS = 750


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/preds", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: pd.Series, y_pred: pd.Series):
    mask = (~y_true.isna()) & (~y_pred.isna())
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "n": int(mask.sum()),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": rmse(yt, yp),
    }


def dm_simple(loss_diff: pd.Series):
    d = loss_diff.dropna()
    T = len(d)
    if T < 20:
        return np.nan, np.nan, T

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))

    if var_d == 0:
        if mean_d > 0:
            return np.inf, 0.0, T
        elif mean_d < 0:
            return -np.inf, 0.0, T
        else:
            return 0.0, 1.0, T

    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value, T


def expanding_forecast(df, features, target):
    """
    Expanding window forecast:
    Train on [0:t) and predict at t, starting from INITIAL_TRAIN_LEN.
    """
    y = df[target].to_numpy(dtype=float)
    X = df[features].to_numpy(dtype=float)

    preds = np.full(len(df), np.nan)

    for t in range(INITIAL_TRAIN_LEN, len(df) - TEST_HORIZON + 1):
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:t+1]

        if len(y_train) < MIN_TRAIN_OBS:
            continue

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds[t] = model.predict(X_test)[0]

    return pd.Series(preds, index=df.index)


def main():
    ensure_dirs()

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()

    # Feature sets (same as before)
    har = ["vol_lag1", "vol_week", "vol_month"]
    market = ["VIX", "OVX"]
    uncertainty = ["EPU", "GPR"]
    news_lags = [f"news_tone_mean_l{i}" for i in range(1, 8)] + [f"news_volume_l{i}" for i in range(1, 8)]
    all_feats = list(dict.fromkeys(har + market + uncertainty + news_lags))

    # Drop NA rows for modeling
    df = df.dropna(subset=[TARGET] + all_feats).copy()

    y_true = df[TARGET]

    # Forecasts
    pred_har = expanding_forecast(df, har, TARGET)
    pred_all = expanding_forecast(df, all_feats, TARGET)

    # Save preds
    out_preds = pd.DataFrame({
        "y_true": y_true,
        "pred_HAR": pred_har,
        "pred_HAR+All": pred_all
    }, index=df.index)

    preds_path = f"outputs/preds/preds_expanding_{TARGET}.csv"
    out_preds.to_csv(preds_path)

    # Metrics
    m_har = evaluate(y_true, pred_har)
    m_all = evaluate(y_true, pred_all)

    metrics = pd.DataFrame([
        {"model": "HAR", **m_har},
        {"model": "HAR+All", **m_all},
    ]).set_index("model").sort_values("RMSE")

    metrics_path = f"outputs/tables/oos_expanding_{TARGET}.csv"
    metrics.to_csv(metrics_path)

    print(f"\n=== Expanding-Window OOS Results (Target: {TARGET}) ===")
    print(metrics)
    print(f"\nSaved preds:   {preds_path}")
    print(f"Saved metrics: {metrics_path}")

    # DM tests (squared loss)
    e_har_sq = (y_true - pred_har) ** 2
    e_all_sq = (y_true - pred_all) ** 2
    diff_sq = (e_har_sq - e_all_sq).dropna()
    dm_sq, p_sq, n_sq = dm_simple(diff_sq)

    # DM tests (MAE loss)
    e_har_abs = (y_true - pred_har).abs()
    e_all_abs = (y_true - pred_all).abs()
    diff_abs = (e_har_abs - e_all_abs).dropna()
    dm_abs, p_abs, n_abs = dm_simple(diff_abs)

    dm_table = pd.DataFrame([
        {"loss": "squared", "n": n_sq, "DM_stat": dm_sq, "p_value": p_sq},
        {"loss": "absolute", "n": n_abs, "DM_stat": dm_abs, "p_value": p_abs},
    ])

    dm_path = f"outputs/tables/dm_expanding_{TARGET}.csv"
    dm_table.to_csv(dm_path, index=False)

    print(f"\n=== Expanding-Window DM Tests: HAR vs HAR+All ({TARGET}) ===")
    print(dm_table)
    print(f"\nSaved DM table: {dm_path}")

    # Cumulative loss plot (squared)
    cum = diff_sq.cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(cum, label="Cumulative Loss Diff (HAR - HAR+All)")
    plt.axhline(0, linestyle="--")
    plt.title("Expanding Window: Cumulative Squared Error Difference\n(HAR vs HAR+All)")
    plt.ylabel("Cumulative Difference")
    plt.legend()
    plt.tight_layout()
    fig_path = f"outputs/figures/cum_loss_expanding_{TARGET}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
