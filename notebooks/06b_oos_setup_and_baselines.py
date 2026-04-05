import os
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data_final/master_daily.csv"

# Rolling window parameters
TRAIN_YEARS = 5
TEST_HORIZON = 1
MIN_TRAIN_OBS = 750
TRADING_DAYS = 252


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rolling_forecast(df, features, model_builder, target):
    """
    Rolling-window 1-step-ahead forecasts.
    Train on the previous TRAIN_YEARS*252 observations and predict at time t.
    Returns a Series aligned to df.index.
    """
    y = df[target].to_numpy(dtype=float)
    X = df[features].to_numpy(dtype=float)

    dates = df.index
    preds = np.full(len(df), np.nan)

    train_len = TRAIN_YEARS * TRADING_DAYS

    for t in range(train_len, len(df) - TEST_HORIZON + 1):
        train_start = t - train_len
        train_end = t
        test_idx = t

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_idx:test_idx + 1]

        if len(y_train) < MIN_TRAIN_OBS:
            continue

        model = model_builder()
        model.fit(X_train, y_train)
        preds[test_idx] = model.predict(X_test)[0]

    return pd.Series(preds, index=dates)


def evaluate(y_true: pd.Series, y_pred: pd.Series):
    mask = (~y_true.isna()) & (~y_pred.isna())
    yt = y_true[mask]
    yp = y_pred[mask]

    return {
        "n": int(mask.sum()),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": rmse(yt, yp),
    }


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/preds", exist_ok=True)


def build_models():
    har = ["vol_lag1", "vol_week", "vol_month"]
    market = ["VIX", "OVX"]
    uncertainty = ["EPU", "GPR"]
    news_lags = [f"news_tone_mean_l{i}" for i in range(1, 8)] + \
                [f"news_volume_l{i}" for i in range(1, 8)]

    models = {}

    models["HAR"] = (har, lambda: LinearRegression())
    models["HAR+Market"] = (har + market, lambda: LinearRegression())
    models["HAR+Uncertainty"] = (har + uncertainty, lambda: LinearRegression())
    models["HAR+News"] = (har + news_lags, lambda: LinearRegression())
    models["HAR+All"] = (har + market + uncertainty + news_lags, lambda: LinearRegression())

    def lasso_builder():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(
                cv=5,
                random_state=42,
                alphas=100,
                max_iter=30000
            ))
        ])

    models["Lasso(All)"] = (har + market + uncertainty + news_lags, lasso_builder)

    all_features = list(dict.fromkeys(har + market + uncertainty + news_lags))
    return models, all_features


def run_for_target(df_raw: pd.DataFrame, target: str):
    models, all_features = build_models()

    needed_cols = [target] + all_features
    df = df_raw.dropna(subset=needed_cols).copy()

    pred_df = pd.DataFrame(index=df.index)
    pred_df["y_true"] = df[target]

    metrics_rows = []

    for name, (feats, builder) in models.items():
        t0 = time.time()
        pred = rolling_forecast(df, feats, builder, target=target)
        elapsed = time.time() - t0
        print(f"[TIME] {name} | target={target} | {elapsed:.2f}s")

        pred_df[f"pred_{name}"] = pred

        m = evaluate(df[target], pred)
        m["model"] = name
        metrics_rows.append(m)

    metrics = pd.DataFrame(metrics_rows).set_index("model").sort_values("RMSE")

    metrics_path = f"outputs/tables/oos_baselines_{target}.csv"
    preds_path = f"outputs/preds/preds_{target}.csv"

    metrics.to_csv(metrics_path)
    pred_df.to_csv(preds_path)

    print(f"\n=== Rolling OOS Results (Target: {target}) ===")
    print(metrics)
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved preds:   {preds_path}")

    return metrics, pred_df


def main():
    ensure_dirs()

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]) \
           .set_index("Date") \
           .sort_index()

    run_for_target(df, "vol_abs")
    run_for_target(df, "vol_rv5")


if __name__ == "__main__":
    main()
