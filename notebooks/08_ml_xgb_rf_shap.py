import os
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

import shap

# Force a non-interactive backend to avoid Tk/Tcl thread shutdown issues on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})


DATA_PATH = "data_final/master_daily.csv"

# Rolling window settings (match your baselines)
TRAIN_YEARS = 5
TRADING_DAYS = 252
TRAIN_LEN = TRAIN_YEARS * TRADING_DAYS
TEST_HORIZON = 1
MIN_TRAIN_OBS = 750

# Script runs both targets
TARGETS = ["vol_rv5", "vol_abs"]

# Only compute SHAP for this target (publication target)
SHAP_TARGET = "vol_rv5"

RANDOM_STATE = 42


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


def feature_sets():
    har = ["vol_lag1", "vol_week", "vol_month"]
    market = ["VIX", "OVX"]
    uncertainty = ["EPU", "GPR"]
    news_lags = [f"news_tone_mean_l{i}" for i in range(1, 8)] + \
                [f"news_volume_l{i}" for i in range(1, 8)]
    all_feats = list(dict.fromkeys(har + market + uncertainty + news_lags))
    return har, all_feats


def rolling_forecast(df, features, model_builder, target):
    """
    Rolling-window 1-step ahead predictions.
    Train on the previous TRAIN_LEN obs and predict next day at index t.
    Returns predictions as a Series aligned to df.index.
    """
    y = df[target].to_numpy(dtype=float)
    X = df[features].to_numpy(dtype=float)
    idx = df.index

    preds = np.full(len(df), np.nan)

    for t in range(TRAIN_LEN, len(df) - TEST_HORIZON + 1):
        train_start = t - TRAIN_LEN
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

    return pd.Series(preds, index=idx)


def build_models():
    """
    Returns dict: name -> builder()
    Keep these stable and defensible for the paper.
    """
    def lin_har_all():
        return LinearRegression()

    def rf():
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def xgb():
        return XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return {
        "HAR+All (OLS)": lin_har_all,
        "RandomForest": rf,
        "XGBoost": xgb,
    }


def run_target(target: str):
    _, all_feats = feature_sets()
    models = build_models()

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.dropna(subset=[target] + all_feats).copy()

    y_true = df[target]
    pred_df = pd.DataFrame(index=df.index)
    pred_df["y_true"] = y_true

    rows = []
    for name, builder in models.items():
        feats = all_feats  # all models use the same info set for fair comparison

        t0 = time.time()
        pred = rolling_forecast(df, feats, builder, target)
        elapsed = time.time() - t0
        print(f"[TIME] {name} | target={target} | {elapsed:.2f}s")

        pred_df[f"pred_{name}"] = pred

        m = evaluate(y_true, pred)
        m["model"] = name
        rows.append(m)

    metrics = pd.DataFrame(rows).set_index("model").sort_values("RMSE")

    metrics_path = f"outputs/tables/oos_ml_{target}.csv"
    preds_path = f"outputs/preds/preds_ml_{target}.csv"
    metrics.to_csv(metrics_path)
    pred_df.to_csv(preds_path)

    print(f"\n=== Rolling OOS Results (ML) Target: {target} ===")
    print(metrics)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved preds:   {preds_path}")

    if target == SHAP_TARGET:
        make_shap_explanations(df, all_feats, target, tag=target)
    else:
        print(f"\nSHAP skipped for {target} (configured SHAP_TARGET={SHAP_TARGET}).")

    return metrics


def make_shap_explanations(df, features, target, tag="vol_rv5"):
    """
    Fit a single XGBoost model on the last rolling training window (same as OOS setup),
    then compute SHAP on a manageable sample. Saves plots + tables.
    """
    last_test_idx = len(df) - 1
    train_start = max(0, last_test_idx - TRAIN_LEN)
    train_end = last_test_idx

    X_train = df[features].iloc[train_start:train_end]
    y_train = df[target].iloc[train_start:train_end]

    model = build_models()["XGBoost"]()
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    sample_n = min(800, len(X_train))

    # Sample with an explicit RNG (future-proof; avoids NumPy global RNG warnings)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(X_train.index.to_numpy(), size=sample_n, replace=False)
    X_shap = X_train.loc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    imp_path = f"outputs/tables/shap_importance_{tag}.csv"
    imp.to_csv(imp_path, index=False)

    # SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    bar_path = f"outputs/figures/shap_bar_{tag}.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # SHAP beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    swarm_path = f"outputs/figures/shap_beeswarm_{tag}.png"
    plt.tight_layout()
    plt.savefig(swarm_path, dpi=200)
    plt.close()

    print(f"\nSHAP saved for {tag}:")
    print(f"- {imp_path}")
    print(f"- {bar_path}")
    print(f"- {swarm_path}")


def main():
    ensure_dirs()
    for tgt in TARGETS:
        run_target(tgt)


if __name__ == "__main__":
    main()
