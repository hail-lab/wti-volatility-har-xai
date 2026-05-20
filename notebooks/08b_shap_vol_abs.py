import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

import shap
from xgboost import XGBRegressor

DATA_PATH = "data_final/master_daily.csv"
TARGET = "vol_abs"
TRAIN_LEN = 5 * 252   # 1260 trading days
RANDOM_STATE = 42

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()

    har = ["vol_lag1", "vol_week", "vol_month"]
    market = ["VIX", "OVX"]
    uncertainty = ["EPU", "GPR"]
    news_lags = [f"news_tone_mean_l{i}" for i in range(1, 8)] + \
                [f"news_volume_l{i}" for i in range(1, 8)]
    all_feats = list(dict.fromkeys(har + market + uncertainty + news_lags))

    df = df.dropna(subset=[TARGET] + all_feats).copy()

    # Last rolling training window
    last_idx = len(df) - 1
    train_start = max(0, last_idx - TRAIN_LEN)
    X_train = df[all_feats].iloc[train_start:last_idx]
    y_train = df[TARGET].iloc[train_start:last_idx]

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Model fitted.")

    # SHAP on 800-sample subset
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(X_train.index.to_numpy(), size=min(800, len(X_train)), replace=False)
    X_shap = X_train.loc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    print("SHAP computed.")

    tag = TARGET

    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    bar_path = f"outputs/figures/shap_bar_{tag}.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    swarm_path = f"outputs/figures/shap_beeswarm_{tag}.png"
    plt.tight_layout()
    plt.savefig(swarm_path, dpi=200)
    plt.close()

    print(f"Saved: {bar_path}")
    print(f"Saved: {swarm_path}")


if __name__ == "__main__":
    main()
