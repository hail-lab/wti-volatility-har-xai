import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

HAR_RMSE = 0.008439  # from Table 8 baseline HAR

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

# Load predictions
df = pd.read_csv("outputs/preds/preds_vol_rv5_finbert.csv", parse_dates=["Date"])

# Identify the VADER OOS mask (the 424 region)
mask = df["pred_HAR+All (VADER)"].notna()

# Apply SAME mask to FinBERT
df_eval = df.loc[mask].copy()

y = df_eval["y_true"]

finbert_pred = df_eval["pred_HAR+All (FinBERT)"]

mae_finbert = mean_absolute_error(y, finbert_pred)
rmse_finbert = rmse(y, finbert_pred)

delta_rmse = 100 * (HAR_RMSE - rmse_finbert) / HAR_RMSE

print("FinBERT (aligned 424 OOS sample)")
print("MAE:", round(mae_finbert, 6))
print("RMSE:", round(rmse_finbert, 6))
print("ΔRMSE vs HAR (%):", round(delta_rmse, 2))
print("n used:", len(df_eval))
