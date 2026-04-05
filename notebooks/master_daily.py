import pandas as pd
import numpy as np
from scipy.stats import skew

df = pd.read_csv("data_final/master_daily.csv", parse_dates=["Date"])

# Ensure consistent naming
df = df.rename(columns={"ret": "WTI_return"})

vars_used = [
    "vol_rv5", "vol_abs",
    "VIX", "OVX", "WTI_return",
    "EPU", "GPR",
    "news_volume",
    "news_tone_mean",
    "news_tone_median"
]

df_clean = df[vars_used].dropna()

# --- Descriptive statistics ---
desc = df_clean.describe().T
desc["Skewness"] = df_clean.skew()

print("\nDESCRIPTIVE STATS\n")
print(desc[["mean", "std", "min", "50%", "max", "Skewness"]])

# --- Correlation matrix ---
corr = df_clean.corr()
print("\nCORRELATION MATRIX\n")
print(corr.round(3))
