import pandas as pd
import numpy as np

# Load market data
df = pd.read_csv("data_raw/market_data.csv", parse_dates=["Date"]).set_index("Date")

# 1. Compute log returns
# Defensive cleaning: drop non-positive prices (log undefined)
df["WTI"] = pd.to_numeric(df["WTI"], errors="coerce")
df.loc[df["WTI"] <= 0, "WTI"] = np.nan

df["ret"] = np.log(df["WTI"] / df["WTI"].shift(1))

# 2. Main volatility target: absolute returns
df["vol_abs"] = df["ret"].abs()

# 3. Robustness volatility target: 5-day realized volatility
df["vol_rv5"] = np.sqrt((df["ret"]**2).rolling(5).sum())

# 4. HAR-style lag structure (very important for credibility)
df["vol_lag1"] = df["vol_abs"].shift(1)
df["vol_week"] = df["vol_abs"].rolling(5).mean().shift(1)
df["vol_month"] = df["vol_abs"].rolling(22).mean().shift(1)

# Clean
df = df.dropna()

# Save interim dataset
df.to_csv("data_interim/volatility_features.csv")

print(df.tail())
print("\nDate range:", df.index.min().date(), "to", df.index.max().date())
print("\nColumns:", df.columns.tolist())
