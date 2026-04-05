import pandas as pd

base = pd.read_csv("data_interim/vol_plus_epu_gpr.csv", parse_dates=["Date"]).set_index("Date").sort_index()
news = pd.read_csv("data_interim/news_daily_sentiment.csv", parse_dates=["Date"]).set_index("Date").sort_index()

df = base.join(news, how="left")

# Short forward fill (weekends)
df[["news_volume", "news_tone_mean", "news_tone_median"]] = \
    df[["news_volume", "news_tone_mean", "news_tone_median"]].ffill(limit=2)

# Lags 1–7
for lag in range(1, 8):
    df[f"news_tone_mean_l{lag}"] = df["news_tone_mean"].shift(lag)
    df[f"news_volume_l{lag}"] = df["news_volume"].shift(lag)

# Drop early missing lag rows
df = df.dropna(subset=[f"news_tone_mean_l7", f"news_volume_l7"])

df.to_csv("data_final/master_daily.csv")

print("Saved: data_final/master_daily.csv")
print("Final shape:", df.shape)
print("Date range:", df.index.min().date(), "to", df.index.max().date())
