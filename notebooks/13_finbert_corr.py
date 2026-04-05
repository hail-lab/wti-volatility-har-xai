import pandas as pd

df = pd.read_csv("data_final/master_daily_finbert.csv", parse_dates=["Date"])
tmp = df[["news_tone_mean", "finbert_mean"]].dropna()

corr = tmp["news_tone_mean"].corr(tmp["finbert_mean"])
print("VADER mean vs FinBERT mean correlation:", round(corr, 3))
print("N used:", len(tmp))
