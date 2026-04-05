import pandas as pd

MASTER_PATH = "data_final/master_daily.csv"
FINBERT_PATH = "data_interim/finbert_daily_sentiment.csv"
OUT_PATH = "data_final/master_daily_finbert.csv"

def normalize_date(s):
    # Robust: parses strings/datetimes, drops timezone, normalizes to midnight
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)  # drop tz if present
    return dt.dt.normalize()

def main():
    master = pd.read_csv(MASTER_PATH)
    finbert = pd.read_csv(FINBERT_PATH)

    # Normalize Date columns
    master["Date"] = normalize_date(master["Date"])
    finbert["Date"] = normalize_date(finbert["Date"])

    # Keep only what we need from FinBERT
    finbert = finbert[["Date", "finbert_mean", "finbert_median", "finbert_volume"]].copy()

    # Merge (LEFT join: preserve master trading dates)
    merged = master.merge(finbert, on="Date", how="left")

    # Report coverage
    total = len(merged)
    matched = merged["finbert_mean"].notna().sum()
    print(f"Total master rows: {total}")
    print(f"FinBERT matched rows: {matched} ({matched/total:.1%})")

    # Save
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")

    # Show last few rows for sanity
    print(merged[["Date","news_tone_mean","finbert_mean","news_volume","finbert_volume"]].tail(10))

if __name__ == "__main__":
    main()
