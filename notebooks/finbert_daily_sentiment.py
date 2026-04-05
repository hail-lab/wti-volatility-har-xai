import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm

IN_PATH = "data_raw/gdelt_titles_raw.csv"
OUT_DAILY = "data_interim/finbert_daily_sentiment.csv"

# FinBERT label -> signed score
LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# Batch size: tune for your RAM/CPU (32–128 typical)
BATCH_SIZE = 64

def main():
    # Load raw titles
    df = pd.read_csv(IN_PATH)
    if "Date" not in df.columns or "title" not in df.columns:
        raise ValueError(f"Expected columns ['Date','title'] in {IN_PATH}, got {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["title"] = df["title"].astype(str).str.strip()
    df = df.dropna(subset=["Date"])
    df = df[df["title"].ne("")]

    # Load FinBERT
    finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True  # let HF handle truncation safely
    )

    titles = df["title"].tolist()
    scores = []

    # Batch scoring (much faster than apply)
    for i in tqdm(range(0, len(titles), BATCH_SIZE), desc="Scoring titles"):
        batch = titles[i:i + BATCH_SIZE]
        try:
            preds = finbert(batch)
            for p in preds:
                sign = LABEL_MAP.get(p["label"].lower(), 0.0)
                scores.append(sign * float(p["score"]))
        except Exception:
            # If batch fails, fall back to per-item scoring to avoid losing work
            for t in batch:
                try:
                    p = finbert(t)[0]
                    sign = LABEL_MAP.get(p["label"].lower(), 0.0)
                    scores.append(sign * float(p["score"]))
                except Exception:
                    scores.append(np.nan)

    df["finbert_score"] = scores

    # Daily aggregation
    daily = (
        df.groupby(df["Date"].dt.date)
          .agg(
              finbert_mean=("finbert_score", "mean"),
              finbert_median=("finbert_score", "median"),
              finbert_volume=("finbert_score", "count")
          )
          .reset_index()
          .rename(columns={"Date": "Date"})
    )

    daily["Date"] = pd.to_datetime(daily["Date"])
    daily.to_csv(OUT_DAILY, index=False)
    print(f"Saved: {OUT_DAILY}")
    print(daily.tail())

if __name__ == "__main__":
    main()
