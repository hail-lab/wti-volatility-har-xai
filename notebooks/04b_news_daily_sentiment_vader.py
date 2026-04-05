import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUT_PATH = "data_interim/news_daily_sentiment.csv"

QUERY = '(oil OR "crude oil" OR WTI OR brent OR opec OR "energy prices")'
BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

MAX_ARTICLES_PER_DAY = 150  # faster than 250
analyzer = SentimentIntensityAnalyzer()


def fetch_day(date_yyyy_mm_dd: str):
    start = date_yyyy_mm_dd.replace("-", "") + "000000"
    end = date_yyyy_mm_dd.replace("-", "") + "235959"

    params = {
        "query": QUERY,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
        "maxrecords": MAX_ARTICLES_PER_DAY,
        "sort": "HybridRel",
    }

    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    return js.get("articles", [])


def compute_title_sentiment(title):
    if not isinstance(title, str) or title.strip() == "":
        return np.nan
    score = analyzer.polarity_scores(title)
    return score["compound"]


def main():
    base_df = pd.read_csv("data_interim/vol_plus_epu_gpr.csv", parse_dates=["Date"])
    start_date = base_df["Date"].min().date()
    end_date = base_df["Date"].max().date()

    dates = []
    volumes = []
    tone_means = []
    tone_medians = []

    cur = start_date
    total_days = (end_date - start_date).days + 1

    for _ in tqdm(range(total_days)):
        d = cur.strftime("%Y-%m-%d")

        try:
            articles = fetch_day(d)
            titles = [a.get("title", "") for a in articles]

            sentiments = [compute_title_sentiment(t) for t in titles]
            sentiments = np.array([s for s in sentiments if np.isfinite(s)])

            vol = len(titles)

            if sentiments.size > 0:
                tone_mean = float(np.mean(sentiments))
                tone_median = float(np.median(sentiments))
            else:
                tone_mean = np.nan
                tone_median = np.nan

        except Exception:
            vol = np.nan
            tone_mean = np.nan
            tone_median = np.nan

        dates.append(cur)
        volumes.append(vol)
        tone_means.append(tone_mean)
        tone_medians.append(tone_median)

        cur += timedelta(days=1)

    out = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "news_volume": volumes,
        "news_tone_mean": tone_means,
        "news_tone_median": tone_medians,
    }).sort_values("Date")

    out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(out.tail(10))


if __name__ == "__main__":
    main()
