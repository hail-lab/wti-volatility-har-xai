import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

OUT_PATH = "data_interim/gdelt_daily_sentiment.csv"

# Oil-focused query (broad enough to get daily coverage)
QUERY = '(oil OR "crude oil" OR WTI OR brent OR "energy prices" OR opec)'

# GDELT DOC 2.1 endpoint
BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Cap articles processed per day (keeps runtime manageable)
MAX_ARTICLES_PER_DAY = 250


def fetch_day(date_yyyy_mm_dd: str):
    """
    Fetch up to MAX_ARTICLES_PER_DAY articles for a given day from GDELT DOC API.
    Returns list of items (each item has tone-related fields sometimes).
    """
    # DOC API uses datetime strings like YYYYMMDDHHMMSS
    start = date_yyyy_mm_dd.replace("-", "") + "000000"
    end = date_yyyy_mm_dd.replace("-", "") + "235959"

    params = {
        "query": QUERY,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
        "maxrecords": MAX_ARTICLES_PER_DAY,
        "sort": "HybridRel",  # relevance-ish
    }

    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    articles = js.get("articles", [])
    return articles


def get_tone(article):
    """
    Try several known fields that sometimes appear.
    DOC API often provides 'tone' in 'tone' or inside 'sourceCountry' etc depending on data.
    We'll safely extract numeric tone if present.
    """
    # Most common in some responses:
    for key in ["tone", "Tone", "V2Tone", "v2tone"]:
        if key in article:
            try:
                return float(article[key])
            except Exception:
                pass

    # Sometimes tone is embedded in "socialimage" etc: ignore.
    return np.nan


def main():
    # Match your existing project range (from your merged dataset)
    df_base = pd.read_csv("data_interim/vol_plus_epu_gpr.csv", parse_dates=["Date"])
    start_date = df_base["Date"].min().date()
    end_date = df_base["Date"].max().date()

    dates = []
    tone_means = []
    tone_medians = []
    volumes = []

    cur = start_date
    for _ in tqdm(range((end_date - start_date).days + 1)):
        d = cur.strftime("%Y-%m-%d")
        try:
            articles = fetch_day(d)
            tones = [get_tone(a) for a in articles]
            tones = np.array([t for t in tones if np.isfinite(t)], dtype=float)

            vol = len(articles)
            if tones.size > 0:
                tone_mean = float(np.mean(tones))
                tone_median = float(np.median(tones))
            else:
                tone_mean = np.nan
                tone_median = np.nan

            dates.append(cur)
            volumes.append(vol)
            tone_means.append(tone_mean)
            tone_medians.append(tone_median)

        except Exception:
            # If API hiccups for a day, keep NaNs but preserve the date
            dates.append(cur)
            volumes.append(np.nan)
            tone_means.append(np.nan)
            tone_medians.append(np.nan)

        cur = cur + timedelta(days=1)

    out = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "gdelt_volume": volumes,
        "gdelt_tone_mean": tone_means,
        "gdelt_tone_median": tone_medians,
    }).sort_values("Date")

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print(out.tail(10))


if __name__ == "__main__":
    main()
