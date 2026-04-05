import requests
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

OUT_DAILY_PATH = "data_interim/gdelt_daily_sentiment.csv"
OUT_TITLES_PATH = "data_raw/gdelt_titles_raw.csv"

QUERY = '(oil OR "crude oil" OR WTI OR brent OR "energy prices" OR opec)'
BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_ARTICLES_PER_DAY = 250


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


def get_tone(article):
    for key in ["tone", "Tone", "V2Tone", "v2tone"]:
        if key in article:
            try:
                return float(article[key])
            except Exception:
                pass
    return np.nan


def get_title(article):
    t = article.get("title", "")
    if t is None:
        t = ""
    return str(t).strip()


def main():
    # Use same date range as your pipeline
    df_base = pd.read_csv("data_interim/vol_plus_epu_gpr.csv", parse_dates=["Date"])
    start_date = df_base["Date"].min().date()
    end_date = df_base["Date"].max().date()

    daily_rows = []
    title_rows = []

    cur = start_date
    for _ in tqdm(range((end_date - start_date).days + 1)):
        d = cur.strftime("%Y-%m-%d")
        try:
            articles = fetch_day(d)

            # --- save raw titles ---
            for a in articles:
                title = get_title(a)
                if title:
                    title_rows.append({"Date": d, "title": title})

            # --- compute daily aggregates (same as before) ---
            tones = [get_tone(a) for a in articles]
            tones = np.array([t for t in tones if np.isfinite(t)], dtype=float)

            vol = len(articles)
            tone_mean = float(np.mean(tones)) if tones.size > 0 else np.nan
            tone_median = float(np.median(tones)) if tones.size > 0 else np.nan

            daily_rows.append({
                "Date": d,
                "gdelt_volume": vol,
                "gdelt_tone_mean": tone_mean,
                "gdelt_tone_median": tone_median
            })

        except Exception:
            daily_rows.append({
                "Date": d,
                "gdelt_volume": np.nan,
                "gdelt_tone_mean": np.nan,
                "gdelt_tone_median": np.nan
            })

        cur = cur + timedelta(days=1)

    # write outputs
    pd.DataFrame(daily_rows).to_csv(OUT_DAILY_PATH, index=False)
    pd.DataFrame(title_rows).to_csv(OUT_TITLES_PATH, index=False)

    print(f"Saved daily aggregates: {OUT_DAILY_PATH}")
    print(f"Saved raw titles:       {OUT_TITLES_PATH}")


if __name__ == "__main__":
    main()
