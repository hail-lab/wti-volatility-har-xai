import yfinance as yf
import pandas as pd

start = "2013-04-01"

tickers = ["CL=F", "^VIX", "^OVX"]

raw = yf.download(
    tickers,
    start=start,
    auto_adjust=False,   # keep raw OHLCV
    group_by="column"    # keeps fields grouped (often MultiIndex)
)

print("Columns:", raw.columns)
print(raw.tail())

def pick_price(df, ticker):
    """
    Return a 1D price series for a ticker.
    Works whether columns are MultiIndex or single-level.
    Prefers Adj Close, falls back to Close.
    """
    # MultiIndex case: columns like ('Adj Close','CL=F') or ('Close','CL=F')
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            raise KeyError(f"No Adj Close/Close found for {ticker}. Available fields: {df.columns.levels[0].tolist()}")
        return s

    # Single-level case: already only one ticker downloaded
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    if "Close" in df.columns:
        return df["Close"]
    raise KeyError("No Adj Close/Close in columns: " + str(df.columns))

df = pd.DataFrame({
    "WTI": pick_price(raw, "CL=F"),
    "VIX": pick_price(raw, "^VIX"),
    "OVX": pick_price(raw, "^OVX"),
})

# Clean missing
df = df.sort_index().dropna(how="all")

df.to_csv("data_raw/market_data.csv")

print("\nSaved: data_raw/market_data.csv")
print("Date range:", df.index.min().date(), "to", df.index.max().date())
print(df.tail())
