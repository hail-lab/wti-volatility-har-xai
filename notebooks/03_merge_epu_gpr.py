import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# Robust CSV reader
# =========================
def robust_read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV with unknown delimiter/encoding robustly.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    seps = [None, ",", "\t", ";", "|"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as e:
                last_err = e
    raise last_err


# =========================
# EPU loader (your exact format)
# =========================
def load_epu_daily(path: str) -> pd.DataFrame:
    """
    EPU file format you have:
      columns: day, month, year, daily_policy_index
    Builds a strict YYYY-MM-DD date index.
    """
    df = robust_read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    print("\n--- AUDIT EPU ---")
    print("Columns:", df.columns.tolist())
    print(df.head(10))

    # Validate expected columns
    required = ["day", "month", "year", "daily_policy_index"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"EPU: missing column '{col}'. Found: {df.columns.tolist()}")

    y = pd.to_numeric(df["year"], errors="coerce")
    m = pd.to_numeric(df["month"], errors="coerce")
    d = pd.to_numeric(df["day"], errors="coerce")
    v = pd.to_numeric(df["daily_policy_index"], errors="coerce")

    # Build strict date strings (version-proof)
    ys = y.astype("Int64").astype(str)
    ms = m.astype("Int64").astype(str).str.zfill(2)
    ds = d.astype("Int64").astype(str).str.zfill(2)
    date_strings = ys + "-" + ms + "-" + ds

    dates = pd.to_datetime(date_strings, format="%Y-%m-%d", errors="coerce")

    print(
        "EPU numeric counts:",
        "year", y.notna().sum(),
        "month", m.notna().sum(),
        "day", d.notna().sum(),
        "value", v.notna().sum(),
        "dates", pd.Series(dates).notna().sum(),
    )

    # IMPORTANT: avoid pandas index alignment (Series index 0..n vs DatetimeIndex)
    out = pd.DataFrame({"EPU": v.to_numpy()}, index=dates)
    out = out.dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    print("EPU built rows:", len(out))
    if len(out) > 0:
        print("EPU date range:", out.index.min().date(), "to", out.index.max().date())

    return out


# =========================
# GPR loader (Excel)
# =========================
def load_gpr_excel(path: str) -> pd.DataFrame:
    """
    Loads GPR from Excel file (data_raw/gpr.xlsx).
    Uses 'month' as date and 'GPR' as the global index.
    Drops metadata rows where var_name/var_label exist.
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    print("\n--- AUDIT GPR ---")
    print("Using file:", path)
    print("Columns:", df.columns.tolist())
    print(df.head(10))

    if "month" not in df.columns or "GPR" not in df.columns:
        raise ValueError("GPR: expected columns 'month' and 'GPR' in the Excel file.")

    dates = pd.to_datetime(df["month"], errors="coerce")
    gpr_values = pd.to_numeric(df["GPR"], errors="coerce")

    # Drop metadata rows if present
    if "var_name" in df.columns and "var_label" in df.columns:
        mask_data = df["var_name"].isna() & df["var_label"].isna()
        dates = dates[mask_data]
        gpr_values = gpr_values[mask_data]

    # IMPORTANT: avoid pandas index alignment
    out = pd.DataFrame({"GPR": gpr_values.to_numpy()}, index=dates)
    out = out.dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    print("GPR built rows:", len(out))
    if len(out) > 0:
        print("GPR date range:", out.index.min().date(), "to", out.index.max().date())

    return out


# =========================
# Main merge pipeline
# =========================
def main():
    # Load trading-day volatility dataset created in Step 4
    vol_path = Path("data_interim/volatility_features.csv")
    if not vol_path.exists():
        raise FileNotFoundError("Missing data_interim/volatility_features.csv. Run 02_build_targets.py first.")

    vol = pd.read_csv(vol_path, parse_dates=["Date"]).set_index("Date").sort_index()

    # Load EPU
    epu_path = Path("data_raw/epu_us_daily.csv")
    if not epu_path.exists():
        raise FileNotFoundError("Missing data_raw/epu_us_daily.csv")

    epu = load_epu_daily(str(epu_path))

    # Load GPR from Excel
    gpr_path = Path("data_raw/gpr.xlsx")
    if not gpr_path.exists():
        raise FileNotFoundError("Missing data_raw/gpr.xlsx. Save the GPR file as an Excel workbook (.xlsx).")

    gpr = load_gpr_excel(str(gpr_path))

    # Merge on trading days
    df = vol.join(epu, how="left").join(gpr, how="left")

    # GPR is monthly; forward-fill onto trading days (cap avoids huge carryovers)
    df[["EPU", "GPR"]] = df[["EPU", "GPR"]].ffill(limit=60)

    # Drop rows where either series still missing (e.g., before 1985 or before 2013 merge window)
    df = df.dropna(subset=["EPU", "GPR"])

    # Save
    out_path = Path("data_interim/vol_plus_epu_gpr.csv")
    df.to_csv(out_path)

    print("\nSaved:", out_path)
    print(df.tail())
    print("\nFinal merged date range:", df.index.min().date(), "to", df.index.max().date())
    print("Rows/Cols:", df.shape)


if __name__ == "__main__":
    main()
