import pandas as pd
import numpy as np
from scipy import stats

TARGET = "vol_rv5"
PRED_PATH = f"outputs/preds/preds_{TARGET}.csv"

BASELINE_MODEL = "HAR"
COMP_MODEL = "HAR+All"


def dm_simple(loss_diff: pd.Series):
    """
    Simple DM-style test (normal approx) on loss differentials.
    Handles zero-variance edge cases safely.
    """
    d = loss_diff.dropna()
    T = len(d)

    if T < 20:
        return np.nan, np.nan, T

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))

    # Edge case: zero variance
    if var_d == 0:
        if mean_d > 0:
            return np.inf, 0.0, T
        elif mean_d < 0:
            return -np.inf, 0.0, T
        else:
            return 0.0, 1.0, T

    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value, T


def main():
    df = pd.read_csv(PRED_PATH, parse_dates=["Date"]).sort_values("Date").set_index("Date")

    col_base = f"pred_{BASELINE_MODEL}"
    col_comp = f"pred_{COMP_MODEL}"

    if col_base not in df.columns:
        raise KeyError(f"Missing column '{col_base}'. Available columns: {df.columns.tolist()}")
    if col_comp not in df.columns:
        raise KeyError(f"Missing column '{col_comp}'. Available columns: {df.columns.tolist()}")

    # Squared errors
    e_base = (df["y_true"] - df[col_base]) ** 2
    e_comp = (df["y_true"] - df[col_comp]) ** 2

    # Loss differential: positive => COMP is better (lower loss)
    loss_diff = (e_base - e_comp).dropna()

    # Split AFTER dropping NaNs (important)
    T = len(loss_diff)
    midpoint = T // 2

    early = loss_diff.iloc[:midpoint]
    late = loss_diff.iloc[midpoint:]

    rows = []
    for label, series in [("Early OOS (1st half)", early), ("Late OOS (2nd half)", late)]:
        stat, p, n = dm_simple(series)
        rows.append({"period": label, "n": n, "DM_stat": stat, "p_value": p})

    res = pd.DataFrame(rows)

    print("\n=== Subsample DM Results (Squared Loss): HAR vs HAR+All ===")
    print(res)

    out_path = f"outputs/tables/dm_subsample_{TARGET}.csv"
    res.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
