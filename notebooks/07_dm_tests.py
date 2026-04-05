import numpy as np
import pandas as pd
from scipy import stats

PRED_PATH_ABS = "outputs/preds/preds_vol_abs.csv"
PRED_PATH_RV5 = "outputs/preds/preds_vol_rv5.csv"


def diebold_mariano(e1, e2, h=1):
    """
    Diebold-Mariano test (two-sided).
    e1, e2 = forecast errors (y - yhat)
    h = forecast horizon (1 here)
    """

    d = e1**2 - e2**2  # loss differential
    d = d.dropna()

    T = len(d)
    mean_d = np.mean(d)

    # HAC variance estimate (Newey-West with lag = h-1)
    gamma = []
    for lag in range(h):
        cov = np.cov(d[lag:], d[:T-lag], bias=True)[0, 1]
        gamma.append(cov)

    var_d = gamma[0] + 2 * sum(gamma[1:])
    dm_stat = mean_d / np.sqrt(var_d / T)

    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


def run_dm_tests(pred_path, target_name):
    df = pd.read_csv(pred_path, parse_dates=["Date"]).set_index("Date")

    y = df["y_true"]

    results = []

    comparisons = [
        ("HAR", "HAR+News"),
        ("HAR", "HAR+All"),
        ("HAR+Market", "HAR+All"),
    ]

    for m1, m2 in comparisons:
        e1 = y - df[f"pred_{m1}"]
        e2 = y - df[f"pred_{m2}"]

        stat, p = diebold_mariano(e1, e2, h=1)

        results.append({
            "comparison": f"{m1} vs {m2}",
            "DM_stat": stat,
            "p_value": p
        })

    res = pd.DataFrame(results)

    print(f"\n=== Diebold–Mariano Tests ({target_name}) ===")
    print(res)

    return res


def main():
    res_abs = run_dm_tests(PRED_PATH_ABS, "vol_abs")
    res_rv5 = run_dm_tests(PRED_PATH_RV5, "vol_rv5")

    res_abs.to_csv("outputs/tables/dm_tests_vol_abs.csv", index=False)
    res_rv5.to_csv("outputs/tables/dm_tests_vol_rv5.csv", index=False)

    print("\nSaved DM test tables.")


if __name__ == "__main__":
    main()
