import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PRED_PATH = "outputs/preds/preds_vol_rv5.csv"  # use the significant target

def main():
    df = pd.read_csv(PRED_PATH, parse_dates=["Date"]).set_index("Date")

    y = df["y_true"]
    e_har = y - df["pred_HAR+All"]   # note: change if needed
    e_baseline = y - df["pred_HAR"]  # HAR baseline

    # Squared loss differential
    d = (e_baseline**2 - e_har**2)

    d = d.dropna()

    cum_d = d.cumsum()

    plt.figure(figsize=(10,6))
    plt.plot(cum_d, label="Cumulative Loss Difference (HAR - HAR+All)")
    plt.axhline(0, linestyle="--")
    plt.title("Cumulative Squared Error Difference\n(HAR vs HAR+All)")
    plt.ylabel("Cumulative Difference")
    plt.legend()
    plt.tight_layout()

    out_path = "outputs/figures/cumulative_loss_vol_rv5.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
