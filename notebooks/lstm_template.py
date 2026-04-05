"""
LSTM Rolling-Window OOS Forecasting Template (FIXED)
====================================================
This script runs an LSTM model under the same rolling/expanding-window OOS protocol
used for the HAR and tree-based models in the paper.

What was fixed:
- Robust sequence creation (safe when X/y lengths differ)
- Hyperparameter selection window alignment (no IndexError)
- Early-stopping best_state fallback (no NoneType crash)
- Skips invalid tiny train/val splits in grid search
- Removes look-ahead in the OOS prediction slice (no using row train_end)

Requirements:
    pip install pandas numpy scikit-learn torch
"""

import time
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data_final/master_daily.csv"
DATE_COL = "Date"
TARGET_COL = "vol_rv5"  # or "vol_abs"

FEATURE_COLS = [
    # HAR components (already in your dataset)
    "vol_lag1", "vol_week", "vol_month",
    # Market
    "VIX", "OVX", "ret",
    # Uncertainty
    "EPU", "GPR",
    # NLP sentiment
    "news_volume", "news_tone_mean", "news_tone_median"
]

# OOS split
N_OOS = 424
ROLLING_WINDOW = None  # set to int for fixed rolling window, None = expanding

# Hyperparameter grid
LOOKBACKS = [5, 10, 22]
HIDDEN_UNITS = [32, 64, 128]
DROPOUTS = [0.1, 0.2, 0.3]
LEARNING_RATES = [0.001, 0.0005]

BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
VAL_FRACTION = 0.1

# Robust minimums to avoid degenerate splits
MIN_TRAIN_SEQ = 80
MIN_VAL_SEQ = 20

# Reproducibility
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# LSTM MODEL
# ============================================================
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)     # h_n: (1, batch, hidden)
        h_n = h_n.squeeze(0)           # (batch, hidden)
        h_n = self.dropout(h_n)
        return self.fc(h_n).squeeze(-1)


# ============================================================
# HELPERS
# ============================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """
    Create (X_seq, y_target) pairs with given lookback.
    Robust to X/y length mismatch by truncating to the common length.
    """
    n = min(len(X), len(y))
    if n <= lookback:
        return np.empty((0, lookback, X.shape[1])), np.empty((0,))
    Xs, ys = [], []
    for i in range(lookback, n):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def train_lstm(X_train, y_train, X_val, y_val, hidden, dropout, lr):
    """
    Train LSTM with early stopping on validation loss.
    FIX: Always has a fallback best_state, so it never crashes.
    """
    input_dim = X_train.shape[2]
    model = LSTMForecaster(input_dim, hidden, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_val_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())  # fallback to init weights
    patience_counter = 0

    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        # Guard against NaNs/Infs
        if not np.isfinite(val_loss):
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def select_hyperparams(X_full, y_full, lookback_options, hidden_options, dropout_options, lr_options):
    """
    Grid search on the FIRST training window's internal validation set.
    FIX: Proper alignment for validation X/y so sequences are valid.
    """
    n_val = max(50, int(len(X_full) * VAL_FRACTION))

    X_tr_raw = X_full[:-n_val]
    y_tr_raw = y_full[:-n_val]

    X_vl_raw = X_full[-n_val:]
    y_vl_raw = y_full[-n_val:]

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_s = scaler.transform(X_tr_raw)
    X_vl_s = scaler.transform(X_vl_raw)

    best_score = np.inf
    best_params = None

    for L in lookback_options:
        X_tr_seq, y_tr_seq = create_sequences(X_tr_s, y_tr_raw, L)
        X_vl_seq, y_vl_seq = create_sequences(X_vl_s, y_vl_raw, L)

        # Skip degenerate cases
        if len(X_tr_seq) < MIN_TRAIN_SEQ or len(X_vl_seq) < MIN_VAL_SEQ:
            continue

        for h in hidden_options:
            for d in dropout_options:
                for lr in lr_options:
                    model = train_lstm(X_tr_seq, y_tr_seq, X_vl_seq, y_vl_seq, h, d, lr)
                    model.eval()
                    with torch.no_grad():
                        pred = model(torch.tensor(X_vl_seq, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                    mse = mean_squared_error(y_vl_seq, pred)

                    if mse < best_score:
                        best_score = mse
                        best_params = {"lookback": L, "hidden": h, "dropout": d, "lr": lr}
                        print(f"  New best: L={L}, h={h}, d={d}, lr={lr}, val_MSE={mse:.10f}")

    if best_params is None:
        raise RuntimeError(
            "Hyperparameter search found no valid configuration. "
            "Try lowering MIN_TRAIN_SEQ / MIN_VAL_SEQ or reducing LOOKBACKS."
        )

    print(f"\nSelected hyperparameters: {best_params}")
    return best_params


# ============================================================
# MAIN: OOS experiment
# ============================================================
def main():
    set_seed(SEED)

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Ensure numeric and no missing values in used columns
    use_cols = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=use_cols).reset_index(drop=True)

    X_all = df[FEATURE_COLS].astype(float).values
    y_all = df[TARGET_COL].astype(float).values

    n = len(df)
    if n <= N_OOS + 100:
        raise ValueError(f"Not enough observations after dropna: n={n}, N_OOS={N_OOS}")

    n_train_start = n - N_OOS
    print(f"Total obs: {n}, OOS steps: {N_OOS}, First training window: {n_train_start}")

    # --- Hyperparameter selection on first window ---
    print("\n--- Hyperparameter selection ---")
    X_first = X_all[:n_train_start]
    y_first = y_all[:n_train_start]

    params = select_hyperparams(X_first, y_first, LOOKBACKS, HIDDEN_UNITS, DROPOUTS, LEARNING_RATES)
    L = params["lookback"]
    H = params["hidden"]
    D = params["dropout"]
    LR = params["lr"]

    # --- Expanding/Rolling OOS forecasting ---
    print("\n--- OOS forecasting ---")
    predictions, actuals = [], []
    start_time = time.time()

    for t in range(N_OOS):
        if ROLLING_WINDOW is not None:
            train_start = max(0, n_train_start + t - ROLLING_WINDOW)
        else:
            train_start = 0

        train_end = n_train_start + t  # train uses [train_start : train_end)

        X_train_raw = X_all[train_start:train_end]
        y_train_raw = y_all[train_start:train_end]

        scaler = StandardScaler().fit(X_train_raw)
        X_train_s = scaler.transform(X_train_raw)

        X_tr_seq, y_tr_seq = create_sequences(X_train_s, y_train_raw, L)
        if len(X_tr_seq) < (MIN_TRAIN_SEQ + MIN_VAL_SEQ):
            # Not enough data for this step; skip safely
            continue

        n_val = max(MIN_VAL_SEQ, int(len(X_tr_seq) * VAL_FRACTION))
        n_val = min(n_val, len(X_tr_seq) - 10)  # ensure at least 10 train seqs
        X_tr, y_tr = X_tr_seq[:-n_val], y_tr_seq[:-n_val]
        X_vl, y_vl = X_tr_seq[-n_val:], y_tr_seq[-n_val:]

        model = train_lstm(X_tr, y_tr, X_vl, y_vl, H, D, LR)

        # ---- Predict y at index train_end using info up to train_end-1 ----
        # FIX: no look-ahead; slice excludes train_end row
        X_pred_raw = X_all[train_end - L:train_end]  # shape (L, features)
        if len(X_pred_raw) != L:
            continue
        X_pred_s = scaler.transform(X_pred_raw)
        X_pred_seq = torch.tensor(X_pred_s.reshape(1, L, -1), dtype=torch.float32).to(DEVICE)

        model.eval()
        with torch.no_grad():
            yhat = model(X_pred_seq).item()

        y_true = y_all[train_end]
        predictions.append(yhat)
        actuals.append(y_true)

        if (t + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {t+1}/{N_OOS} ({elapsed:.1f}s elapsed)")

    total_time = time.time() - start_time

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    print("\n" + "=" * 60)
    print("LSTM OOS RESULTS")
    print("=" * 60)
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print("\nSelected hyperparameters:")
    print(f"  lookback (L):    {L}")
    print(f"  hidden_units:    {H}")
    print(f"  dropout:         {D}")
    print(f"  learning_rate:   {LR}")
    print(f"  batch_size:      {BATCH_SIZE}")
    print(f"  max_epochs:      {MAX_EPOCHS}")
    print(f"  early_stopping:  {PATIENCE}")

    results_df = pd.DataFrame({"actual": actuals, "predicted": predictions})
    results_df.to_csv("lstm_oos_predictions.csv", index=False)
    print("\nPredictions saved to lstm_oos_predictions.csv")
    print("\n>>> Paste MAE/RMSE + training time + hyperparameters into the paper tables <<<")


if __name__ == "__main__":
    main()
