# WTI Crude Oil Volatility Forecasting — Explainable ML Framework

Replication code for:

> **Explainable Machine Learning for WTI Crude Oil Realized Volatility Forecasting Using HAR Models, NLP Sentiment, and Uncertainty Indices**  
> S. Aljaloud, W. Alghassab  
> *Applied Sciences* (MDPI) — submitted 2026

---

## Overview

End-to-end pipeline that fuses NLP-based news sentiment (GDELT/VADER/FinBERT), macroeconomic uncertainty indices (EPU, GPR), and market indicators (VIX, OVX) into an augmented HAR baseline. Five model families are benchmarked under rolling- and expanding-window real-time OOS protocols and SHAP is used for transparent feature attribution.

**Data sources:** WTI daily prices (EIA), OVX/VIX (CBOE), EPU (policyuncertainty.com), GPR (matteoiacoviello.com), GDELT GKG  
**Sample period:** 2013-04-01 → 2026-02-28 (N = 1,848 trading days)  
**OOS window:** n = 424 steps (June 2023 – February 2026)  
**Primary target:** `vol_rv5` (5-day realized volatility = sqrt(Σr²), i=0..4)  
**Secondary target:** `vol_abs` (absolute daily return)

**Why 2013-04-01?** GDELT Global Knowledge Graph (GKG) starts April 1, 2013.

---

## Repository Structure

```
notebooks/
  01_pull_fred.py                 – pull WTI prices from FRED/EIA
  02_build_targets.py             – construct vol_rv5 and vol_abs targets
  03_merge_epu_gpr.py             – merge EPU and GPR indices
  04_gdelt_daily_sentiment.py     – GDELT query + VADER scoring pipeline
  04b_news_daily_sentiment_vader.py – alternative VADER aggregation
  04c_gdelt_titles_raw.py         – raw title extraction
  05_build_master_dataset.py      – merge all inputs → master_daily.csv
  06_oos_setup_and_baselines.py   – HAR baseline rolling-window OOS
  06b/06c_oos_setup_and_baselines.py – variant baseline setups
  07_dm_tests.py                  – Diebold-Mariano tests (squared + absolute loss)
  08_ml_xgb_rf_shap.py            – rolling OOS for XGBoost, RF, HAR+All (OLS);
                                    SHAP bar + beeswarm for vol_rv5 (final window snapshot)
  08b_shap_vol_abs.py             – SHAP figures for vol_abs (standalone, final window)
  09_cumulative_loss_plot.py      – cumulative squared-loss differential plots (rolling)
  09_subsample_dm_tests.py        – subsample DM tests (Early/Late, Pre/Post-Election)
  10_dm_mae.py                    – DM tests under absolute loss
  11_expanding_window_oos.py      – expanding-window OOS + DM + cumulative loss plot
  12_merge_finbert_into_master.py – merge FinBERT scores into master dataset
  13_finbert_corr.py              – VADER vs FinBERT daily correlation
  14_compute_finbert_table9.py    – FinBERT OOS accuracy table
  finbert_daily_sentiment.py      – FinBERT batch scoring script (GPU/CPU)
  lstm_template.py                – LSTM rolling-window OOS
  master_daily.py                 – dataset assembly utility

data_final/
  master_daily.csv                – merged daily dataset (not tracked; regenerate via scripts)

outputs/
  figures/                        – all paper figures (PNG, 300 dpi)
  tables/                         – OOS metrics, SHAP importance CSVs
  preds/                          – rolling prediction series
```

---

## Key Results

| Model | MAE | RMSE | ΔMAE vs HAR | ΔRMSE vs HAR |
|---|---|---|---|---|
| HAR (baseline) | 0.006407 | 0.008439 | — | — |
| HAR+All | 0.006077 | 0.007753 | +5.15% | +8.13% |
| Lasso(All) | 0.005981 | 0.007756 | +6.65% | +8.09% |
| **XGBoost** | **0.005882** | 0.008044 | **+8.19%** | +4.68% |
| RF | 0.007041 | 0.008989 | −9.89% | −6.52% |
| LSTM | 0.007036 | 0.009655 | −9.82% | −14.41% |

DM test (HAR+All vs HAR): p = 0.012** (rolling, squared loss), p = 0.007*** (expanding)  
DM test (HAR+News vs HAR): p = 0.018** (squared), p < 0.001*** (absolute)

VADER vs FinBERT: near-identical ΔRMSE (7.60% vs 7.47%) at ~210× lower compute cost.

---

## Reproducing the Paper

```bash
# 1. Build dataset
python notebooks/01_pull_fred.py
python notebooks/02_build_targets.py
python notebooks/03_merge_epu_gpr.py
python notebooks/04_gdelt_daily_sentiment.py
python notebooks/05_build_master_dataset.py

# 2. Baseline OOS + DM tests
python notebooks/06_oos_setup_and_baselines.py
python notebooks/07_dm_tests.py
python notebooks/09_subsample_dm_tests.py
python notebooks/10_dm_mae.py
python notebooks/11_expanding_window_oos.py

# 3. ML benchmarks + SHAP figures
python notebooks/08_ml_xgb_rf_shap.py       # ~10 min on 12-core CPU
python notebooks/08b_shap_vol_abs.py         # ~3 min

# 4. Cumulative loss figures
python notebooks/09_cumulative_loss_plot.py

# 5. FinBERT robustness (optional; requires GPU or ~3.5 hr CPU)
python notebooks/finbert_daily_sentiment.py
python notebooks/12_merge_finbert_into_master.py
python notebooks/14_compute_finbert_table9.py
```

---

## Requirements

```
pandas numpy scikit-learn xgboost shap matplotlib
torch torchvision transformers  # for FinBERT only
statsmodels arch                # for DM tests + stationarity
```

---

## Notes

- SHAP values are computed on a 800-observation random sample from the **final rolling training window** (last 1,260 trading days) as a representative snapshot. This is consistent with the paper's methodology disclosure.
- All rolling-window experiments use a 5-year (1,260 trading-day) training window with 1-step-ahead forecasts and strict real-time (no look-ahead) feature construction.
- EPU/GPR are monthly series mapped to daily frequency (value assigned to all trading days within the calendar month).
