# Scalable Explainable Machine Learning for WTI Crude Oil Volatility Forecasting via NLP–HAR Fusion
Replication Code

This repository contains the replication code for the empirical analysis in the paper:

**"Scalable Explainable Machine Learning for WTI Crude Oil Volatility Forecasting via NLP–HAR Fusion"**

by S. Aljaloud and W. Alghassab — IEEE Access.

All data are retrieved from public sources (EIA, CBOE, GDELT, policyuncertainty.com, matteoiacoviello.com). Running the scripts in order reproduces the cleaned datasets, output tables, and figures reported in the paper.

## Repository Structure

```
wti-volatility-har-xai/
├── data_raw/           # raw downloads (git-ignored)
├── data_interim/       # intermediate processed files
├── data_final/         # final modelling dataset
├── notebooks/          # replication scripts (numbered pipeline)
├── outputs/
│   ├── figures/        # generated figures
│   ├── tables/         # generated tables
│   └── preds/          # forecast files
├── src/                # shared utilities
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproduction

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Pull raw data
```
python notebooks/01_pull_fred.py
python notebooks/02_build_targets.py
python notebooks/03_merge_epu_gpr.py
python notebooks/04c_gdelt_titles_raw.py
python notebooks/04_gdelt_daily_sentiment.py
python notebooks/04b_news_daily_sentiment_vader.py
python notebooks/finbert_daily_sentiment.py
python notebooks/12_merge_finbert_into_master_fixed.py
```

### 3. Build master dataset
```
python notebooks/05_build_master_dataset.py
python notebooks/master_daily.py
```

### 4. Run analyses
```
python notebooks/06b_oos_setup_and_baselines.py
python notebooks/lstm_template.py
python notebooks/08_ml_xgb_rf_shap.py
python notebooks/07_dm_tests.py
python notebooks/09_cumulative_loss_plot.py
python notebooks/09_subsample_dm_tests.py
python notebooks/10_dm_mae.py
python notebooks/11_expanding_window_oos.py
python notebooks/13_finbert_corr.py
python notebooks/14_compute_finbert_table9.py
```

All outputs (CSV tables, PNG figures, forecasts) are saved to the `outputs/` directory.

## Analysis Code (`notebooks/`)

### Data Collection and Processing
- `01_pull_fred.py` — retrieves WTI price and VIX/OVX data from FRED/EIA
- `02_build_targets.py` — constructs realized volatility targets (vol_rv5 and alternatives)
- `03_merge_epu_gpr.py` — merges EPU and GPR indices into the panel
- `04c_gdelt_titles_raw.py` — fetches raw GDELT news title data
- `04_gdelt_daily_sentiment.py` — computes VADER daily sentiment from GDELT titles
- `04b_news_daily_sentiment_vader.py` — alternative VADER sentiment pipeline
- `finbert_daily_sentiment.py` — computes FinBERT daily sentiment scores
- `12_merge_finbert_into_master_fixed.py` — merges FinBERT sentiment into master panel
- `05_build_master_dataset.py` — assembles the full master dataset
- `master_daily.py` — finalises aligned daily master panel

### HAR Baselines and OOS Evaluation
- `06b_oos_setup_and_baselines.py` — rolling and expanding OOS evaluation for HAR baseline and augmented models

### Machine Learning and Explainability
- `08_ml_xgb_rf_shap.py` — XGBoost and Random Forest benchmarks with SHAP analysis
- `lstm_template.py` — LSTM deep learning benchmark

### Forecast Comparison and Robustness
- `07_dm_tests.py` — Diebold–Mariano forecast comparison tests (MSE)
- `09_cumulative_loss_plot.py` — cumulative loss differential plots
- `09_subsample_dm_tests.py` — subsample DM tests for stability
- `10_dm_mae.py` — DM tests under MAE loss
- `11_expanding_window_oos.py` — expanding-window OOS evaluation with rolling re-estimation

### NLP Sentiment Analysis
- `13_finbert_corr.py` — correlation analysis between VADER and FinBERT sentiment
- `14_compute_finbert_table9.py` — FinBERT model comparison table

## Requirements

Install with:
```
pip install -r requirements.txt
```

## Contact

For questions: s.aljaloud@uoh.edu.sa
