# Regime-Adaptive Transformer (RAMT)

Forecast short-horizon **next-period return direction** from OHLCV sequences. A small transformer encoder learns temporal structure; a **mixture-of-experts (MoE)** head adapts predictions to latent “regimes” (soft routing, not explicit labels).

## Problem

- **Input:** Rolling windows of price-derived features (returns, volatility proxies, momentum).
- **Target:** Binary label: next period log-return \(>\) 0 (after optional neutral band).
- **Why MoE:** Different market conditions favor different linear/nonlinear patterns; gating lets the model specialize subnetworks without hand-labeled regimes.

## Approach

1. **Data:** `yfinance` daily bars → CSV under `data/raw/`.
2. **Features:** Z-scored sequence features per window (`features/feature_engineering.py`).
3. **Models:**
   - **XGBoost** on the last timestep (tabular baseline).
   - **LSTM** sequence baseline.
   - **RAMT:** transformer encoder + MoE classifier (`models/ramt/`).

## Results

Out-of-sample metrics depend on tickers, dates, and seeds. After training, run `evaluate.py` for accuracy, F1, and ROC-AUC. Typical next steps: walk-forward splits, transaction costs, and calibration.

## Setup

```bash
cd regime-adaptive-transformer
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Fetch data

RAMT default tickers (JPM, Indian ADR-style NSE names) with diagnostics and `*_raw.csv` for EDA:

```bash
python data/download.py
```

### Train

Training script in progress — baseline models available in `models/baseline_xgboost.py` and `models/baseline_lstm.py`.

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt
# After XGBoost training:
python evaluate.py --checkpoint checkpoints/xgboost.joblib
```

### EDA

```bash
jupyter notebook eda/eda.ipynb
```

## Layout

| Path | Role |
|------|------|
| `data/download.py` | Yahoo Finance: `--ramt` or generic `--tickers` |
| `data/raw/` | Downloaded CSV (gitignored except structure) |
| `data/processed/` | Optional derived tables / parquet |
| `eda/eda_notebook.ipynb` | Exploratory plots on RAMT `*_raw.csv` |
| `features/feature_engineering.py` | Windows, scaling, labels |
| `models/baseline_*.py` | XGBoost / LSTM |
| `models/ramt/` | Encoder, MoE, full RAMT |
| `train.py` / `evaluate.py` | CLI training and metrics |
| `checkpoints/` | Saved models (artifacts gitignored) |
