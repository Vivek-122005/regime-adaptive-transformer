## RAMT Monthly Ranking — Features & Regime Model

This note documents **exactly what we feed to the RAMT model** (inputs), **what we predict** (target), and **how the HMM regime model works** (and how it is used for portfolio sizing).

**Current status (2026-04-15)**:

- The **monthly ranking training** path exists in `models/ramt/train_ranking.py` and writes `results/ramt/ranking_predictions.csv` when you have many processed tickers with `Monthly_Alpha` (standalone `__main__`); the final runner writes strategy CSVs under `results/final_strategy/`.
- The **portfolio backtest** in `models/backtest.py` is currently a **scaffold** (not wired end-to-end yet).
- Separately, the repo also contains a **daily next-day return** path (XGBoost/LSTM/RAMT) that writes `results/*_predictions.csv`. That daily target is **not** `Monthly_Alpha`.

Source of truth in code:
- Feature engineering: `features/feature_engineering.py`
- Model feature list (what gets loaded as inputs): `models/ramt/dataset.py`
- Regime usage in strategy/backtest: `models/backtest.py`

---

## 1) What is the target we predict?

We train a regression model to predict:

- **`Monthly_Alpha`** = (stock **forward 21 trading days** log return) − (NIFTY **forward 21 trading days** log return)

In code (`features/feature_engineering.py` → `compute_monthly_target`):
- Stock forward return is computed as: rolling sum of `Log_Return` over 21 days, then shifted by `-21`
- NIFTY forward return is computed the same way on NIFTY `Log_Return`
- `Monthly_Alpha = stock_fwd_21d − nifty_fwd_21d`

Interpretation:
- **Positive `Monthly_Alpha`**: stock expected to beat NIFTY over the next “month” (≈21 trading days)
- **Negative `Monthly_Alpha`**: stock expected to lag NIFTY

There is also a derived label:
- **`Beat_NIFTY`** = 1 if `Monthly_Alpha > 0` else 0  
This is **not** used as the training target in the current pipeline (it’s saved for analysis).

### Risk-adjusted target (new)

We also compute an optional, more “production friendly” target:

- **`Monthly_Alpha_Z`** = `Monthly_Alpha / (trailing 21d volatility)`

This penalizes very volatile “junk” outperformers and tends to prefer smoother outperformance.

---

## 2) What features are fed into the model?

The model input matrix for each sample is built from **exactly 42 columns** defined in:
- `models/ramt/dataset.py` → `ALL_FEATURE_COLS`

Each training example uses a **sequence** of these features of length `seq_len` (e.g. 30 days), and predicts the scalar target `Monthly_Alpha` aligned to the sample date.

### 2.1 The 42 input features (exact list, grouped)

#### A) Lagged returns (6)
- `Return_Lag_1`
- `Return_Lag_2`
- `Return_Lag_3`
- `Return_Lag_5`
- `Return_Lag_10`
- `Return_Lag_20`

#### B) Volatility / range (5)
- `Realized_Vol_5`
- `Realized_Vol_20`
- `Realized_Vol_60`
- `Garman_Klass_Vol`
- `Vol_Ratio` (=`Realized_Vol_5 / Realized_Vol_20`)

#### C) Technical indicators (8)
- `RSI_14`
- `MACD`
- `MACD_Signal`
- `MACD_Hist`
- `BB_Upper`
- `BB_Lower`
- `BB_Width`
- `BB_Position`

#### D) Momentum & reversal (4)
- `Momentum_5`
- `Momentum_20`
- `Momentum_60`
- `ROC_10`

#### E) Volume (2)
- `Volume_MA_Ratio`
- `Volume_Log`

#### F) Regime code used for conditioning (1)
- `HMM_Regime`

#### G) Cross-asset linkage (1)
- `Rolling_Corr_Index` (60d rolling correlation of stock returns vs an index return series)

#### H) Relative momentum vs NIFTY (7)
- `RelMom_5d`
- `RelMom_21d`
- `RelMom_63d`
- `RelMom_126d`
- `RelMom_252d`
- `Mom_12_1` (12-month minus last 1-month momentum, “skip last month”)
- `RelMom_12_1` (relative 12–1 momentum vs NIFTY)

#### I) Macro features (8)
Computed as 1d and 5d log-return sums for each macro series:
- `Macro_USDINR_Ret1d`, `Macro_USDINR_Ret5d`
- `Macro_CRUDE_Ret1d`, `Macro_CRUDE_Ret5d`
- `Macro_GOLD_Ret1d`, `Macro_GOLD_Ret5d`
- `Macro_USVIX_Ret1d`, `Macro_USVIX_Ret5d`

### 2.2 Notes on scaling / leakage

Feature engineering creates raw numeric columns.

Scaling happens at training time:
- **`RobustScaler`** (IQR-based) is **fit on training rows only** for features and the monthly alpha label. For the combined NIFTY200 trainer, `transform` runs inside `LazyMultiTickerSequenceDataset.__getitem__` so val/test/inference never reuse training-only statistics implicitly. Single-ticker loaders still use `RAMTDataModule.get_fold_loaders` in `models/ramt/dataset.py`.

---

## 3) How the HMM regime model works (the “old regime model”)

Regime labeling is computed during feature engineering (`features/feature_engineering.py` → `add_hmm_regimes`).

### 3.1 What the HMM is trained on

For each ticker (and separately for NIFTY), we fit a **3-state Gaussian HMM** using `hmmlearn` on **two daily inputs**:
- \(x_1\) = `Log_Return`
- \(x_2\) = `Realized_Vol_20`

Before fitting, these two columns are standardized (z-scored) using the mean and std computed on the available (non-NaN) rows for that ticker:
- \(X_{scaled} = (X - \mu) / \sigma\)

### 3.2 What the HMM outputs

The HMM produces a **raw state** per day: `0`, `1`, or `2`.
Raw state IDs have no inherent meaning.

### 3.3 How raw states become semantic regimes

We compute the **mean log return** inside each raw state and remap:
- Highest mean return state → **Bull** (semantic code `1`)
- Lowest mean return state → **Bear** (semantic code `2`)
- Middle mean return state → **High-vol** (semantic code `0`)

Saved columns:
- `HMM_Regime` (0/1/2)
- `HMM_Regime_Label` (`high_vol` / `bull` / `bear`)

### 3.4 How regime is used in the portfolio strategy

In `models/backtest.py` (especially `run_backtest_daily`), on each rebalance date we read **NIFTY’s** `HMM_Regime` and apply sizing rules:

- **Bull (`1`)**: invest 100%, pick **top 5** stocks
- **High-vol (`0`)**: invest 50%, pick **top 3** stocks
- **Bear (`2`)**: stay in cash

The model still produces rankings (predicted `Monthly_Alpha`) every rebalance date; regime only changes **how much** we invest and **how many** positions we take.

---

## 4) Quick mental model (end-to-end)

- Feature engineering builds daily features for every stock + NIFTY + macro series.
- RAMT learns: “given the last `seq_len` days of features, predict next-month relative outperformance (`Monthly_Alpha`).”
- Each rebalance window (every ~21 trading days):
  - Use predicted `Monthly_Alpha` to **rank stocks**
  - Use NIFTY `HMM_Regime` to **size exposure / reduce risk**

