# RAMT — Regime-Adaptive Stock Ranking System
## Monthly Portfolio Construction for NIFTY 50

### What It Does
Ranks NIFTY 50 stocks by expected monthly
performance using a Transformer + MoE architecture,
conditioned on HMM market regime detection.

### Strategy
- Monthly rebalancing
- Top 5 stocks by RAMT ranking score
- Position sized by regime:
  Bull: 100% | High-Vol: 50% | Bear: Cash

### Results
**Current (as committed in `results/`):**

- **XGBoost (walk-forward, 5 tickers)**: avg **RMSE 0.0176**, **MAE 0.0118**, **DA% 52.30**, **Sharpe 0.46**
- **LSTM (walk-forward, 5 tickers)**: avg **RMSE 0.0221**, **MAE 0.0163**, **DA% 49.84**, **Sharpe 0.04**
- **RAMT (walk-forward, per-ticker deep model)**: `results/ramt_predictions.csv` currently contains **JPM only** (avg **RMSE 0.0178**, **MAE 0.0121**, **DA% 52.00**, **Sharpe 0.13**, **MaxDD -0.5749**, **ProfitFactor 1.03**, **Calmar 0.08**)

**Notes**

- The **XGBoost** and **LSTM** scripts are run across 5 tickers by default; the **RAMT** script in `models/ramt/train.py` is currently configured with `TICKERS = ["JPM"]`.
- A separate **monthly ranking** training script exists (`models/ramt/train_ranking.py`) for **NIFTY-50-style ranking**, but the **portfolio backtest** is currently scaffolded (see `models/backtest.py`) and not end-to-end wired yet.

---

## Quickstart (data + baselines + RAMT)

Run in this exact order:

1. `python data/download.py`  
   Verify: 50 tickers downloaded

2. `python features/feature_engineering.py`  
   Verify: `Monthly_Alpha` column exists  
   Verify: `RelMom` columns exist

3. Check one processed CSV:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/processed/TCS_NS_features.csv',
                 index_col=0)
print('Shape:', df.shape)
print('Target sample:', df['Monthly_Alpha'].dropna().head())
print('New cols:', [c for c in df.columns
                   if 'RelMom' in c or 'Macro' in c])
"
```

---

## Table of contents

1. [Project overview](#1-project-overview)  
2. [Why this project matters](#2-why-this-project-matters)  
3. [Features](#3-features)  
4. [Tech stack](#4-tech-stack)  
5. [System architecture](#5-system-architecture)  
6. [Dataset](#6-dataset)  
7. [Data preprocessing](#7-data-preprocessing)  
8. [Machine learning concepts](#8-machine-learning-concepts-used)  
9. [Deep learning concepts](#9-deep-learning-concepts-used)  
10. [Model selection reasoning](#10-model-selection-reasoning)  
11. [Training process](#11-training-process)  
12. [Evaluation & results](#12-evaluation--results)  
13. [Folder structure](#13-folder-structure)  
14. [Installation](#14-installation-guide)  
15. [Usage](#15-usage-guide)  
16. [Deployment & scaling](#16-deployment--scaling--operations)  
17. [Challenges & mitigations](#17-challenges-faced)  
18. [Future improvements](#18-future-improvements)  
19. [Resume & recruiter notes](#19-resume--recruiter-section)  
20. [Contributing & license](#20-contributing--license)  

**Additional docs**: [FEATURES_AND_REGIMES.md](FEATURES_AND_REGIMES.md) (monthly ranking target/features + HMM regime notes).

---

## 1. Project overview

| Item | Description |
|------|-------------|
| **Name** | Regime-Adaptive Multimodal Transformer (RAMT) |
| **Problem** | Forecast **next-day log returns** from multivariate daily inputs when markets are **non-stationary**: volatility clusters, correlation breaks, and regime shifts make a single static model suboptimal. |
| **Approach** | Engineer **interpretable features** (including **Gaussian HMM** regimes), evaluate with **walk-forward** splits, compare **XGBoost** and **LSTM** baselines to **RAMT** (multimodal encoder + **positional encoding** + **MoE** transformers + **regime-conditioned gating**). |
| **Users / use case** | **Quants / ML researchers**, students, and **hiring managers** reviewing a serious forecasting pipeline—not a toy dataset. Industry use cases include **research backtests**, **signal generation** (with proper risk controls), and **regime-aware model risk** analysis. |
| **Value proposition** | **Explicit regime structure**, **honest time-series evaluation** (no random train/test split on shuffled days), **reproducible scripts**, and a **deep model** whose components (encoders, gates, experts) map to clear hypotheses about market behavior. |

---

## 2. Why this project matters

### Market need

Short-horizon return prediction is **hard**; naive i.i.d. ML assumptions fail on **sequential, heavy-tailed** financial data. Practitioners need pipelines that **respect time** and **report metrics aligned with trading** (direction, risk-adjusted returns), not accuracy alone.

### Pain points addressed

| Pain point | How this repo addresses it |
|------------|----------------------------|
| **Look-ahead bias** | **Walk-forward** folds; **StandardScaler fit on training rows only** per fold (RAMT/LSTM-style pipeline in `dataset.py` / training scripts). |
| **Ignoring regimes** | **HMM_Regime** features and **regime embedding + gating** in RAMT. |
| **Only point forecasts** | **Directional accuracy** and **Sharpe-style** metrics alongside RMSE/MAE. |
| **Black-box only** | Baselines + **gate weights** from MoE for interpretability of expert mixing. |

### Why not “only traditional methods”

**Linear / ARIMA** struggle with nonlinear interactions and high-dimensional inputs. **Tree ensembles (XGBoost)** handle tabular nonlinearities well and are strong baselines. **Deep sequence models** can learn temporal patterns and **multi-head attention** can weight relevant days. **RAMT** adds **structured multimodal fusion** and **explicit expert routing**—a hypothesis-driven architecture, not a generic MLP on raw prices.

---

## 3. Features

### Core

- End-to-end **data download** (`yfinance`) and **feature engineering** (lags, vol, technicals, momentum, volume, **HMM regimes**, rolling correlation).
- **Walk-forward validation** with expanding training windows and rolling test blocks (aligned across baselines and RAMT training).
- **Metrics**: RMSE, MAE, **directional accuracy (DA%)**, **Sharpe** (simple strategy proxy), plus **Max Drawdown, Profit Factor, Calmar** in RAMT training script.

### “Smart” / AI

- **HMM** regime labels as latent **market state** features.
- **RAMT**: per-group encoders, **categorical regime embedding**, **transformer experts**, **soft gating** over experts using **context + regime**.

### Automation

- Scriptable pipeline: download → features → baselines → RAMT train → CSV outputs.
- Module-level **self-tests** (`python -m models.ramt.model`, etc.).

### User-facing

- This repository is **CLI- and notebook-oriented** with optional UIs:
  - **Streamlit dashboard** under `dashboard/app.py` (run via `./run_dashboard.sh`)
  - **FastAPI API** under `models/api.py` (lightweight endpoints for viewing metrics / features; not a production serving stack)
  Outputs are primarily **CSV results** and console logs.

### Developer / admin

- Clear **module boundaries** (`dataset`, `encoder`, `moe`, `model`, `losses`, `train`).
- Documentation: **README** + supporting notes in `FEATURES_AND_REGIMES.md`.

---

## 4. Tech stack

| Tool | Role | Why we use it | Alternatives | Why this choice here |
|------|------|---------------|--------------|----------------------|
| **Python 3** | Language | Ecosystem for ML, data, and PyTorch | R, Julia | Broad libraries, hiring signal, PyTorch-first DL |
| **NumPy / Pandas** | Arrays & time series | Fast columnar ops, alignment | Polars | Pandas ubiquitous for finance tutorials and team familiarity |
| **yfinance** | OHLCV download | Free, simple | Polygon, Refinitiv | Zero API keys for coursework/research |
| **scikit-learn** | Scaling, metrics, splits | `StandardScaler`, metric helpers | — | Industry default for preprocessing |
| **XGBoost** | Gradient boosted trees baseline | Strong tabular performance, fast | LightGBM, CatBoost | Mature, well-documented; easy to justify |
| **hmmlearn** | Gaussian HMM | Regime discovery from returns/vol | Bayesian HMM, HDP-HMM | Simple, fits pipeline scope |
| **PyTorch** | Deep learning | Dynamic graphs, research flexibility | TensorFlow/JAX | Standard for custom architectures (MoE, encoders) |
| **Matplotlib / Seaborn** | EDA plots | Publication-style plots | Plotly | Notebook-friendly |
| **SciPy / statsmodels** | Stats / diagnostics | Complements EDA | — | Optional depth in notebooks |
| **Jupyter** | Exploratory work | Interactive EDA | VS Code only | Standard for quantitative research |

**Not used in this repo (by design):**

| Category | Note |
|----------|------|
| **Backend / REST API** | No FastAPI/Flask service; batch inference via scripts. |
| **Database** | Data on disk as CSV; no PostgreSQL/Redis. |
| **Cloud SDK** | No vendor lock-in; run locally or bring your own container. |
| **Dedicated experiment tracking** | No Weights & Biases / MLflow in `requirements.txt`; easy to add. |

---

## 5. System architecture

This project is a **batch ML pipeline**, not a client-server product. There is **no frontend** in the repository.

```mermaid
flowchart LR
  subgraph sources["Data sources"]
    YF[yfinance APIs]
  end

  subgraph local["Local pipeline"]
    RAW[data/raw CSVs]
    FE[feature_engineering.py]
    PROC[data/processed CSVs]
    BL[baselines: XGBoost / LSTM]
    RAMT[models/ramt/train.py]
    RES[results/*.csv]
  end

  YF --> RAW --> FE --> PROC --> BL --> RES
  PROC --> RAMT --> RES
```

### Model training pipeline (RAMT)

1. **Load** processed features for ticker `T` → `RAMTDataModule`.  
2. **Define folds** (`get_walk_forward_indices`): expanding train, fixed step test.  
3. **Per fold:** split train into train/val; **fit scaler on train**; build `DataLoader`s.  
4. **Initialize** `RAMTModel` (fresh weights per fold in `train.py`).  
5. **Optimize** `CombinedLoss` with **AdamW**, **cosine warm restarts**, **gradient clipping**, **early stopping** on validation loss.  
6. **Predict** on test window; append to out-of-sample CSV.

### Inference pipeline

- **Inference = same forward pass** as training without gradients: `model(X, regime)` → prediction + gates. Implemented in `predict()` inside `models/ramt/train.py`. No separate GPU serving layer.

### Data flow (conceptual)

```
Raw OHLCV → engineered features + regimes → scaled sequences → encoder → +position → MoE → ŷ
                                              ↑
                                    regime labels (parallel path)
```

---

## 6. Dataset

### Source & format

- **Source:** Public market data via **`yfinance`** (`data/download.py`).  
- **Format:** Per-ticker CSV under `data/processed/` named `{TICKER}_features.csv` (e.g. `JPM_features.csv`, `RELIANCE_NS_features.csv`). **Git ignores** `data/processed/*.csv` by default (large files).

### Scale (approximate)

- **Horizon:** Configurable; default download window is roughly **2010–2026** for most tickers (see `data/download.py`; EPIGRAL uses a shorter window).  
- **Rows:** Order of **thousands** of trading days per ticker after cleaning (exact count depends on listing and NaN drop). Example: ~**3.9k** rows for JPM in development logs.  
- **RAMT input width:** **27** numeric features (`ALL_FEATURE_COLS` in `models/ramt/dataset.py`). The full engineered table has **more columns** (e.g. ~36 features in the engineering script); RAMT uses the **27-column** subset consistent across encoders.

### Features & labels

- **Features (groups):** lagged returns, realized volatility, Garman–Klass, vol ratio, RSI/MACD/Bollinger, momentum/ROC, volume ratios, **HMM_Regime**, **Rolling_Corr_Index** (see `features/feature_engineering.py` and `README` feature table below).  
- **Label:** **Next-day** log return (`Log_Return` shifted), aligned so each `(X, y)` pair is **causal**.

### Data quality & risks

| Topic | Practice in this repo |
|-------|------------------------|
| **Missing values** | Engineering uses rolling windows; early rows may drop; pipeline aligns with supervised target. Phase 2 plan notes **zero NaNs** target in processed CSVs after completion. |
| **Class imbalance** | **Regression** task; not class-balanced. XGBoost baseline may use **sample weights** on large moves (`baseline_xgboost.py`). |
| **Noise** | Returns are **high noise**; metrics emphasize **robust** measures (DA%, Sharpe proxy) alongside RMSE. |
| **Bias / survivorship** | Single-name equities; **not** a universe study—results **do not** claim market-wide generalization without further work. |
| **Corporate actions** | Yahoo-adjusted prices typical via yfinance; **verify** for production use. |

### Train / validation / test logic

- **Walk-forward:** Initial train fraction (e.g. **60%** of timeline), then **rolling test** blocks (e.g. **63** days), train expanding. **No random shuffle** of dates.  
- **Validation:** Held out from the **training** segment of each fold (e.g. last **15%** of train indices in `RAMTDataModule.get_fold_loaders`).  
- **Why these ratios:** They balance **enough history** to fit regimes/scaler with **many out-of-sample tests**—standard in finance backtesting (not arbitrary 80/10/10 i.i.d. splits).

---

## 7. Data preprocessing

| Step | What | Why | Effect |
|------|------|-----|--------|
| **Returns** | Log returns from closes | Stationarity-ish, scale-free | Stable inputs across tickers |
| **Rolling indicators** | Vol, RSI, MACD, Bollinger, etc. | Encode momentum, risk, positioning | Nonlinear market state |
| **HMM** | Gaussian HMM on selected series | Discrete **regime** proxy | Allows regime features + gating |
| **Cross-asset correlation** | Rolling corr vs benchmark | Context for single-stock moves | Extra context feature |
| **Scaling (RAMT/LSTM)** | `StandardScaler` **fit on train only** | Comparable feature scales for neural nets | **Prevents leakage** from test into normalization |
| **Sequence construction** | Last `seq_len` days → `X` | Temporal context for transformers/LSTM | Standard sequence modeling |
| **Regime in RAMT** | Integer embedding + column in `X` | Categorical vs continuous treatment | Avoids false ordinality in embedding path |

**Outliers:** No aggressive winsorization in core scripts; **financial extremes** are often informative—mitigated by **scaling**, **regularization**, and **walk-forward** testing.

---

## 8. Machine learning concepts used

**Extended definitions:** (glossary docs directory is currently not present in-tree; rely on this README + code as source of truth).

| Concept | Definition / role | Applied here |
|---------|-------------------|--------------|
| **Supervised learning** | Learn \(f(X) \approx y\) from pairs | Next-day return regression |
| **Regression** | Continuous target | RMSE, MAE |
| **Bias–variance** | Underfit vs overfit tradeoff | Walk-forward + early stopping + dropout |
| **Overfitting** | Memorizing noise | Mitigated by **val early stopping**, **regularization**, **simple baselines** |
| **Cross-validation** | Not i.i.d. k-fold | **Walk-forward** only—time order preserved |
| **Feature engineering** | Domain inputs vs raw prices | Lags, vol, technicals, HMM, correlation |
| **Hyperparameters** | Config outside training | Learning rate, heads, experts, `lambda_dir` in scripts |
| **Ensemble** | Combine models | **MoE** = weighted ensemble of **experts** (learned weights) |
| **Regularization** | Penalize complexity | Weight decay, dropout, gradient clipping |
| **Evaluation metrics** | Success criteria | RMSE/MAE for fit; **DA%** for direction; **Sharpe/Calmar** for risk-adjusted narrative |

---

## 9. Deep learning concepts used

**Extended definitions:** (glossary docs directory is currently not present in-tree; rely on this README + code as source of truth).

| Concept | Theory (brief) | In this codebase |
|---------|----------------|------------------|
| **Neural network** | Composed linear + nonlinear layers | Encoders, experts, gates |
| **Layers** | Linear, LayerNorm, Dropout, etc. | `encoder.py`, `moe.py`, `model.py` |
| **Activations** | ReLU, softmax | ReLU in MLPs; **softmax** on expert gates |
| **Forward pass** | Compute ŷ from X | `RAMTModel.forward` |
| **Backpropagation** | Chain rule for gradients | `loss.backward()` in training |
| **Optimizers** | AdamW (default in RAMT train) | Weight decay for regularization |
| **Loss** | MSE + directional penalty | `CombinedLoss` |
| **Transformer** | Self-attention over sequence | `ExpertTransformer` uses `nn.TransformerEncoder` |
| **LSTM** | Recurrent baseline | `baseline_lstm.py` |
| **Dropout** | Random unit drops | Encoders, MoE, positional dropout |
| **Layer normalization** | Stabilize activations | Throughout encoder/MoE |
| **Embeddings** | Discrete → vector | **Regime** embedding; **positional** embedding |
| **Attention** | Weighted aggregation over time | Transformer encoder in each expert |
| **MoE** | Multiple experts + router | `MixtureOfExperts` + `GatingNetwork` |

**Transfer learning** is **planned** as cross-market transfer in project docs; the current training script is **per-ticker** walk-forward (verify `train.py` for any multi-ticker joint training).

---

## 10. Model selection reasoning

| Model | Status | Rationale |
|-------|--------|-----------|
| **XGBoost** | Implemented | Strong **tabular** baseline, fast, interpretable feature importance. |
| **LSTM** | Implemented | Classic **sequence** baseline without attention/MoE complexity. |
| **RAMT** | Implemented | Tests hypothesis: **structured multimodal fusion** + **regime routing** helps vs single trunk. |
| **Pure ARIMA / linear** | Not emphasized | Baseline gap for nonlinear multivariate setup. |
| **Single transformer** | Subsumed | Experts **are** transformers; MoE adds **capacity** without one giant model. |

**Tradeoffs:** RAMT has **more parameters** and **longer train** per fold than XGBoost; **interpretability** comes from **gates** and **regimes** vs tree **feature importance**. Speed vs accuracy is **tunable** (embed dim, layers, experts).

---

## 11. Training process (RAMT)

Configured in `models/ramt/train.py` (subject to change in code):

| Item | Typical value | Purpose |
|------|----------------|--------|
| **Epochs** | Up to 50 | Enough to converge with early stopping |
| **Batch size** | 32 | Standard GPU/CPU balance |
| **Learning rate** | 1e-3 | AdamW default scale |
| **Weight decay** | 1e-4 | L2 regularization |
| **Scheduler** | CosineAnnealingWarmRestarts | Escape local minima |
| **Early stopping** | Patience 10 on **val loss** | Reduce overfit |
| **Gradient clip** | 1.0 | Stability with transformers |
| **Hardware** | CPU or CUDA if available | `DEVICE = cuda if available` |
| **Checkpoints** | Directory `checkpoints/` created; **best weights per fold** held in memory | Full walk-forward retrain each fold by default |
| **Experiment tracking** | Console + CSV | Add W&B/MLflow if needed |

**Training time** depends on CPU/GPU, number of folds, and ticker length—not fixed in README.

---

## 12. Evaluation & results

### Metrics

| Metric | Meaning | Why it matters |
|--------|---------|----------------|
| **RMSE / MAE** | Error magnitude | Standard regression fit |
| **DA%** | % correct **sign** | Directional trading relevance |
| **Sharpe** (proxy) | Risk-adjusted return of simple strategy using **predictions as sizing** | Sanity check beyond raw error |
| **MaxDD / Calmar / Profit Factor** | Risk and payoff asymmetry | From `compute_metrics` in `train.py` |

**Confusion matrix:** Not primary for **regression**; direction could be turned into binary classification for extra analysis (not automated in `evaluate.py`).

### Published baseline snapshot (XGBoost, walk-forward)

Illustrative numbers from the project README history (re-run after data refresh):

| Ticker | RMSE | MAE | DA% | Sharpe |
|--------|------|-----|-----|--------|
| JPM | 0.0194 | 0.0127 | 52.13 | 0.52 |
| RELIANCE.NS | 0.0180 | 0.0121 | 52.25 | 0.54 |
| TCS.NS | 0.0151 | 0.0106 | 53.44 | 0.82 |
| HDFCBANK.NS | 0.0165 | 0.0111 | 51.52 | 0.04 |
| EPIGRAL.NS | 0.0243 | 0.0166 | 51.32 | -0.56 |
| **Average** | **0.0187** | **0.0126** | **52.13** | **0.27** |

**RAMT** results are written to `results/ramt_predictions.csv` when running `models/ramt/train.py`; compare against baselines on the **same walk-forward philosophy**.

**Interpretation:** ~**52% directional accuracy** is only slightly above coin flip—**realistic** for daily returns. The project demonstrates **rigorous methodology** more than guaranteed alpha.

---

## 13. Folder structure

```
regime-adaptive-transformer/
├── data/
│   ├── download.py           # Download OHLCV + benchmarks → data/raw/
│   ├── raw/                  # Raw CSVs (gitignored)
│   └── processed/            # Engineered features per ticker (gitignored)
├── features/
│   └── feature_engineering.py  # HMM, technicals, correlation → processed/
├── eda/
│   ├── eda.ipynb             # Exploration
│   └── plots/                # Figures / summary exports
├── models/
│   ├── baseline_xgboost.py   # Walk-forward XGBoost
│   ├── baseline_lstm.py      # Walk-forward LSTM
│   └── ramt/
│       ├── dataset.py        # Column defs, RAMTDataModule, sequences
│       ├── encoder.py        # Multimodal + regime encoders
│       ├── moe.py            # Positional encoding, experts, gating, MoE
│       ├── model.py          # Full RAMTModel
│       ├── losses.py         # MSE + directional
│       └── train_ranking.py  # Monthly ranking training + CSV export
├── FEATURES_AND_REGIMES.md   # Monthly target/features + HMM regime notes
├── results/                  # Predictions & metric tables (gitignored CSVs)
├── checkpoints/              # Saved weights (optional; .gitkeep pattern in .gitignore)
├── evaluate.py               # Summarize baseline predictions CSV
├── requirements.txt
├── README.md                 # This file (project entry point)
├── docs/                     # (currently empty / optional)
└── Phase1_report.tex / .pdf  # Academic reporting artifacts
```

---

## 14. Installation guide

### Prerequisites

- **Python 3.10+** recommended (project uses modern typing; venv may use 3.14 per environment).  
- **Git**  
- Optional: **CUDA** for faster PyTorch training

### Steps

After cloning this repository from your Git hosting provider:

```bash
cd regime-adaptive-transformer

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Data

```bash
python data/download.py
python features/feature_engineering.py
```

Processed CSVs appear under `data/processed/`. **No `.env` file is required** for the default Yahoo pipeline.

---

## 15. Usage guide

### Baselines

```bash
# XGBoost walk-forward; writes predictions under results/
python models/baseline_xgboost.py

python evaluate.py   # aggregates results/xgboost_predictions.csv if present
```

```bash
python models/baseline_lstm.py
```

### RAMT (full walk-forward)

```bash
python -m models.ramt.train
# or
python models/ramt/train.py
```

Outputs: **`results/ramt_predictions.csv`**, console metric table.

### Monthly ranking (NIFTY-50-style, combined training)

```bash
python -m models.ramt.train_ranking
```

Outputs: **`results/ranking_predictions.csv`** (if the processed CSVs contain `Monthly_Alpha` and are present for many tickers).

### Dashboard (optional)

There is a Streamlit dashboard under `dashboard/app.py` that visualizes predictions and regimes. It currently expects:

- `data/processed/{TICKER}_features.csv`
- one or more of: `results/xgboost_predictions.csv`, `results/lstm_predictions.csv`, `results/ramt_predictions.csv`
- optional: `results/monthly_rankings.csv` for the monthly portfolio view (may not exist yet)

Run:

```bash
./run_dashboard.sh
```

### Module tests (sanity checks)

```bash
python -m models.ramt.dataset
python -m models.ramt.encoder
python -m models.ramt.moe
python -m models.ramt.model
```

### API / UI

- **None** in-repo. Batch inference is via **Python scripts** and **saved CSVs**.

### Retraining on new data

1. Refresh raw data (`download.py`).  
2. Rebuild features (`feature_engineering.py`).  
3. Re-run baselines and/or `train.py`.

---

## 16. Deployment, scaling & operations

| Aspect | Current state | Production-oriented path |
|--------|----------------|---------------------------|
| **Deployment** | Research scripts | Export **TorchScript** or **ONNX**; wrap in **FastAPI** for internal batch scoring |
| **Containerization** | Not in-repo | Add `Dockerfile` (Python + CUDA optional), pin `requirements.txt` |
| **Cloud** | Local | S3 for data, SageMaker/Vertex for training |
| **CI/CD** | Not configured | GitHub Actions: lint, unit tests, smoke `python -m models.ramt.model` |
| **Scaling inference** | N/A | **Batch** job per universe; **streaming** needs online feature pipeline |
| **Monitoring** | Manual CSV | Track **prediction drift**, **regime distribution**, **Sharpe** in production |

---

## 17. Challenges faced

| Challenge | Mitigation |
|-----------|------------|
| **Non-stationarity** | Walk-forward, regime features, regularization |
| **Leakage** | Scaler fit on train only per fold |
| **Volatile small caps** | EPIGRAL shows weaker metrics—documented honestly |
| **Transformer stability** | Pre-norm encoder, grad clip, dropout |
| **Class imbalance (regression)** | Directional loss + finance metrics |

---

## 18. Future improvements

- **Unified experiment tracking** (W&B, MLflow).  
- **Hyperparameter search** (Optuna) with walk-forward CV.  
- **Attention visualization** and **gate analysis** dashboards.  
- **Cross-market transfer** training (multi-ticker joint or pretrain/finetune).  
- **Production API** + **Docker** + **scheduled retraining**.  
- **Explainability** (SHAP on baselines; gate attribution for RAMT).  
- **License file** and **contributing guidelines** for open-source release.

---

## 19. Resume / recruiter section

**Why this project stands out**

- Demonstrates **end-to-end ML engineering**: data acquisition, **feature engineering with HMM**, **time-series validation that is not naive k-fold**, **multiple model families**, and **deep learning architecture design** (multimodal encoders, transformers, MoE).  
- Shows awareness of **finance-specific pitfalls** (leakage, direction vs magnitude, Sharpe).  
- **Code organization** matches how research teams structure repos (modules, scripts, docs).

**Skills demonstrated**

- Python, PyTorch, scikit-learn, XGBoost, pandas  
- **Time-series ML**, **walk-forward validation**, **metrics** beyond accuracy  
- **Transformer architecture**, **MoE**, **embeddings**, **regularization**  
- Technical writing (**README**, workflow, architecture docs)

**Business relevance**

- Quantitative finance and **model risk** contexts value **transparent methodology** and **honest metrics** over overclaimed accuracy.

---

## 20. Contributing & license

### Contributing

1. Fork the repository.  
2. Create a branch for your change.  
3. Keep commits focused; match existing style.  
4. Run module self-tests after changes to `models/ramt/*`.  
5. Open a pull request with a clear description of motivation and validation.

### License

No `LICENSE` file is present in the repository as of this writing. Before public distribution or reuse, **add an explicit open-source license** (e.g. **MIT** for permissive academic use) and ensure **data terms** from Yahoo/vendors are respected.

---

## Team & institution

| Name | Role |
|------|------|
| Vivek (230119) | Literature review, LaTeX report, theory |
| Shivansh Gupta (230054) | Data pipeline, feature engineering, models |

**Institution:** B.Tech Computer Science and Artificial Intelligence, Rishihood University, Sonipat, India.

---

## Feature groups (reference)

| Group | Examples | Purpose |
|-------|----------|---------|
| Lagged returns | Return_Lag_1 … 20 | Price memory |
| Volatility | Realized vol, Garman–Klass, vol ratio | Risk regime |
| Technical | RSI, MACD, Bollinger | Short-horizon positioning |
| Momentum | Momentum_5/20/60, ROC_10 | Trend |
| Volume | Volume_MA_Ratio, Volume_Log | Participation |
| HMM | HMM_Regime | Latent state |
| Cross-asset | Rolling_Corr_Index | Benchmark context |

---

*This README describes the repository as implemented; scripts and hyperparameters may evolve—prefer reading source files for authoritative behavior.*

**Last updated:** 2026-04-15
