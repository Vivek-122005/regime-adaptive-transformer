# Repository audit (read-only)

**Generated:** 2026-04-20  
**Update:** 2026-04-20 — The `results/` directory was reorganized into phase-based folders (`final_strategy/`, `ramt/`, `hmm_ablation/`, etc.). CSV/JSON artifacts written by older runs may still embed **pre-reorganization** paths in their cells; that is intentional provenance. See `results/README_PATHS.md`.  
**Scope:** Full filesystem inventory under `/Users/shivanshgupta/regime-adaptive-transformer`. Nested copies of local Python environments (`.venv/`, `needed/`, `once/`, `only/`, `#/`) are summarized as bloat but not line-audited. No files were modified for this report.

---

## Repository overview

This repository is an undergraduate capstone codebase for **Regime-Adaptive Multimodal Transformer (RAMT)** research on **Indian equities (NIFTY 200–style universe)**. The narrative arc (documented in `README.md` and `report.tex`) is: Phase 1 daily baselines (XGBoost/LSTM), Phase 2 monthly targets, RAMT training with walk-forward checkpoints, honest backtests with friction and regime rules, a pivot to **momentum + HMM regime sizing + sector cap**, and supporting ablations (HMM vs flat sizing, parameter sensitivity, historical Yahoo windows).

**Authoritative “live” path today:** feature engineering under `features/`, portfolio simulation in `models/backtest.py`, final runner `models/run_final_2024_2026.py`, momentum-based `results/final_strategy/ranking_predictions.csv` from `scripts/build_momentum_predictions.py`, and exports in `results/final_strategy/backtest_results.csv`. **RAMT** code under `models/ramt/` and artifacts under `results/`, `models/ramt/artifacts/`, and `ramt model results/` are retained for thesis/ablation; the README states the production signal is no longer the transformer.

**Dashboard:** `dashboard/app.py` (Streamlit) visualizes `results/final_strategy/backtest_results.csv`, optional `results/ramt/`, and walk-forward baselines under `results/phase1_baselines/` — **no live inference** in the default narrative.

---

## STEP 2 — CATEGORIZE EVERY FILE

### 2.1 — Python source code (models, features, utilities)

Conventions: **Phase** = `phase1_daily` | `phase2_monthly` | `ramt` | `diagnostic` | `final_strategy` | `shared_utility` | `orphaned`. **Imported by** lists project modules that import this file (static scan of `import` / `from` lines to project packages only). **Dead code** = not referenced as a module by any other project file *and* not a documented CLI/script entry — does *not* mean unused on purpose (many scripts are runners).

| Path | Size (B) | Last modified | Description | Keep / Delete |
|------|------------|---------------|-------------|----------------|
| `dashboard/__init__.py` | 51 | 2026-04-16 | Package marker. | Keep |
| `dashboard/app.py` | 51050 | 2026-04-20 | Streamlit UI: RAMT section, production momentum strategy tabs, Phase 1/2 blocks for LSTM & XGBoost, model comparison, research notes. | Keep |
| `dashboard/market_pulse.py` | 183 | 2026-04-16 | Imports `fetch_live_macro_data_engine` from `market_scraper` for optional live macro UI. | Keep |
| `dashboard/market_scraper.py` | 15196 | 2026-04-16 | HTTP/HTML helpers to scrape or fetch macro pages; reads processed feature parquets for alignment checks. | Keep |
| `data/__init__.py` | 0 | 2026-03-22 | Empty package file. | Keep |
| `data/download.py` | 15454 | 2026-04-17 | Downloads Yahoo series into `data/raw/` parquets (benchmarks + tickers). | Keep |
| `features/__init__.py` | 0 | 2026-03-22 | Empty package file. | Keep |
| `features/feature_engineering.py` | 24600 | 2026-04-19 | Builds per-ticker feature panels: returns, technicals, macro alignment, **GaussianHMM** regime labels on NIFTY, alphas. | Keep |
| `features/sectors.py` | 6846 | 2026-04-16 | Static NSE sector mapping for diversification rules. | Keep |
| `models/__init__.py` | 16 | 2026-03-22 | Package exports placeholder. | Keep |
| `models/attention_consistency_report.py` | 3359 | 2026-04-15 | Compares attention patterns across dates/tickers using saved RAMT weights. | Keep (diagnostic) |
| `models/backtest.py` | 30694 | 2026-04-19 | **`run_backtest_daily`**: rebalance grid, friction, regime sizing, sector cap, stops flags, Kelly helpers, portfolio returns. | Keep |
| `models/baseline_lstm.py` | 18075 | 2026-04-20 | Walk-forward LSTM baseline; writes `results/phase1_baselines/lstm_predictions.csv` and metrics JSON. | Keep |
| `models/baseline_xgboost.py` | 15777 | 2026-04-20 | Walk-forward XGBoost baseline; writes `results/phase1_baselines/xgboost_predictions.csv` and metrics JSON. | Keep |
| `models/inspect_attention.py` | 5762 | 2026-04-15 | Loads RAMT checkpoint and plots attention diagnostics (Plotly). | Keep (diagnostic) |
| `models/permutation_importance.py` | 6211 | 2026-04-15 | Permutation importance on RAMT inputs. | Keep (diagnostic) |
| `models/ramt/__init__.py` | 262 | 2026-04-15 | Re-exports encoder, model, MoE classes. | Keep |
| `models/ramt/dataset.py` | 17420 | 2026-04-17 | Lazy multi-ticker sequence dataset, feature column definitions, scalers. | Keep |
| `models/ramt/encoder.py` | 5513 | 2026-04-17 | Multimodal encoder + regime cross-attention wiring. | Keep |
| `models/ramt/losses.py` | 7166 | 2026-04-19 | Tournament ranking loss and combined training objective. | Keep |
| `models/ramt/model.py` | 11304 | 2026-04-19 | `build_ramt` / `RAMTModel` — transformer stack with MoE path. | Keep |
| `models/ramt/moe.py` | 21061 | 2026-04-19 | Mixture-of-experts transformer blocks and positional encoding. | Keep |
| `models/ramt/train_ranking.py` | 52201 | 2026-04-19 | Walk-forward training, export of predictions, scalers, and training plots. | Keep |
| `models/run_final_2024_2026.py` | 8046 | 2026-04-19 | CLI: train RAMT walk-forward and/or **`--backtest-only`** using existing `ranking_predictions.csv`; writes main `results/` CSVs. | Keep |
| `scripts/baseline_feature_ic.py` | 5653 | 2026-04-19 | IC / Spearman diagnostic comparing features, LightGBM, momentum, RAMT exports. | Keep |
| `scripts/build_momentum_predictions.py` | 1601 | 2026-04-19 | Builds **`results/final_strategy/ranking_predictions.csv`** from `Ret_21d` aligned to existing RAMT dates. | Keep |
| `scripts/build_processed_range.py` | 2180 | 2026-04-19 | Invokes feature engineering for a date range / folder tag (batch builder). | Keep |
| `scripts/fetch_nifty200.py` | 16024 | 2026-04-19 | Fetches index constituents and price history; universe files under `scripts/universe/`. | Keep |
| `scripts/hmm_vs_flat_backtest.py` | 7537 | 2026-04-19 | Compares HMM-conditioned vs flat regime sizing for a predictions file; writes under `results/hmm_ablation/<phase>/` (or custom `--out-dir`). | Keep |
| `scripts/momentum_predictions_from_features.py` | 3131 | 2026-04-19 | Alternative builder for momentum-style predictions from processed features (CLI). | Keep |
| `scripts/parameter_sensitivity_backtest.py` | 6486 | 2026-04-19 | Grid over top-N, stops, sector cap, flat sizing; writes `results/final_strategy/sensitivity/`. | Keep |
| `scripts/regenerate_ramt_outputs.py` | 16507 | 2026-04-20 | Regenerates **`results/ramt/`** predictions/metrics/backtest from disk checkpoints only (no training). | Keep |
| `scripts/run_nifty500_annual_hmm_ablation.py` | 8539 | 2026-04-20 | Orchestrates annual ablation runs (subprocess driver). | Keep |
| `scripts/run_yf_hmm_ablation.py` | 4359 | 2026-04-19 | Historical Yahoo window pipeline + HMM vs flat backtests for a tag. | Keep |

**Per-file detail (phase / imports / dead-code signal)**

| Path | Phase | Imported by (project) | Dead? |
|------|--------|------------------------|-------|
| `dashboard/app.py` | shared_utility | *(entry — Streamlit)* | No |
| `dashboard/market_scraper.py` | shared_utility | `dashboard/market_pulse.py` | No |
| `dashboard/market_pulse.py` | shared_utility | *(entry via app if used)* | Possibly thin; Keep |
| `data/download.py` | shared_utility | — | CLI |
| `features/feature_engineering.py` | shared_utility | `dashboard/app.py`, `models/backtest.py`, `models/run_final_2024_2026.py` | No |
| `features/sectors.py` | shared_utility | `dashboard/app.py`, `features/feature_engineering.py`, `models/backtest.py`, `models/ramt/dataset.py`, `models/ramt/train_ranking.py` | No |
| `models/backtest.py` | final_strategy | `models/run_final_2024_2026.py`, `scripts/hmm_vs_flat_backtest.py`, `scripts/parameter_sensitivity_backtest.py`, `scripts/regenerate_ramt_outputs.py` | No |
| `models/baseline_lstm.py` | phase1_daily / baseline | — | CLI |
| `models/baseline_xgboost.py` | phase1_daily / baseline | — | CLI |
| `models/ramt/train_ranking.py` | ramt | `models/run_final_2024_2026.py`, `models/permutation_importance.py`, `scripts/regenerate_ramt_outputs.py` | No |
| `models/ramt/model.py` | ramt | Multiple RAMT tools | No |
| `models/ramt/dataset.py` | ramt | `dashboard/market_scraper.py`, RAMT stack | No |
| `models/attention_consistency_report.py` | diagnostic | — | CLI |
| `models/inspect_attention.py` | diagnostic | `models/attention_consistency_report.py` | No |
| `models/permutation_importance.py` | diagnostic | — | CLI |
| `scripts/*.py` | mixed | — | All CLIs / one-off; not “dead” |

---

### 2.2 — Trained model artifacts (`.pt`, `.joblib`, `.pkl`, `.onnx`, `.h5`, `.ckpt`)

**RAMT `.pt` files** (all ~1.1 MB each, same architecture family): require **`models/ramt/model.py`** (`build_ramt`), **`models/ramt/train_ranking.py`** (checkpoint payload format with `model_state_dict`), and matching **`joblib`** scalers. Loading pattern: `torch.load` → `payload["model_state_dict"]` (see `scripts/regenerate_ramt_outputs.py`, `models/inspect_attention.py`).

| Path | Model | Phase | Size (B) | Main vs segment | Notes |
|------|--------|--------|-----------|-----------------|--------|
| `checkpoints/best.pt` | **Unclear** (no loader in repo) | legacy | 2148415 | Unknown | **No `torch.load` reference** to this path in `.py` — see §3.1 |
| `checkpoints/xgboost.joblib` | XGBoost sklearn-style | baseline | 826821 | Legacy checkpoint | Referenced in narrative only; not loaded in audited `.py` |
| `results/ramt/ramt_model_state.pt` | RAMT | ramt | 1125029 | Aggregated / latest-style | Duplicate of artifacts below |
| `results/ramt/ramt_model_state_wf_seg_01.pt` … `07.pt` | RAMT | ramt | ~1126593 each | **Walk-forward segment** | Segments 01–07 |
| `models/ramt/artifacts/ramt_model_state*.pt` | RAMT | ramt | same | Duplicate copy of `results/` set | `ramt_metrics.json` points here |
| `ramt model results/ramt/ramt_model_state*.pt` | RAMT | ramt | same | **Third full duplicate** of checkpoints | Folder name contains a space |
| `results/ramt/ramt_scaler*.joblib`, `results/ramt/ramt_y_scaler*.joblib` | RobustScaler / y-scaler | ramt | 1e5–1e6 order | Per-segment + global | Pair with matching `.pt` segment |
| `models/ramt/artifacts/ramt_*scaler*.joblib` | same | ramt | same | Duplicate | |
| `ramt model results/ramt_*scaler*.joblib` | same | ramt | same | Duplicate | |

**No** `.pkl`, `.onnx`, `.h5`, or `.ckpt` files were found outside environment trees.

---

### 2.3 — Data files (`.csv`, `.parquet`, `.feather`, `.json`, `.jsonl`)

#### Raw data (`data/raw/*`)

| Metric | Value |
|--------|--------|
| **Parquet tickers / files** | **205** files, **14,420,863** bytes total |
| **CSV files** (`*_raw.csv` etc.) | **158** files, **49,195,981** bytes total |
| **Typical CSV header** | `Date,Open,High,Low,Close,Volume,Log_Return,Ticker` (sampled first row) |
| **Typical Parquet** | OHLCV + `Ticker` / `Adj Close` style series (per earlier pipeline; not re-read row-by-row here for all 205) |

*Individual listing omitted by design — hundreds of tickers; counts and totals are complete.*

#### Processed features (`data/processed/*` and `data/processed_yf_*`)

| Directory | File count | Total size (B) |
|-----------|--------------|----------------|
| `data/processed/` | 201 | 53,741,703 |
| `data/processed_yf_2008_2010` | 139 | 18,267,392 |
| `data/processed_yf_2010` | 283 | 14,614,845 |
| `data/processed_yf_2010_2012` | 79 | 10,723,276 |
| `data/processed_yf_2011` | 290 | 14,635,355 |
| `data/processed_yf_2012` | 297 | 15,014,164 |
| `data/processed_yf_2013_2015` | 80 | 10,867,240 |
| *(other `processed_yf_*` may exist — audit used `os.walk` on `data/`)* | | |

**Representative Parquet schema** (`results/final_strategy/ranking_predictions.csv` driver): columns sampled via dashboard code include `Date`, `Ret_21d`, `Sector_Alpha`, `Monthly_Alpha` on processed files.

#### Predictions (model inference outputs)

| Path | Model / phase | Header (first line) | Sample content |
|------|----------------|---------------------|----------------|
| `results/final_strategy/ranking_predictions.csv` | **Momentum** (current pivot) or prior RAMT export | `Date,Ticker,predicted_alpha,actual_alpha,Period` | Momentum signal as `predicted_alpha` when built via `build_momentum_predictions.py` |
| `results/final_strategy/monthly_rankings.csv` | Final runner export | `Date,Ticker,score,actual_alpha,Period,momentum` | Scores aligned to strategy |
| `results/phase1_baselines/xgboost_predictions.csv` | Phase 1 / WF XGBoost | `Date,Ticker,predicted_alpha,actual_alpha,Period` | OOS test rows from walk-forward |
| `results/phase1_baselines/lstm_predictions.csv` | Phase 1 / WF LSTM | same | same |
| `results/ramt/ranking_predictions.csv` | RAMT | `Date,Ticker,predicted_alpha,actual_alpha,Period,Segment` | Includes `Segment` e.g. `WF 1/7` |
| `results/hmm_ablation/**/backtest_*.csv` | Backtest outputs (HMM vs flat studies) | Variants: portfolio value series per script | See §2.3 metrics |
| `results/final_strategy/sensitivity/runs/backtest_*.csv` | Parameter sensitivity | Per-variant backtest windows | Linked from `parameter_sensitivity_summary.csv` |

**Missing path (dashboard):** `results/phase2_monthly/*.csv` — directory **does not exist**; Phase 2 XGBoost/LSTM **tabs** in Streamlit expect files there (see §3.5).

#### Metrics / summary files

| Path | Produced by | Notes |
|------|-------------|-------|
| `results/phase1_baselines/xgboost_metrics.json` | `models/baseline_xgboost.py` | Contains `predictions_csv` path |
| `results/phase1_baselines/lstm_metrics.json` | `models/baseline_lstm.py` | Same structure |
| `results/ramt/ramt_metrics.json` | `scripts/regenerate_ramt_outputs.py` or training | Blind-test metrics, walk-forward metadata |
| `results/final_strategy/sensitivity/parameter_sensitivity_meta.json` | `scripts/parameter_sensitivity_backtest.py` | Caveats on stop-loss implementation |
| `results/final_strategy/sensitivity/parameter_sensitivity_summary.csv` | same | Links to per-variant CSVs |
| `results/hmm_ablation/**/hmm_vs_flat_meta.json` | `scripts/hmm_vs_flat_backtest.py` / `run_yf_hmm_ablation.py` | Window metadata |
| `results/hmm_ablation/**/hmm_vs_flat_summary.csv` | same | Aggregated Sharpe/CAGR etc. |
| `data/raw_yf_*/*_fetch_stats.json` | `scripts/fetch_nifty200.py` / download helpers | Per-run fetch stats |

#### Backtest results

| Path | Role |
|------|------|
| `results/final_strategy/backtest_results.csv` | **Authoritative** production strategy (momentum + regime + sector) |
| `results/ramt/backtest_results.csv` | RAMT-only backtest for dashboard section |
| `results/final_strategy/sensitivity/runs/backtest_*.csv` | Sensitivity grid |
| `results/hmm_ablation/**/backtest_*.csv` | Regime ablations |

**Verified `results/final_strategy/backtest_results.csv`:** columns include `date`, `portfolio_return`, `portfolio_value`, `regime`, `stocks_held`, friction columns — first rows dated **2024-01-10** onward.

#### Configuration files (data / universe)

| Path | Purpose |
|------|---------|
| `data/nifty200_tickers.txt` | Static universe list |
| `scripts/universe/nifty100_nse_survivorship_proxy.txt` | Proxy universe for historical ablations |
| `scripts/universe/nifty500_nse_survivorship_proxy.txt` | NIFTY 500 proxy |
| `scripts/universe/README.txt` | Universe notes |

---

### 2.4 — Notebooks

| Path | Purpose | vs current scripts | Outputs saved? |
|------|---------|---------------------|----------------|
| `models/baseline_xgboost.ipynb` | Walk-forward XGBoost baseline; markdown says writes `results/xgboost_predictions.csv` | **Outdated path**: repo uses `results/phase1_baselines/xgboost_predictions.csv` per `models/baseline_xgboost.py` | **Yes** — large execution output in notebook JSON |

**Note:** `README.md` references `RAMT_Monolith_Trainer.ipynb` and `RAMT_Production_Pipeline.ipynb` — **these files are not present** in the workspace (see §3.4).

---

### 2.5 — Documentation and reports

| Path | Type | Status | Contradictions |
|------|------|--------|----------------|
| `README.md` | Main narrative + structure | **Current** story; detailed | Claims notebooks not in repo; NIFTY Sharpe wording vs `report.tex` abstract (~0.65 vs ~0.5) |
| `RESULTS.md` | Results log | **Current** | Momentum window CSVs live under `results/final_strategy/momentum_rankings_yf_<tag>.csv` |
| `RAMT_CORE_AUDIT.md` | Design / audit | Reference | — |
| `FEATURES_AND_REGIMES.md` | Features | Reference | — |
| `ATTENTION_EXPLAINABILITY.md` | Attention notes | Reference | — |
| `report.tex` | IEEE-style paper | **Draft/final thesis** | Abstract numbers should be checked against `results/final_strategy/backtest_results.csv` |
| `Phase1_report.pdf` | PDF report | Static snapshot | — |
| `requirements.txt` | Dependencies | **Current** | — |

---

### 2.6 — Dashboard code (`dashboard/*.py`)

| File | Section / tab | Data files read | Runs? |
|------|-----------------|-----------------|-------|
| `dashboard/app.py` | Sidebar: RAMT, Production strategy, LSTM, XGBoost, Model comparison | `results/final_strategy/backtest_results.csv`, `data/raw/_NSEI.parquet`, optional `results/ramt/*`, `results/phase1_baselines/*`, **`results/phase2_monthly/*` (missing)**, optional `results/archive/*` (missing), processed parquets via helpers | **Imports OK** under `.venv/bin/python` (`import dashboard.app`); Streamlit warnings only |
| `dashboard/market_scraper.py` | Helper for macro / feature parity | `pd.read_parquet` on `data/processed/*_features.parquet` | Used by `market_pulse` |
| `dashboard/market_pulse.py` | Thin re-export | — | Depends on `requests`, `bs4` |

---

### 2.7 — Scripts (utilities, fetchers, runners)

| Path | Role | Still needed? |
|------|------|----------------|
| `run_dashboard.sh` | Launches Streamlit | Yes, convenience |
| `scripts/*.py` | See §2.1 | Yes for reproduction / thesis |
| `models/run_final_2024_2026.py` | Primary backtest / training orchestration | Yes |

---

### 2.8 — Configuration and environment

| Path | Purpose |
|------|---------|
| `requirements.txt` | Pip dependencies |
| `.gitignore` | Ignores `.venv/`, `checkpoints/*.pt`, `eda/eda_executed.ipynb`, etc. |
| **No** `pyproject.toml`, `setup.cfg`, `.env`, or `package.json` at repo root |
| `once/pyvenv.cfg`, `needed/pyvenv.cfg`, etc. | Stray **duplicate virtualenv** metadata (see §2.10) |

---

### 2.9 — Images and figures

| Path | Content | Referenced in docs? |
|------|---------|---------------------|
| `ramt_prediction_collapse.png` | Training / prediction collapse visualization | Cited in README narrative |
| `results/ramt/training_dashboard.png` | Training curves | README / dashboard RAMT section |
| `ramt model results/training_dashboard.png` | Duplicate | Unlikely |
| `eda/plots/*.png` | EDA: volatility, correlation, distributions | Local EDA only |
| `Phase1_report.pdf` | Phase 1 write-up | External PDF |

---

### 2.10 — Junk, caches, temp files

| Category | Approx. size | Count / notes | Delete? |
|----------|----------------|---------------|---------|
| `.venv/` | **~1.5 GB** | Full Python env | **Do not commit** — keep local only; gitignored |
| `needed/`, `once/`, `only/`, `#/` | **~13 MB each** | Duplicate mini-venvs (`bin/`, `lib/python3.14/...`) | **P0 candidate** — redundant with `.venv` |
| `dashboard/.vite/` | small | Frontend tooling cache | **P0 candidate** |
| `.DS_Store` | ~14 KB | macOS | P0 |
| `__pycache__/` under `data/` | ~21 KB | bytecode | P0 |
| `checkpoints/best.pt` | ~2.1 MB | Legacy `.pt` | P1/P3 — verify before delete (§3.1) |
| `node_modules` / `.pytest_cache` | **not found** in project tree | — | — |

---

### 2.11 — Orphaned or unclear files

| Path | Issue |
|------|--------|
| `checkpoints/best.pt` | No Python reference; unknown architecture without inspection |
| `ramt model results/` (space in name) | Full duplicate of RAMT artifacts — triplication with `results/` and `models/ramt/artifacts/` |
| `dashboard/.vite/` | Unexpected Vite deps cache inside repo |
| `eda/` | Only PNGs — no checked-in notebook (`.gitignore` mentions `eda/eda_executed.ipynb`) |

---

## STEP 3 — CROSS-REFERENCE CHECKS

### 3.1 — Model artifact → code match

| Artifact | Match? |
|----------|--------|
| `results/ramt/ramt_model_state*.pt`, `models/ramt/artifacts/*.pt`, `ramt model results/*.pt` | **Yes** — `models/ramt/model.py` + `train_ranking` / `regenerate_ramt_outputs` |
| `checkpoints/best.pt` | **ORPHANED** — no `.py` references; not loaded by `inspect_attention` / `regenerate` (they take explicit paths). **Action:** inspect `torch.load` keys or delete/archive after confirmation. |
| `checkpoints/xgboost.joblib` | No loader in audited code — **legacy** |

### 3.2 — Predictions CSV → generator script

| CSV | Generator exists? |
|-----|-------------------|
| `results/phase1_baselines/xgboost_predictions.csv` | **Yes** — `models/baseline_xgboost.py` |
| `results/phase1_baselines/lstm_predictions.csv` | **Yes** — `models/baseline_lstm.py` |
| `results/final_strategy/ranking_predictions.csv` | **Yes** — `scripts/build_momentum_predictions.py` or `models/run_final_2024_2026.py` |
| `results/ramt/ranking_predictions.csv` | **Yes** — `scripts/regenerate_ramt_outputs.py` |
| `results/phase2_monthly/*` | **STALE / missing** — directory absent; dashboard expects Phase 2 files here |

### 3.3 — Metrics JSON → predictions CSV

| Metrics JSON | Backing CSV exists? |
|--------------|---------------------|
| `xgboost_metrics.json` → path in JSON | **Yes** — `results/phase1_baselines/xgboost_predictions.csv` |
| `lstm_metrics.json` | **Yes** — `lstm_predictions.csv` |
| `ramt/ramt_metrics.json` | **Yes** — `results/ramt/ranking_predictions.csv` |

**No UNREPRODUCIBLE** flags for these three.

### 3.4 — README / docs → reality

| Claim | Match? |
|-------|--------|
| “`RAMT_Monolith_Trainer.ipynb`, `RAMT_Production_Pipeline.ipynb`” | **Mismatch** — not in repo |
| “`results/phase2_monthly/`” (implied by dashboard README structure for Phase 2) | **Mismatch** — folder missing |
| Sharpe **0.83** / CAGR **13.5%** for final strategy | **Consistent** with `results/final_strategy/sensitivity/parameter_sensitivity_summary.csv` baseline row and README table |
| “7 walk-forward segments” | **Consistent** with `ramt_metrics.json` (`n_segments`: 7) |
| `RESULTS.md` references `momentum_rankings_yf_<tag>.csv` | **Mismatch** — files not found |

### 3.5 — Dashboard → data match

| Expected path | Status |
|---------------|--------|
| `results/final_strategy/backtest_results.csv` | **Exists** |
| `data/raw/_NSEI.parquet` | **Exists** (required for benchmark) |
| `results/ramt/ramt_metrics.json`, `ranking_predictions.csv`, `backtest_results.csv` | **Exist** |
| `results/phase1_baselines/xgboost_predictions.csv`, `lstm_predictions.csv` | **Exist** (Phase 1 tabs) |
| `results/phase2_monthly/xgboost_predictions.csv` (and siblings) | **Missing** — Phase 2 tabs show “not reproducible” |
| `results/archive/ramt_backtest_results.csv`, `momentum_regime_no_sector_backtest.csv` | **Missing** — optional |

### 3.6 — Duplicate or conflicting files

- **Triple duplicate:** RAMT `.pt` / `.joblib` in `results/`, `models/ramt/artifacts/`, `ramt model results/`.
- **Duplicate predictions:** `results/final_strategy/ranking_predictions.csv` vs `results/ramt/ranking_predictions.csv` (different roles — momentum vs RAMT).
- **Notebook vs script:** `baseline_xgboost.ipynb` output references old CSV paths vs `baseline_walkforward/`.

### 3.7 — Import graph (key entry points)

**`dashboard/app.py`:** `features.feature_engineering`, `features.sectors` → pulls in numpy/pandas/streamlit only from stdlib/third-party beyond that.

**`models/ramt/train_ranking.py`:** `models.ramt.dataset`, `losses`, `model` → full RAMT stack.

**`models/backtest.py`:** `features.sectors` only.

**`models/run_final_2024_2026.py`:** `models.backtest`, `models.ramt` (train_ranking), `features.feature_engineering`.

**`scripts/run_yf_hmm_ablation.py`**, **`scripts/run_nifty500_annual_hmm_ablation.py`:** subprocess / CLI drivers — do not import full model graph.

**Files not imported by any other module** (standalone CLIs / diagnostics): most `scripts/*.py`, `models/baseline_*.py`, `models/attention_consistency_report.py`, `models/permutation_importance.py`, `models/inspect_attention.py`, `data/download.py`. These are **expected** runners, not deletion candidates.

---

## STEP 4 — SIZE AND BLOAT ANALYSIS

### Top 20 largest files (excluding `.venv/`, `needed/`, `once/`, `only/`, `#/`)

| Size (B) | Path |
|----------|------|
| 2148415 | `checkpoints/best.pt` |
| 1155765 | `results/final_strategy/monthly_rankings.csv` |
| 1126593 | `ramt model results/ramt_model_state_wf_seg_07.pt` |
| 1126593 | `ramt model results/ramt_model_state_wf_seg_03.pt` |
| 1126593 | `ramt model results/ramt_model_state_wf_seg_02.pt` |
| 1126593 | `ramt model results/ramt_model_state_wf_seg_06.pt` |
| 1126593 | `ramt model results/ramt_model_state_wf_seg_01.pt` |
| 1126593 | `ramt model results/ramt_model_state_wf_seg_05.pt` |
| 1126593 | `ramt model results/ramt_model_state_wf_seg_04.pt` |
| 1126593 | `results/ramt/ramt_model_state_wf_seg_07.pt` |
| 1126593 | `results/ramt/ramt_model_state_wf_seg_03.pt` |
| 1126593 | `results/ramt/ramt_model_state_wf_seg_06.pt` |
| 1126593 | `results/ramt/ramt_model_state_wf_seg_05.pt` |
| 1126593 | `results/ramt/ramt_model_state_wf_seg_04.pt` |
| 1126593 | `models/ramt/artifacts/ramt_model_state_wf_seg_07.pt` |
| 1126593 | `models/ramt/artifacts/ramt_model_state_wf_seg_03.pt` |
| 1126593 | `models/ramt/artifacts/ramt_model_state_wf_seg_06.pt` |
| 1126593 | `models/ramt/artifacts/ramt_model_state_wf_seg_05.pt` |
| 1126593 | `models/ramt/artifacts/ramt_model_state_wf_seg_04.pt` |
| 1125029 | `ramt model results/ramt/ramt_model_state.pt` |

**No project file exceeds 100 MB** outside `.venv/`. Largest data risk is **duplicate environments** + **triplicated RAMT weights** (~18+ MB × 3 for `.pt` alone, plus joblibs).

### Total size by category (excluding `.venv`, `needed`, `once`, `only`, `#`)

| Category | Approx. bytes |
|----------|----------------|
| Raw + processed data under `data/` | ~241,000,000 |
| `results/` | ~12,600,000 |
| `ramt model results/` | ~9,200,000 |
| `models/` (code + artifacts) | ~7,400,000 |
| `checkpoints/` | ~3,000,000 |
| `eda/` | ~993,000 |
| Docs / images at root | ~150,000 |
| Dashboard | ~137,000 |
| Scripts + features | ~110,000 |

*(From `du` / walk; rounded.)*

---

## STEP 5 — RECOMMENDED ACTIONS TABLE

| Priority | Action | Path(s) | Reason | Est. size recovered |
|----------|--------|-----------|--------|---------------------|
| P0 | Delete duplicate virtualenv trees | `needed/`, `once/`, `only/`, `#/` | Redundant with `.venv` | ~52 MB |
| P0 | Remove `dashboard/.vite/` cache | `dashboard/.vite/` | Accidental tooling artifact | &lt; 1 MB |
| P0 | Delete `__pycache__`, `.DS_Store` | repo root, `data/__pycache__` | Regenerable / OS junk | &lt; 100 KB |
| P1 | Archive or delete **one** duplicate RAMT tree | Keep e.g. `models/ramt/artifacts/` OR `results/`; remove `ramt model results/` and duplicates | Triplicated `.pt`/`.joblib` | ~9–18 MB+ |
| P1 | Move `checkpoints/best.pt` to archive after identity check | `checkpoints/best.pt` | ORPHANED loader; `.gitignore` already ignores `*.pt` here | ~2.1 MB |
| P2 | Create `results/phase2_monthly/` **or** change dashboard paths | `dashboard/app.py`, new folder under `results/` | Dashboard Phase 2 tabs broken | — |
| P2 | Align `RESULTS.md` with on-disk CSV names or generate missing `momentum_rankings_yf_*.csv` | `RESULTS.md`, `scripts/run_yf_hmm_ablation.py` | Doc drift | — |
| P3 | Keep or delete `checkpoints/xgboost.joblib` | `checkpoints/` | No loader located | ~0.8 MB |
| P3 | Resolve README notebook names | `README.md` | Listed notebooks missing | — |

---

## STEP 6 — OPEN QUESTIONS FOR YOU

- Should **`needed/`, `once/`, `only/`, `#/`** be removed? They look like accidental duplicate venvs beside `.venv/`.
- Which **single canonical directory** should hold RAMT checkpoints: `results/`, `models/ramt/artifacts/`, or `ramt model results/`? (Triplication is error-prone.)
- What is **`checkpoints/best.pt`**? Safe to delete or must it be archived for grading?
- Do you want **`results/phase2_monthly/`** populated (new runs) or should the **dashboard** be pointed at `results/phase1_baselines/` for Phase 2?
- **`RESULTS.md`** references `momentum_rankings_yf_<tag>.csv` — should those files be generated, or should the doc be edited?
- Should **`README.md`** drop or update references to **missing notebooks**?

---

## EXECUTIVE SUMMARY (one page)

### Totals

| Metric | Value |
|--------|--------|
| **Total files** (excluding `.git`, including `.venv`) | **64,120** |
| **Total files** (excluding `.venv`, `needed/`, `once/`, `only/`, `#/`, `.git`) | **4,835** |
| **Total size** (same exclusion) | **~292 MB** |
| **Total size** with `.venv` | **~2.0 GB** |

### Recommendations count

| Bucket | Approx. count / scope |
|--------|----------------------|
| **P0 — Delete immediately** | Hundreds of files if counting duplicate envs (`needed/`, `once/`, `only/`, `#/` — each ~488+ files); plus `.vite` cache, `__pycache__`, `.DS_Store` |
| **P1 — Archive** | Dozens of duplicate `.pt`/`.joblib` after choosing one tree; `checkpoints/best.pt` after confirmation |
| **P3 — Need your decision** | 3+ items (canonical RAMT folder, `best.pt`, Phase 2 dashboard path, README notebooks) |

### Top 3 risks in current repo state

1. **Triplicated RAMT artifacts** — version confusion and wasted space; risk of loading the wrong checkpoint in reports.
2. **Dashboard / docs drift** — Phase 2 paths and `RESULTS.md` references point to **missing** files; easy to misinterpret what was actually run.
3. **Orphan `checkpoints/best.pt`** — unknown provenance; could be mistaken for current RAMT if not labeled.

### Five most important questions (before cleanup)

1. Which RAMT artifact directory is **canonical** for the thesis?
2. Is **`checkpoints/best.pt`** still required for any course deliverable?
3. Should **Phase 2** metrics live under a new `results/phase2_monthly/` (dashboard expectation) or should the dashboard change?
4. Should **README** be updated to remove **missing notebook** names?
5. Are **`needed/`, `once/`, `only/`, `#/`** safe to delete entirely?

### Estimated post-cleanup size (order of magnitude)

- Removing duplicate venvs + one copy of RAMT weights + junk: **~60–80 MB** recovered from the ~292 MB project tree (excluding `.venv`).
- Keeping **one** `.venv` for work: total disk **~1.55 GB** typical dev checkout.
- **No single file &gt; 100 MB** in project data; bloat is **duplication and envs**, not one huge parquet.

---

*End of audit. Path: `/Users/shivanshgupta/regime-adaptive-transformer/REPO_AUDIT.md`*
