# Results directory layout and path history

**Reorganization date:** 2026-04-20

All result files were moved with `git mv` so Git history for each file is preserved. **No contents inside existing CSV, JSON, or training log artifacts were modified** as part of the move; those files may still contain string columns or metadata fields that reference **older paths** (for example `results/hmm_vs_flat/...` or `results/ranking_predictions.csv`). Those embedded strings are part of the experiment record and were left unchanged on purpose.

## Current layout (summary)

| Directory | Purpose |
|-----------|---------|
| `final_strategy/` | Production pipeline: backtest export, ranking/momentum CSVs, parameter sensitivity |
| `ramt/` | RAMT checkpoints, scalers, training logs, RAMT-only backtest and metrics |
| `ramt/attention/` | Attention diagnostics (optional outputs from inspect scripts) |
| `phase1_baselines/` | Phase 1 walk-forward XGBoost and LSTM predictions and metrics |
| `hmm_ablation/` | Four-window HMM vs flat regime-sizing ablation (`2008_2010`, `2010_2012`, `2013_2015`, `2024_2026`) |
| `archive/hmm_vs_flat/` | Single-calendar-year HMM vs flat runs (superseded by multi-year windows but kept) |
| `lightgbm/` | Placeholder for future LightGBM diagnostic exports |
| `archive/` (optional CSVs) | Optional archived strategy comparison CSVs referenced by the dashboard |

## Mapping: old path → new path

| Before | After |
|--------|--------|
| `results/backtest_results.csv` | `results/final_strategy/backtest_results.csv` |
| `results/ranking_predictions.csv` | `results/final_strategy/ranking_predictions.csv` |
| `results/monthly_rankings.csv` | `results/final_strategy/monthly_rankings.csv` |
| `results/momentum_rankings_yf_*.csv` | `results/final_strategy/momentum_rankings_yf_*.csv` |
| `results/sensitivity/` | `results/final_strategy/sensitivity/` |
| `results/ramt_model_state*.pt`, `results/ramt_scaler*.joblib`, … (repo root) | `results/ramt/` |
| `results/training_history.csv`, `results/training_dashboard.png`, `results/training_log_*.txt` | `results/ramt/` |
| `results/baseline_walkforward/` | `results/phase1_baselines/` |
| `results/hmm_vs_flat/yf_2008_2010/...` | `results/hmm_ablation/2008_2010/...` |
| `results/hmm_vs_flat/yf_2010_2012/...` | `results/hmm_ablation/2010_2012/...` |
| `results/hmm_vs_flat/yf_2013_2015/...` | `results/hmm_ablation/2013_2015/...` |
| `results/hmm_vs_flat/2024-01-01_2025-12-31/...` | `results/hmm_ablation/2024_2026/...` |
| `results/hmm_vs_flat/yf_2010` (single year) | `results/archive/hmm_vs_flat/yf_2010` (and similarly `yf_2011`, `yf_2012`) |

Duplicate `results/training_dashboard.png` at the repo root was identical byte-for-byte to `results/ramt/training_dashboard.png` and was removed from the root to avoid duplication; history for the PNG is retained via `results/ramt/training_dashboard.png`.
