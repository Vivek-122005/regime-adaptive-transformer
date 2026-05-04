# Phase Index — where everything lives

This document maps every file in the repository to the phase it belongs to in the
project report (`report/report.tex` / `report/report.pdf`). The repo is organized
**by function** at the file-system level (`models/`, `scripts/`, `features/`,
`results/`, `dashboard/`, …) for engineering reasons (shared imports, single
backtest engine, single dashboard). This index is the **phase view** — an
evaluator can use it to find anything cited in the report without digging.

If you only have time to read four things:

1. [report/report.pdf](report/report.pdf) — the IEEE writeup (10 pages).
2. [docs/architecture_final.png](docs/architecture_final.png) — final hybrid architecture.
3. [results/ablation_summary.json](results/ablation_summary.json) — every ablation scenario in one file.
4. `streamlit run dashboard/app.py` — every metric and chart, live.

### Per-phase deliverables (presentations + reports)

| Phase | Presentation | Report |
|---|---|---|
| Phase 1 | [report/phase1/Phase1_PPT.pdf](report/phase1/Phase1_PPT.pdf) | (covered in main IEEE paper §Phase 1) |
| Phase 2 | (covered in main IEEE paper §Phase 2) | [report/phase2/Phase2_Report.pdf](report/phase2/Phase2_Report.pdf) |
| Phase 3 | [report/phase3/Phase3_PPT.pdf](report/phase3/Phase3_PPT.pdf) | [report/phase3/Phase3_Report.pdf](report/phase3/Phase3_Report.pdf) |
| All phases (consolidated) | — | [report/report.pdf](report/report.pdf) (IEEE format, the canonical one) |

---

## Repository layout at a glance

| Top-level dir | Phase relevance | Purpose |
|---|---|---|
| `data/` | All phases (shared) | Raw OHLCV, processed features, manifest fingerprints, sentiment data |
| `features/` | All phases (shared) | Feature engineering, sector mapping, sentiment integration |
| `backtest/` | All phases (shared) | The single backtest engine (`run_backtest_daily`) every phase uses |
| `models/` | Phase 1 + Phase 2 + Phase 3 | All model code (organized below by phase) |
| `scripts/` | All phases | Pipeline orchestration scripts (organized below by phase) |
| `results/` | All phases | All artefacts — metrics, predictions, backtest CSVs, training curves |
| `dashboard/` | All phases | Streamlit reviewer dashboard (phase-grouped sidebar) |
| `report/` | Final deliverable | IEEE LaTeX paper + compiled PDF |
| `docs/` | Supporting documentation | Literature review, architecture, design notes, audits |
| `config/` | All phases | YAML pipeline configuration |
| `eda/` | All phases | Exploratory plots |
| `report/`, `docs/`, `data/manifest.csv`, `requirements.txt`, `Dockerfile`, `.github/workflows/ci.yml` | Cross-cutting | Reproducibility infrastructure |

---

## Phase 1 — Foundational ML baselines

**Report section:** §Phase 1 — Baseline Models (`report/report.tex` line 117).

**Goal:** Establish whether daily-return prediction is tractable before pivoting to monthly cross-sectional alpha. Both XGBoost and LSTM converge near-random — the label was the bug, not the algorithm.

### Source code
| File | Purpose |
|---|---|
| [models/baseline_xgboost.py](models/baseline_xgboost.py) | XGBoost daily-return baseline trainer + walk-forward eval |
| [models/baseline_xgboost.ipynb](models/baseline_xgboost.ipynb) | Notebook variant, same model |
| [models/baseline_lstm.py](models/baseline_lstm.py) | LSTM daily-return baseline trainer |
| [models/hmm_train_2011.py](models/hmm_train_2011.py) | 3-state HMM regime detector trained on pre-2012 NIFTY |
| [features/feature_engineering.py](features/feature_engineering.py) | Builds the 10-feature panel both baselines consume |
| [features/sectors.py](features/sectors.py) | Sector mapping used by Phase 1 metrics |

### Results / artefacts
| File | What it shows |
|---|---|
| [results/backtesting/phase1_baselines/xgboost_metrics.json](results/backtesting/phase1_baselines/xgboost_metrics.json) | DA, mean IC, RMSE for XGBoost daily |
| [results/backtesting/phase1_baselines/xgboost_predictions.csv](results/backtesting/phase1_baselines/xgboost_predictions.csv) | Per-row predicted vs actual daily return |
| [results/backtesting/phase1_baselines/lstm_metrics.json](results/backtesting/phase1_baselines/lstm_metrics.json) | DA, mean IC, RMSE for LSTM daily |
| [results/backtesting/phase1_baselines/lstm_predictions.csv](results/backtesting/phase1_baselines/lstm_predictions.csv) | Per-row predicted vs actual daily return |
| [checkpoints/xgboost.joblib](checkpoints/xgboost.joblib) | Saved Phase 1 XGBoost model |
| [checkpoints/best.pt](checkpoints/best.pt) | Saved Phase 1 LSTM weights |

### Documentation & deliverables
| File | Purpose |
|---|---|
| [docs/README_PHASE1.md](docs/README_PHASE1.md) | Phase 1 narrative + how to reproduce |
| [docs/FEATURES_AND_REGIMES.md](docs/FEATURES_AND_REGIMES.md) | Feature definitions for the panel both baselines use |
| [report/phase1/Phase1_PPT.pdf](report/phase1/Phase1_PPT.pdf) | **Phase 1 presentation deck** |

### How to reproduce
```bash
python models/baseline_xgboost.py
python models/baseline_lstm.py
```

### Headline number
Both baselines: directional accuracy ≈ 50%, mean IC essentially zero. **The IC≈0 result motivated the Phase 2 re-specification to 21-day forward sector alpha.**

### Where it appears in the dashboard
Sidebar → **Phase 1 — Foundational ML** → Overview, XGBoost (daily), LSTM (daily).

---

## Phase 2 — RAMT (Regime-Adaptive Multimodal Transformer)

**Report section:** §Phase 2 — RAMT Architecture and Failure (`report/report.tex` line 127).

**Goal:** Build a custom multimodal transformer with regime cross-attention and a tournament-ranking loss. RAMT collapses on out-of-sample (predictions converge to the mean, pairwise margins vanish, gradients vanish). A 60-line LightGBM diagnostic on the same features confirms the failure was architectural, not a feature problem.

### Source code (RAMT model itself)
| File | Purpose |
|---|---|
| [models/ramt/model.py](models/ramt/model.py) | Top-level RAMT module composition |
| [models/ramt/encoder.py](models/ramt/encoder.py) | `MultimodalEncoder` (4 modalities → 64-dim embeddings) |
| [models/ramt/moe.py](models/ramt/moe.py) | Mixture-of-experts transformer block (regime-gated) |
| [models/ramt/losses.py](models/ramt/losses.py) | `TournamentRankingLoss` + auxiliary MSE anchor |
| [models/ramt/dataset.py](models/ramt/dataset.py) | Walk-forward sequence dataset (200 tickers × 30-day windows) |
| [models/ramt/train_ranking.py](models/ramt/train_ranking.py) | Training loop with walk-forward orchestration |

### Source code (RAMT diagnostics + analysis)
| File | Purpose |
|---|---|
| [models/inspect_attention.py](models/inspect_attention.py) | Attention-weight inspection (the heatmaps in `docs/ATTENTION_EXPLAINABILITY.md`) |
| [models/attention_consistency_report.py](models/attention_consistency_report.py) | Cross-fold attention stability check |
| [models/permutation_importance.py](models/permutation_importance.py) | Feature attribution via permutation |
| [scripts/regenerate_ramt_outputs.py](scripts/regenerate_ramt_outputs.py) | Re-runs RAMT inference on saved checkpoints to regenerate metrics + predictions |

### Pipeline orchestration
| File | Purpose |
|---|---|
| [main.py](main.py) `--task validate` | Validates feature/prediction date alignment |
| [scripts/check_pipeline_health.py](scripts/check_pipeline_health.py) | Sanity checks before training |

### Results / artefacts
| File | What it shows |
|---|---|
| [results/models/ramt/ramt_metrics.json](results/models/ramt/ramt_metrics.json) | DA 47.2%, mean IC -0.016, Sharpe 0.49, MaxDD -6.4% (the failure numbers) |
| [results/models/ramt/ranking_predictions.csv](results/models/ramt/ranking_predictions.csv) | 5 200 OOS predictions — σ(predicted_alpha) ≈ 0.0048 (collapse evidence) |
| [results/models/ramt/training_history.csv](results/models/ramt/training_history.csv) | Per-epoch train + val loss + LR |
| [results/models/ramt/training_dashboard.png](results/models/ramt/training_dashboard.png) | Visual loss curves |
| [results/models/ramt/backtest_results.csv](results/models/ramt/backtest_results.csv) | Per-rebalance equity curve from RAMT predictions |
| `results/models/ramt/ramt_model_state*.pt` | Saved walk-forward checkpoints (7 segments) |
| `results/models/ramt/ramt_scaler*.joblib`, `ramt_y_scaler*.joblib` | Saved feature/target scalers |
| [results/models/ramt/attention/](results/models/ramt/attention/) | Saved attention weight tensors |

### Documentation & deliverables
| File | Purpose |
|---|---|
| [docs/RAMT_CORE_AUDIT.md](docs/RAMT_CORE_AUDIT.md) | The architectural audit that names the collapse mode |
| [docs/ATTENTION_EXPLAINABILITY.md](docs/ATTENTION_EXPLAINABILITY.md) | Attention heatmaps + interpretation |
| [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) | Vaswani, Lim et al. (TFT), Hamilton (HMM), Chronos, LoRA citations |
| [report/phase2/Phase2_Report.pdf](report/phase2/Phase2_Report.pdf) | **Phase 2 standalone report** |

### How to reproduce
```bash
python -m models.ramt.train_ranking      # train all walk-forward segments
python scripts/regenerate_ramt_outputs.py  # produce metrics + predictions CSV
```

### Headline number
**Mean IC = -0.016** on OOS — RAMT is anti-correlated with the realized ranked target. **σ(predictions) ≈ 0.0048** — the prediction-collapse smoking gun.

### Where it appears in the dashboard
Sidebar → **Phase 2 — Deep Learning** → Overview (collapse evidence + Sharpe pivot bar + training curves), RAMT transformer (live conviction table + per-ticker scatter).

---

## Phase 3 — Foundation model with LoRA + hybrid system

**Report section:** §Phase 3 — Foundation Model with LoRA (`report/report.tex` line 169) plus §Final Strategy (line 198) plus §Ablation (line 217).

**Goal:** Replace the bespoke transformer with a frozen Chronos-T5-Small backbone (~190M params) and attach LoRA adapters (rank 8) — only ~0.1% of weights trainable. Combine the foundation model's alpha signal with momentum and an HMM regime gate. Run a full ablation across 8 component scenarios + 4 historical HMM windows.

### Source code (foundation model + LoRA)
| File | Purpose |
|---|---|
| [models/lora_experiment/chronos_lora.py](models/lora_experiment/chronos_lora.py) | LoRA v1 module + adapter wiring on Chronos attention layers |
| [models/lora_experiment/chronos_lora_v2.py](models/lora_experiment/chronos_lora_v2.py) | LoRA v2 (the version cited in the report) |
| [models/lora_experiment/train_lora.py](models/lora_experiment/train_lora.py) | LoRA training loop on monthly-alpha targets |
| [models/lora_experiment/train_chronos_lora_2012.py](models/lora_experiment/train_chronos_lora_2012.py) | Pre-2012 LoRA training for the historical stress test |
| [models/lora_experiment/chronos_v2_adapter.pt](models/lora_experiment/chronos_v2_adapter.pt) | Trained LoRA adapter weights |
| [scripts/train_chronos_lora.py](scripts/train_chronos_lora.py) | Top-level LoRA training driver |
| [scripts/generate_chronos_predictions.py](scripts/generate_chronos_predictions.py) | Produces the 108k OOS Chronos prediction CSV |
| [scripts/explain_chronos.py](scripts/explain_chronos.py) | Feature-attribution analysis on Chronos predictions |

### Source code (production hybrid strategy)
| File | Purpose |
|---|---|
| [backtest/core/momentum_hmm_strategy.py](backtest/core/momentum_hmm_strategy.py) | The deployed Momentum + HMM strategy (production) |
| [backtest/core/backtest.py](backtest/core/backtest.py) | The single `run_backtest_daily` engine — every Phase 3 result uses this |
| [backtest/scripts/hmm_vs_flat_backtest.py](backtest/scripts/hmm_vs_flat_backtest.py) | The 4-window HMM-vs-flat ablation driver |
| [backtest/scripts/parameter_sensitivity_backtest.py](backtest/scripts/parameter_sensitivity_backtest.py) | Sensitivity sweep over stop-loss / position-count / sector-cap / regime-sizing |
| [features/sentiment_integration.py](features/sentiment_integration.py) | FinBERT sentiment features (Phase 3 extra) |

### Source code (ablation orchestration)
| File | Purpose |
|---|---|
| [models/ablation_engine.py](models/ablation_engine.py) | Configurable ablation harness — toggles momentum / HMM / sentiment / LoRA |
| [scripts/run_ablation_study.py](scripts/run_ablation_study.py) | Top-level ablation driver (writes `results/ablation_report.csv`) |
| [scripts/run_diagnostic_ablation.py](scripts/run_diagnostic_ablation.py) | 4-way diagnostic ablation (Momentum / Foundation / Simple Hybrid / Triple-Expert) |
| [scripts/run_hybrid_minus_momentum.py](scripts/run_hybrid_minus_momentum.py) | Generates the "Hybrid − Momentum" ablation row (Chronos + HMM, no momentum) |
| [scripts/pipeline_sentiment.py](scripts/pipeline_sentiment.py) | FinBERT sentiment pipeline for the sentiment ablation toggle |
| [scripts/generate_sentiment_features.py](scripts/generate_sentiment_features.py) | Generate sentiment features per ticker per date |
| [scripts/explain_sentiment.py](scripts/explain_sentiment.py) | Sentiment-layer explainability |

### Pipeline orchestration
| File | Purpose |
|---|---|
| [main.py](main.py) `--task all` | End-to-end pipeline: features → ablation → diagram → report → dashboard |
| [main.py](main.py) `--task momentum` | Production Mom + HMM strategy backtest |
| [main.py](main.py) `--task diagnostic` | Pipeline health + 4-way diagnostic ablation |
| [main.py](main.py) `--task explain` | Chronos feature-importance |
| [scripts/generate_architecture_diagram.py](scripts/generate_architecture_diagram.py) | Renders [docs/architecture_final.png](docs/architecture_final.png) from Mermaid |
| [scripts/generate_final_report.py](scripts/generate_final_report.py) | Builds [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) from artefacts |

### Results / artefacts
| File | What it shows |
|---|---|
| [results/backtesting/final_strategy/backtest_results.csv](results/backtesting/final_strategy/backtest_results.csv) | Production strategy: 25 monthly rebalances, Sharpe 0.83, CAGR 13.5%, MaxDD -18.7% |
| [results/backtesting/final_strategy/backtest_results_weekly_2023_2026.csv](results/backtesting/final_strategy/backtest_results_weekly_2023_2026.csv) | Weekly-cadence sensitivity variant |
| [results/backtesting/final_strategy/backtest_results_weekly_ret5d_2023_2026.csv](results/backtesting/final_strategy/backtest_results_weekly_ret5d_2023_2026.csv) | Weekly + 5-day signal sensitivity |
| [results/backtesting/final_strategy/monthly_rankings.csv](results/backtesting/final_strategy/monthly_rankings.csv) | Top-5 picks per month |
| [results/backtesting/final_strategy/momentum_rankings_yf_*.csv](results/backtesting/final_strategy/) | Historical momentum rankings (2008-2015) |
| [results/backtesting/final_strategy/sensitivity/](results/backtesting/final_strategy/sensitivity/) | Parameter sensitivity sweep |
| [results/backtesting/hybrid_lora/backtest_results.csv](results/backtesting/hybrid_lora/backtest_results.csv) | Hybrid + LoRA backtest (Sharpe 1.05) |
| [results/backtesting/hybrid_lora/hybrid_rankings.csv](results/backtesting/hybrid_lora/hybrid_rankings.csv) | Top-5 picks under the hybrid scoring rule |
| [results/ablation/backtest_hybrid_minus_momentum.csv](results/ablation/backtest_hybrid_minus_momentum.csv) | "Hybrid − Momentum" backtest (Chronos + HMM, no momentum) |
| [results/ablation/predictions_hybrid_minus_momentum.csv](results/ablation/predictions_hybrid_minus_momentum.csv) | Predictions backing the row above |
| [results/models/lora/lora_predictions.csv](results/models/lora/lora_predictions.csv) | 108k Chronos-LoRA predictions over 2024-2026 OOS |
| [results/models/lora/lora_metrics.json](results/models/lora/lora_metrics.json) | LoRA v1 metrics (DA 47.5%, IC +0.009) |
| [results/models/lora/lora_v2_predictions.csv](results/models/lora/lora_v2_predictions.csv) | LoRA v2 predictions |
| [results/models/lora/lora_v2_metrics.json](results/models/lora/lora_v2_metrics.json) | LoRA v2 metrics (DA 47.0%, IC +0.002) |
| [results/models/lora/chronos_lora_adapter.pt](results/models/lora/chronos_lora_adapter.pt) | Trained LoRA adapter for production inference |
| [results/models/lora/lora_v2_predictions_2012_2015.csv](results/models/lora/lora_v2_predictions_2012_2015.csv) | LoRA v2 inference on the 2012-2015 historical window |
| [results/models/lora/lora_v2_predictions_2020_2022.csv](results/models/lora/lora_v2_predictions_2020_2022.csv) | LoRA v2 inference on 2020-2022 |
| [results/models/explainability/feature_importance_plot.png](results/models/explainability/feature_importance_plot.png) | Chronos feature importance bar chart |
| [results/models/explainability/chronos_feature_importance.json](results/models/explainability/chronos_feature_importance.json) | Underlying numerical feature attributions |
| [results/ablation_summary.json](results/ablation_summary.json) | **Master ablation file — 8 scenarios + 4-window HMM study** |
| [results/ablation_summary_2020_2022.json](results/ablation_summary_2020_2022.json) | Ablation rerun on 2020-2022 |
| [results/ablation_report.csv](results/ablation_report.csv) | Tabular ablation summary |
| [results/backtesting/hmm_ablation/2008_2010/](results/backtesting/hmm_ablation/2008_2010/) | HMM vs flat sizing, 2008-2010 |
| [results/backtesting/hmm_ablation/2010_2012/](results/backtesting/hmm_ablation/2010_2012/) | HMM vs flat sizing, 2010-2012 |
| [results/backtesting/hmm_ablation/2013_2015/](results/backtesting/hmm_ablation/2013_2015/) | HMM vs flat sizing, 2013-2015 |
| [results/backtesting/hmm_ablation/2024_2026/](results/backtesting/hmm_ablation/2024_2026/) | HMM vs flat sizing, 2024-2026 |
| [results/backtesting/historical_2012/backtest_summary_2012_2015.csv](results/backtesting/historical_2012/backtest_summary_2012_2015.csv) | 2012-2015 stress test summary |
| [results/backtesting/historical_2012/backtest_*_2012_2015.csv](results/backtesting/historical_2012/) | Per-strategy 2012-2015 backtests (4 variants) |
| [results/backtesting/historical_2012/hmm_pre2012.pkl](results/backtesting/historical_2012/hmm_pre2012.pkl) | HMM trained on pre-2012 data only |

### Documentation & deliverables
| File | Purpose |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Hybrid architecture description + Mermaid source |
| [docs/architecture_final.png](docs/architecture_final.png) | Final architecture diagram (rasterized) |
| [docs/architecture_final.svg](docs/architecture_final.svg) | Final architecture diagram (vector) |
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | Markdown summary of every Phase 3 result |
| [docs/RESULTS.md](docs/RESULTS.md) | Numerical results table (mirrors the report) |
| [report/phase3/Phase3_PPT.pdf](report/phase3/Phase3_PPT.pdf) | **Phase 3 presentation deck** |
| [report/phase3/Phase3_Report.pdf](report/phase3/Phase3_Report.pdf) | **Phase 3 standalone report** |

### Configuration
| File | Purpose |
|---|---|
| [config/hybrid_config.yaml](config/hybrid_config.yaml) | Pipeline paths, model knobs, ablation toggles |

### How to reproduce
```bash
# Train LoRA adapter (or use the committed one):
python scripts/train_chronos_lora.py
python scripts/generate_chronos_predictions.py

# Run ablations:
python scripts/run_diagnostic_ablation.py
python scripts/run_hybrid_minus_momentum.py
python -m backtest.scripts.hmm_vs_flat_backtest

# Production strategy:
python main.py --task momentum

# Full pipeline (everything above + diagram + report + dashboard):
python main.py --task all
```

### Headline numbers
- Production (Mom + HMM): **Sharpe 0.83, CAGR 13.5%, MaxDD -18.7%**
- Foundation Only (Chronos-LoRA): **Sharpe 1.34, CAGR 23.5%** (best single signal in OOS window)
- Hybrid + Chronos-LoRA: **Sharpe 1.05** (live `results/backtesting/hybrid_lora/`)
- Simple Hybrid (50/50 Momentum + Chronos): Sharpe 0.91, CAGR 22.8%
- Triple-Expert (Mom + Chronos + HMM): Sharpe 0.54 (HMM caps upside in clean bull)
- 4-window HMM study: HMM saves 9.4 pp drawdown in 2008-2010, turns -3.0 Sharpe into +0.79 in 2010-2012, costs upside in 2024-2026

### Where it appears in the dashboard
Sidebar → **Phase 3 — Hybrid system** → Overview (architecture + equity overlay + 3-panel ablation + 4-window HMM small-multiples), Production strategy (Mom + HMM), Foundation expert (Chronos + LoRA), Triple-Expert diagnostic, Historical stress test (2012-2015), Master comparison.

---

## Cross-cutting infrastructure (used by every phase)

### Data
| File | Purpose |
|---|---|
| [data/manifest.csv](data/manifest.csv) | md5 fingerprints for all 200 ticker feature files |
| [data/nifty200_tickers.txt](data/nifty200_tickers.txt) | Static NIFTY 200 universe (2026 snapshot) |
| [data/download.py](data/download.py) | yfinance downloader |
| `data/raw/` | Per-ticker OHLCV + macro parquets (gitignored, regenerable) |
| `data/processed/` | Engineered feature parquets per ticker (gitignored, regenerable) |
| `data/processed/sentiment/` | FinBERT sentiment features (gitignored) |

### Reproducibility
| File | Purpose |
|---|---|
| [requirements.txt](requirements.txt) | Pinned Python dependencies (Python 3.11, 23 packages) |
| [Dockerfile](Dockerfile) | Single-stage container — `docker build && docker run -p 8501:8501` |
| [.dockerignore](.dockerignore) | Keeps build context lean (~5 MB) |
| [docker-compose.yml](docker-compose.yml) | Multi-service compose if needed |
| [setup.sh](setup.sh) | Creates venv, installs deps, runs smoke test |
| [run_all.sh](run_all.sh) | Full pipeline launcher |
| [run_dashboard.sh](run_dashboard.sh) | Streamlit launcher |
| [demo_walkthrough.sh](demo_walkthrough.sh) | Builds final report + boots dashboard |
| [main.py](main.py) | Single entry point — `python main.py --task {all,data,sentiment,validate,ablation,diagnostic,diagram,explain,momentum,smoke-test}` |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | CI: lint, structure, install, docker build |

### Reporting & deliverables
| File | Purpose |
|---|---|
| [report/report.tex](report/report.tex) | **Canonical IEEE LaTeX paper** (10 pages, covers all phases) |
| [report/report.pdf](report/report.pdf) | Compiled canonical paper |
| [report/references.bib](report/references.bib) | Bibliography |
| [report/phase1/Phase1_PPT.pdf](report/phase1/Phase1_PPT.pdf) | Phase 1 presentation deck |
| [report/phase2/Phase2_Report.pdf](report/phase2/Phase2_Report.pdf) | Phase 2 standalone report |
| [report/phase3/Phase3_PPT.pdf](report/phase3/Phase3_PPT.pdf) | Phase 3 presentation deck |
| [report/phase3/Phase3_Report.pdf](report/phase3/Phase3_Report.pdf) | Phase 3 standalone report |
| [LICENSE](LICENSE) | License |
| [CITATION.cff](CITATION.cff) | Citation metadata |

### Frontend
| File | Purpose |
|---|---|
| [dashboard/app.py](dashboard/app.py) | **Phase-grouped Streamlit reviewer dashboard** — sidebar maps directly to this index |
| [dashboard/market_pulse.py](dashboard/market_pulse.py) | Live market pulse helper |
| [dashboard/market_scraper.py](dashboard/market_scraper.py) | Market data scraper |

### EDA
| File | Purpose |
|---|---|
| [eda/plots/](eda/plots/) | Exploratory plots (regime distributions, return histograms) |

---

## Quick map — "where would I find …?"

| Looking for … | Look here |
|---|---|
| The IEEE paper | [report/report.pdf](report/report.pdf) |
| Architecture diagram | [docs/architecture_final.png](docs/architecture_final.png), [docs/architecture.md](docs/architecture.md) |
| Phase 1 baseline numbers | [results/backtesting/phase1_baselines/](results/backtesting/phase1_baselines/) |
| Phase 2 RAMT failure | [results/models/ramt/ramt_metrics.json](results/models/ramt/ramt_metrics.json), [docs/RAMT_CORE_AUDIT.md](docs/RAMT_CORE_AUDIT.md) |
| Phase 3 ablation table | [results/ablation_summary.json](results/ablation_summary.json) |
| 4-window HMM study | [results/backtesting/hmm_ablation/](results/backtesting/hmm_ablation/) |
| Production strategy code | [backtest/core/momentum_hmm_strategy.py](backtest/core/momentum_hmm_strategy.py) |
| Production strategy backtest | [results/backtesting/final_strategy/backtest_results.csv](results/backtesting/final_strategy/backtest_results.csv) |
| Chronos-LoRA training script | [scripts/train_chronos_lora.py](scripts/train_chronos_lora.py) |
| Chronos-LoRA predictions | [results/models/lora/lora_predictions.csv](results/models/lora/lora_predictions.csv) |
| Hybrid system backtest | [results/backtesting/hybrid_lora/backtest_results.csv](results/backtesting/hybrid_lora/backtest_results.csv) |
| Live demo | `streamlit run dashboard/app.py` |
| Reproducibility entry point | [main.py](main.py) |
| CI workflow | [.github/workflows/ci.yml](.github/workflows/ci.yml) |
