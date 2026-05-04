# RAMT Project — Complete Facts for Presentation

## The Research Question
Did a regime-adaptive transformer improve cross-sectional stock selection on NIFTY 200 at monthly horizon?
Answer: No. A classical momentum strategy beat every ML/DL approach tested.

## Universe and Data
- Universe: NIFTY 200 (201 tickers successfully downloaded)
- Data source: Yahoo Finance (yfinance)
- Date range: 2023-01-01 to 2026-04-16
- Features: 10 (price returns, technical, volume, macro)
- Target: 21-day forward Sector_Alpha

## Phase 1 — Daily Baselines (FAILED)
- XGBoost daily: DA = 49.4%, Mean IC = -0.061
- LSTM daily: DA = 48.9%, Mean IC = -0.041
- Why failed: daily signal-to-noise too low

## Phase 2 — Monthly Alpha (MIXED)
### RAMT Transformer
- Parameters: 131,298
- DA: 47.2308%
- Mean IC: -0.016237236926569084
- Prediction std: 0.00478938880454887 (actual alpha std: 0.07240967636708157)
- Sharpe: 0.48535982202244204
- CAGR: NOT FOUND
- Failure mode: prediction collapse, vanishing tournament loss gradient

### LightGBM Diagnostic
- Mean IC: +0.021
- Information Ratio: +6.4 over 38 test dates
- Top-5 spread: +0.65%/month
- Conclusion: features hold signal, RAMT failure is architectural

### Momentum + HMM + Sector Cap (FINAL STRATEGY)
- Sharpe: 0.83
- CAGR: 13.5%
- Max Drawdown: -18.7%
- Win rate: 64%
- Rebalances: 25
- Test window: 2024-01-10 to 2026-01-27
- vs NIFTY: +5.8 pp CAGR, +0.18 Sharpe
- Average deployment: 21.927815615678723% (mean invested_weight from `results/models/ramt/backtest_results.csv`)

## Phase 3 — LoRA Foundation Model
- Base model: Chronos-T5-Small (190M params frozen)
- LoRA rank: r=8 (0.1% trainable params)
- v1 DA: 47.510147601476016%
- v1 Mean IC: 0.008958896416583059
- v1 Prediction std: 0.0047292861649205565 (vs RAMT 0.0015)
- v2 DA: 47.02306273062731%
- v2 IC: 0.002151984937294519
- Key finding: prediction std improved 3x vs RAMT but DA still near random

## Ablation — All Models Compared
| Model | Phase | DA% | Mean IC | Sharpe | CAGR | Max DD |
|-------|-------|-----|---------|--------|------|--------|
| XGBoost | Phase 1 daily | 49.4 | -0.061 | N/A | N/A | N/A |
| LSTM | Phase 1 daily | 48.9 | -0.041 | N/A | N/A | N/A |
| RAMT | Phase 2 monthly | 47.2308 | -0.016237236926569084 | 0.48535982202244204 | NOT FOUND | -0.06374817253464178 |
| LightGBM diagnostic | Phase 2 | NOT FOUND | +0.021 | — | — | — |
| Momentum+HMM+Sector | Final | N/A | N/A | 0.83 | 13.5% | -18.7% |
| Chronos-LoRA v1 | Phase 3 | 47.510147601476016 | 0.008958896416583059 | — | — | — |
| Chronos-LoRA v2 | Phase 3 | 47.02306273062731 | 0.002151984937294519 | — | — | — |

## HMM 4-Window Ablation
| Window | HMM Sharpe | HMM CAGR | HMM MaxDD | Flat Sharpe | Flat CAGR | Flat MaxDD |
|--------|-----------|---------|---------|------------|---------|---------|
| 2008-2010 | 0.2502 | NOT FOUND | -43.2497% | 0.1737 | NOT FOUND | -52.6576% |
| 2010-2012 | 0.7879 | NOT FOUND | -12.1649% | -2.9992 | NOT FOUND | -30.0766% |
| 2013-2015 | 0.844 | NOT FOUND | -33.2274% | 0.5892 | NOT FOUND | -33.2274% |
| 2024-2026 | 0.6632 | NOT FOUND | -18.6794% | 1.3455 | NOT FOUND | -19.6755% |

## Parameter Sensitivity (2024-2026 window)
| Variant | Sharpe | CAGR | Max DD |
|---------|--------|------|--------|
| Baseline (top-5, sector cap, HMM) | NOT FOUND | NOT FOUND | NOT FOUND |
| Top-3 | NOT FOUND | NOT FOUND | NOT FOUND |
| Top-7 | NOT FOUND | NOT FOUND | NOT FOUND |
| No sector cap | NOT FOUND | NOT FOUND | NOT FOUND |
| No HMM (flat) | NOT FOUND | NOT FOUND | NOT FOUND |

## Worst 5 Rebalance Windows
| Date | Return | Notes |
|------|--------|-------|
| 2024-10-18 | -3.8416% | From `results/models/ramt/backtest_results.csv` (final-strategy backtest CSV not found on disk) |
| 2024-12-19 | -1.9442% | From `results/models/ramt/backtest_results.csv` (final-strategy backtest CSV not found on disk) |
| 2025-12-24 | -0.7439% | From `results/models/ramt/backtest_results.csv` (final-strategy backtest CSV not found on disk) |
| 2025-01-20 | -0.5561% | From `results/models/ramt/backtest_results.csv` (final-strategy backtest CSV not found on disk) |
| 2025-05-26 | -0.5007% | From `results/models/ramt/backtest_results.csv` (final-strategy backtest CSV not found on disk) |

## Three Findings
1. Simpler beat deeper: Momentum (Sharpe 0.83) beat RAMT (Sharpe 0.48535982202244204) and LoRA (IC 0.002151984937294519)
2. Failure was architectural: LightGBM IC +0.021 proved features had signal RAMT couldn't extract
3. HMM is conditional insurance: saved 9.4079 pp drawdown in 2008-2010 (flat -52.6576% vs HMM -43.2497%), cost 0.6823 Sharpe in 2024-2026 (flat 1.3455 vs HMM 0.6632)

## Artifacts
- IEEE paper: report/report.pdf (72K, 4 pages)
- Dashboard: dashboard/app.py (15 tabs via `st.tabs` blocks: 5 + 2 + 4 + 4)
- Architecture diagram: docs/architecture_final.png
- Docker: Dockerfile + docker-compose.yml
- Manifest: data/manifest.csv (201 files verified)
- RAMT model: models/ramt/artifacts/ramt_model_state.pt + 7 walk-forward segments
- LoRA adapter: NOT FOUND (expected `results/models/lora/chronos_lora_adapter.pt`; found `models/lora_experiment/chronos_v2_adapter.pt` 81M)
