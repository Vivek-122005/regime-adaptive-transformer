# Results Directory Structure

This directory contains all results from the regime-adaptive transformer project organized by category.

## Directory Structure

```
results/
├── README.md                    # This file
├── README_PATHS.md              # Original paths documentation
├── ablation_report.csv          # Summary of ablation studies
├── ablation_summary*.json       # Ablation results summaries
├── summary_infographic.png      # Project summary visualization
├── backtesting/                 # All backtesting results
│   ├── final_strategy/          # Main strategy backtest results
│   ├── hybrid_lora/             # Hybrid LoRA backtest results
│   ├── historical_2012/         # Historical validation backtests
│   ├── hmm_ablation/            # HMM regime ablation studies
│   ├── ablation/                # Various ablation study results
│   ├── phase1_baselines/        # Phase 1 baseline comparisons
│   └── archive/                 # Archived backtest results
└── models/                      # Model training and evaluation results
    ├── lora/                    # LoRA model results and metrics
    ├── ramt/                    # RAMT model states and artifacts
    └── explainability/          # Model explainability results
```

## Categories

### Backtesting Results (`backtesting/`)
Contains all portfolio backtesting results across different strategies and time periods:
- **final_strategy/**: Main momentum + HMM strategy results
- **hybrid_lora/**: Hybrid strategy combining RAMT + LoRA + HMM
- **historical_2012/**: Pre-2012 blind backtest validation
- **hmm_ablation/**: HMM regime vs flat sizing comparisons
- **ablation/**: Various component ablation studies
- **phase1_baselines/**: Early phase baseline comparisons
- **archive/**: Older archived results

### Model Results (`models/`)
Contains trained model artifacts and evaluation metrics:
- **lora/**: LoRA adapter weights, predictions, and metrics
- **ramt/**: RAMT model states, scalers, and training artifacts
- **explainability/**: Feature importance and explainability analyses

## Key Files

- `ablation_report.csv`: Summary of all ablation study results
- `ablation_summary*.json`: Detailed ablation metrics
- `summary_infographic.png`: Visual summary of project results

## Usage

Results are organized by phase and methodology:
1. **Phase 1**: Baseline momentum strategies (see `phase1_baselines/`)
2. **Phase 2**: RAMT model integration (see `models/ramt/`)
3. **Phase 3**: LoRA fine-tuning (see `models/lora/`)
4. **Final**: Hybrid strategy with all components (see `backtesting/final_strategy/`)

Each subdirectory contains CSV files with detailed results, JSON files with metrics, and relevant visualizations.
