#!/usr/bin/env python3
"""Verify that key results/ paths referenced by the reorganized layout exist."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    ROOT / "results/final_strategy/backtest_results.csv",
    ROOT / "results/final_strategy/ranking_predictions.csv",
    ROOT / "results/final_strategy/monthly_rankings.csv",
    ROOT / "results/ramt/ramt_model_state.pt",
    ROOT / "results/ramt/training_dashboard.png",
    ROOT / "results/phase1_baselines/xgboost_predictions.csv",
    ROOT / "results/phase1_baselines/lstm_predictions.csv",
    ROOT / "results/hmm_ablation/2008_2010/2008-01-01_2010-12-31/hmm_vs_flat_summary.csv",
    ROOT / "results/lightgbm/.gitkeep",
    ROOT / "results/README_PATHS.md",
]


def main() -> None:
    missing = [p for p in REQUIRED if not p.is_file()]
    if missing:
        print("Missing expected files:")
        for p in missing:
            print(f"  {p.relative_to(ROOT)}")
        raise SystemExit(1)
    print(f"OK: {len(REQUIRED)} paths exist under results/.")


if __name__ == "__main__":
    main()
