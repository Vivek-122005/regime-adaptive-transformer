"""
Final split runner (no future leakage)

Train: 2015-2023
Test/backtest: 2024-2026 (until available data end)

Produces:
- results/ranking_predictions.csv
- results/monthly_rankings.csv
- results/backtest_results.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.backtest import run_backtest_daily  # noqa: E402
from models.ramt import train_ranking as tr  # noqa: E402


def add_momentum_column(rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a momentum score for dashboard display.
    Uses RelMom_12_1 if present, else RelMom_252d, else 0.
    """
    if rankings.empty:
        rankings["momentum"] = []
        return rankings

    # Lazy per-ticker cache of processed feature frames
    cache: dict[str, pd.DataFrame] = {}
    moms = []
    for _, row in rankings.iterrows():
        d = pd.to_datetime(row["Date"])
        t = row["Ticker"]
        if t not in cache:
            p = ROOT / "data/processed" / f"{t}_features.csv"
            df = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
            df = df.set_index("Date")
            cache[t] = df
        df = cache[t]
        # last available on/before date
        sub = df.loc[:d]
        if sub.empty:
            moms.append(0.0)
            continue
        last = sub.iloc[-1]
        if "RelMom_12_1" in last:
            moms.append(float(last["RelMom_12_1"]))
        elif "RelMom_252d" in last:
            moms.append(float(last["RelMom_252d"]))
        else:
            moms.append(0.0)

    rankings = rankings.copy()
    rankings["momentum"] = moms
    return rankings


def main():
    os.makedirs("results", exist_ok=True)

    # Strict split
    train_start = "2015-01-01"
    train_end = "2023-12-31"
    test_start = "2024-01-01"
    test_end = "2026-01-01"

    # Training hyperparams (adjust up for a true final run)
    tr.MAX_EPOCHS = 10
    tr.PATIENCE = 3
    tr.BATCH_SIZE = 128

    print("Training fixed combined model (no leakage) and predicting test period...", flush=True)
    preds = tr.train_fixed_and_predict(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        step_size=21,
        max_epochs=tr.MAX_EPOCHS,
    )

    preds_out = ROOT / "results/ranking_predictions.csv"
    preds.to_csv(preds_out, index=False)
    print(f"Saved: {preds_out}", flush=True)

    rankings = preds.rename(columns={"predicted_alpha": "score"})[
        ["Date", "Ticker", "score", "actual_alpha", "fold_train_end"]
    ]
    rankings = add_momentum_column(rankings)
    rankings_out = ROOT / "results/monthly_rankings.csv"
    rankings.to_csv(rankings_out, index=False)
    print(f"Saved: {rankings_out}", flush=True)

    print("Running daily-price backtest with risk rules...", flush=True)
    bt = run_backtest_daily(
        predictions_df=preds[["Date", "Ticker", "predicted_alpha", "actual_alpha"]],
        nifty_features_path=str(ROOT / "data/processed/NIFTY50_features.csv"),
        raw_dir=str(ROOT / "data/raw"),
        start=test_start,
        end=test_end,
        step_size=21,
        top_n=5,
        capital=100000,
        stop_loss=0.07,
        max_weight=0.20,
        portfolio_dd_cash_trigger=0.15,
    )
    bt_out = ROOT / "results/backtest_results.csv"
    bt.to_csv(bt_out, index=False)
    print(f"Saved: {bt_out}", flush=True)

    print("\nDone. View results in Streamlit:", flush=True)
    print("  streamlit run dashboard/app.py", flush=True)


if __name__ == "__main__":
    main()

