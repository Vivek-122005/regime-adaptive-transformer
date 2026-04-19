"""
Build a 'predictions' file using pure Ret_21d momentum as the signal.
Drop-in replacement for ranking_predictions.csv so the existing backtest runs unchanged.
"""

from pathlib import Path

import pandas as pd

files = [
    f
    for f in Path("data/processed").glob("*_features.parquet")
    if not f.stem.startswith("_")
]
dfs = []
for f in files:
    d = pd.read_parquet(
        f, columns=["Date", "Ticker", "Ret_21d", "Sector_Alpha", "Monthly_Alpha"]
    )
    dfs.append(d)
panel = pd.concat(dfs, ignore_index=True)
panel["Date"] = pd.to_datetime(panel["Date"])

# Use same target as RAMT
target = "Sector_Alpha" if panel["Sector_Alpha"].notna().any() else "Monthly_Alpha"
panel = panel.dropna(subset=["Ret_21d", target])

# Filter to rebalance dates (same ones as RAMT produced)
try:
    ramt = pd.read_csv("results/ranking_predictions.csv", parse_dates=["Date"])
    rebal_dates = sorted(ramt["Date"].unique())
    panel = panel[panel["Date"].isin(rebal_dates)]
except FileNotFoundError:
    pass

panel["predicted_alpha"] = panel["Ret_21d"]  # pure momentum signal
panel["actual_alpha"] = panel[target]
panel["Period"] = panel["Date"].apply(
    lambda d: "Test" if d >= pd.Timestamp("2023-01-01") else "Train"
)

out = panel[["Date", "Ticker", "predicted_alpha", "actual_alpha", "Period"]]
out = out.sort_values(["Date", "predicted_alpha"], ascending=[True, False])
out.to_csv("results/ranking_predictions.csv", index=False)
print(f"Wrote {len(out):,} predictions across {out['Date'].nunique()} rebalance dates.")
print("Now run: python models/run_final_2024_2026.py --backtest-only")
