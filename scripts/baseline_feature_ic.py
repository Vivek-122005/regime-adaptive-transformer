"""
Baseline diagnostic: do the features have cross-sectional predictive power?

Trains a simple LightGBM on the same features RAMT uses. If this gets meaningful
IC, the features work and RAMT is broken. If this also gets near-zero IC, the
features/target themselves lack signal.
"""

from __future__ import annotations
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURES = [
    "Ret_1d",
    "Ret_5d",
    "Ret_21d",
    "RSI_14",
    "BB_Dist",
    "Volume_Surge",
    "Macro_INDIAVIX_Ret1d_L1",
    "Macro_CRUDE_Ret1d_L1",
    "Macro_USDINR_Ret1d_L1",
    "Macro_SP500_Ret1d_L1",
]

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"


def load_panel():
    files = [
        f
        for f in Path("data/processed").glob("*_features.parquet")
        if not f.stem.startswith("_")
    ]
    print(f"Loading {len(files)} tickers...")
    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        dfs.append(d)
    panel = pd.concat(dfs, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    return panel


def choose_target(panel):
    if "Sector_Alpha" in panel.columns and panel["Sector_Alpha"].notna().any():
        return "Sector_Alpha"
    return "Monthly_Alpha"


def evaluate_predictions(test_df, target, pred_col="pred"):
    ics, spreads, top5_pos_rates = [], [], []
    for d, g in test_df.groupby("Date"):
        if len(g) < 10 or g[target].std() == 0:
            continue
        ic, _ = spearmanr(g[pred_col], g[target])
        ics.append(ic)
        top5 = g.nlargest(5, pred_col)
        bot5 = g.nsmallest(5, pred_col)
        spreads.append(top5[target].mean() - bot5[target].mean())
        top5_pos_rates.append((top5[target] > 0).mean())
    ics = np.array(ics)
    spreads = np.array(spreads)
    top5_pos = np.array(top5_pos_rates)
    return ics, spreads, top5_pos


def main():
    panel = load_panel()
    target = choose_target(panel)
    print(f"Target column: {target}")
    panel = panel.dropna(subset=FEATURES + [target])
    train = panel[panel["Date"] <= TRAIN_END].copy()
    test = panel[panel["Date"] >= TEST_START].copy()
    print(f"Train rows: {len(train):,}   Test rows: {len(test):,}")
    print(f"Train dates: {train['Date'].nunique():,}  Test dates: {test['Date'].nunique():,}")

    # Try LightGBM first; fall back to Ridge if not installed
    try:
        import lightgbm as lgb

        print("\nFitting LightGBM (500 trees, depth 5)...")
        model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            random_state=42,
            verbose=-1,
        )
        model.fit(train[FEATURES], train[target])
        test["pred"] = model.predict(test[FEATURES])
        model_name = "LightGBM"
    except ImportError:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import RobustScaler

        print("\nLightGBM not installed, using Ridge regression instead...")
        sc = RobustScaler()
        Xtr = sc.fit_transform(train[FEATURES])
        Xte = sc.transform(test[FEATURES])
        model = Ridge(alpha=1.0)
        model.fit(Xtr, train[target])
        test["pred"] = model.predict(Xte)
        model_name = "Ridge"

    # Also do a pure-momentum baseline (use Ret_21d as the score)
    test["momentum_pred"] = test["Ret_21d"]

    print("\n" + "=" * 60)
    print(f"BASELINE: {model_name}")
    print("=" * 60)
    ics, spreads, top5_pos = evaluate_predictions(test, target, "pred")
    print(f"Mean IC (Spearman, cross-sectional): {ics.mean():+.4f}")
    print(f"Median IC:                            {np.median(ics):+.4f}")
    print(f"% dates with IC > 0:                  {(ics > 0).mean() * 100:.1f}%")
    print(f"IR (mean/std * sqrt(N)):              {ics.mean() / ics.std() * np.sqrt(len(ics)):+.2f}")
    print(f"Top5 - Bot5 mean alpha spread:        {spreads.mean() * 100:+.2f}% per month")
    print(f"Top5 positive alpha rate:             {top5_pos.mean() * 100:.1f}%")

    print("\n" + "=" * 60)
    print("SANITY: Pure Ret_21d momentum (no model)")
    print("=" * 60)
    ics_m, spreads_m, top5_pos_m = evaluate_predictions(test, target, "momentum_pred")
    print(f"Mean IC:                              {ics_m.mean():+.4f}")
    print(f"Top5 - Bot5 spread:                   {spreads_m.mean() * 100:+.2f}% per month")
    print(f"Top5 positive alpha rate:             {top5_pos_m.mean() * 100:.1f}%")

    print("\n" + "=" * 60)
    print("RAMT (from ranking_predictions.csv, for comparison)")
    print("=" * 60)
    try:
        preds = pd.read_csv("results/ranking_predictions.csv", parse_dates=["Date"])
        # Panel files use yfinance-style tickers (e.g. SBIN.NS); exports may use SBIN_NS
        preds = preds.copy()
        preds["Ticker"] = preds["Ticker"].str.replace("_NS", ".NS", regex=False)
        rt = preds[preds["Period"] == "Test"].merge(
            test[["Date", "Ticker", target]], on=["Date", "Ticker"], how="inner"
        )
        rt = rt.rename(columns={"predicted_alpha": "pred"})
        ics_r, spreads_r, top5_pos_r = evaluate_predictions(rt, target, "pred")
        print(f"Mean IC:                              {ics_r.mean():+.4f}")
        print(f"Top5 - Bot5 spread:                   {spreads_r.mean() * 100:+.2f}% per month")
        print(f"Top5 positive alpha rate:             {top5_pos_r.mean() * 100:.1f}%")
    except Exception as e:
        print(f"Could not load RAMT predictions: {e}")


if __name__ == "__main__":
    main()
