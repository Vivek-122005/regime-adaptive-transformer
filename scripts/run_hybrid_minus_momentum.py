"""
User Story:
As a researcher closing the Phase 3 ablation, I need the missing "Hybrid - Momentum"
row regenerated so the ablation table is complete. This row uses Chronos-T5 (LoRA)
as the only alpha signal, with HMM regime sizing kept on (the "sizing-only" leg of
the hybrid). It complements:
  - Hybrid - HMM      (alias: Simple Hybrid 50/50, removes the regime gate)
  - Hybrid - Chronos  (alias: Mom+HMM production, removes the foundation model)
  - Hybrid - Momentum (THIS, removes the momentum signal, keeps Chronos + HMM)

Implementation Approach:
Reuse models/ablation_engine.compute_metrics and backtest.core.backtest.run_backtest_daily
exactly as the other scenarios used. Load Chronos predictions, rename columns to
the schema the backtester expects, run with flat_regime_sizing=False, then patch
results/ablation_summary.json in place (preserving every other field).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.core.backtest import run_backtest_daily  # noqa: E402


def compute_metrics(bt: pd.DataFrame, capital: float) -> dict[str, float]:
    """Inlined from models.ablation_engine (which has a stale import). Identical math."""
    if bt.empty or len(bt) < 2:
        return {"CAGR": 0.0, "Sharpe_Net": 0.0, "Max_Drawdown": 0.0, "Win_Rate": 0.0}
    r = bt["portfolio_return"].astype(float).fillna(0.0)
    nav = bt["portfolio_value"].astype(float).to_numpy()
    start_ts = pd.to_datetime(bt["date"].iloc[0])
    end_ts = pd.to_datetime(bt["date"].iloc[-1])
    years = max((end_ts - start_ts).days / 365.25, 1e-9)
    total_ret = float(nav[-1] / float(capital) - 1.0)
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(12.0))
    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min())
    win_rate = float((r > 0).mean())
    return {
        "CAGR": cagr,
        "Sharpe_Net": sharpe,
        "Max_Drawdown": max_dd,
        "Win_Rate": win_rate,
    }

CHRONOS_PREDS = ROOT / "results" / "models" / "lora" / "lora_predictions.csv"
NIFTY_FEATURES = ROOT / "data" / "processed" / "_NSEI_features.parquet"
RAW_DIR = ROOT / "data" / "raw"
ABLATION_SUMMARY = ROOT / "results" / "ablation_summary.json"
BT_OUTPUT = ROOT / "results" / "ablation" / "backtest_hybrid_minus_momentum.csv"
PRED_OUTPUT = ROOT / "results" / "ablation" / "predictions_hybrid_minus_momentum.csv"
PRODUCTION_BT = ROOT / "results" / "backtesting" / "final_strategy" / "backtest_results.csv"


def _load_production_rebalance_dates(path: Path) -> list[pd.Timestamp]:
    """Read the production strategy's rebalance dates so this row is cadence-comparable."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"]).dt.tz_localize(None).sort_values().unique().tolist()
    return [pd.Timestamp(d) for d in dates]


def _snap_to_monthly_rebalances(preds: pd.DataFrame, anchors: list[pd.Timestamp]) -> pd.DataFrame:
    """Keep only the prediction rows whose Date matches a production rebalance date.

    For each anchor date we keep predictions on that exact date if available,
    otherwise the latest available prediction strictly before the anchor.
    """
    if not anchors:
        return preds
    preds = preds.sort_values("Date")
    keep_dates: set[pd.Timestamp] = set()
    available = sorted(preds["Date"].unique())
    for anchor in anchors:
        if anchor in available:
            keep_dates.add(anchor)
            continue
        # snap back to last available date <= anchor
        prior = [d for d in available if d <= anchor]
        if prior:
            keep_dates.add(prior[-1])
    return preds[preds["Date"].isin(keep_dates)].reset_index(drop=True)


def _load_chronos_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    if "predicted" in df.columns and "predicted_alpha" not in df.columns:
        rename["predicted"] = "predicted_alpha"
    if "actual" in df.columns and "actual_alpha" not in df.columns:
        rename["actual"] = "actual_alpha"
    if rename:
        df = df.rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().str.replace(".", "_", regex=False)
    df["predicted_alpha"] = pd.to_numeric(df["predicted_alpha"], errors="coerce")
    df["actual_alpha"] = pd.to_numeric(df["actual_alpha"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["predicted_alpha"])
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def _patch_ablation_summary(metrics: dict, start: str, end: str) -> None:
    if not ABLATION_SUMMARY.exists():
        raise SystemExit(f"Missing {ABLATION_SUMMARY.relative_to(ROOT)}")

    with ABLATION_SUMMARY.open() as f:
        summary = json.load(f)

    new_row = {
        "scenario": "Hybrid minus Momentum (Chronos + HMM sizing only)",
        "key": "hybrid_minus_momentum",
        "CAGR": float(metrics["CAGR"]),
        "Sharpe_Net": float(metrics["Sharpe_Net"]),
        "Max_Drawdown": float(metrics["Max_Drawdown"]),
        "Win_Rate": float(metrics["Win_Rate"]),
        "source": "scripts/run_hybrid_minus_momentum.py",
        "window": f"{start} to {end}",
        "interpretation": (
            "Chronos-T5 (LoRA) as the sole alpha signal with HMM regime sizing kept on. "
            "Compared against Foundation Only (Chronos with flat_regime_sizing=True), this "
            "isolates the cost/benefit of the HMM gate when momentum is removed."
        ),
    }

    scenarios = summary.get("scenarios", [])
    replaced = False
    for i, row in enumerate(scenarios):
        if row.get("key") == "hybrid_minus_momentum":
            scenarios[i] = new_row
            replaced = True
            break
    if not replaced:
        scenarios.append(new_row)
    summary["scenarios"] = scenarios

    notes = summary.get("_notes", [])
    drop_note_marker = "hybrid_minus_momentum is not_run"
    notes = [n for n in notes if drop_note_marker not in n]
    notes.append(
        "hybrid_minus_momentum was regenerated by scripts/run_hybrid_minus_momentum.py "
        "(Chronos predictions + HMM regime sizing, no momentum signal). It complements the "
        "Foundation Only row (same Chronos signal, no HMM)."
    )
    summary["_notes"] = notes

    with ABLATION_SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-04-15")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics without patching the JSON.")
    args = parser.parse_args()

    if not CHRONOS_PREDS.exists():
        raise SystemExit(f"Missing Chronos predictions at {CHRONOS_PREDS.relative_to(ROOT)}")
    if not NIFTY_FEATURES.exists():
        raise SystemExit(f"Missing NIFTY features at {NIFTY_FEATURES.relative_to(ROOT)}")

    print(f"[info] Loading Chronos predictions from {CHRONOS_PREDS.relative_to(ROOT)}")
    preds = _load_chronos_predictions(CHRONOS_PREDS)
    print(f"[info] {len(preds)} rows, {preds['Ticker'].nunique()} tickers, "
          f"{preds['Date'].min().date()} → {preds['Date'].max().date()}")

    anchors = _load_production_rebalance_dates(PRODUCTION_BT)
    if anchors:
        print(f"[info] Snapping to {len(anchors)} production rebalance dates "
              f"({anchors[0].date()} → {anchors[-1].date()}) for cadence comparability.")
        preds = _snap_to_monthly_rebalances(preds, anchors)
        print(f"[info] After snap: {len(preds)} rows on {preds['Date'].nunique()} rebalance dates")
    else:
        print(f"[warn] No production rebalance dates available; running daily cadence.")

    print("[info] Running backtest: Chronos signal, HMM regime sizing ON (flat=False), "
          "top_n=%d, capital=%s, friction=0.22%%" % (args.top_n, f"{args.capital:,.0f}"))
    bt = run_backtest_daily(
        predictions_df=preds[["Date", "Ticker", "predicted_alpha", "actual_alpha"]],
        nifty_features_path=str(NIFTY_FEATURES),
        raw_dir=str(RAW_DIR),
        start=args.start,
        end=args.end,
        top_n=int(args.top_n),
        capital=float(args.capital),
        stop_loss=0.07,
        stop_loss_bear=0.05,
        max_weight=0.25,
        portfolio_dd_cash_trigger=0.15,
        rebalance_friction_rate=0.0022,
        turnover_penalty_score=0.0,
        kelly_p=0.5238,
        kelly_use_predicted_margin=True,
        kelly_scale_position=True,
        use_sector_cap=True,
        flat_regime_sizing=False,
    )

    metrics = compute_metrics(bt, capital=float(args.capital))
    print("\n=== Hybrid - Momentum (Chronos + HMM sizing only) ===")
    print(f"  Window:        {args.start} → {args.end}")
    print(f"  Rebalances:    {len(bt)}")
    print(f"  CAGR:          {metrics['CAGR']:.4f}  ({metrics['CAGR']*100:+.2f}%)")
    print(f"  Sharpe (net):  {metrics['Sharpe_Net']:.4f}")
    print(f"  Max DD:        {metrics['Max_Drawdown']:.4f}  ({metrics['Max_Drawdown']*100:+.2f}%)")
    print(f"  Win rate:      {metrics['Win_Rate']:.4f}  ({metrics['Win_Rate']*100:.2f}%)")

    if args.dry_run:
        print("\n[dry-run] Skipping JSON patch and CSV writes.")
        return

    BT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(BT_OUTPUT, index=False)
    preds[["Date", "Ticker", "predicted_alpha", "actual_alpha"]].to_csv(PRED_OUTPUT, index=False)
    print(f"\n[info] Wrote {BT_OUTPUT.relative_to(ROOT)}")
    print(f"[info] Wrote {PRED_OUTPUT.relative_to(ROOT)}")

    _patch_ablation_summary(metrics, args.start, args.end)
    print(f"[info] Patched {ABLATION_SUMMARY.relative_to(ROOT)} (hybrid_minus_momentum row replaced)")


if __name__ == "__main__":
    main()
