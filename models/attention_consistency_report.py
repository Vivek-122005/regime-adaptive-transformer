"""
Generate an attention consistency report across many rebalance dates.

This answers: "Is the model consistently focusing on similar days, or is it random?"

Outputs:
  - results/ramt/attention/attention_consistency.csv

Usage:
  .venv/bin/python models/attention_consistency_report.py --ticker TCS_NS --n 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.inspect_attention import _extract_sequence, _load_processed, _mean_attention_from_model  # noqa: E402
from models.ramt.dataset import TICKER_TO_ID  # noqa: E402
from models.ramt.model import build_ramt  # noqa: E402


def _rebalance_dates_from_predictions() -> list[pd.Timestamp]:
    p = ROOT / "results" / "final_strategy" / "ranking_predictions.csv"
    df = pd.read_csv(p, parse_dates=["Date"])
    dates = sorted(pd.to_datetime(df["Date"]).dropna().unique().tolist())
    return [pd.Timestamp(d) for d in dates]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, type=str)
    ap.add_argument("--n", default=12, type=int, help="number of rebalance dates to sample (from the end)")
    ap.add_argument("--seq-len", default=30, type=int)
    args = ap.parse_args()

    ticker = args.ticker
    n = int(args.n)
    seq_len = int(args.seq_len)

    state_path = ROOT / "results" / "ramt" / "ramt_model_state.pt"
    scaler_path = ROOT / "results" / "ramt" / "ramt_scaler.joblib"
    if not state_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Missing artifacts. Run `python models/run_final_2024_2026.py` first."
        )

    payload = torch.load(state_path, map_location="cpu")
    scaler = joblib.load(scaler_path)

    model = build_ramt({"seq_len": seq_len, "explainable_attn": True})
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    df = _load_processed(ticker)
    dates = _rebalance_dates_from_predictions()
    dates = dates[-n:] if n > 0 else dates

    rows = []
    for d in dates:
        try:
            Xseq_sc, regime = _extract_sequence(df, d, seq_len, scaler)
        except Exception:
            continue

        X = torch.from_numpy(Xseq_sc).unsqueeze(0)
        r = torch.tensor([int(regime)], dtype=torch.long)
        tid = torch.tensor([int(TICKER_TO_ID.get(ticker, 0))], dtype=torch.long)

        with torch.no_grad():
            _pred_m, _pred_d, _g = model(X, r, ticker_id=tid)
        _attn_mean, last_token = _mean_attention_from_model(model)

        # summarize peaks
        top3 = np.argsort(-last_token)[:3].tolist()
        rows.append(
            {
                "Date": d.date().isoformat(),
                "top_day_1": int(top3[0]),
                "top_day_2": int(top3[1]),
                "top_day_3": int(top3[2]),
                "mass_last_5_days": float(last_token[-5:].sum()),
                "mass_last_10_days": float(last_token[-10:].sum()),
                "entropy": float(-(last_token * np.log(last_token + 1e-12)).sum()),
            }
        )

    out = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "ramt" / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attention_consistency.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

