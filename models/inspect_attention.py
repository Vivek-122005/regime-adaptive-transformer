"""
Inspect RAMT attention over the 30-day input window.

This script loads:
  - results/ramt_model_state.pt
  - results/ramt_scaler.joblib
and produces:
  - results/attention_last_token.csv  (importance of each day to the last day)
  - results/attention_map_mean.csv    (seq_len x seq_len mean attention)

Usage:
  .venv/bin/python models/inspect_attention.py --ticker TCS_NS --date 2024-10-09
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import plotly.express as px

from models.ramt.dataset import ALL_FEATURE_COLS, TICKER_TO_ID
from models.ramt.model import build_ramt


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


def _load_processed(ticker: str) -> pd.DataFrame:
    p = ROOT / "data" / "processed" / f"{ticker}_features.csv"
    df = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
    df = df.set_index("Date")
    return df


def _extract_sequence(
    df: pd.DataFrame, d: pd.Timestamp, seq_len: int, scaler
) -> tuple[np.ndarray, int]:
    # last available on/before d
    sub = df.loc[:d]
    if len(sub) < seq_len + 1:
        raise ValueError(f"Not enough history before {d.date()} (need >= {seq_len+1} rows)")
    X = sub[ALL_FEATURE_COLS].values.astype(np.float32)
    # use last seq_len rows ending at d (exclusive of target, but we only need inputs)
    Xseq = X[-seq_len:, :]
    Xseq_sc = scaler.transform(Xseq)
    regime = int(np.round(float(sub["HMM_Regime"].iloc[-1])))
    regime = int(np.clip(regime, 0, 2))
    return Xseq_sc.astype(np.float32), regime


def _mean_attention_from_model(model) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      attn_mean: (seq_len, seq_len) mean attention across experts/layers/heads
      last_token: (seq_len,) attention from last timestep to each source timestep
    """
    # Collect from MoE experts
    attn_by_expert = model.moe.get_last_attention()  # expert -> [layer tensors]
    mats = []
    last_rows = []
    for layers in attn_by_expert:
        for w in layers:
            # w: (batch, heads, tgt_len, src_len)
            w0 = w[0].detach().cpu().numpy()  # (heads, tgt, src)
            w0m = w0.mean(axis=0)  # (tgt, src)
            mats.append(w0m)
            last_rows.append(w0m[-1, :])
    if not mats:
        raise RuntimeError(
            "No attention weights captured. Make sure the model was built with explainable_attn=True."
        )
    attn_mean = np.mean(np.stack(mats, axis=0), axis=0)
    last_token = np.mean(np.stack(last_rows, axis=0), axis=0)
    return attn_mean, last_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, type=str)
    ap.add_argument("--date", required=True, type=str, help="rebalance date like 2024-10-09")
    ap.add_argument("--seq-len", default=30, type=int)
    ap.add_argument(
        "--out-prefix",
        default="attention",
        type=str,
        help="output filename prefix inside results/",
    )
    args = ap.parse_args()

    ticker = args.ticker
    d = pd.Timestamp(args.date)
    seq_len = int(args.seq_len)
    out_prefix = str(args.out_prefix).strip() or "attention"

    state_path = ROOT / "results" / "ramt_model_state.pt"
    scaler_path = ROOT / "results" / "ramt_scaler.joblib"
    if not state_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Missing artifacts. Run `python models/run_final_2024_2026.py` first "
            "to generate results/ramt_model_state.pt and results/ramt_scaler.joblib."
        )

    payload = torch.load(state_path, map_location="cpu")
    scaler = joblib.load(scaler_path)

    df = _load_processed(ticker)
    Xseq_sc, regime = _extract_sequence(df, d, seq_len, scaler)

    # Build explainable model and load weights
    model = build_ramt({"seq_len": seq_len, "explainable_attn": True})
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    X = torch.from_numpy(Xseq_sc).unsqueeze(0)  # (1, seq, feat)
    r = torch.tensor([regime], dtype=torch.long)
    tid = torch.tensor([int(TICKER_TO_ID.get(ticker, 0))], dtype=torch.long)

    with torch.no_grad():
        _pred_m, _pred_d, _g = model(X, r, ticker_id=tid)

    attn_mean, last_token = _mean_attention_from_model(model)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save matrix and last-token vector
    map_path = out_dir / f"{out_prefix}_map_mean.csv"
    last_path = out_dir / f"{out_prefix}_last_token.csv"
    pd.DataFrame(attn_mean).to_csv(map_path, index=False)
    pd.DataFrame({"day_index": list(range(seq_len)), "attn_from_last_token": last_token}).to_csv(
        last_path, index=False
    )

    # Plotly heatmap (png + html)
    fig = px.imshow(
        attn_mean,
        color_continuous_scale="Viridis",
        origin="lower",
        aspect="auto",
        title=f"RAMT Attention Map (mean) — {ticker} @ {d.date()}",
        labels=dict(x="source_day", y="target_day", color="attn"),
    )
    html_path = out_dir / f"{out_prefix}_heatmap.html"
    png_path = out_dir / f"{out_prefix}_heatmap.png"
    fig.write_html(html_path)
    # PNG export requires kaleido; if not available, keep HTML only.
    try:
        fig.write_image(png_path, width=900, height=750, scale=2)
        png_msg = f"  {png_path}"
    except Exception:
        png_msg = "  (PNG export skipped: install `kaleido` if you want PNGs)"

    print("Saved:")
    print(f"  {map_path}")
    print(f"  {last_path}")
    print(f"  {html_path}")
    print(png_msg)
    print("Tip: day_index 0 is oldest in the 30-day window; day_index 29 is most recent.")


if __name__ == "__main__":
    main()

