"""
Permutation importance audit for RAMT features (monthly head).

Goal:
  Identify features that do NOT help (e.g. some macro series),
  so we can drop them and reduce noise + compute.

How it works:
  1) Load trained model + scaler artifacts from results/ramt/
  2) Build a validation sample set from the last part of the training period
  3) Compute baseline metric (Spearman IC between pred_monthly and y_monthly)
  4) For each feature column:
       shuffle that feature across samples (for each timestep) and recompute metric
       importance = baseline - shuffled

Usage:
  .venv/bin/python models/permutation_importance.py --target Monthly_Alpha_Z --max-samples 8000
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

from models.ramt.dataset import ALL_FEATURE_COLS  # noqa: E402
from models.ramt.model import build_ramt  # noqa: E402
from models.ramt.train_ranking import (  # noqa: E402
    SEQ_LEN,
    TickerData,
    _apply_scaler,
    _build_sample_keys,
    _fit_scaler_on_train,
    _load_all_tickers,
)


def _spearman_ic(a: np.ndarray, b: np.ndarray) -> float:
    # Spearman via rank correlation
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    if np.std(ar) < 1e-12 or np.std(br) < 1e-12:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


@torch.no_grad()
def _predict_monthly(model, X: np.ndarray, regime: np.ndarray, ticker_id: np.ndarray) -> np.ndarray:
    model.eval()
    bs = 256
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs]).float()
        rb = torch.from_numpy(regime[i : i + bs]).long()
        tb = torch.from_numpy(ticker_id[i : i + bs]).long()
        pm, _pd, _g = model(xb, rb, ticker_id=tb)
        out.append(pm.cpu().numpy().squeeze())
    return np.concatenate(out)


def _make_val_batch(
    data: dict[str, TickerData],
    target_col: str,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Use the last ~120 calendar days of the training period as a simple val window.
    # (We keep it cheap and purely for feature audit.)
    train_end = pd.Timestamp("2023-12-31")
    val_start = train_end - pd.Timedelta(days=180)

    keys = []
    for _t, td in data.items():
        keys.extend(_build_sample_keys(td, val_start, train_end + pd.Timedelta(days=1), SEQ_LEN))
    keys = sorted(keys)
    if max_samples > 0:
        keys = keys[-max_samples:]

    Xseq = []
    y = []
    r = []
    tid = []

    for t, i in keys:
        td = data[t]
        Xseq.append(td.X[i - SEQ_LEN : i])
        y.append(td.y_monthly[i])
        r.append(int(td.regime[i]))
        tid.append(int(td.ticker_id))

    Xseq = np.asarray(Xseq, dtype=np.float32)  # (N, seq, F)
    y = np.asarray(y, dtype=np.float32)
    r = np.asarray(r, dtype=np.int64)
    tid = np.asarray(tid, dtype=np.int64)
    return Xseq, y, r, tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Monthly_Alpha_Z", choices=["Monthly_Alpha", "Monthly_Alpha_Z"])
    ap.add_argument("--max-samples", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    state_path = ROOT / "results" / "ramt" / "ramt_model_state.pt"
    scaler_path = ROOT / "results" / "ramt" / "ramt_scaler.joblib"
    if not state_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Missing results artifacts. Run models/run_final_2024_2026.py first.")

    # Load data
    data = _load_all_tickers("data/processed")

    # Patch monthly target to requested (keeps daily head intact)
    patched = {}
    for t, td in data.items():
        p = ROOT / "data" / "processed" / f"{t}_features.csv"
        df = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").set_index("Date", drop=True)
        df = df.dropna(subset=[args.target, "Daily_Return"])
        patched[t] = TickerData(
            ticker=t,
            ticker_id=td.ticker_id,
            dates=pd.DatetimeIndex(df.index),
            X=df[list(ALL_FEATURE_COLS)].values.astype(np.float32),
            y_monthly=df[args.target].values.astype(np.float32),
            y_daily=df["Daily_Return"].values.astype(np.float32),
            regime=df["HMM_Regime"].values.astype(np.int64),
        )
    data = patched

    # Use scaler from training artifacts if present; otherwise fit on a broad train window.
    scaler = joblib.load(scaler_path)
    data_sc = _apply_scaler(data, scaler)

    # Load model
    payload = torch.load(state_path, map_location="cpu")
    model = build_ramt({"seq_len": SEQ_LEN})
    model.load_state_dict(payload["model_state_dict"], strict=True)

    Xseq, y, r, tid = _make_val_batch(data_sc, args.target, args.max_samples)

    # Baseline
    baseline_pred = _predict_monthly(model, Xseq, r, tid)
    baseline_ic = _spearman_ic(baseline_pred, y)
    print(f"Baseline Spearman IC: {baseline_ic:.4f}  (N={len(y)})")

    results = []
    F = Xseq.shape[-1]
    for j in range(F):
        Xp = Xseq.copy()
        # shuffle feature j independently per timestep
        for t_step in range(Xp.shape[1]):
            np.random.shuffle(Xp[:, t_step, j])
        p_shuf = _predict_monthly(model, Xp, r, tid)
        ic_shuf = _spearman_ic(p_shuf, y)
        results.append(
            {
                "feature": ALL_FEATURE_COLS[j],
                "baseline_ic": baseline_ic,
                "shuffled_ic": ic_shuf,
                "ic_drop": baseline_ic - ic_shuf,
            }
        )

    out = pd.DataFrame(results).sort_values("ic_drop", ascending=False)
    out_path = ROOT / "results" / "ramt" / "permutation_importance.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("\nTop 15 important features (by IC drop):")
    print(out.head(15)[["feature", "ic_drop", "shuffled_ic"]].to_string(index=False))
    print("\nBottom 15 (likely noise):")
    print(out.tail(15)[["feature", "ic_drop", "shuffled_ic"]].to_string(index=False))


if __name__ == "__main__":
    main()

