#!/usr/bin/env python3
"""
Regenerate RAMT walk-forward predictions and backtest from saved checkpoints only (no training).

Mirrors ``combined_walk_forward`` calendar logic in ``models/ramt/train_ranking.py``:
for each segment, load ``ramt_model_state_wf_seg_XX.pt`` + matching scalers and run
``_predict_rows_for_dates`` on that segment's rebalance grid.

Outputs:
  - results/ramt/ranking_predictions.csv
  - results/ramt/ramt_metrics.json
  - results/ramt/backtest_results.csv
  - copies training_dashboard.png into results/ramt/ when present in --artifact-dir
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib  # noqa: E402
import torch  # noqa: E402

from models.backtest import run_backtest_daily  # noqa: E402
from models.ramt.model import build_ramt  # noqa: E402
from models.ramt import train_ranking as tr  # noqa: E402
from models.ramt.train_ranking import (  # noqa: E402
    LazyTickerStore,
    _full_nifty_trading_calendar,
    _last_trading_day_before,
    _nifty_raw_path,
    _predict_rows_for_dates,
    _rebalance_dates_21d,
)


def _default_artifact_dir() -> Path:
    cand = ROOT / "models" / "ramt" / "artifacts"
    if cand.is_dir() and any(cand.glob("ramt_model_state_wf_seg_01.pt")):
        return cand
    fall = ROOT / "results" / "ramt"
    if fall.is_dir() and any(fall.glob("ramt_model_state_wf_seg_01.pt")):
        return fall
    return Path("ramt model results")


def _max_wf_seg_index(artifact_dir: Path) -> int:
    tags = list(artifact_dir.glob("ramt_model_state_wf_seg_*.pt"))
    if not tags:
        return 0
    m = 0
    for p in tags:
        mo = re.search(r"wf_seg_(\d+)\.pt$", p.name)
        if mo:
            m = max(m, int(mo.group(1)))
    return m


def infer_training_step(
    *,
    nifty_path: str,
    test_start: str,
    test_end: str,
    n_wf_segments: int,
) -> int:
    """
    Outer walk-forward stride (trading days between segment starts) must yield exactly
    ``n_wf_segments`` starts — must match the number of ``wf_seg_XX`` checkpoints.

    Prefer 252 (annual retrain), then 126 (≈6 months), then the smallest step that fits.
    """
    if n_wf_segments < 1:
        raise ValueError("n_wf_segments must be >= 1")

    def n_for(step: int) -> int:
        return len(_rebalance_dates_21d(nifty_path, test_start, test_end, step_size=int(step)))

    for step in [252, 126, 168, 210, 84, 105]:
        if n_for(step) == n_wf_segments:
            return int(step)
    for step in range(21, 253):
        if n_for(step) == n_wf_segments:
            return int(step)
    got = [n_for(s) for s in (252, 126)]
    raise RuntimeError(
        f"Cannot find training_step in [21,252] that yields exactly {n_wf_segments} WF segments "
        f"for test window {test_start}…{test_end}. (For reference: step 252→{got[0]} segments, "
        f"step 126→{got[1]} segments.) Extend ``--test-end`` / NIFTY history or pass an explicit "
        f"--training-step after checking ``models/ramt/train_ranking.py`` WF logic."
    )


def _resolve_ramt_config(state_dict: dict, raw: dict) -> dict:
    """
    Some older WF checkpoints omit ``num_experts`` / ``num_transformer_layers`` in ``config``
    even though weights use MoE. Infer missing fields from key names so ``build_ramt`` matches.
    """
    base = {
        "seq_len": 30,
        "num_heads": 8,
        "num_transformer_layers": 2,
        "num_experts": 1,
        "dropout": 0.2,
    }
    merged = {**base, **raw}
    keys = list(state_dict.keys())
    experts_idx = {
        int(m.group(1)) for k in keys for m in [re.match(r"^moe\.experts\.(\d+)\.", k)] if m
    }
    if experts_idx:
        merged["num_experts"] = max(experts_idx) + 1
    layer_ids: set[int] = set()
    for k in keys:
        m = re.search(r"\.transformer\.layers\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    if layer_ids and "num_transformer_layers" not in raw:
        merged["num_transformer_layers"] = max(layer_ids) + 1
    return merged


def load_fold(
    artifact_dir: Path,
    fold_tag: str,
) -> tuple[object, object, object, float, float]:
    """Load RAMT + scalers + winsor bounds for one walk-forward fold tag (e.g. wf_seg_01)."""
    pt = artifact_dir / f"ramt_model_state_{fold_tag}.pt"
    if not pt.is_file():
        raise FileNotFoundError(pt)
    payload = torch.load(pt, map_location=tr.DEVICE, weights_only=False)
    cfg = _resolve_ramt_config(payload["model_state_dict"], payload["config"])
    model = build_ramt(cfg).to(tr.DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    scaler = joblib.load(artifact_dir / f"ramt_scaler_{fold_tag}.joblib")
    y_scaler = joblib.load(artifact_dir / f"ramt_y_scaler_{fold_tag}.joblib")
    lo_b = float(payload["y_winsor_lo"])
    hi_b = float(payload["y_winsor_hi"])
    return model, scaler, y_scaler, lo_b, hi_b


def verify_main_checkpoint(artifact_dir: Path) -> dict[str, object]:
    """Load ``ramt_model_state.pt`` and return a small status dict (architecture check)."""
    path = artifact_dir / "ramt_model_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Main checkpoint missing: {path}")
    payload = torch.load(path, map_location=tr.DEVICE, weights_only=False)
    cfg = _resolve_ramt_config(payload["model_state_dict"], payload["config"])
    model = build_ramt(cfg).to(tr.DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    n_params = model.count_parameters()
    return {
        "path": str(path.resolve()),
        "n_parameters": int(n_params),
        "config": payload["config"],
        "resolved_config": cfg,
        "fold_label": payload.get("fold_label"),
    }


def _mean_rank_ic(df: pd.DataFrame) -> float:
    ics: list[float] = []
    for _, g in df.groupby("Date"):
        if len(g) < 4:
            continue
        ic = g["predicted_alpha"].corr(g["actual_alpha"], method="spearman")
        if ic == ic:
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else float("nan")


def _top5_positive_rate(df: pd.DataFrame) -> float:
    hits = 0
    n = 0
    for _, g in df.groupby("Date"):
        g = g.sort_values("predicted_alpha", ascending=False).head(5)
        if len(g) < 1:
            continue
        n += 1
        if float(g["actual_alpha"].mean()) > 0:
            hits += 1
    return float(hits / n) if n else float("nan")


def run_walk_forward_inference(
    artifact_dir: Path,
    *,
    training_step: int,
    rebalance_step: int,
    inference_warmup_days: int,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    store = LazyTickerStore("data/processed", cache_size=200)
    tickers = list(tr.TICKERS)
    if not tickers:
        raise FileNotFoundError("No processed parquets under data/processed.")

    nifty_path = _nifty_raw_path()
    full_cal = _full_nifty_trading_calendar(nifty_path)

    segment_starts = _rebalance_dates_21d(
        nifty_path, test_start, test_end, step_size=int(training_step)
    )
    if len(segment_starts) == 0:
        raise RuntimeError(f"No walk-forward segments for TEST_START={test_start} TEST_END={test_end}")

    n_seg = len(segment_starts)
    n_tag = _max_wf_seg_index(artifact_dir)
    if n_seg != n_tag:
        raise RuntimeError(
            f"Calendar produced {n_seg} WF segments but artifact dir has wf_seg_01..{n_tag:02d}. "
            f"training_step={training_step} test window {test_start}…{test_end} — "
            f"try --training-step or --wf-test-end."
        )
    all_rows: list[dict[str, object]] = []

    for seg_idx, seg_start in enumerate(segment_starts):
        fold_tag = f"wf_seg_{seg_idx + 1:02d}"
        seg_label = f"WF {seg_idx + 1}/{n_seg}"
        model, scaler, y_scaler, lo_b, hi_b = load_fold(artifact_dir, fold_tag)

        seg_end_ts = (
            pd.Timestamp(segment_starts[seg_idx + 1])
            if seg_idx + 1 < len(segment_starts)
            else pd.Timestamp(test_end) + pd.Timedelta(days=1)
        )
        pred_end = _last_trading_day_before(full_cal, seg_end_ts)
        pred_dates = _rebalance_dates_21d(
            nifty_path,
            str(pd.Timestamp(seg_start).date()),
            str(pred_end.date()),
            step_size=int(rebalance_step),
        )

        fold_rows = _predict_rows_for_dates(
            pred_dates,
            store=store,
            tickers=tickers,
            model=model,
            scaler=scaler,
            y_scaler=y_scaler,
            lo_b=lo_b,
            hi_b=hi_b,
            inference_warmup_days=inference_warmup_days,
        )
        for r in fold_rows:
            r["Segment"] = seg_label
        all_rows.extend(fold_rows)

        if seg_idx == 0:
            train_pred_dates = _rebalance_dates_21d(
                nifty_path, tr.TRAIN_START, tr.TRAIN_END, step_size=int(rebalance_step)
            )
            train_rows = _predict_rows_for_dates(
                train_pred_dates,
                store=store,
                tickers=tickers,
                model=model,
                scaler=scaler,
                y_scaler=y_scaler,
                lo_b=lo_b,
                hi_b=hi_b,
                inference_warmup_days=inference_warmup_days,
            )
            for r in train_rows:
                r["Segment"] = seg_label
            all_rows.extend(train_rows)

        if tr.DEVICE.type == "mps":
            torch.mps.empty_cache()

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    return df.sort_values(["Date", "predicted_alpha"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate RAMT predictions from disk checkpoints.")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory with ramt_model_state_wf_seg_*.pt and matching joblibs (default: auto)",
    )
    parser.add_argument(
        "--training-step",
        type=int,
        default=None,
        help="Outer WF stride (trading days). Default: infer from count of wf_seg_*.pt files.",
    )
    parser.add_argument("--rebalance-step", type=int, default=21)
    parser.add_argument("--inference-warmup-days", type=int, default=30)
    parser.add_argument(
        "--wf-test-start",
        type=str,
        default=None,
        help="TEST_START for segment calendar (default: train_ranking.TEST_START).",
    )
    parser.add_argument(
        "--wf-test-end",
        type=str,
        default=None,
        help="TEST_END for segment calendar (default: train_ranking.TEST_END).",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="2024-01-01",
        help="Blind-test metrics & backtest start (aligned with models/run_final_2024_2026.py).",
    )
    parser.add_argument("--test-end", type=str, default="2026-04-16")
    parser.add_argument("--out-dir", type=str, default="results/ramt")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else _default_artifact_dir()
    artifact_dir = artifact_dir.resolve()
    if not artifact_dir.is_dir():
        raise SystemExit(f"Artifact dir not found: {artifact_dir}")

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wf_test_start = args.wf_test_start or tr.TEST_START
    wf_test_end = args.wf_test_end or tr.TEST_END
    n_ckpt = _max_wf_seg_index(artifact_dir)
    if n_ckpt < 1:
        raise SystemExit(f"No ramt_model_state_wf_seg_*.pt files under {artifact_dir}")

    if args.training_step is not None:
        training_step = int(args.training_step)
    else:
        training_step = infer_training_step(
            nifty_path=_nifty_raw_path(),
            test_start=str(wf_test_start),
            test_end=str(wf_test_end),
            n_wf_segments=n_ckpt,
        )
    print(
        f"Walk-forward calendar: TEST {wf_test_start} … {wf_test_end}  "
        f"training_step={training_step}  (matches {n_ckpt} checkpoint file(s))",
        flush=True,
    )

    print("Verifying main checkpoint (architecture match)...", flush=True)
    main_info = verify_main_checkpoint(artifact_dir)
    print(
        f"  OK: {main_info['path']}  params={main_info['n_parameters']:,}  "
        f"last_fold_label={main_info.get('fold_label')!r}",
        flush=True,
    )

    dash_src = artifact_dir / "training_dashboard.png"
    if not dash_src.is_file():
        dash_src = ROOT / "results" / "ramt" / "training_dashboard.png"
    if dash_src.is_file():
        import shutil

        shutil.copy2(dash_src, out_dir / "training_dashboard.png")
        print(f"Copied training dashboard: {out_dir / 'training_dashboard.png'}", flush=True)

    print("Walk-forward inference (no training)...", flush=True)
    df = run_walk_forward_inference(
        artifact_dir,
        training_step=training_step,
        rebalance_step=int(args.rebalance_step),
        inference_warmup_days=int(args.inference_warmup_days),
        test_start=str(wf_test_start),
        test_end=str(wf_test_end),
    )
    if df.empty:
        raise SystemExit("Inference produced no rows — check data/processed and date ranges.")

    pred_path = out_dir / "ranking_predictions.csv"
    df.to_csv(pred_path, index=False)
    print(f"Saved: {pred_path}  rows={len(df)}", flush=True)

    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    blind = df[(df["Period"] == "Test") & (df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()
    pred_a = blind["predicted_alpha"].astype(float).values
    act_a = blind["actual_alpha"].astype(float).values
    da = float(np.mean((pred_a * act_a) > 0)) if len(blind) else float("nan")
    rmse = float(np.sqrt(np.mean((pred_a - act_a) ** 2))) if len(blind) else float("nan")
    mae = float(np.mean(np.abs(pred_a - act_a))) if len(blind) else float("nan")
    mic = _mean_rank_ic(blind)
    t5 = _top5_positive_rate(blind)
    n_dates = int(blind["Date"].nunique())

    metrics_prelim = {
        "artifact_dir": str(artifact_dir),
        "walk_forward": {
            "test_start": str(wf_test_start),
            "test_end": str(wf_test_end),
            "training_step": training_step,
            "n_segments": n_ckpt,
        },
        "main_checkpoint": main_info,
        "blind_test_window": {"start": str(test_start.date()), "end": str(test_end.date())},
        "DA_pct": round(100.0 * da, 4),
        "RMSE": rmse,
        "MAE": mae,
        "mean_IC": mic,
        "top5_positive_rate": t5,
        "n_blind_rows": int(len(blind)),
        "n_blind_rebalance_dates": n_dates,
    }

    print("Running backtest on RAMT Test-period predictions...", flush=True)
    bt_in = blind[["Date", "Ticker", "predicted_alpha", "actual_alpha"]].copy()
    bt = run_backtest_daily(
        predictions_df=bt_in,
        nifty_features_path=str(ROOT / "data/processed/_NSEI_features.parquet"),
        raw_dir=str(ROOT / "data/raw"),
        start=str(test_start.date()),
        end=str(test_end.date()),
        top_n=5,
        capital=100_000,
        stop_loss=0.07,
        max_weight=0.25,
        portfolio_dd_cash_trigger=0.15,
        rebalance_friction_rate=0.0022,
        turnover_penalty_score=0.0,
        kelly_p=0.5238,
        kelly_use_predicted_margin=True,
        kelly_scale_position=True,
        use_sector_cap=True,
    )
    bt_path = out_dir / "backtest_results.csv"
    bt.to_csv(bt_path, index=False)
    print(f"Saved: {bt_path}  windows={len(bt)}", flush=True)

    strat = pd.read_csv(bt_path, parse_dates=["date"])
    r = strat["portfolio_return"].dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if len(r) > 1 and r.std() > 0 else float("nan")
    nav = strat["portfolio_value"].astype(float).values
    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min()) if len(nav) else float("nan")

    metrics_prelim["Sharpe"] = sharpe
    metrics_prelim["MaxDD"] = max_dd
    metrics_prelim["n_rebalances"] = int(len(bt))

    mpath = out_dir / "ramt_metrics.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics_prelim, f, indent=2)
    print(f"Saved: {mpath}", flush=True)


if __name__ == "__main__":
    main()
