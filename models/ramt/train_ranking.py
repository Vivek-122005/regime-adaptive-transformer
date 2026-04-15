"""
RAMT Ranking Model Training

Trains on ALL NIFTY 50 stocks combined.
Target: Monthly alpha vs NIFTY (beat/miss benchmark)
Loss: MSE on monthly alpha + ranking loss
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.ramt.dataset import ALL_FEATURE_COLS, TICKER_TO_ID, build_ticker_universe
from models.ramt.losses import CombinedLoss
from models.ramt.model import build_ramt


TICKERS = [
    t
    for t in os.listdir("data/processed")
    if t.endswith("_features.csv") and "JPM" not in t
]  # Indian only


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 30
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 8
GRAD_CLIP = 1.0
LAMBDA_DIR = 0.3


def _safe_ticker_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    if stem.endswith("_features"):
        stem = stem[: -len("_features")]
    return stem


def _rebalance_dates_21d(
    nifty_raw_path: str, start: str, end: str, step_size: int = 21
) -> pd.DatetimeIndex:
    """
    Rebalance every `step_size` trading days using NIFTY trading calendar.
    """
    df = pd.read_csv(nifty_raw_path, parse_dates=["Date"]).sort_values("Date")
    df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
    dates = pd.DatetimeIndex(df["Date"].unique())
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    return dates[::step_size]


@dataclass(frozen=True)
class TickerData:
    ticker: str
    ticker_id: int
    dates: pd.DatetimeIndex
    X: np.ndarray  # (N, F) float32
    y: np.ndarray  # (N,) float32
    regime: np.ndarray  # (N,) int64


class MultiTickerSequenceDataset(Dataset):
    """
    Overlapping sequences from multiple tickers.

    Each sample uses one ticker's timeline:
      X: (seq_len, F)
      y: scalar Monthly_Alpha at time i
      regime: HMM_Regime at time i
      ticker_id: integer id
    """

    def __init__(self, data: dict[str, TickerData], sample_keys: list[tuple[str, int]], seq_len: int):
        self.data = data
        self.sample_keys = sample_keys
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        ticker, i = self.sample_keys[idx]
        td = self.data[ticker]
        X = td.X[i - self.seq_len : i]
        y = td.y[i]
        r = td.regime[i]
        t = td.ticker_id
        d = td.dates[i].value  # int64 ns since epoch
        return (
            torch.from_numpy(X).float(),
            torch.tensor([y], dtype=torch.float32),
            torch.tensor([r], dtype=torch.long),
            torch.tensor([t], dtype=torch.long),
            torch.tensor([d], dtype=torch.long),
        )


def _build_sample_keys(
    td: TickerData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seq_len: int,
) -> list[tuple[str, int]]:
    # indices where date in [start, end)
    mask = (td.dates >= start) & (td.dates < end)
    idxs = np.where(mask)[0]
    idxs = idxs[idxs >= seq_len]
    return [(td.ticker, int(i)) for i in idxs]


def _load_all_tickers(processed_dir: str = "data/processed") -> dict[str, TickerData]:
    out: dict[str, TickerData] = {}
    pdir = Path(processed_dir)
    allowed = set(build_ticker_universe(processed_dir))
    files = sorted([p for p in pdir.glob("*_features.csv") if _safe_ticker_from_filename(p.name) in allowed])
    for p in files:
        ticker = _safe_ticker_from_filename(p.name)
        df = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        df = df.set_index("Date", drop=True)

        needed = list(ALL_FEATURE_COLS) + ["Monthly_Alpha", "HMM_Regime"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker}: missing columns: {missing}")

        df = df.dropna(subset=["Monthly_Alpha"])
        dates = pd.DatetimeIndex(df.index)
        X = df[list(ALL_FEATURE_COLS)].values.astype(np.float32)
        y = df["Monthly_Alpha"].values.astype(np.float32)
        r = df["HMM_Regime"].values.astype(np.int64)

        tid = int(TICKER_TO_ID.get(ticker, -1))
        if tid < 0:
            # Should not happen if using build_ticker_universe(), but keep safe.
            tid = int(abs(hash(ticker)) % max(1, len(allowed)))

        out[ticker] = TickerData(ticker=ticker, ticker_id=tid, dates=dates, X=X, y=y, regime=r)
    return out


def _fit_scaler_on_train(data: dict[str, TickerData], train_keys: list[tuple[str, int]]) -> StandardScaler:
    scaler = StandardScaler()
    rows = []
    for ticker, i in train_keys:
        td = data[ticker]
        rows.append(td.X[i])  # single row features at time i
    X_train = np.vstack(rows) if rows else np.empty((0, len(ALL_FEATURE_COLS)), dtype=np.float32)
    scaler.fit(X_train)
    return scaler


def _apply_scaler(data: dict[str, TickerData], scaler: StandardScaler) -> dict[str, TickerData]:
    out: dict[str, TickerData] = {}
    for t, td in data.items():
        Xs = scaler.transform(td.X).astype(np.float32)
        out[t] = TickerData(
            ticker=td.ticker,
            ticker_id=td.ticker_id,
            dates=td.dates,
            X=Xs,
            y=td.y,
            regime=td.regime,
        )
    return out


def _pairwise_rank_loss(pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Pairwise ranking loss (logistic) for a set of items from the same rebalance date.
    Encourages correct ordering when y_true differs.
    """
    # pred, y_true: (n, 1)
    y = y_true.squeeze(-1)
    p = pred.squeeze(-1)
    # choose pairs by sorting and taking extremes to keep it cheap
    n = y.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=pred.device)
    k = min(10, n // 2)
    top_idx = torch.topk(y, k=k, largest=True).indices
    bot_idx = torch.topk(y, k=k, largest=False).indices
    top_p = p[top_idx].unsqueeze(1)
    bot_p = p[bot_idx].unsqueeze(0)
    # want top_p > bot_p
    margin = top_p - bot_p
    return torch.nn.functional.softplus(-margin).mean()


def _train_one_epoch(model, loader, optimizer, criterion, lambda_rank: float = 0.2):
    model.train()
    total = 0.0
    n = 0
    for Xb, yb, rb, tb, db in tqdm(loader, desc="train", leave=False, mininterval=0.5):
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE)
        rb = rb.squeeze(-1).to(DEVICE)
        tb = tb.squeeze(-1).to(DEVICE)
        db = db.squeeze(-1).to(DEVICE)

        optimizer.zero_grad()
        pred, _ = model(Xb, rb, ticker_id=tb)
        mse_dir, _, _ = criterion(pred, yb)
        # ranking loss within each rebalance date (db groups)
        rank_losses = []
        for d in torch.unique(db):
            m = db == d
            if int(m.sum()) >= 4:
                rank_losses.append(_pairwise_rank_loss(pred[m], yb[m]))
        rank_loss = torch.stack(rank_losses).mean() if rank_losses else torch.tensor(0.0, device=DEVICE)
        loss = mse_dir + lambda_rank * rank_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def _eval_loss(model, loader, criterion):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for Xb, yb, rb, tb, _db in loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            rb = rb.squeeze(-1).to(DEVICE)
            tb = tb.squeeze(-1).to(DEVICE)
            pred, _ = model(Xb, rb, ticker_id=tb)
            loss, _, _ = criterion(pred, yb)
            total += float(loss.item())
            n += 1
    return total / max(n, 1)


def _predict(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    actuals = []
    ticker_ids = []
    with torch.no_grad():
        for Xb, yb, rb, tb in loader:
            Xb = Xb.to(DEVICE)
            rb = rb.squeeze(-1).to(DEVICE)
            tb = tb.squeeze(-1).to(DEVICE)
            pred, _ = model(Xb, rb, ticker_id=tb)
            preds.append(pred.cpu().numpy().squeeze())
            actuals.append(yb.numpy().squeeze())
            ticker_ids.append(tb.cpu().numpy().squeeze())
    return np.concatenate(preds), np.concatenate(actuals), np.concatenate(ticker_ids)


def combined_walk_forward(
    start: str = "2016-01-01",
    end: str = "2024-12-31",
    test_steps: int = 3,
    step_size: int = 21,
) -> pd.DataFrame:
    """
    Walk-forward on combined dataset.

    Key difference from old approach:
    Train on ALL tickers simultaneously.
    Test on ALL tickers.

    At each fold:
    1. Combine training data from all tickers
    2. Train one RAMT model
    3. Score all tickers on test period
    4. Rank by predicted monthly alpha
    5. Evaluate: did top 5 beat NIFTY?
    """
    data = _load_all_tickers("data/processed")

    # Determine fold boundaries by trading-day steps (21 trading days ≈ 1 month)
    rebal_dates = _rebalance_dates_21d("data/raw/_NSEI_raw.csv", start, end, step_size=step_size)
    predictions_rows: list[dict[str, object]] = []

    for fold_i in range(6, len(rebal_dates) - test_steps):
        train_start = pd.Timestamp(start)
        train_end = pd.Timestamp(rebal_dates[fold_i])
        test_start = train_end
        test_end = pd.Timestamp(rebal_dates[fold_i + test_steps])

        # Build per-ticker sample keys (daily training, daily val/test)
        train_keys: list[tuple[str, int]] = []
        test_keys: list[tuple[str, int]] = []
        for t, td in data.items():
            train_keys.extend(_build_sample_keys(td, train_start, train_end, SEQ_LEN))
            test_keys.extend(_build_sample_keys(td, test_start, test_end, SEQ_LEN))

        if len(train_keys) < 1000 or len(test_keys) < 200:
            continue

        # Validation = last 15% of train keys (time-ordered within each ticker not guaranteed,
        # but combined training is robust; we keep deterministic split)
        val_size = max(1, int(len(train_keys) * 0.15))
        train_keys_sorted = sorted(train_keys, key=lambda k: (k[0], k[1]))
        val_keys = train_keys_sorted[-val_size:]
        train_keys_final = train_keys_sorted[:-val_size]

        scaler = _fit_scaler_on_train(data, train_keys_final)
        data_sc = _apply_scaler(data, scaler)

        train_ds = MultiTickerSequenceDataset(data_sc, train_keys_final, SEQ_LEN)
        val_ds = MultiTickerSequenceDataset(data_sc, val_keys, SEQ_LEN)
        test_ds = MultiTickerSequenceDataset(data_sc, test_keys, SEQ_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model = build_ramt({"seq_len": SEQ_LEN}).to(DEVICE)
        criterion = CombinedLoss(lambda_dir=LAMBDA_DIR)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best = float("inf")
        best_state = None
        patience = 0

        print(
            f"\nFold {fold_i}: train<{train_start.date()}→{train_end.date()}> "
            f"test<{test_start.date()}→{test_end.date()}> "
            f"train_samples={len(train_ds)} val_samples={len(val_ds)} test_samples={len(test_ds)}",
            flush=True,
        )

        for epoch in range(MAX_EPOCHS):
            _ = _train_one_epoch(model, train_loader, optimizer, criterion)
            v = _eval_loss(model, val_loader, criterion)
            if v < best:
                best = v
                patience = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(f"  epoch {epoch+1:02d}/{MAX_EPOCHS} val_loss={v:.6f}", flush=True)
            if patience >= PATIENCE:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Predict on rebalance dates inside test window
        for t, td in data_sc.items():
            ds = rebal_dates[(rebal_dates >= test_start) & (rebal_dates < test_end)]
            if len(ds) == 0:
                continue
            for d in ds:
                try:
                    i = int(td.dates.get_loc(d))
                except KeyError:
                    continue
                if i < SEQ_LEN:
                    continue
                Xseq = torch.from_numpy(td.X[i - SEQ_LEN : i]).float().unsqueeze(0).to(DEVICE)
                r = torch.tensor([int(td.regime[i])], dtype=torch.long).to(DEVICE)
                tid = torch.tensor([int(td.ticker_id)], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    pred, _ = model(Xseq, r, ticker_id=tid)
                predictions_rows.append(
                    {
                        "Date": pd.Timestamp(d),
                        "Ticker": t,
                        "predicted_alpha": float(pred.cpu().numpy().squeeze()),
                        "actual_alpha": float(td.y[i]),
                        "fold_train_end": train_end,
                    }
                )

    if not predictions_rows:
        return pd.DataFrame(
            columns=["Date", "Ticker", "predicted_alpha", "actual_alpha", "fold_train_end"]
        )

    preds_df = pd.DataFrame(predictions_rows).sort_values(
        ["Date", "predicted_alpha"], ascending=[True, False]
    )
    return preds_df


def train_fixed_and_predict(
    train_start: str = "2016-01-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2025-12-31",
    step_size: int = 21,
    max_epochs: int | None = None,
) -> pd.DataFrame:
    """
    Train ONE combined RAMT model on [train_start, train_end], then predict
    on rebalance dates in [test_start, test_end].

    This enforces a strict no-lookahead split for final backtests.
    """
    data = _load_all_tickers("data/processed")

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    train_keys: list[tuple[str, int]] = []
    for _t, td in data.items():
        train_keys.extend(_build_sample_keys(td, train_start_ts, train_end_ts + pd.Timedelta(days=1), SEQ_LEN))

    if len(train_keys) < 5000:
        raise ValueError(f"Not enough training samples: {len(train_keys)}")

    # Validation window = last ~3 rebalance steps worth of days from the training period
    rebal_train = _rebalance_dates_21d("data/raw/_NSEI_raw.csv", train_start, train_end, step_size=step_size)
    if len(rebal_train) >= 4:
        val_start = pd.Timestamp(rebal_train[-3])
    else:
        val_start = train_end_ts - pd.Timedelta(days=120)

    train_keys_final: list[tuple[str, int]] = []
    val_keys: list[tuple[str, int]] = []
    for t, td in data.items():
        train_keys_final.extend(_build_sample_keys(td, train_start_ts, val_start, SEQ_LEN))
        val_keys.extend(_build_sample_keys(td, val_start, train_end_ts + pd.Timedelta(days=1), SEQ_LEN))

    scaler = _fit_scaler_on_train(data, train_keys_final)
    data_sc = _apply_scaler(data, scaler)

    train_ds = MultiTickerSequenceDataset(data_sc, sorted(train_keys_final), SEQ_LEN)
    val_ds = MultiTickerSequenceDataset(data_sc, sorted(val_keys), SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = build_ramt({"seq_len": SEQ_LEN}).to(DEVICE)
    criterion = CombinedLoss(lambda_dir=LAMBDA_DIR)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best = float("inf")
    best_state = None
    patience = 0
    epochs = MAX_EPOCHS if max_epochs is None else int(max_epochs)

    print(
        f"\nFixed-train: train<{train_start_ts.date()}→{train_end_ts.date()}> "
        f"val_start<{val_start.date()}> "
        f"train_samples={len(train_ds)} val_samples={len(val_ds)}",
        flush=True,
    )

    for epoch in range(epochs):
        _ = _train_one_epoch(model, train_loader, optimizer, criterion)
        v = _eval_loss(model, val_loader, criterion)
        if v < best:
            best = v
            patience = 0
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        else:
            patience += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:02d}/{epochs} val_loss={v:.6f}", flush=True)
        if patience >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on rebalance dates in test period
    rebal_test = _rebalance_dates_21d("data/raw/_NSEI_raw.csv", test_start, test_end, step_size=step_size)
    rows: list[dict[str, object]] = []
    for t, td in data_sc.items():
        for d in rebal_test:
            if d < test_start_ts or d > test_end_ts:
                continue
            try:
                i = int(td.dates.get_loc(pd.Timestamp(d)))
            except KeyError:
                continue
            if i < SEQ_LEN:
                continue
            Xseq = torch.from_numpy(td.X[i - SEQ_LEN : i]).float().unsqueeze(0).to(DEVICE)
            r = torch.tensor([int(td.regime[i])], dtype=torch.long).to(DEVICE)
            tid = torch.tensor([int(td.ticker_id)], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                pred, _ = model(Xseq, r, ticker_id=tid)
            rows.append(
                {
                    "Date": pd.Timestamp(d),
                    "Ticker": t,
                    "predicted_alpha": float(pred.cpu().numpy().squeeze()),
                    "actual_alpha": float(td.y[i]),
                    "fold_train_end": train_end_ts,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "predicted_alpha", "actual_alpha", "fold_train_end"])
    return pd.DataFrame(rows).sort_values(["Date", "predicted_alpha"], ascending=[True, False])


if __name__ == "__main__":
    print("Combined ranking training (walk-forward).", flush=True)
    df = combined_walk_forward()
    os.makedirs("results", exist_ok=True)
    out = "results/ranking_predictions.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
