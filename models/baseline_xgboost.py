"""
RAMT Baseline Model — XGBoost
Walk-forward validation with expanding training window.
Evaluates RMSE, MAE, Directional Accuracy, and Sharpe Ratio
on out-of-sample predictions for all 4 tickers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data" / "processed").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_FILES = [
    ("JPM", "JPM_features.csv"),
    ("RELIANCE_NS", "RELIANCE_NS_features.csv"),
    ("TCS_NS", "TCS_NS_features.csv"),
    ("HDFCBANK_NS", "HDFCBANK_NS_features.csv"),
]

EXCLUDE_FROM_X = {
    "Date",
    "Log_Return",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Ticker",
    "HMM_Regime_Label",
}

INITIAL_TRAIN_FRAC = 0.6
STEP_DAYS = 63
TEST_DAYS = 63
VAL_FRAC_WITHIN_TRAIN = 0.2

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Target = next-day log return; align rows where y is defined."""
    df = df.sort_values("Date").reset_index(drop=True)
    y = df["Log_Return"].shift(-1)
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    dates = df["Date"]
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]
    X = df[feat_cols]
    return X, y, dates


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    ticker: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expanding window: first train ends at INITIAL_TRAIN_FRAC * n; each step adds
    TEST_DAYS to the training end. No test rows appear in training for that fold.
    Returns parallel arrays: date, y_true, y_pred for all OOS test rows.
    """
    n = len(X)
    n0 = int(n * INITIAL_TRAIN_FRAC)
    if n0 < 50 or n0 + TEST_DAYS > n:
        raise ValueError(f"{ticker}: insufficient rows (n={n}, n0={n0})")

    oos_dates: list = []
    oos_y_true: list = []
    oos_y_pred: list = []

    train_end = n0
    while train_end + TEST_DAYS <= n:
        test_start = train_end
        test_end = train_end + TEST_DAYS

        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]

        n_tr = len(X_tr)
        n_val = max(1, int(np.ceil(VAL_FRAC_WITHIN_TRAIN * n_tr)))
        n_fit = n_tr - n_val
        if n_fit < 10:
            raise ValueError(f"{ticker}: training split too small at train_end={train_end}")

        X_fit = X_tr.iloc[:n_fit]
        y_fit = y_tr.iloc[:n_fit]
        X_val = X_tr.iloc[n_fit:]
        y_val = y_tr.iloc[n_fit:]

        X_te = X.iloc[test_start:test_end]
        y_te = y.iloc[test_start:test_end]
        d_te = dates.iloc[test_start:test_end]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred = model.predict(X_te)

        oos_dates.extend(d_te.tolist())
        oos_y_true.extend(y_te.to_numpy().tolist())
        oos_y_pred.extend(pred.tolist())

        train_end += TEST_DAYS

    return (
        np.array(oos_dates, dtype="datetime64[ns]"),
        np.array(oos_y_true, dtype=float),
        np.array(oos_y_pred, dtype=float),
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(mean_absolute_error(a, b))


def directional_accuracy_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    st = np.sign(y_true)
    sp = np.sign(y_pred)
    return float(np.mean(st == sp) * 100.0)


def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    strat = y_true * np.sign(y_pred)
    mu = float(np.mean(strat))
    sig = float(np.std(strat, ddof=0))
    if sig == 0.0:
        return float("nan")
    return mu / sig * np.sqrt(252.0)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    metrics_rows: list[tuple[str, float, float, float, float]] = []

    for ticker, fname in FEATURE_FILES:
        path = PROCESSED_DIR / fname
        if not path.is_file():
            raise FileNotFoundError(path)

        raw = pd.read_csv(path, parse_dates=["Date"])
        X, y, dates = prepare_xy(raw)

        dates_oos, y_true, y_pred = walk_forward_predict(X, y, dates, ticker)

        r = rmse(y_true, y_pred)
        m = mae(y_true, y_pred)
        da = directional_accuracy_pct(y_true, y_pred)
        sh = sharpe_ratio(y_true, y_pred)

        metrics_rows.append((ticker, r, m, da, sh))

        for d, yt, yp in zip(dates_oos, y_true, y_pred):
            all_rows.append(
                {
                    "Date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    out_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "xgboost_predictions.csv"
    out_df.to_csv(out_path, index=False)

    print()
    print(f"Saved predictions → {out_path.resolve()}")
    print()
    print("Ticker       | RMSE   | MAE    | DA%   | Sharpe")
    print("-------------|--------|--------|-------|-------")

    rmses, maes, das, shs = [], [], [], []
    for ticker, r, m, da, sh in metrics_rows:
        rmses.append(r)
        maes.append(m)
        das.append(da)
        shs.append(sh if not np.isnan(sh) else np.nan)
        sh_str = f"{sh:>6.2f}" if not np.isnan(sh) else "   nan"
        print(
            f"{ticker:<12} | {r:.4f} | {m:.4f} | {da:>5.2f} | {sh_str}"
        )

    avg_r = float(np.nanmean(rmses))
    avg_m = float(np.nanmean(maes))
    avg_da = float(np.nanmean(das))
    avg_sh = float(np.nanmean([s for s in shs if not np.isnan(s)]))
    if np.all(np.isnan(shs)):
        avg_sh_str = "   nan"
    else:
        avg_sh_str = f"{avg_sh:>6.2f}"
    print(
        f"{'Average':<12} | {avg_r:.4f} | {avg_m:.4f} | {avg_da:>5.2f} | {avg_sh_str}"
    )
    print()


def last_timestep(X: np.ndarray) -> np.ndarray:
    """Legacy helper for sequence tensors: take last timestep."""
    return X[:, -1, :]


def evaluate_classifier(clf: XGBClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """Legacy classifier metrics for `evaluate.py` + saved XGBoost checkpoints."""
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
    }


if __name__ == "__main__":
    main()
