"""
Portfolio Backtest 2015-2024

Monthly rebalancing strategy:
1. First trading day of month
2. Check HMM regime
3. If BULL: top N RAMT-ranked stocks at 100% allocation
4. If HIGH_VOL: top 3 at 50% allocation
5. If BEAR: top 5 at 20% allocation (5% one-period loss floor on portfolio return)
6. Hold until next month
7. Repeat

Metrics to report:
- Annual return vs NIFTY
- Sharpe ratio
- Max drawdown
- Win rate (% months beating NIFTY)
- Best/worst month
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_nifty_benchmark_raw(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load NIFTY benchmark OHLCV from Parquet (preferred) or legacy `_NSEI_raw.csv`.

    Ensures `Adj Close` exists for downstream feature helpers.
    """
    rdir = Path(raw_dir)
    pq = rdir / "_NSEI.parquet"
    csv = rdir / "_NSEI_raw.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
        df["Date"] = pd.to_datetime(df["Date"])
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"].astype(float)
        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                df[col] = df["Adj Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 1.0
    else:
        raise FileNotFoundError(
            f"NIFTY benchmark not found: expected {pq} or {csv}"
        )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def ensure_nifty_features_parquet(processed_dir: str | Path, raw_dir: str | Path) -> str:
    """
    Return path to ``_NSEI_features.parquet``, building it if missing.

    Uses the same pipeline as ``features.feature_engineering`` so the index file has
    the full feature column set (including ``HMM_Regime``) as stock files.
    """
    pdir = Path(processed_dir)
    rdir = Path(raw_dir)
    out = pdir / "_NSEI_features.parquet"
    if out.exists():
        return str(out)

    pdir.mkdir(parents=True, exist_ok=True)

    from features.feature_engineering import (
        _download_benchmark_if_missing,
        _read_raw_equity,
        load_macro_series,
        process_raw_equity_path,
    )

    bench_path = _download_benchmark_if_missing()
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Cannot build NIFTY features: missing benchmark Parquet/CSV under {rdir}"
        )

    nifty_df = _read_raw_equity(bench_path)
    macro_data = load_macro_series(rdir)
    _, written = process_raw_equity_path(bench_path, nifty_df, macro_data, pdir)
    if written is None or not written.exists():
        raise RuntimeError("Failed to materialize _NSEI_features.parquet from raw NIFTY data.")
    return str(written.resolve())


def resolve_nifty_features_path(nifty_features_path: str, raw_dir: str) -> str:
    """Use existing processed Parquet if present; otherwise build from raw NIFTY."""
    p = Path(nifty_features_path)
    if p.exists():
        return str(p.resolve())
    processed_dir = Path(raw_dir).resolve().parent / "processed"
    return ensure_nifty_features_parquet(processed_dir, raw_dir)


def _load_price_series(raw_path: str) -> pd.Series:
    p = pd.read_parquet(raw_path)
    p["Date"] = pd.to_datetime(p["Date"])
    p = p.sort_values("Date")
    # Use Adj Close for integrity under splits/bonuses
    s = p.set_index("Date")["Adj Close"].astype(float)
    return s


def build_rebalance_regime_df(
    nifty_features_path: str,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build regime series aligned to explicit rebalance dates.
    If exact date is missing in features, use last available previous value.
    """
    f = pd.read_parquet(nifty_features_path)
    f["Date"] = pd.to_datetime(f["Date"])
    f = f.sort_values("Date").set_index("Date")["HMM_Regime"].astype(float).ffill()
    out = []
    for d in rebalance_dates:
        # take last available on/before date
        sel = f.loc[:d]
        if sel.empty:
            continue
        out.append({"Date": pd.Timestamp(d), "regime": int(sel.iloc[-1])})
    return pd.DataFrame(out)


def build_monthly_regime_df(
    nifty_features_path: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Build a month-start regime series from processed NIFTY features.

    Returns DataFrame with columns: Date, regime
    where Date is month start (first trading day present in that month).
    """
    df = pd.read_parquet(nifty_features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
    if df.empty or "HMM_Regime" not in df.columns:
        return pd.DataFrame(columns=["Date", "regime"])

    df["Month"] = df["Date"].dt.to_period("M")
    first_rows = df.groupby("Month", as_index=False).head(1)
    out = first_rows[["Date", "HMM_Regime"]].rename(columns={"HMM_Regime": "regime"})
    out["regime"] = out["regime"].astype(int)
    return out.reset_index(drop=True)


def compute_nifty_monthly_returns(
    nifty_raw_path: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Approximate month-ahead NIFTY return as 21-trading-day forward log return.

    Returns DataFrame: date, nifty_return
    where date is month start (first trading day present in that month).
    """
    df = pd.read_parquet(nifty_raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
    if df.empty:
        return pd.DataFrame(columns=["date", "nifty_return"])

    px = df["Adj Close"].astype(float).replace(0.0, np.nan)
    r1 = px / px.shift(1)
    lr = np.log(r1.where(r1 > 0.0))
    df["fwd_21"] = lr.rolling(21).sum().shift(-21)
    df["Month"] = df["Date"].dt.to_period("M")
    ms = df.groupby("Month", as_index=False).head(1)
    out = ms[["Date", "fwd_21"]].rename(columns={"Date": "date", "fwd_21": "nifty_return"})
    out["nifty_return"] = out["nifty_return"].fillna(0.0).astype(float)
    return out.reset_index(drop=True)


def run_backtest(
    predictions_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    start: str = "2016-01-01",
    end: str = "2024-12-31",
    top_n: int = 5,
    capital: float = 50000,
) -> pd.DataFrame:
    """
    Full portfolio backtest.

    predictions_df columns:
      Date, Ticker, predicted_alpha, actual_alpha

    regime_df columns:
      Date, regime (0/1/2)

    Returns:
      Monthly portfolio returns
      NIFTY benchmark returns
      All trade history
    """
    results: list[dict[str, object]] = []
    monthly_dates = pd.date_range(start, end, freq="MS")

    predictions_df = predictions_df.copy()
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
    regime_df = regime_df.copy()
    regime_df["Date"] = pd.to_datetime(regime_df["Date"])

    for i, date in enumerate(monthly_dates[:-1]):
        _next_date = monthly_dates[i + 1]

        month_regime = regime_df[regime_df["Date"] == date]["regime"].values
        if len(month_regime) == 0:
            continue
        regime = int(month_regime[0])

        if regime == 2:  # Bear — fractional toe-hold + loss floor (trailing-stop proxy)
            position_size = 0.2
            top_n_regime = min(5, top_n)
        elif regime == 0:  # High vol
            position_size = 0.5
            top_n_regime = 3
        else:  # Bull
            position_size = 1.0
            top_n_regime = top_n

        month_preds = predictions_df[predictions_df["Date"] == date].nlargest(
            top_n_regime, "predicted_alpha"
        )
        if month_preds.empty:
            continue

        actual_returns = month_preds["actual_alpha"].values.astype(float)
        portfolio_return = float(np.mean(actual_returns) * position_size)
        if regime == 2:
            portfolio_return = float(max(portfolio_return, -0.05))

        results.append(
            {
                "date": date,
                "portfolio_return": portfolio_return,
                "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                "stocks_held": month_preds["Ticker"].tolist(),
                "cash": False,
            }
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    results_df["cumulative_return"] = (1 + results_df["portfolio_return"]).cumprod() - 1

    monthly_returns = results_df["portfolio_return"].values.astype(float)
    annual_return = ((1 + np.mean(monthly_returns)) ** 12 - 1) * 100
    sharpe = (np.mean(monthly_returns) / (np.std(monthly_returns) + 1e-8)) * np.sqrt(12)
    max_dd = compute_max_drawdown(monthly_returns)
    win_rate = float(np.mean(monthly_returns > 0) * 100)

    print(f"\n{'='*50}")
    print("BACKTEST RESULTS 2016-2024")
    print(f"{'='*50}")
    print(f"Annual Return:  {annual_return:.1f}%")
    print(f"Sharpe Ratio:   {sharpe:.2f}")
    print(f"Max Drawdown:   {max_dd*100:.1f}%")
    print(f"Win Rate:       {win_rate:.1f}% months")
    print(
        f"Capital: ₹{capital:,} → "
        f"₹{capital*(1+results_df['cumulative_return'].iloc[-1]):,.0f}"
    )

    return results_df


def run_backtest_daily(
    predictions_df: pd.DataFrame,
    nifty_features_path: str,
    raw_dir: str,
    start: str,
    end: str,
    step_size: int = 21,
    top_n: int = 5,
    capital: float = 100000,
    stop_loss: float = 0.07,
    stop_loss_bear: float = 0.05,
    max_weight: float = 0.20,
    portfolio_dd_cash_trigger: float = 0.15,
    trade_cost_bps: float = 15.0,
    slippage_bps: float = 10.0,
    turnover_penalty_score: float = 0.0,
) -> pd.DataFrame:
    """
    Daily-price backtest with risk rules.

    - Rebalance every `step_size` trading days on NIFTY calendar.
    - Stop-loss per stock (intraperiod): ``stop_loss`` by default; in BEAR use ``stop_loss_bear`` (5%).
    - Max weight per stock: cap at `max_weight`, remainder stays cash.
    - If portfolio return <= -portfolio_dd_cash_trigger in a window:
        force next window to cash.

    Returns one row per rebalance window with trade/hold details.
    """
    preds = predictions_df.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    nifty_features_path = resolve_nifty_features_path(nifty_features_path, raw_dir)

    nifty_raw = _load_nifty_benchmark_raw(raw_dir)
    cal = pd.DatetimeIndex(nifty_raw["Date"])
    cal = cal[(cal >= start_ts) & (cal <= end_ts)]
    rebal = cal[::step_size]
    if len(rebal) < 2:
        return pd.DataFrame()

    regime_df = build_rebalance_regime_df(nifty_features_path, rebal)
    regime_df = regime_df.set_index("Date")["regime"]

    # Preload price series lazily
    price_cache: dict[str, pd.Series] = {}

    def get_prices(ticker: str) -> pd.Series:
        if ticker in price_cache:
            return price_cache[ticker]
        path = f"{raw_dir}/{ticker}.parquet"
        price_cache[ticker] = _load_price_series(path)
        return price_cache[ticker]

    forced_cash_next = False
    results: list[dict[str, object]] = []
    pv = float(capital)
    prev_holdings: set[str] = set()

    for i in range(len(rebal) - 1):
        d0 = pd.Timestamp(rebal[i])
        d1 = pd.Timestamp(rebal[i + 1])

        regime = int(regime_df.loc[:d0].iloc[-1]) if not regime_df.loc[:d0].empty else 1
        if forced_cash_next:
            results.append(
                {
                    "date": d0,
                    "portfolio_return": 0.0,
                    "regime": "RISK_OFF",
                    "stocks_held": [],
                    "cash": True,
                    "portfolio_value": pv,
                }
            )
            forced_cash_next = False
            prev_holdings = set()
            continue

        if regime == 2:
            position_size = 0.2
            n_sel = min(5, top_n)
            sl_stock = stop_loss_bear
        elif regime == 0:
            position_size = 0.5
            n_sel = 3
            sl_stock = stop_loss
        else:
            position_size = 1.0
            n_sel = top_n
            sl_stock = stop_loss

        month_df = preds[preds["Date"] == d0].copy()
        if month_df.empty:
            results.append(
                {
                    "date": d0,
                    "portfolio_return": 0.0,
                    "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                    "stocks_held": [],
                    "cash": True,
                    "portfolio_value": pv,
                }
            )
            prev_holdings = set()
            continue

        # Optional turnover-aware selection: penalize NEW names slightly so we don't churn
        if turnover_penalty_score > 0 and prev_holdings:
            month_df["score_adj"] = month_df["predicted_alpha"]
            month_df.loc[~month_df["Ticker"].isin(prev_holdings), "score_adj"] = (
                month_df.loc[~month_df["Ticker"].isin(prev_holdings), "score_adj"] - turnover_penalty_score
            )
            month_preds = month_df.nlargest(n_sel, "score_adj")
        else:
            month_preds = month_df.nlargest(n_sel, "predicted_alpha")
        if month_preds.empty:
            results.append(
                {
                    "date": d0,
                    "portfolio_return": 0.0,
                    "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                    "stocks_held": [],
                    "cash": True,
                    "portfolio_value": pv,
                }
            )
            continue

        tickers = month_preds["Ticker"].tolist()
        base_w = 1.0 / len(tickers)
        w = min(base_w, max_weight)
        invested = w * len(tickers)
        invested *= position_size
        cash_weight = 1.0 - invested

        # Turnover and friction costs (approx):
        # - turnover = fraction of invested basket that changes
        # - cost applied on notional traded (entry/exit) at rebalance
        new_holdings = set(tickers)
        if not prev_holdings:
            turnover = invested  # entering positions from cash
        else:
            # approximate: changed names / current names
            changed = len(new_holdings.symmetric_difference(prev_holdings))
            denom = max(1, len(new_holdings.union(prev_holdings)))
            turnover = invested * (changed / denom)

        # total bps cost on traded notional (STT+fees+slippage proxy)
        total_cost_rate = (trade_cost_bps + slippage_bps) / 10000.0
        friction_cost = pv * turnover * total_cost_rate

        stock_rets = []
        stopped = []
        for t in tickers:
            px = get_prices(t)
            window = px.loc[(px.index >= d0) & (px.index < d1)]
            if window.empty:
                stock_rets.append(0.0)
                continue
            entry = float(window.iloc[0])
            # stop-loss check
            min_px = float(window.min())
            if (min_px / entry - 1.0) <= -sl_stock:
                stock_rets.append(-sl_stock)
                stopped.append(t)
            else:
                exit_px = float(window.iloc[-1])
                stock_rets.append(exit_px / entry - 1.0)

        gross_stock_ret = float(np.mean(stock_rets)) if stock_rets else 0.0
        port_ret_gross = invested * gross_stock_ret  # cash returns 0
        port_ret = port_ret_gross - (friction_cost / pv if pv > 0 else 0.0)
        if regime == 2:
            port_ret = float(max(port_ret, -0.05))

        if port_ret <= -portfolio_dd_cash_trigger:
            forced_cash_next = True

        pv = pv * (1.0 + port_ret)
        results.append(
            {
                "date": d0,
                "portfolio_return": port_ret,
                "portfolio_return_gross": port_ret_gross,
                "friction_cost": float(friction_cost),
                "turnover": float(turnover),
                "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                "stocks_held": tickers,
                "stops_hit": stopped,
                "cash": False,
                "portfolio_value": pv,
                "cash_weight": cash_weight,
                "invested_weight": invested,
            }
        )
        prev_holdings = new_holdings

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["cumulative_return"] = (df["portfolio_value"] / float(capital)) - 1.0
    return df


def compute_max_drawdown(returns) -> float:
    cumulative = np.cumprod(1 + np.array(returns, dtype=float))
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


if __name__ == "__main__":
    print("Loading predictions...")
    # Load from results/ranking_predictions.csv
    # Run backtest
    # Print results
    raise SystemExit(
        "Backtest module scaffolded. "
        "Not runnable until ranking predictions + regime series are produced."
    )
