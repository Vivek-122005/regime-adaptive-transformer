"""
Portfolio Backtest 2015-2024

Monthly rebalancing strategy:
1. First trading day of month
2. Check HMM regime
3. If BULL: buy top 5 RAMT-ranked stocks
4. If HIGH_VOL: buy top 3
5. If BEAR: cash
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

import numpy as np
import pandas as pd


def _load_price_series(raw_path: str) -> pd.Series:
    df = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date")
    s = df.set_index("Date")["Close"].astype(float)
    return s


def build_rebalance_regime_df(
    nifty_features_path: str,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build regime series aligned to explicit rebalance dates.
    If exact date is missing in features, use last available previous value.
    """
    f = pd.read_csv(nifty_features_path, parse_dates=["Date"]).sort_values("Date")
    f = f.set_index("Date")["HMM_Regime"].astype(float).ffill()
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
    df = pd.read_csv(nifty_features_path, parse_dates=["Date"]).sort_values("Date")
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
    df = pd.read_csv(nifty_raw_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
    if df.empty:
        return pd.DataFrame(columns=["date", "nifty_return"])

    if "Log_Return" not in df.columns:
        # compute if missing
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["fwd_21"] = df["Log_Return"].rolling(21).sum().shift(-21)
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

        if regime == 2:  # Bear
            position_size = 0.0
            top_n_regime = 0
        elif regime == 0:  # High vol
            position_size = 0.5
            top_n_regime = 3
        else:  # Bull
            position_size = 1.0
            top_n_regime = top_n

        if position_size == 0.0:
            results.append(
                {
                    "date": date,
                    "portfolio_return": 0.0,
                    "regime": "BEAR",
                    "stocks_held": [],
                    "cash": True,
                }
            )
            continue

        month_preds = predictions_df[predictions_df["Date"] == date].nlargest(
            top_n_regime, "predicted_alpha"
        )
        if month_preds.empty:
            continue

        actual_returns = month_preds["actual_alpha"].values.astype(float)
        portfolio_return = float(np.mean(actual_returns) * position_size)

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
    max_weight: float = 0.20,
    portfolio_dd_cash_trigger: float = 0.15,
) -> pd.DataFrame:
    """
    Daily-price backtest with risk rules.

    - Rebalance every `step_size` trading days on NIFTY calendar.
    - Stop-loss per stock (intramonth): exit at -stop_loss and hold cash.
    - Max weight per stock: cap at `max_weight`, remainder stays cash.
    - If portfolio return <= -portfolio_dd_cash_trigger in a window:
        force next window to cash.

    Returns one row per rebalance window with trade/hold details.
    """
    preds = predictions_df.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    nifty_raw = pd.read_csv(f"{raw_dir}/_NSEI_raw.csv", parse_dates=["Date"]).sort_values("Date")
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
        path = f"{raw_dir}/{ticker}_raw.csv"
        price_cache[ticker] = _load_price_series(path)
        return price_cache[ticker]

    forced_cash_next = False
    results: list[dict[str, object]] = []
    pv = float(capital)

    for i in range(len(rebal) - 1):
        d0 = pd.Timestamp(rebal[i])
        d1 = pd.Timestamp(rebal[i + 1])

        regime = int(regime_df.loc[:d0].iloc[-1]) if not regime_df.loc[:d0].empty else 1
        if forced_cash_next:
            regime = 2
            forced_cash_next = False

        if regime == 2:
            results.append(
                {
                    "date": d0,
                    "portfolio_return": 0.0,
                    "regime": "BEAR",
                    "stocks_held": [],
                    "cash": True,
                    "portfolio_value": pv,
                }
            )
            continue

        if regime == 0:
            position_size = 0.5
            n_sel = 3
        else:
            position_size = 1.0
            n_sel = top_n

        month_preds = preds[preds["Date"] == d0].nlargest(n_sel, "predicted_alpha")
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
            if (min_px / entry - 1.0) <= -stop_loss:
                stock_rets.append(-stop_loss)
                stopped.append(t)
            else:
                exit_px = float(window.iloc[-1])
                stock_rets.append(exit_px / entry - 1.0)

        gross_stock_ret = float(np.mean(stock_rets)) if stock_rets else 0.0
        port_ret = invested * gross_stock_ret  # cash returns 0

        if port_ret <= -portfolio_dd_cash_trigger:
            forced_cash_next = True

        pv = pv * (1.0 + port_ret)
        results.append(
            {
                "date": d0,
                "portfolio_return": port_ret,
                "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                "stocks_held": tickers,
                "stops_hit": stopped,
                "cash": False,
                "portfolio_value": pv,
                "cash_weight": cash_weight,
                "invested_weight": invested,
            }
        )

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
