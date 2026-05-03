"""
Core backtesting engine for daily portfolio rebalancing with HMM regime adaptation.

This module provides the main run_backtest_daily function used throughout the codebase
for portfolio backtesting with realistic friction, stops, and regime-based position sizing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from features.sectors import get_sector


# Constants
REAL_2026_REBALANCE_FRICTION_RATE = 0.0022
LEGACY_REBALANCE_FRICTION_RATE = 0.002


def _load_price_series(ticker: str, raw_dir: str) -> pd.Series:
    """Load price series for a ticker from raw data directory."""
    raw_path = Path(raw_dir)
    
    # Try different naming conventions
    price_path = raw_path / f"{ticker}.parquet"
    if not price_path.exists():
        price_path = raw_path / f"{ticker}.NS.parquet"
    
    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found for {ticker} in {raw_dir}")
    
    df = pd.read_parquet(price_path)
    df["Date"] = pd.to_datetime(df["Date"])
    prices = df.set_index("Date")["Adj Close"]
    
    return prices


def _raw_ticker_stem(ticker: str) -> str:
    """Convert ticker to filename stem for price data."""
    return ticker.replace(".", "_")


def _window_period_return(
    prices: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    stop_loss: float,
) -> float:
    """Calculate return for a period with stop-loss protection."""
    if len(prices) < 2:
        return 0.0
    
    # Get price series for the period
    period_prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
    
    if len(period_prices) < 2:
        return 0.0
    
    start_price = period_prices.iloc[0]
    
    # Check stop-loss
    for price in period_prices.iloc[1:]:
        if (price - start_price) / start_price <= -stop_loss:
            return -stop_loss
    
    end_price = period_prices.iloc[-1]
    return (end_price - start_price) / start_price


def run_backtest_daily(
    predictions_df: pd.DataFrame,
    nifty_features_path: str,
    raw_dir: str,
    start: str,
    end: str,
    capital: float = 100_000,
    top_n: int = 5,
    stop_loss: float = 0.07,
    stop_loss_bear: float = 0.05,
    max_weight: float = 0.25,
    portfolio_dd_cash_trigger: float = 0.15,
    rebalance_friction_rate: float = REAL_2026_REBALANCE_FRICTION_RATE,
    turnover_penalty_score: float = 0.0,
    kelly_p: float = 0.5238,
    kelly_use_predicted_margin: bool = True,
    kelly_scale_position: bool = True,
    use_sector_cap: bool = True,
    flat_regime_sizing: bool = False,
) -> pd.DataFrame:
    """
    Run daily backtest with HMM regime-based position sizing.
    
    This is the main backtesting function used throughout the codebase.
    It implements realistic portfolio rebalancing with friction, stops,
    and regime-based position sizing using NIFTY HMM regimes.
    
    Args:
        predictions_df: DataFrame with Date, Ticker, predicted_alpha, actual_alpha
        nifty_features_path: Path to NIFTY features with HMM_Regime column
        raw_dir: Directory with raw price data parquet files
        start: Backtest start date (YYYY-MM-DD)
        end: Backtest end date (YYYY-MM-DD)
        capital: Initial capital
        top_n: Maximum number of stocks per rebalance
        stop_loss: Stop loss threshold (bull markets)
        stop_loss_bear: Stop loss threshold (bear markets)
        max_weight: Maximum weight per stock
        portfolio_dd_cash_trigger: Drawdown trigger for forced cash
        rebalance_friction_rate: Transaction cost rate
        turnover_penalty_score: Turnover penalty (not implemented)
        kelly_p: Kelly criterion parameter
        kelly_use_predicted_margin: Use predicted margin for Kelly
        kelly_scale_position: Scale positions with Kelly
        use_sector_cap: Apply sector caps (max 1 stock per sector)
        flat_regime_sizing: If True, ignore HMM for sizing (flat 1.0/1.0/1.0)
        
    Returns:
        DataFrame with backtest results per rebalance period
    """
    # Load NIFTY features for regime detection
    nifty_features = pd.read_parquet(nifty_features_path)
    nifty_features["Date"] = pd.to_datetime(nifty_features["Date"])
    
    # Prepare predictions
    preds = predictions_df.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    
    # Filter to backtest period
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    preds = preds[(preds["Date"] >= start_ts) & (preds["Date"] <= end_ts)]
    
    if preds.empty:
        return pd.DataFrame()
    
    # Get unique rebalance dates
    rebalance_dates = sorted(preds["Date"].unique())
    
    # Load HMM regimes
    regime_df = nifty_features[["Date", "HMM_Regime"]].copy()
    regime_df["Date"] = pd.to_datetime(regime_df["Date"])
    
    # Merge regime information
    preds = preds.merge(regime_df, on="Date", how="left")
    preds["HMM_Regime"] = preds["HMM_Regime"].fillna(0.0)  # Default to HIGH_VOL
    
    # Initialize results
    results = []
    portfolio_value = capital
    forced_cash_next = False
    
    # Cache for price data
    price_cache: Dict[str, pd.Series] = {}
    
    for i, rebalance_date in enumerate(rebalance_dates):
        if i == len(rebalance_dates) - 1:
            break  # Skip last rebalance (no future period)
        
        next_date = rebalance_dates[i + 1]
        
        # Get regime for this rebalance
        regime_data = preds[preds["Date"] == rebalance_date]
        if regime_data.empty:
            continue
        
        regime = int(regime_data["HMM_Regime"].iloc[0])
        
        # Determine position sizing
        if flat_regime_sizing:
            position_size = 1.0
            n_stocks = top_n
            effective_stop_loss = stop_loss
        else:
            # Regime-based sizing
            if regime == 2:  # BEAR
                position_size = 0.2
                n_stocks = min(5, top_n)
                effective_stop_loss = stop_loss_bear
            elif regime == 0:  # HIGH_VOL
                position_size = 0.5
                n_stocks = 3
                effective_stop_loss = stop_loss
            else:  # BULL
                position_size = 1.0
                n_stocks = top_n
                effective_stop_loss = stop_loss
        
        # Select top stocks
        top_stocks = regime_data.nlargest(n_stocks, "predicted_alpha")
        
        # Apply sector caps if enabled
        if use_sector_cap:
            selected_stocks = []
            used_sectors = set()
            
            for _, stock in top_stocks.iterrows():
                sector = get_sector(stock["Ticker"])
                
                if sector not in used_sectors:
                    selected_stocks.append(stock)
                    used_sectors.add(sector)
                
                if len(selected_stocks) >= n_stocks:
                    break
            
            top_stocks = pd.DataFrame(selected_stocks)
        
        if top_stocks.empty or forced_cash_next:
            result = {
                "date": rebalance_date,
                "portfolio_return": 0.0,
                "regime": "HIGH_VOL" if regime == 0 else "BULL" if regime == 1 else "BEAR",
                "stocks_held": [],
                "cash": True,
                "portfolio_value_start": portfolio_value,
                "portfolio_value": portfolio_value,
                "position_size": 0.0,
                "n_stocks": 0,
                "hmm_regime": regime,
            }
            results.append(result)
            forced_cash_next = False
            continue
        
        # Calculate portfolio return
        portfolio_return = 0.0
        stock_returns = []
        
        for _, stock in top_stocks.iterrows():
            ticker = stock["Ticker"]
            
            # Load price data
            if ticker not in price_cache:
                try:
                    price_cache[ticker] = _load_price_series(ticker, raw_dir)
                except FileNotFoundError:
                    continue
            
            prices = price_cache[ticker]
            stock_return = _window_period_return(
                prices, rebalance_date, next_date, effective_stop_loss
            )
            stock_returns.append(stock_return)
        
        if stock_returns:
            # Equal weight allocation
            portfolio_return = np.mean(stock_returns)
        
        # Apply friction
        trade_value = portfolio_value * position_size
        friction_cost = trade_value * rebalance_friction_rate
        net_return = portfolio_return - friction_cost / portfolio_value
        
        # Update portfolio value
        portfolio_value_start = portfolio_value
        portfolio_value *= (1 + net_return)
        
        # Check drawdown trigger
        if portfolio_value <= capital * (1 - portfolio_dd_cash_trigger):
            forced_cash_next = True
        
        result = {
            "date": rebalance_date,
            "portfolio_return": net_return,
            "portfolio_return_gross": portfolio_return,
            "trade_value": trade_value,
            "friction_cost": friction_cost,
            "regime": "HIGH_VOL" if regime == 0 else "BULL" if regime == 1 else "BEAR",
            "stocks_held": top_stocks["Ticker"].tolist(),
            "cash": False,
            "portfolio_value_start": portfolio_value_start,
            "portfolio_value": portfolio_value,
            "position_size": position_size,
            "n_stocks": len(top_stocks),
            "hmm_regime": regime,
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Calculate cumulative returns
    if not results_df.empty:
        results_df["cumulative_return"] = (results_df["portfolio_value"] / capital) - 1
    
    return results_df


def run_backtest_monthly(
    predictions_df: pd.DataFrame,
    nifty_features_path: str,
    raw_dir: str,
    start: str,
    end: str,
    **kwargs
) -> pd.DataFrame:
    """
    Run monthly backtest (wrapper around run_backtest_daily).
    
    This function aggregates daily results to monthly frequency and computes
    additional metrics like Sharpe ratio and maximum drawdown.
    """
    # Run daily backtest
    daily_results = run_backtest_daily(
        predictions_df, nifty_features_path, raw_dir, start, end, **kwargs
    )
    
    if daily_results.empty:
        return daily_results
    
    # Convert to monthly frequency
    daily_results["date"] = pd.to_datetime(daily_results["date"])
    daily_results.set_index("date", inplace=True)
    
    # Resample to monthly
    monthly_results = daily_results.resample("M").last()
    monthly_results.reset_index(inplace=True)
    
    # Calculate monthly returns
    monthly_returns = monthly_results["portfolio_return"].dropna()
    
    if len(monthly_returns) > 1:
        # Calculate additional metrics
        sharpe = (monthly_returns.mean() / monthly_returns.std() * np.sqrt(12)) if monthly_returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + monthly_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (monthly_returns > 0).mean()
        
        # Add metrics to the last row
        monthly_results.loc[monthly_results.index[-1], "sharpe_ratio"] = sharpe
        monthly_results.loc[monthly_results.index[-1], "max_drawdown"] = max_drawdown
        monthly_results.loc[monthly_results.index[-1], "win_rate"] = win_rate
    
    return monthly_results
