"""
Regime-Adaptive Momentum Strategy

A unified implementation of the momentum + HMM regime-based position sizing strategy.
This combines 21-day momentum ranking with Hidden Markov Model regime detection
for dynamic position allocation across bull, high-volatility, and bear market states.

Strategy Logic:
- Signal: 21-day momentum (Ret_21d) for stock ranking
- Regime Detection: 3-state HMM on returns and volatility
- Position Sizing: 100% in bull, 50% in high-vol, 20% in bear
- Risk Controls: 7% stops (5% in bear), sector caps, drawdown protection
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from features.sectors import get_sector


class RegimeAdaptiveMomentumStrategy:
    """
    Unified momentum + HMM regime-adaptive strategy implementation.
    
    This class encapsulates:
    1. Momentum signal generation from 21-day returns
    2. HMM regime detection on market states
    3. Regime-based position sizing and risk management
    4. Complete backtesting with realistic friction and stops
    """
    
    # Strategy parameters
    MOMENTUM_WINDOW = 21  # 21-day momentum
    TOP_N_STOCKS = 5       # Maximum stocks per rebalance
    FRICTION_RATE = 0.0022 # 0.22% transaction cost
    STOP_LOSS_BULL = 0.07  # 7% stop loss
    STOP_LOSS_BEAR = 0.05  # 5% stop loss in bear markets
    MAX_WEIGHT = 0.20      # Maximum 20% per stock
    DRAWDOWN_TRIGGER = 0.15 # 15% portfolio drawdown trigger
    
    # HMM parameters
    HMM_STATES = 3
    HMM_MIN_OBS = 252      # Minimum observations for HMM fitting
    
    # Regime mapping (semantic labels)
    REGIME_LABELS = {0: "HIGH_VOL", 1: "BULL", 2: "BEAR"}
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        """
        Initialize the strategy with data paths.
        
        Args:
            data_dir: Directory containing raw and processed data
            results_dir: Directory for output results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Cache for loaded data
        self._price_cache: Dict[str, pd.Series] = {}
        self._nifty_features: Optional[pd.DataFrame] = None
        
    def generate_momentum_signals(self) -> pd.DataFrame:
        """
        Generate momentum-based stock predictions using 21-day returns.
        
        Returns:
            DataFrame with columns: Date, Ticker, predicted_alpha, actual_alpha, Period
        """
        print("Generating momentum signals...")
        
        # Load all processed feature files
        files = [
            f for f in self.processed_dir.glob("*_features.parquet")
            if not f.stem.startswith("_")
        ]
        
        if not files:
            raise FileNotFoundError(f"No feature files found in {self.processed_dir}")
        
        dfs = []
        for f in files:
            d = pd.read_parquet(
                f, columns=["Date", "Ticker", "Ret_21d", "Sector_Alpha", "Monthly_Alpha"]
            )
            dfs.append(d)
        
        panel = pd.concat(dfs, ignore_index=True)
        panel["Date"] = pd.to_datetime(panel["Date"])
        
        # Use sector-neutralized alpha as target if available
        target = "Sector_Alpha" if panel["Sector_Alpha"].notna().any() else "Monthly_Alpha"
        panel = panel.dropna(subset=["Ret_21d", target])
        
        # Filter to rebalance dates (21-day intervals)
        panel = self._filter_rebalance_dates(panel)
        
        # Pure momentum signal
        panel["predicted_alpha"] = panel["Ret_21d"]
        panel["actual_alpha"] = panel[target]
        panel["Period"] = panel["Date"].apply(
            lambda d: "Test" if d >= pd.Timestamp("2023-01-01") else "Train"
        )
        
        # Sort by date and momentum rank
        out = panel[["Date", "Ticker", "predicted_alpha", "actual_alpha", "Period"]]
        out = out.sort_values(["Date", "predicted_alpha"], ascending=[True, False])
        
        print(f"Generated {len(out):,} predictions across {out['Date'].nunique()} rebalance dates")
        return out
    
    def _filter_rebalance_dates(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Filter panel to 21-day rebalance dates."""
        # Get unique dates and sort
        dates = sorted(panel["Date"].unique())
        
        # Select every 21st trading day
        rebalance_dates = dates[::21]
        
        # Filter panel to rebalance dates
        return panel[panel["Date"].isin(rebalance_dates)]
    
    def detect_hmm_regimes(self, nifty_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes using 3-state Hidden Markov Model.
        
        Args:
            nifty_data: NIFTY index data with returns and volatility
            
        Returns:
            DataFrame with Date and HMM_Regime columns
        """
        print("Detecting HMM regimes...")
        
        # Prepare features for HMM
        lr = nifty_data["Ret_1d"]
        rv20 = nifty_data["Realized_Vol_20"]
        mask = lr.notna() & rv20.notna()
        
        regime_df = nifty_data[["Date"]].copy()
        regime_df["HMM_Regime"] = 0.0  # Default to HIGH_VOL
        
        if mask.sum() < self.HMM_MIN_OBS:
            print(f"Insufficient data for HMM: {mask.sum()} < {self.HMM_MIN_OBS}")
            return regime_df
        
        # Prepare data
        idx = nifty_data.index[mask]
        lr_vals = lr[mask].to_numpy(dtype=float)
        rv_vals = rv20[mask].to_numpy(dtype=float)
        X_raw = np.column_stack([lr_vals, rv_vals])
        
        # Fit HMM on expanding window
        prev_hmm = None
        regimes = []
        
        for end_pos in range(self.HMM_MIN_OBS - 1, len(idx)):
            X_hist_raw = X_raw[: end_pos + 1]
            
            # Standardize
            mu = X_hist_raw.mean(axis=0)
            sigma = X_hist_raw.std(axis=0)
            sigma = np.where(sigma == 0.0, 1.0, sigma)
            X_hist = (X_hist_raw - mu) / sigma
            
            try:
                # Fit HMM
                hmm = self._build_hmm(prev_hmm)
                hmm.fit(X_hist)
                raw_states = hmm.predict(X_hist)
                
                # Map states to semantic regimes
                means = {}
                for state in range(self.HMM_STATES):
                    sel = raw_states == state
                    means[state] = float(np.mean(lr_vals[:end_pos+1][sel])) if np.any(sel) else float("nan")
                
                mapping = self._semantic_regime_mapping(means)
                regime = float(mapping[int(raw_states[-1])])
                prev_hmm = hmm
                
            except Exception as e:
                print(f"HMM fit failed at position {end_pos}: {e}")
                regime = 0.0  # Default to HIGH_VOL
            
            regimes.append(regime)
        
        # Assign regimes to dates
        regime_dates = idx[self.HMM_MIN_OBS-1:]
        regime_df.loc[regime_df["Date"].isin(nifty_data.loc[regime_dates, "Date"]), "HMM_Regime"] = regimes
        
        print(f"HMM regime detection completed. Regime distribution:")
        for regime, label in self.REGIME_LABELS.items():
            count = (regime_df["HMM_Regime"] == regime).sum()
            print(f"  {label}: {count} days")
        
        return regime_df
    
    def _build_hmm(self, prev_hmm: Optional[GaussianHMM] = None) -> GaussianHMM:
        """Build or update HMM model."""
        if prev_hmm is not None:
            return prev_hmm
        
        return GaussianHMM(
            n_components=self.HMM_STATES,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            verbose=False
        )
    
    def _semantic_regime_mapping(self, means: Dict[int, float]) -> Dict[int, int]:
        """
        Map HMM states to semantic regimes based on mean returns.
        
        Args:
            means: Dictionary of state -> mean return
            
        Returns:
            Dictionary mapping raw state to semantic regime (0=HIGH_VOL, 1=BULL, 2=BEAR)
        """
        # Sort states by mean return
        sorted_states = sorted(means.items(), key=lambda x: x[1], reverse=True)
        
        # Map: highest return -> BULL (1), middle -> HIGH_VOL (0), lowest -> BEAR (2)
        mapping = {}
        mapping[sorted_states[0][0]] = 1  # BULL
        mapping[sorted_states[1][0]] = 0  # HIGH_VOL  
        mapping[sorted_states[2][0]] = 2  # BEAR
        
        return mapping
    
    def get_regime_position_config(self, regime: int) -> Tuple[float, int, float]:
        """
        Get position sizing configuration for a given regime.
        
        Args:
            regime: HMM regime (0=HIGH_VOL, 1=BULL, 2=BEAR)
            
        Returns:
            Tuple of (position_size, n_stocks, stop_loss)
        """
        if regime == 2:  # BEAR
            return 0.2, min(5, self.TOP_N_STOCKS), self.STOP_LOSS_BEAR
        elif regime == 0:  # HIGH_VOL
            return 0.5, 3, self.STOP_LOSS_BULL
        else:  # BULL
            return 1.0, self.TOP_N_STOCKS, self.STOP_LOSS_BULL
    
    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
        start_date: str = "2024-01-01",
        end_date: str = "2026-04-16",
        capital: float = 100000,
    ) -> pd.DataFrame:
        """
        Run complete backtest with regime-based position sizing.
        
        Args:
            predictions_df: DataFrame with momentum predictions
            start_date: Backtest start date
            end_date: Backtest end date
            capital: Initial capital
            
        Returns:
            DataFrame with backtest results
        """
        print("Running backtest with regime-adaptive position sizing...")
        
        # Load NIFTY features for regime detection
        nifty_features = self._load_nifty_features()
        regime_df = self.detect_hmm_regimes(nifty_features)
        
        # Prepare backtest data
        preds = predictions_df.copy()
        preds["Date"] = pd.to_datetime(preds["Date"])
        
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        # Get rebalance dates
        rebalance_dates = preds["Date"].drop_duplicates().sort_values()
        rebalance_dates = rebalance_dates[(rebalance_dates >= start_ts) & (rebalance_dates <= end_ts)]
        
        if len(rebalance_dates) < 2:
            print("Insufficient rebalance dates for backtest")
            return pd.DataFrame()
        
        # Merge regime information
        regime_df["Date"] = pd.to_datetime(regime_df["Date"])
        preds = preds.merge(regime_df[["Date", "HMM_Regime"]], on="Date", how="left")
        preds["HMM_Regime"] = preds["HMM_Regime"].fillna(0.0)
        
        # Run backtest
        results = []
        portfolio_value = capital
        forced_cash_next = False
        
        for i, rebalance_date in enumerate(rebalance_dates):
            if i == len(rebalance_dates) - 1:
                break  # Skip last rebalance (no future period)
            
            next_date = rebalance_dates[i + 1]
            
            # Get regime for this rebalance
            regime_data = preds[preds["Date"] == rebalance_date]
            if regime_data.empty:
                continue
            
            regime = int(regime_data["HMM_Regime"].iloc[0])
            position_size, n_stocks, stop_loss = self.get_regime_position_config(regime)
            
            # Get top stocks by momentum
            top_stocks = self._select_top_stocks(regime_data, n_stocks)
            
            if top_stocks.empty or forced_cash_next:
                result = {
                    "date": rebalance_date,
                    "portfolio_return": 0.0,
                    "regime": self.REGIME_LABELS[regime],
                    "stocks_held": [],
                    "cash": True,
                    "portfolio_value_start": portfolio_value,
                    "portfolio_value": portfolio_value,
                    "position_size": 0.0,
                    "n_stocks": 0,
                }
                results.append(result)
                forced_cash_next = False
                continue
            
            # Calculate portfolio return for the period
            portfolio_return = self._calculate_period_return(
                top_stocks, rebalance_date, next_date, stop_loss, portfolio_value * position_size
            )
            
            # Apply friction
            trade_value = portfolio_value * position_size
            friction_cost = trade_value * self.FRICTION_RATE
            net_return = portfolio_return - friction_cost / portfolio_value
            
            # Update portfolio value
            portfolio_value_start = portfolio_value
            portfolio_value *= (1 + net_return)
            
            # Check drawdown trigger
            if portfolio_value <= capital * (1 - self.DRAWDOWN_TRIGGER):
                forced_cash_next = True
            
            result = {
                "date": rebalance_date,
                "portfolio_return": net_return,
                "portfolio_return_gross": portfolio_return,
                "trade_value": trade_value,
                "friction_cost": friction_cost,
                "regime": self.REGIME_LABELS[regime],
                "stocks_held": top_stocks["Ticker"].tolist(),
                "cash": False,
                "portfolio_value_start": portfolio_value_start,
                "portfolio_value": portfolio_value,
                "position_size": position_size,
                "n_stocks": len(top_stocks),
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Calculate cumulative returns
        if not results_df.empty:
            results_df["cumulative_return"] = (results_df["portfolio_value"] / capital) - 1
        
        print(f"Backtest completed: {len(results_df)} periods")
        print(f"Final portfolio value: ₹{portfolio_value:,.0f}")
        print(f"Total return: {(portfolio_value/capital - 1)*100:.1f}%")
        
        return results_df
    
    def _load_nifty_features(self) -> pd.DataFrame:
        """Load NIFTY features for regime detection."""
        if self._nifty_features is None:
            nifty_path = self.processed_dir / "_NSEI_features.parquet"
            if not nifty_path.exists():
                raise FileNotFoundError(f"NIFTY features not found at {nifty_path}")
            
            self._nifty_features = pd.read_parquet(nifty_path)
        
        return self._nifty_features
    
    def _select_top_stocks(self, regime_data: pd.DataFrame, n_stocks: int) -> pd.DataFrame:
        """Select top stocks by momentum with sector caps."""
        if regime_data.empty:
            return pd.DataFrame()
        
        # Sort by momentum
        sorted_data = regime_data.sort_values("predicted_alpha", ascending=False)
        
        # Apply sector caps (max 1 stock per sector)
        selected_stocks = []
        used_sectors = set()
        
        for _, row in sorted_data.iterrows():
            sector = get_sector(row["Ticker"])
            
            if sector not in used_sectors:
                selected_stocks.append(row)
                used_sectors.add(sector)
            
            if len(selected_stocks) >= n_stocks:
                break
        
        return pd.DataFrame(selected_stocks)
    
    def _calculate_period_return(
        self,
        stocks: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        stop_loss: float,
        capital: float,
    ) -> float:
        """Calculate portfolio return for a period with stops."""
        if stocks.empty:
            return 0.0
        
        # Equal weight allocation
        weight_per_stock = 1.0 / len(stocks)
        portfolio_return = 0.0
        
        for _, stock in stocks.iterrows():
            ticker = stock["Ticker"]
            stock_return = self._get_stock_return(ticker, start_date, end_date, stop_loss)
            portfolio_return += weight_per_stock * stock_return
        
        return portfolio_return
    
    def _get_stock_return(
        self,
        ticker: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        stop_loss: float,
    ) -> float:
        """Get stock return for a period with stop-loss protection."""
        try:
            prices = self._load_price_series(ticker)
            
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
            
        except Exception:
            return 0.0
    
    def _load_price_series(self, ticker: str) -> pd.Series:
        """Load price series for a ticker with caching."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        
        # Load from raw data
        price_path = self.raw_dir / f"{ticker}.parquet"
        if not price_path.exists():
            # Try alternative naming
            price_path = self.raw_dir / f"{ticker}.NS.parquet"
        
        if not price_path.exists():
            raise FileNotFoundError(f"Price data not found for {ticker}")
        
        df = pd.read_parquet(price_path)
        df["Date"] = pd.to_datetime(df["Date"])
        prices = df.set_index("Date")["Adj Close"]
        
        # Cache for future use
        self._price_cache[ticker] = prices
        return prices
    
    def run_complete_strategy(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2026-04-16",
        capital: float = 100000,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Run the complete strategy from signal generation to backtesting.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            capital: Initial capital
            save_results: Whether to save results to file
            
        Returns:
            DataFrame with complete backtest results
        """
        print("=" * 60)
        print("REGIME-ADAPTIVE MOMENTUM STRATEGY")
        print("=" * 60)
        
        # Step 1: Generate momentum signals
        predictions = self.generate_momentum_signals()
        
        # Step 2: Run backtest
        results = self.run_backtest(predictions, start_date, end_date, capital)
        
        # Step 3: Save results
        if save_results and not results.empty:
            output_path = self.results_dir / "momentum_hmm_backtest.csv"
            results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
            
            # Save predictions
            pred_path = self.results_dir / "momentum_hmm_predictions.csv"
            predictions.to_csv(pred_path, index=False)
            print(f"Predictions saved to {pred_path}")
        
        # Step 4: Print summary
        self._print_performance_summary(results, capital)
        
        return results
    
    def _print_performance_summary(self, results: pd.DataFrame, capital: float) -> None:
        """Print performance summary statistics."""
        if results.empty:
            print("No results to summarize")
            return
        
        print("\n" + "=" * 40)
        print("PERFORMANCE SUMMARY")
        print("=" * 40)
        
        # Basic metrics
        final_value = results["portfolio_value"].iloc[-1]
        total_return = (final_value / capital) - 1
        cagr = (final_value / capital) ** (252 / len(results)) - 1
        
        # Risk metrics
        returns = results["portfolio_return"]
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        print(f"Initial Capital: ₹{capital:,.0f}")
        print(f"Final Value: ₹{final_value:,.0f}")
        print(f"Total Return: {total_return*100:.1f}%")
        print(f"CAGR: {cagr*100:.1f}%")
        print(f"Volatility: {volatility*100:.1f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.1f}%")
        print(f"Win Rate: {win_rate*100:.1f}%")
        
        # Regime analysis
        print("\nRegime Analysis:")
        regime_stats = results.groupby("regime")["portfolio_return"].agg(["count", "mean", "std"])
        for regime, stats in regime_stats.iterrows():
            print(f"  {regime}: {stats['count']} periods, "
                  f"avg return {stats['mean']*100:.2f}%, "
                  f"vol {stats['std']*100:.2f}%")


def main():
    """Main function to run the strategy."""
    strategy = RegimeAdaptiveMomentumStrategy()
    
    # Run complete strategy
    results = strategy.run_complete_strategy(
        start_date="2024-01-01",
        end_date="2026-04-16", 
        capital=100000,
        save_results=True
    )
    
    return results


if __name__ == "__main__":
    main()
