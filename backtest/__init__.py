"""
Backtest module for regime-adaptive trading strategies.

This module provides organized backtesting capabilities including:
- Core backtesting engine with HMM regime adaptation
- Momentum + HMM strategy implementation
- Parameter sensitivity analysis scripts
- Ablation study utilities

Key components:
- backtest.core.backtest: Main run_backtest_daily function
- backtest.core.momentum_hmm_strategy: Unified momentum + HMM strategy
- backtest.scripts: Analysis and comparison scripts
"""

from __future__ import annotations

from .core.backtest import run_backtest_daily, run_backtest_monthly
from .core.momentum_hmm_strategy import RegimeAdaptiveMomentumStrategy

__all__ = [
    "run_backtest_daily",
    "run_backtest_monthly", 
    "RegimeAdaptiveMomentumStrategy",
]
