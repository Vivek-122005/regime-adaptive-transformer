"""
Core backtesting components.

This submodule contains the fundamental backtesting engine and strategy implementations.
"""

from __future__ import annotations

from .backtest import run_backtest_daily, run_backtest_monthly
from .momentum_hmm_strategy import RegimeAdaptiveMomentumStrategy

__all__ = [
    "run_backtest_daily",
    "run_backtest_monthly",
    "RegimeAdaptiveMomentumStrategy",
]
