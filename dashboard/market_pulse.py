"""Compatibility shim — market pulse lives in `market_scraper.py`."""

from dashboard.market_scraper import fetch_live_macro_data_engine

__all__ = ["fetch_live_macro_data_engine"]
