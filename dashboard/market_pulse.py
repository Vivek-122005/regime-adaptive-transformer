"""
yfinance market pulse for RAMT Live Predictor.

Cached at the Streamlit layer (ttl=3600) in app.py — this module stays free of st.*.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# Mirrors features/feature_engineering.MACRO_TICKERS (SP500 uses ^GSPC in pipeline)
TICKER_VIX = "^INDIAVIX"
TICKER_CRUDE = "CL=F"
TICKER_NIFTY = "^NSEI"
TICKER_INR = "INR=X"
TICKER_SP500 = "^GSPC"

MACRO_ORDER = ("INDIAVIX", "CRUDE", "USDINR", "SP500")
YF_MACRO = {
    "INDIAVIX": TICKER_VIX,
    "CRUDE": TICKER_CRUDE,
    "USDINR": TICKER_INR,
    "SP500": TICKER_SP500,
}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _download_adj_close(symbol: str, period: str = "3mo") -> tuple[pd.Series, pd.Timestamp | None]:
    """Return Adj Close series indexed by naive date + last valid timestamp."""
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval="1d", auto_adjust=False)
    except Exception:
        return pd.Series(dtype=float), None
    if hist is None or hist.empty:
        return pd.Series(dtype=float), None
    hist = _flatten_columns(hist)
    if "Adj Close" not in hist.columns and "Close" in hist.columns:
        hist["Adj Close"] = hist["Close"].astype(float)
    if "Adj Close" not in hist.columns:
        return pd.Series(dtype=float), None
    s = hist["Adj Close"].astype(float).dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    last_ts = s.index.max() if len(s) else None
    return s, last_ts


def _macro_ret1d_l1(px: pd.Series) -> float:
    """Training pipeline: log(px/px.shift(1)).shift(1) — last scalar."""
    if px is None or len(px) < 3:
        return float("nan")
    r1 = px.astype(float) / px.shift(1).astype(float)
    mret = np.log(r1.where(r1 > 0.0)).shift(1)
    v = mret.dropna()
    if v.empty:
        return float("nan")
    return float(v.iloc[-1])


def fetch_live_macro_data_engine() -> dict[str, Any]:
    """
    Pull macro + NIFTY spot context. Returns structured dict; never raises
    (errors are surfaced via ok=False).
    """
    out: dict[str, Any] = {
        "ok": False,
        "error": None,
        "fetched_at_utc": datetime.now(timezone.utc),
        "vix_level": float("nan"),
        "crude_level": float("nan"),
        "inr_level": float("nan"),
        "sp500_level": float("nan"),
        "nifty_level": float("nan"),
        "nifty_ret_1d_pct": float("nan"),
        "nifty_mom_20_pct": float("nan"),
        "macro_features": {},
        "last_bar_dates": {},
        "latest_data_date": None,
        "stale": True,
    }

    try:
        nifty, nd = _download_adj_close(TICKER_NIFTY, "6mo")
        if nifty is None or len(nifty) < 22:
            out["error"] = "Insufficient ^NSEI history from yfinance."
            return out

        c = nifty.astype(float)
        ret_1d = (c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0
        mom_20 = (c.iloc[-1] / c.iloc[-22] - 1.0) * 100.0
        out["nifty_level"] = float(c.iloc[-1])
        out["nifty_ret_1d_pct"] = float(ret_1d)
        out["nifty_mom_20_pct"] = float(mom_20)
        out["last_bar_dates"]["NIFTY"] = c.index[-1]

        macro_features: dict[str, float] = {}
        last_dates: list[pd.Timestamp] = []

        for name, sym in YF_MACRO.items():
            s, _ = _download_adj_close(sym, "6mo")
            col = f"Macro_{name}_Ret1d_L1"
            if s is None or len(s) < 3:
                macro_features[col] = float("nan")
                continue
            macro_features[col] = _macro_ret1d_l1(s)
            last_dates.append(s.index.max())
            if name == "INDIAVIX":
                out["vix_level"] = float(s.iloc[-1])
            elif name == "CRUDE":
                out["crude_level"] = float(s.iloc[-1])
            elif name == "USDINR":
                out["inr_level"] = float(s.iloc[-1])
            elif name == "SP500":
                out["sp500_level"] = float(s.iloc[-1])
            out["last_bar_dates"][name] = s.index.max()

        out["macro_features"] = macro_features
        if last_dates:
            out["latest_data_date"] = max(last_dates + [c.index.max()])

        lb = out["latest_data_date"]
        if lb is not None:
            lag = (pd.Timestamp.today().normalize() - pd.Timestamp(lb).normalize()).days
            out["stale"] = bool(lag > 1)

        out["ok"] = True
        return out

    except Exception as e:  # pragma: no cover
        out["error"] = str(e)
        return out
