"""
HTML scraping engine for Live Predictor market pulse (requests + BeautifulSoup).

Targets: Yahoo Finance quotes (crude, NIFTY, INR, SP500, India VIX), with fallbacks
to Investing.com / Moneycontrol for India VIX.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 5
MAX_ATTEMPTS = 3  # initial + 2 retries

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


def _log_scrape_ok(ticker: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = f"Successfully scraped {ticker} at {ts}"
    print(msg, flush=True)
    logger.info(msg)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_yahoo_quote_html(html: str) -> dict[str, float | None]:
    """
    Yahoo Finance embeds quote JSON in the HTML. Extract regularMarketPrice and
    regularMarketChangePercent raw values.
    """
    out: dict[str, float | None] = {"price": None, "change_pct": None}
    if not html:
        return out
    # Common patterns in QuoteSummary / streamer bootstrap JSON
    mp = re.search(
        r'"regularMarketPrice"\s*:\s*\{\s*"raw"\s*:\s*([0-9.eE+-]+)',
        html,
    )
    cp = re.search(
        r'"regularMarketChangePercent"\s*:\s*\{\s*"raw"\s*:\s*([0-9.eE+-]+)',
        html,
    )
    if mp:
        out["price"] = _safe_float(mp.group(1))
    if cp:
        out["change_pct"] = _safe_float(cp.group(1))
    # Fallback: meta / fin-streamer text
    if out["price"] is None:
        m2 = re.search(r'data-field="regularMarketPrice"[^>]*value="([0-9.eE+-]+)"', html)
        if m2:
            out["price"] = _safe_float(m2.group(1))
    return out


@dataclass
class YahooQuote:
    symbol: str
    price: float | None = None
    change_pct: float | None = None
    ok: bool = False


class MarketScraper:
    """
    User-agent rotation, 5s timeout, 3 attempts (initial + 2 retries) per URL.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._ua_i = 0

    def _next_headers(self) -> dict[str, str]:
        self._ua_i += 1
        ua = USER_AGENTS[self._ua_i % len(USER_AGENTS)]
        return {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
        }

    def _get_text(self, url: str) -> str | None:
        last_err: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = self._session.get(
                    url,
                    headers=self._next_headers(),
                    timeout=REQUEST_TIMEOUT_S,
                )
                if r.status_code == 200 and len(r.text) > 200:
                    return r.text
            except requests.RequestException as e:
                last_err = e
                time.sleep(0.35 * (attempt + 1))
        if last_err:
            logger.debug("Request failed for %s: %s", url, last_err)
        return None

    def scrape_yahoo_quote(self, symbol: str, label: str) -> YahooQuote:
        """
        symbol: Yahoo symbol, e.g. 'CL=F', '^NSEI', '^INDIAVIX', 'INR=X', '^GSPC'
        """
        url = f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}"
        html = self._get_text(url)
        if not html:
            return YahooQuote(symbol=label, ok=False)
        d = _parse_yahoo_quote_html(html)
        price, ch = d.get("price"), d.get("change_pct")
        if price is None and ch is None:
            return YahooQuote(symbol=label, ok=False)
        if price is not None:
            _log_scrape_ok(label)
        return YahooQuote(symbol=label, price=price, change_pct=ch, ok=price is not None)

    def scrape_investing_india_vix(self) -> tuple[float | None, float | None]:
        """Returns (price, change_pct if found)."""
        url = "https://in.investing.com/indices/india-vix"
        html = self._get_text(url)
        if not html:
            return None, None
        soup = BeautifulSoup(html, "html.parser")
        price: float | None = None
        chg: float | None = None
        node = soup.select_one('[data-test="instrument-price-last"]')
        if node:
            t = node.get_text(strip=True).replace(",", "")
            price = _safe_float(re.sub(r"[^\d.\-]", "", t))
        chg_node = soup.select_one('[data-test="instrument-price-change"]')
        if chg_node:
            t = chg_node.get_text(strip=True).replace(",", "")
            m = re.search(r"([\-0-9.]+)\s*%", t)
            if m:
                chg = _safe_float(m.group(1))
        if price is not None:
            _log_scrape_ok("INDIA_VIX (investing.com)")
        return price, chg

    def scrape_moneycontrol_india_vix(self) -> float | None:
        url = "https://www.moneycontrol.com/indian-indices/india-vix-36.html"
        html = self._get_text(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        # Common MC index price containers (site changes often — try several)
        for sel in (
            "#indim_last_price",
            "#last_price",
            "span.idx_val",
            "div#indices_last_price",
        ):
            node = soup.select_one(sel)
            if node:
                t = re.sub(r"[^\d.\-]", "", node.get_text(strip=True).replace(",", ""))
                v = _safe_float(t)
                if v is not None:
                    _log_scrape_ok("INDIA_VIX (moneycontrol.com)")
                    return v
        # Regex fallback
        m = re.search(
            r"(?:India\s*VIX|VIX)[^0-9]{0,40}([0-9]{1,2}\.[0-9]{2}|[0-9]{1,2})",
            html,
            re.I,
        )
        if m:
            v = _safe_float(m.group(1))
            if v is not None:
                _log_scrape_ok("INDIA_VIX (moneycontrol.com regex)")
                return v
        return None

    def run_pulse(self) -> dict[str, Any]:
        """
        Scrape all primary Yahoo targets + India VIX with Yahoo-first, then backups.
        """
        crude = self.scrape_yahoo_quote("CL=F", "CL=F (Yahoo)")
        nifty = self.scrape_yahoo_quote("^NSEI", "^NSEI (Yahoo)")
        inr = self.scrape_yahoo_quote("INR=X", "INR=X (Yahoo)")
        sp = self.scrape_yahoo_quote("^GSPC", "^GSPC (Yahoo)")
        vix_y = self.scrape_yahoo_quote("^INDIAVIX", "^INDIAVIX (Yahoo)")

        vix_price = vix_y.price
        vix_chg = vix_y.change_pct
        if vix_price is None:
            iv, ic = self.scrape_investing_india_vix()
            vix_price, vix_chg = iv, ic
        if vix_price is None:
            vix_price = self.scrape_moneycontrol_india_vix()
            vix_chg = None

        return {
            "crude": crude,
            "nifty": nifty,
            "inr": inr,
            "sp500": sp,
            "vix_yahoo": vix_y,
            "vix_price": vix_price,
            "vix_change_pct": vix_chg,
        }


def log_return_pct_to_macro_slot(change_pct: float | None) -> float | None:
    """Map daily change % to a log-return style scalar for Macro_* slots (demo bridge)."""
    if change_pct is None:
        return None
    try:
        return float(np.log1p(change_pct / 100.0))
    except Exception:
        return None


def _nifty_index_last_metrics(root: Path) -> tuple[float, float, float | None]:
    """Prefer benchmark _NSEI_features for index 1d / ~21d momentum proxies."""
    p = root / "data" / "processed" / "_NSEI_features.parquet"
    if not p.is_file():
        return float("nan"), float("nan"), None
    df = pd.read_parquet(p)
    if df.empty:
        return float("nan"), float("nan"), None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    last = df.iloc[-1]
    r1 = last.get("Ret_1d")
    r21 = last.get("Ret_21d")
    d1 = float("nan")
    d21 = float("nan")
    if r1 is not None and np.isfinite(r1):
        d1 = float((np.exp(float(r1)) - 1.0) * 100.0)
    if r21 is not None and np.isfinite(r21):
        d21 = float((np.exp(float(r21)) - 1.0) * 100.0)
    return d1, d21, last.get("Date")


def parquet_last_known(root: Path, ticker_stem: str) -> dict[str, Any]:
    """Last processed row for fallback (levels may be absent)."""
    from models.ramt.dataset import MACRO_COLS

    empty = {
        "macro_features": {},
        "nifty_ret_1d_pct": float("nan"),
        "nifty_mom_20_pct": float("nan"),
        "vix_level": float("nan"),
        "crude_level": float("nan"),
        "inr_level": float("nan"),
        "sp500_level": float("nan"),
        "nifty_level": float("nan"),
        "latest_data_date": None,
    }
    n1, n21, nd = _nifty_index_last_metrics(root)
    if np.isfinite(n1):
        empty["nifty_ret_1d_pct"] = n1
    if np.isfinite(n21):
        empty["nifty_mom_20_pct"] = n21
    if nd is not None:
        empty["latest_data_date"] = nd

    pq = root / "data" / "processed" / f"{ticker_stem}_features.parquet"
    if not pq.is_file():
        return empty
    df = pd.read_parquet(pq)
    if df.empty:
        return empty
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    last = df.iloc[-1]

    mf: dict[str, float] = {}
    for c in MACRO_COLS:
        if c in last.index:
            mf[c] = float(last[c])
    # If no NIFTY benchmark file, use stock row as rough proxy for index fields
    if not np.isfinite(empty["nifty_ret_1d_pct"]):
        r1 = last.get("Ret_1d")
        if r1 is not None and np.isfinite(r1):
            empty["nifty_ret_1d_pct"] = float((np.exp(float(r1)) - 1.0) * 100.0)
    if not np.isfinite(empty["nifty_mom_20_pct"]):
        r21 = last.get("Ret_21d")
        if r21 is not None and np.isfinite(r21):
            empty["nifty_mom_20_pct"] = float((np.exp(float(r21)) - 1.0) * 100.0)
    empty["macro_features"] = mf
    if empty["latest_data_date"] is None:
        empty["latest_data_date"] = last.get("Date")
    return empty


def merge_scrape_with_parquet(
    root: Path,
    ticker_stem: str,
    pulse: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the same dict shape as legacy fetch_live_macro_data_engine.
    Per-field: use scrape when ok, else Parquet last-known.
    """
    from models.ramt.dataset import MACRO_COLS

    fb = parquet_last_known(root, ticker_stem)
    out: dict[str, Any] = {
        "ok": True,
        "error": None,
        "fetched_at_utc": datetime.now(timezone.utc),
        "vix_level": fb["vix_level"],
        "crude_level": fb["crude_level"],
        "inr_level": fb["inr_level"],
        "sp500_level": fb["sp500_level"],
        "nifty_level": fb["nifty_level"],
        "nifty_ret_1d_pct": fb["nifty_ret_1d_pct"],
        "nifty_mom_20_pct": fb["nifty_mom_20_pct"],
        "macro_features": dict(fb["macro_features"]),
        "last_bar_dates": {},
        "latest_data_date": fb.get("latest_data_date"),
        "stale": True,
        "sources": {},
    }

    crude: YahooQuote = pulse["crude"]
    nifty: YahooQuote = pulse["nifty"]
    inr: YahooQuote = pulse["inr"]
    sp: YahooQuote = pulse["sp500"]

    if crude.ok and crude.price is not None:
        out["crude_level"] = float(crude.price)
        out["sources"]["CRUDE"] = "yahoo"
    if nifty.ok and nifty.price is not None:
        out["nifty_level"] = float(nifty.price)
        out["sources"]["NIFTY"] = "yahoo"
    if nifty.change_pct is not None:
        out["nifty_ret_1d_pct"] = float(nifty.change_pct)
    if inr.ok and inr.price is not None:
        out["inr_level"] = float(inr.price)
    if sp.ok and sp.price is not None:
        out["sp500_level"] = float(sp.price)

    vp = pulse.get("vix_price")
    if vp is not None and np.isfinite(vp):
        out["vix_level"] = float(vp)

    mf = dict(out["macro_features"])
    vix_y = pulse.get("vix_yahoo")
    vix_chg: float | None = None
    if vix_y is not None and vix_y.change_pct is not None:
        vix_chg = float(vix_y.change_pct)
    elif pulse.get("vix_change_pct") is not None:
        vix_chg = float(pulse["vix_change_pct"])

    updates: list[tuple[str, float | None, str]] = [
        ("Macro_CRUDE_Ret1d_L1", crude.change_pct, "CRUDE"),
        ("Macro_INDIAVIX_Ret1d_L1", vix_chg, "INDIAVIX"),
        ("Macro_USDINR_Ret1d_L1", inr.change_pct, "USDINR"),
        ("Macro_SP500_Ret1d_L1", sp.change_pct, "SP500"),
    ]
    for col, chg, src in updates:
        slot = log_return_pct_to_macro_slot(chg)
        if slot is not None and np.isfinite(slot):
            mf[col] = float(slot)
            out["sources"][src] = "scrape_change_pct"

    # Fill any remaining nan macro from fb
    for c in MACRO_COLS:
        if c not in mf or not np.isfinite(mf.get(c, np.nan)):
            if c in fb["macro_features"]:
                mf[c] = float(fb["macro_features"][c])

    out["macro_features"] = mf

    # Momentum: if NIFTY scrape missed, keep fb; optional refine from Ret_21d not available intraday
    if not (nifty.ok and nifty.change_pct is not None):
        out["nifty_ret_1d_pct"] = fb["nifty_ret_1d_pct"]
    out["nifty_mom_20_pct"] = fb["nifty_mom_20_pct"]

    lb = out.get("latest_data_date")
    if lb is not None:
        lag = (pd.Timestamp.today().normalize() - pd.Timestamp(lb).normalize()).days
        out["stale"] = bool(lag > 1)

    return out


def fetch_live_macro_data_engine(ticker_stem: str, root: Path) -> dict[str, Any]:
    """
    Orchestrate MarketScraper + Parquet fallback. No exceptions to caller.
    """
    scraper = MarketScraper()
    try:
        pulse = scraper.run_pulse()
        return merge_scrape_with_parquet(root, ticker_stem, pulse)
    except Exception as e:
        logger.exception("Market scrape failed: %s", e)
        fb = parquet_last_known(root, ticker_stem)
        return {
            "ok": len(fb.get("macro_features", {})) > 0
            or np.isfinite(fb.get("nifty_ret_1d_pct", np.nan)),
            "error": str(e),
            "fetched_at_utc": datetime.now(timezone.utc),
            "vix_level": fb["vix_level"],
            "crude_level": fb["crude_level"],
            "inr_level": fb["inr_level"],
            "sp500_level": fb["sp500_level"],
            "nifty_level": fb["nifty_level"],
            "nifty_ret_1d_pct": fb["nifty_ret_1d_pct"],
            "nifty_mom_20_pct": fb["nifty_mom_20_pct"],
            "macro_features": fb["macro_features"],
            "last_bar_dates": {},
            "latest_data_date": fb.get("latest_data_date"),
            "stale": True,
            "sources": {},
        }
