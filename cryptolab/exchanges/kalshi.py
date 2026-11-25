"""Kalshi public market data (read-only research source).

Uses only public, unauthenticated GET endpoints of https://api.elections.kalshi.com/trade-api/v2.
No account, no keys, no orders. Kalshi prices are prediction-market contract prices (0-1 USD
per contract = market-implied probability of the contract's YES outcome). They are NOT crypto
spot prices: every Kalshi crypto contract settles on CF Benchmarks' BRTI/ETHUSD_RTI index
averaged over the 60 seconds before a stated time, which differs from any exchange's last trade.

Series used:
  KXBTC15M / KXETH15M  "BTC/ETH price up in next 15 mins?"  YES iff the 60-s index average
                       before window close >= the 60-s average before window open
                       (floor_strike = that opening average).
  KXBTCD / KXETHD      hourly (and daily 5 pm ET) "price above strike K at time T?" ladders.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from cryptolab.exchanges.base import fnum
from cryptolab.util.http import HttpClient

log = logging.getLogger(__name__)
BASE = "https://api.elections.kalshi.com/trade-api/v2"
DIRECTION_SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M"}
LADDER_SERIES = {"BTC": "KXBTCD", "ETH": "KXETHD"}


def iso_ms(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def normalize_market(m: dict) -> dict:
    """Flatten the fields we use; prices are YES-contract dollars (= implied probability)."""
    bid, ask = fnum(m.get("yes_bid_dollars")), fnum(m.get("yes_ask_dollars"))
    mid = (bid + ask) / 2 if (bid is not None and ask is not None and ask > 0) else None
    return {
        "ticker": m["ticker"], "event_ticker": m.get("event_ticker"), "series": m["ticker"].split("-")[0],
        "title": m.get("title"), "subtitle": m.get("yes_sub_title") or m.get("subtitle"),
        "status": m.get("status"), "result": m.get("result"),
        "open_ms": iso_ms(m.get("open_time")), "close_ms": iso_ms(m.get("close_time")),
        "strike_type": m.get("strike_type"), "floor_strike": fnum(m.get("floor_strike")),
        "cap_strike": fnum(m.get("cap_strike")),
        "yes_bid": bid, "yes_ask": ask, "yes_mid": mid, "spread": (ask - bid) if mid is not None else None,
        "yes_bid_size": fnum(m.get("yes_bid_size_fp")), "yes_ask_size": fnum(m.get("yes_ask_size_fp")),
        "last_price": fnum(m.get("last_price_dollars")), "volume": fnum(m.get("volume_fp")),
        "volume_24h": fnum(m.get("volume_24h_fp")), "open_interest": fnum(m.get("open_interest_fp")),
        "expiration_value": fnum(m.get("expiration_value")),
        "settlement_rule": (m.get("rules_primary") or "")[:400],
    }


class Kalshi:
    name = "kalshi"

    def __init__(self, http: HttpClient, base: str = BASE) -> None:
        self.http, self.base = http, base.rstrip("/")

    def status(self) -> dict:
        return self.http.get(f"{self.base}/exchange/status")

    def series(self, ticker: str) -> dict:
        return self.http.get(f"{self.base}/series/{ticker}")["series"]

    def markets(self, series_ticker: str, status: Optional[str] = None, max_pages: int = 50,
                min_close_ts: Optional[int] = None, max_close_ts: Optional[int] = None) -> list[dict]:
        out, cursor = [], None
        for _ in range(max_pages):
            p: dict = {"series_ticker": series_ticker, "limit": 1000}
            if status:
                p["status"] = status
            if min_close_ts:
                p["min_close_ts"] = int(min_close_ts)
            if max_close_ts:
                p["max_close_ts"] = int(max_close_ts)
            if cursor:
                p["cursor"] = cursor
            d = self.http.get(f"{self.base}/markets", params=p)
            ms = d.get("markets", [])
            out.extend(normalize_market(m) for m in ms)
            cursor = d.get("cursor")
            if not cursor or not ms:
                break
        return out

    def candlesticks(self, series_ticker: str, ticker: str, start_s: int, end_s: int, period_min: int = 1) -> list[dict]:
        d = self.http.get(f"{self.base}/series/{series_ticker}/markets/{ticker}/candlesticks",
                          params={"start_ts": int(start_s), "end_ts": int(end_s), "period_interval": period_min})
        out = []
        for c in d.get("candlesticks", []):
            yb, ya, pr = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
            out.append({"end_ms": int(c["end_period_ts"]) * 1000,
                        "yes_bid": fnum(yb.get("close_dollars")), "yes_ask": fnum(ya.get("close_dollars")),
                        "last": fnum(pr.get("close_dollars")), "volume": fnum(c.get("volume_fp")),
                        "open_interest": fnum(c.get("open_interest_fp"))})
        return out

    def orderbook(self, ticker: str, depth: int = 10) -> dict:
        return self.http.get(f"{self.base}/markets/{ticker}/orderbook", params={"depth": depth})

    def current_direction_market(self, asset: str, now_ms: int) -> Optional[dict]:
        """The open 15-minute up/down market whose window contains now."""
        ms = self.markets(DIRECTION_SERIES[asset], status="open", max_pages=1)
        live = [m for m in ms if m["open_ms"] and m["close_ms"] and m["open_ms"] <= now_ms < m["close_ms"]]
        return sorted(live, key=lambda m: m["close_ms"])[0] if live else None

    def ladder(self, asset: str, now_ms: int) -> list[dict]:
        """All strikes of the nearest-expiring open above/below event (one expiry time only)."""
        ms = [m for m in self.markets(LADDER_SERIES[asset], status="open", max_pages=3)
              if m["close_ms"] and m["close_ms"] > now_ms]
        if not ms:
            return []
        t = min(m["close_ms"] for m in ms)
        return sorted([m for m in ms if m["close_ms"] == t], key=lambda m: m["floor_strike"] or 0)


def ladder_summary(ladder: list[dict], spot: float) -> dict:
    """Implied P(index above spot at expiry) by interpolating the strike ladder (mid prices,
    two-sided quotes only). Describes ONE expiry; never mix expiries."""
    pts = [(m["floor_strike"], m["yes_mid"]) for m in ladder
           if m.get("floor_strike") and m.get("yes_mid") is not None and (m.get("spread") or 1) <= 0.10
           and m.get("yes_bid", 0) > 0 and m.get("yes_ask", 1) < 1]
    if len(pts) < 3:
        return {"available": False, "n_strikes_quoted": len(pts)}
    pts.sort()
    below = [p for p in pts if p[0] <= spot]
    above = [p for p in pts if p[0] > spot]
    if not below or not above:
        return {"available": False, "reason": "spot outside quoted strikes", "n_strikes_quoted": len(pts)}
    (k0, p0), (k1, p1) = below[-1], above[0]
    w = (spot - k0) / (k1 - k0)
    p_above = p0 + w * (p1 - p0)
    # implied median: strike where P(above) crosses 0.5 (linear interpolation)
    med = None
    for (ka, pa), (kb, pb) in zip(pts, pts[1:]):
        if pa >= 0.5 >= pb and pa != pb:
            med = ka + (pa - 0.5) / (pa - pb) * (kb - ka)
            break
    return {"available": True, "close_ms": ladder[0]["close_ms"], "event_ticker": ladder[0]["event_ticker"],
            "p_above_spot": p_above, "implied_median": med,
            "implied_median_vs_spot_pct": (med / spot - 1) * 100 if med else None,
            "n_strikes_quoted": len(pts), "total_volume": sum((m.get("volume") or 0) for m in ladder)}
