"""Binance spot public market data via the official market-data mirror (data-api.binance.vision).

The main api.binance.com host is geo-restricted from this environment (HTTP 451);
the mirror serves the same public spot market-data endpoints read-only.
Bulk historical files come from data.binance.vision.
"""
from __future__ import annotations

import io
import logging
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

from cryptolab.exchanges.base import Ticker, fnum, ttl_cached
from cryptolab.util.http import HttpClient, HttpError
from cryptolab.util.timeutil import ExchangeClock, now_ms

log = logging.getLogger(__name__)

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


class Binance:
    name = "binance"

    def __init__(self, http: HttpClient, base: str = "https://data-api.binance.vision",
                 bulk_base: str = "https://data.binance.vision") -> None:
        self.http = http
        self.base = base.rstrip("/")
        self.bulk_base = bulk_base.rstrip("/")
        self.clock = ExchangeClock()
        self._info_cache: Optional[dict] = None

    # ------------------------------------------------------------------ basics
    def server_time(self) -> int:
        before = now_ms()
        d = self.http.get(f"{self.base}/api/v3/time")
        after = now_ms()
        self.clock.update(before, after, int(d["serverTime"]))
        return int(d["serverTime"])

    def exchange_info(self, refresh: bool = False) -> dict:
        if refresh:
            self.__dict__.pop("_ttl_cache", None)
        return ttl_cached(self, "exchangeInfo", 6 * 3600, lambda: self.http.get(
            f"{self.base}/api/v3/exchangeInfo", params={"permissions": "SPOT"}))

    def symbols(self, quote: Optional[str] = None, include_delisted: bool = True) -> list[dict]:
        out = []
        for s in self.exchange_info()["symbols"]:
            if quote and s["quoteAsset"] != quote:
                continue
            if not include_delisted and s["status"] != "TRADING":
                continue
            tick = None
            banded = False
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick = fnum(f.get("tickSize"))
                if f.get("filterType") == "PERCENT_PRICE_BY_SIDE":
                    # Tight price bands mark pegged assets and tokenized equities/ETFs (verified
                    # 2026-11-24: 212 symbols, all stablecoins/pegs or "<ticker>B" stock tokens).
                    banded = (fnum(f.get("askMultiplierUp")) or 9) <= 1.2 and (fnum(f.get("bidMultiplierDown")) or 0) >= 0.8
            out.append({"symbol": s["symbol"], "base": s["baseAsset"], "quote": s["quoteAsset"],
                        "status": s["status"], "tick_size": tick, "price_banded": banded})
        return out

    # ----------------------------------------------------------------- tickers
    def tickers(self) -> list[Ticker]:
        """All spot markets: 24h stats + best bid/ask, normalized."""
        stats = self.http.get(f"{self.base}/api/v3/ticker/24hr")
        books = {b["symbol"]: b for b in self.http.get(f"{self.base}/api/v3/ticker/bookTicker")}
        meta = {s["symbol"]: s for s in self.symbols()}
        out = []
        for t in stats:
            m = meta.get(t["symbol"])
            if m is None:
                continue
            b = books.get(t["symbol"], {})
            out.append(Ticker(
                exchange=self.name, symbol=t["symbol"], base=m["base"], quote=m["quote"],
                last=fnum(t.get("lastPrice")), bid=fnum(b.get("bidPrice")), ask=fnum(b.get("askPrice")),
                quote_volume_24h=fnum(t.get("quoteVolume")), ts_ms=int(t.get("closeTime") or 0) or None,
                tick_size=m["tick_size"], status_ok=(m["status"] == "TRADING"),
                extra={"count_24h": t.get("count"), "price_change_pct": fnum(t.get("priceChangePercent")),
                       "price_banded": m["price_banded"]}))
        return out

    # ------------------------------------------------------------------ klines
    def klines(self, symbol: str, interval: str = "1m", start_ms: Optional[int] = None,
               end_ms: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        rows = self.http.get(f"{self.base}/api/v3/klines", params=params)
        return klines_to_df(rows)

    def klines_range(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        step = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000, "1d": 86_400_000}[interval]
        frames, cur = [], start_ms
        while cur < end_ms:
            df = self.klines(symbol, interval, cur, end_ms, 1000)
            if df.empty:
                break
            frames.append(df)
            nxt = int(df["open_time"].iloc[-1]) + step
            if nxt <= cur:
                break
            cur = nxt
            if len(df) < 1000:
                break
        if not frames:
            return klines_to_df([])
        return pd.concat(frames, ignore_index=True).drop_duplicates("open_time").reset_index(drop=True)

    # ------------------------------------------------------------------ trades
    def agg_trades(self, symbol: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None,
                   limit: int = 1000) -> list[dict]:
        params: dict = {"symbol": symbol, "limit": limit}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        rows = self.http.get(f"{self.base}/api/v3/aggTrades", params=params)
        return [{"price": float(r["p"]), "qty": float(r["q"]), "ts_ms": int(r["T"]), "id": int(r["a"]),
                 "buyer_maker": bool(r["m"])} for r in rows]

    def last_trade_at_or_before(self, symbol: str, ts_ms: int, window_ms: int) -> Optional[dict]:
        """Last aggregated trade with timestamp <= ts_ms, searching back `window_ms`.

        aggTrades with startTime/endTime returns trades in ascending order; the range may
        not exceed 1 hour. If the window holds more than 1000 trades we page forward.
        """
        # Search progressively wider windows so busy markets need few pages.
        for w in sorted({min(2_000, window_ms), min(15_000, window_ms), min(60_000, window_ms), window_ms}):
            best = None
            cur = ts_ms - w
            for _ in range(50):
                rows = self.agg_trades(symbol, start_ms=cur, end_ms=ts_ms, limit=1000)
                rows = [r for r in rows if r["ts_ms"] <= ts_ms]
                if not rows:
                    break
                best = rows[-1]
                if len(rows) < 1000:
                    break
                nxt = rows[-1]["ts_ms"]
                if nxt <= cur:  # >1000 trades within one millisecond: take the last returned
                    break
                cur = nxt
            if best is not None:
                return best
        return None

    def latest_trade(self, symbol: str) -> Optional[dict]:
        rows = self.agg_trades(symbol, limit=1)
        return rows[-1] if rows else None

    # ------------------------------------------------------------- order book
    def depth(self, symbol: str, limit: int = 100) -> dict:
        d = self.http.get(f"{self.base}/api/v3/depth", params={"symbol": symbol, "limit": limit})
        return {"bids": [(float(p), float(q)) for p, q in d["bids"]],
                "asks": [(float(p), float(q)) for p, q in d["asks"]],
                "fetched_ms": now_ms()}

    # ------------------------------------------------------------- bulk files
    def bulk_monthly_klines(self, symbol: str, interval: str, year: int, month: int) -> Optional[pd.DataFrame]:
        url = f"{self.bulk_base}/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"
        return self._bulk(url)

    def bulk_daily_klines(self, symbol: str, interval: str, date: str) -> Optional[pd.DataFrame]:
        url = f"{self.bulk_base}/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
        return self._bulk(url)

    def _bulk(self, url: str) -> Optional[pd.DataFrame]:
        try:
            content = self.http.get(url, raw=True, timeout_s=60)
        except HttpError as e:
            if e.status == 404:
                return None
            raise
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                raw = pd.read_csv(fh, header=None)
        # Some files contain a header row
        if isinstance(raw.iloc[0, 0], str) and not str(raw.iloc[0, 0]).isdigit():
            raw = raw.iloc[1:]
        return klines_to_df(raw.values.tolist())


def _normalize_ts(arr: np.ndarray) -> np.ndarray:
    """Binance spot bulk files switched to MICROsecond timestamps from 2025-01-01.
    Normalize anything > 1e14 (i.e. microseconds) to milliseconds."""
    arr = arr.astype("int64")
    return np.where(arr > 10**14, arr // 1000, arr)


def klines_to_df(rows) -> pd.DataFrame:
    if rows is None or len(rows) == 0:
        return pd.DataFrame(columns=KLINE_COLS[:-1])
    df = pd.DataFrame([r[:12] for r in rows], columns=KLINE_COLS)
    df = df.drop(columns=["ignore"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype("int64")
    df["open_time"] = _normalize_ts(pd.to_numeric(df["open_time"]).values)
    df["close_time"] = _normalize_ts(pd.to_numeric(df["close_time"]).values)
    return df
