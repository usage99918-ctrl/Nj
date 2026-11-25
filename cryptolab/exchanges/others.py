"""Public read-only clients for Coinbase Exchange, Kraken, OKX and discovery-only exchanges.

These are used for (1) market discovery counts, (2) cross-exchange quality screening,
(3) cross-exchange price confirmation of the forecast candidate, and (4) derivatives
context (OKX funding rate, open interest, liquidations). No authenticated endpoint is
ever called.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from cryptolab.exchanges.base import Ticker, fnum, ttl_cached
from cryptolab.util.http import HttpClient

log = logging.getLogger(__name__)


def _iso_to_ms(s: str) -> Optional[int]:
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


class Coinbase:
    name = "coinbase"

    def __init__(self, http: HttpClient, base: str = "https://api.exchange.coinbase.com") -> None:
        self.http, self.base = http, base.rstrip("/")

    def products(self) -> list[dict]:
        return ttl_cached(self, "products", 6 * 3600, lambda: self.http.get(f"{self.base}/products"))

    def markets(self) -> list[dict]:
        return [{"symbol": p["id"], "base": p["base_currency"], "quote": p["quote_currency"],
                 "status_ok": p.get("status") == "online" and not p.get("trading_disabled", False),
                 "tick_size": fnum(p.get("quote_increment"))} for p in self.products()]

    def ticker(self, product_id: str) -> Ticker:
        t = self.http.get(f"{self.base}/products/{product_id}/ticker")
        base, quote = product_id.split("-", 1)
        last = fnum(t.get("price"))
        vol = fnum(t.get("volume"))
        return Ticker(self.name, product_id, base, quote, last, fnum(t.get("bid")), fnum(t.get("ask")),
                      (vol * last) if (vol is not None and last) else None, _iso_to_ms(t.get("time", "")))


class Kraken:
    name = "kraken"

    def __init__(self, http: HttpClient, base: str = "https://api.kraken.com") -> None:
        self.http, self.base = http, base.rstrip("/")
        self._pairs: Optional[dict] = None

    @staticmethod
    def _norm(asset: str) -> str:
        a = asset.upper()
        return {"XBT": "BTC", "XDG": "DOGE"}.get(a, a)

    def pairs(self) -> dict:
        return ttl_cached(self, "pairs", 6 * 3600, lambda: self.http.get(f"{self.base}/0/public/AssetPairs")["result"])

    def markets(self) -> list[dict]:
        out = []
        for k, p in self.pairs().items():
            ws = p.get("wsname") or ""
            if "/" not in ws:
                continue
            b, q = ws.split("/", 1)
            out.append({"symbol": k, "base": self._norm(b), "quote": self._norm(q),
                        "status_ok": p.get("status", "online") == "online", "tick_size": fnum(p.get("tick_size"))})
        return out

    def tickers(self) -> list[Ticker]:
        d = self.http.get(f"{self.base}/0/public/Ticker")
        meta = {m["symbol"]: m for m in self.markets()}
        out = []
        for k, t in d.get("result", {}).items():
            m = meta.get(k)
            if not m:
                continue
            last = fnum(t["c"][0])
            vwap = fnum(t["p"][1]) or last
            vol = fnum(t["v"][1])
            out.append(Ticker(self.name, k, m["base"], m["quote"], last, fnum(t["b"][0]), fnum(t["a"][0]),
                              (vol * vwap) if (vol is not None and vwap) else None, None,
                              tick_size=m["tick_size"], status_ok=m["status_ok"]))
        return out


class OKX:
    name = "okx"

    def __init__(self, http: HttpClient, base: str = "https://www.okx.com") -> None:
        self.http, self.base = http, base.rstrip("/")

    def _data(self, path: str, params: Optional[dict] = None) -> list:
        d = self.http.get(f"{self.base}{path}", params=params)
        if str(d.get("code")) != "0":
            raise RuntimeError(f"OKX error {d.get('code')}: {d.get('msg')}")
        return d["data"]

    def markets(self, inst_type: str = "SPOT") -> list[dict]:
        return ttl_cached(self, f"markets:{inst_type}", 6 * 3600, lambda: self._markets(inst_type))

    def _markets(self, inst_type: str) -> list[dict]:
        return [{"symbol": i["instId"], "base": i.get("baseCcy") or i["instId"].split("-")[0],
                 "quote": i.get("quoteCcy") or i.get("settleCcy") or "", "status_ok": i.get("state") == "live",
                 "tick_size": fnum(i.get("tickSz"))}
                for i in self._data("/api/v5/public/instruments", {"instType": inst_type})]

    def tickers(self, inst_type: str = "SPOT") -> list[Ticker]:
        meta = {m["symbol"]: m for m in self.markets(inst_type)}
        out = []
        for t in self._data("/api/v5/market/tickers", {"instType": inst_type}):
            m = meta.get(t["instId"])
            if not m:
                continue
            qv = fnum(t.get("volCcy24h")) if inst_type == "SPOT" else None
            out.append(Ticker(self.name, t["instId"], m["base"], m["quote"], fnum(t.get("last")),
                              fnum(t.get("bidPx")), fnum(t.get("askPx")), qv, int(t["ts"]) if t.get("ts") else None,
                              tick_size=m["tick_size"], status_ok=m["status_ok"]))
        return out

    def funding_rate(self, inst_id: str) -> Optional[dict]:
        try:
            d = self._data("/api/v5/public/funding-rate", {"instId": inst_id})
        except Exception as e:  # instrument may not exist
            log.debug("okx funding %s: %s", inst_id, e)
            return None
        if not d:
            return None
        r = d[0]
        return {"inst_id": inst_id, "funding_rate": fnum(r.get("fundingRate")),
                "next_funding_ms": int(r["fundingTime"]) if r.get("fundingTime") else None}

    def open_interest_all(self) -> dict[str, dict]:
        out = {}
        for r in self._data("/api/v5/public/open-interest", {"instType": "SWAP"}):
            out[r["instId"]] = {"oi_ccy": fnum(r.get("oiCcy")), "oi_usd": fnum(r.get("oiUsd")), "ts_ms": int(r["ts"])}
        return out

    def liquidations(self, underlying: str, limit: int = 100) -> list[dict]:
        try:
            d = self._data("/api/v5/public/liquidation-orders",
                           {"instType": "SWAP", "uly": underlying, "state": "filled", "limit": limit})
        except Exception as e:
            log.debug("okx liquidations %s: %s", underlying, e)
            return []
        out = []
        for blk in d:
            for x in blk.get("details", []):
                out.append({"side": x.get("side"), "pos_side": x.get("posSide"), "px": fnum(x.get("bkPx")),
                            "sz": fnum(x.get("sz")), "ts_ms": int(x.get("ts") or x.get("time") or 0)})
        return out

    def swap_contract_values(self) -> dict[str, float]:
        return {i["instId"]: fnum(i.get("ctVal")) or 0.0
                for i in self._data("/api/v5/public/instruments", {"instType": "SWAP"})}


# ---------------------------------------------------------------- discovery-only
class KuCoin:
    name = "kucoin"

    def __init__(self, http: HttpClient, base: str = "https://api.kucoin.com") -> None:
        self.http, self.base = http, base.rstrip("/")

    def tickers(self) -> list[Ticker]:
        syms = ttl_cached(self, "symbols", 6 * 3600, lambda: {
            s["symbol"]: s for s in self.http.get(f"{self.base}/api/v2/symbols")["data"]})
        d = self.http.get(f"{self.base}/api/v1/market/allTickers")["data"]
        ts = int(d.get("time") or 0) or None
        out = []
        for t in d["ticker"]:
            s = syms.get(t["symbol"])
            if not s:
                continue
            out.append(Ticker(self.name, t["symbol"], s["baseCurrency"], s["quoteCurrency"], fnum(t.get("last")),
                              fnum(t.get("buy")), fnum(t.get("sell")), fnum(t.get("volValue")), ts,
                              tick_size=fnum(s.get("priceIncrement")), status_ok=bool(s.get("enableTrading"))))
        return out


class Gate:
    name = "gate"

    def __init__(self, http: HttpClient, base: str = "https://api.gateio.ws") -> None:
        self.http, self.base = http, base.rstrip("/")

    def tickers(self) -> list[Ticker]:
        pairs = ttl_cached(self, "pairs", 6 * 3600, lambda: {
            p["id"]: p for p in self.http.get(f"{self.base}/api/v4/spot/currency_pairs")})
        out = []
        for t in self.http.get(f"{self.base}/api/v4/spot/tickers"):
            p = pairs.get(t["currency_pair"])
            if not p:
                continue
            prec = p.get("precision")
            tick = 10 ** (-int(prec)) if prec is not None else None
            out.append(Ticker(self.name, t["currency_pair"], p["base"], p["quote"], fnum(t.get("last")),
                              fnum(t.get("highest_bid")), fnum(t.get("lowest_ask")), fnum(t.get("quote_volume")),
                              None, tick_size=tick, status_ok=p.get("trade_status") == "tradable"))
        return out


class MEXC:
    name = "mexc"

    def __init__(self, http: HttpClient, base: str = "https://api.mexc.com") -> None:
        self.http, self.base = http, base.rstrip("/")

    def tickers(self) -> list[Ticker]:
        info = ttl_cached(self, "info", 6 * 3600, lambda: {
            s["symbol"]: s for s in self.http.get(f"{self.base}/api/v3/exchangeInfo", timeout_s=60)["symbols"]})
        out = []
        for t in self.http.get(f"{self.base}/api/v3/ticker/24hr", timeout_s=60):
            s = info.get(t["symbol"])
            if not s:
                continue
            out.append(Ticker(self.name, t["symbol"], s["baseAsset"], s["quoteAsset"], fnum(t.get("lastPrice")),
                              fnum(t.get("bidPrice")), fnum(t.get("askPrice")), fnum(t.get("quoteVolume")),
                              int(t.get("closeTime") or 0) or None,
                              status_ok=str(s.get("status")) in ("1", "ENABLED", "TRADING")))
        return out


class Bitfinex:
    name = "bitfinex"

    def __init__(self, http: HttpClient, base: str = "https://api-pub.bitfinex.com") -> None:
        self.http, self.base = http, base.rstrip("/")

    def tickers(self) -> list[Ticker]:
        out = []
        for t in self.http.get(f"{self.base}/v2/tickers", params={"symbols": "ALL"}):
            sym = t[0]
            if not sym.startswith("t") or len(t) < 11:
                continue
            pair = sym[1:]
            if ":" in pair:
                base, quote = pair.split(":", 1)
            else:
                base, quote = pair[:3], pair[3:]
            last, vol = fnum(t[7]), fnum(t[8])
            out.append(Ticker(self.name, sym, base.replace("UST", "USDT") if base == "UST" else base,
                              {"UST": "USDT"}.get(quote, quote), last, fnum(t[1]), fnum(t[3]),
                              (vol * last) if (vol and last) else None, None))
        return out


class Bitstamp:
    name = "bitstamp"

    def __init__(self, http: HttpClient, base: str = "https://www.bitstamp.net") -> None:
        self.http, self.base = http, base.rstrip("/")

    def tickers(self) -> list[Ticker]:
        info = ttl_cached(self, "info", 6 * 3600, lambda: {
            p["url_symbol"]: p for p in self.http.get(f"{self.base}/api/v2/trading-pairs-info/")})
        out = []
        for t in self.http.get(f"{self.base}/api/v2/ticker/"):
            pair = t.get("pair", "")
            if "/" not in pair:
                continue
            base, quote = pair.split("/", 1)
            p = info.get(pair.replace("/", "").lower(), {})
            last, vol = fnum(t.get("last")), fnum(t.get("volume"))
            out.append(Ticker(self.name, pair, base, quote, last, fnum(t.get("bid")), fnum(t.get("ask")),
                              (vol * last) if (vol and last) else None,
                              int(t["timestamp"]) * 1000 if t.get("timestamp") else None,
                              status_ok=p.get("trading", "Enabled") == "Enabled"))
        return out
