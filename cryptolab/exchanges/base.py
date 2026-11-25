"""Common types for exchange clients."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Optional

# Base assets that are pegged/fiat/commodity tokens or wrapped duplicates. Forecasting
# a stablecoin's price is not a meaningful directional experiment, so they are excluded.
STABLE_OR_PEGGED = {
    "USDT", "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDD", "PYUSD", "USDE", "USD1",
    "USDS", "RLUSD", "FRAX", "LUSD", "GUSD", "SUSD", "USTC", "UST", "EURI", "AEUR", "EUR", "EURC",
    "GBP", "AUD", "TRY", "BRL", "BIDR", "IDRT", "UAH", "NGN", "RUB", "ZAR", "PLN", "RON", "ARS",
    "JPY", "MXN", "COP", "CZK", "USD", "XUSD", "BFUSD", "USDB", "USDG", "EURQ", "USDQ", "EURR", "USDR",
    "U", "USD0", "USDX", "USDF", "SUSDE", "USDTB", "AUSD", "DEUSD", "EURCV", "TGBP", "XSGD",
    "PAXG", "XAUT",            # gold-backed tokens (commodity, not crypto-native)
    "WBTC", "WBETH", "BETH", "WETH", "CBBTC", "STETH", "WSTETH",  # wrapped/staked duplicates
}
# Tokenized equities/ETFs listed as spot "coins" (US ticker + "B" on Binance, "X" xStocks
# elsewhere). They are not crypto-native assets and track equity sessions, so they are
# excluded. The list is explicit (no reliable metadata flag exists); see docs/METHODOLOGY.md.
TOKENIZED_EQUITIES = {
    "SNDKB", "SPCXB", "SOXLB", "MUB", "AAPLB", "TSLAB", "NVDAB", "MSTRB", "COINB", "AMZNB", "GOOGLB",
    "METAB", "MSFTB", "SPYB", "QQQB", "HOODB", "CRCLB", "AMDB", "PLTRB", "NFLXB", "INTCB", "TQQQB",
    "AAPLX", "TSLAX", "NVDAX", "MSTRX", "COINX", "AMZNX", "GOOGLX", "METAX", "MSFTX", "SPYX", "QQQX",
    "HOODX", "CRCLX",
}
# Leveraged / ETF-style tokens (e.g. BTCUP, ETHDOWN, BTC3L, XRPBULL)
_LEVERAGED = re.compile(r"^(?:[A-Z0-9]{2,10}(?:UP|DOWN|BULL|BEAR)|[A-Z0-9]{2,10}\d[LS])$")
_LEV_WHITELIST = {"JUP", "SUP", "PUP", "CUP", "NUP", "GUP", "FUP"}  # real assets that end in UP


def is_excluded_base(base: str) -> Optional[str]:
    """Return an exclusion reason for a base asset, or None if it is eligible."""
    b = base.upper()
    if b in STABLE_OR_PEGGED:
        return "stable/pegged/wrapped"
    if b in TOKENIZED_EQUITIES:
        return "tokenized equity"
    if b not in _LEV_WHITELIST and _LEVERAGED.match(b) and len(b) > 4:
        return "leveraged token"
    return None


@dataclass
class Ticker:
    exchange: str
    symbol: str                   # exchange-native market id
    base: str
    quote: str
    last: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    quote_volume_24h: Optional[float]   # in quote currency
    ts_ms: Optional[int]                # exchange timestamp of the quote/ticker
    tick_size: Optional[float] = None
    status_ok: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def mid(self) -> Optional[float]:
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_bps(self) -> Optional[float]:
        if self.bid and self.ask and self.bid > 0 and self.ask >= self.bid:
            return (self.ask - self.bid) / ((self.ask + self.bid) / 2) * 1e4
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spread_bps"] = self.spread_bps
        return d


def ttl_cached(obj, key: str, ttl_s: float, fn):
    """Per-object metadata cache (exchange listings change rarely; tickers are never cached)."""
    import time as _t
    cache = obj.__dict__.setdefault("_ttl_cache", {})
    hit = cache.get(key)
    if hit is not None and _t.time() - hit[0] < ttl_s:
        return hit[1]
    val = fn()
    cache[key] = (_t.time(), val)
    return val


def fnum(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None
