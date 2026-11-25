"""Multi-exchange market scan with explicit quality checks and honest counts.

Counts reported for every cycle:
  markets_discovered      every spot market listed by every reachable exchange
  markets_quality_passed  markets passing ALL quality checks (USD-quoted, eligible asset,
                          24h volume, spread, staleness, tick size, trading status)
  markets_analyzed        Binance markets actually scored by the forecasting model
                          (top-N by trailing 30-day volume among quality-passed, with
                          complete feature history)
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cryptolab.data.history import PEG_MAX_DAILY_VOL
from cryptolab.exchanges.base import Ticker, is_excluded_base
from cryptolab.exchanges.binance import Binance
from cryptolab.util.timeutil import DAY, MIN, now_ms

log = logging.getLogger(__name__)


@dataclass
class ScanResult:
    started_ms: int
    finished_ms: int = 0
    per_exchange: dict = field(default_factory=dict)
    markets_discovered: int = 0
    markets_quality_passed: int = 0
    unique_assets_passed: int = 0
    binance_passed: list = field(default_factory=list)   # Ticker list (Binance, passed)
    binance_universe: list = field(default_factory=list)  # ranked symbols for forecasting
    warnings: list = field(default_factory=list)
    reject_reasons: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"started_ms": self.started_ms, "finished_ms": self.finished_ms,
                "markets_discovered": self.markets_discovered,
                "markets_quality_passed": self.markets_quality_passed,
                "unique_assets_passed": self.unique_assets_passed,
                "per_exchange": self.per_exchange, "binance_universe": self.binance_universe,
                "reject_reasons": self.reject_reasons, "warnings": self.warnings}


def quality_check(t: Ticker, q: dict, now: int) -> str | None:
    """Return the first failed check (reason) or None if the market passes."""
    if not t.status_ok:
        return "not trading"
    if t.quote.upper() not in q["usd_quotes"]:
        return "non-USD quote"
    if is_excluded_base(t.base):
        return "excluded asset type"
    if t.extra.get("price_banded"):
        return "price-banded (stablecoin/tokenized equity)"
    if not t.last or t.last <= 0:
        return "no price"
    if t.quote_volume_24h is None or t.quote_volume_24h < q["min_quote_volume_24h_usd"]:
        return "low 24h volume"
    sp = t.spread_bps
    if sp is None:
        return "no bid/ask"
    if sp > q["max_spread_bps"]:
        return "wide spread"
    if t.ts_ms is not None and now - t.ts_ms > q["max_staleness_s"] * 1000:
        return "stale"
    if t.tick_size and t.last and t.tick_size / t.last * 1e4 > q["max_tick_bps"]:
        return "coarse tick"
    return None


class Scanner:
    def __init__(self, cfg, binance: Binance, others: list, cache_dir: Path) -> None:
        self.cfg, self.bn, self.others = cfg, binance, others
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _vol30(self, symbols: list[str]) -> pd.DataFrame:
        """Trailing 30-day quote volume + daily vol per symbol (cached for 6 hours)."""
        path = self.cache_dir / "daily_30d.parquet"
        cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["symbol"])
        fresh = cached[cached.get("fetched_ms", pd.Series(dtype="int64")) > now_ms() - 6 * 3600_000] \
            if not cached.empty else cached
        have = set(fresh["symbol"]) if not fresh.empty else set()
        rows = [fresh] if not fresh.empty else []
        end = now_ms()
        start = end - 32 * DAY
        today0 = (end // DAY) * DAY
        for s in symbols:
            if s in have:
                continue
            try:
                d = self.bn.klines(s, "1d", start, end, 40)
            except Exception as e:
                log.warning("daily klines %s: %s", s, e)
                continue
            d = d[d["open_time"] < today0]  # completed days only
            if len(d) < 2:
                continue
            lr = np.diff(np.log(d["close"].values))
            rows.append(pd.DataFrame([{"symbol": s, "qv30": float(d["quote_volume"].tail(30).sum()),
                                       "days": int(min(len(d), 30)), "daily_vol": float(np.std(lr, ddof=1)),
                                       "fetched_ms": now_ms()}]))
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not out.empty:
            out = out.drop_duplicates("symbol", keep="last")
            out.to_parquet(path, index=False)
        return out

    def scan(self) -> ScanResult:
        q = self.cfg["quality"]
        res = ScanResult(started_ms=now_ms())
        assets = set()
        rejects: dict[str, int] = {}
        sources = [self.bn] + self.others

        def fetch(ex):
            t0 = time.time()
            try:
                return ex, ex.tickers(), None, time.time() - t0
            except Exception as e:  # one exchange failing never stops the scan
                return ex, None, e, time.time() - t0

        with ThreadPoolExecutor(len(sources)) as pool:
            fetched = list(pool.map(fetch, sources))
        for ex, tickers, err, secs in fetched:
            name = getattr(ex, "name", str(ex))
            if err is not None:
                res.per_exchange[name] = {"ok": False, "error": str(err)[:200], "seconds": round(secs, 2)}
                res.warnings.append(f"{name}: scan failed ({str(err)[:120]})")
                continue
            t0 = time.time() - secs
            now = now_ms()
            passed = []
            for t in tickers:
                why = quality_check(t, q, now)
                if why is None:
                    passed.append(t)
                else:
                    rejects[why] = rejects.get(why, 0) + 1
            res.per_exchange[name] = {"ok": True, "discovered": len(tickers), "quality_passed": len(passed),
                                      "seconds": round(time.time() - t0, 2)}
            res.markets_discovered += len(tickers)
            res.markets_quality_passed += len(passed)
            assets |= {t.base.upper() for t in passed}
            if ex is self.bn:
                res.binance_passed = [t for t in passed if t.quote == "USDT"]
        res.unique_assets_passed = len(assets)
        res.reject_reasons = dict(sorted(rejects.items(), key=lambda kv: -kv[1]))
        # Forecast universe: same rule as the backtest (top-N by trailing 30-day volume, peg filter)
        cands = [t.symbol for t in res.binance_passed]
        v = self._vol30(cands)
        if v.empty:
            res.warnings.append("could not rank universe by 30-day volume")
        else:
            v = v[v["symbol"].isin(cands) & (v["days"] >= 20) & (v["daily_vol"] >= PEG_MAX_DAILY_VOL)]
            res.binance_universe = v.sort_values("qv30", ascending=False)["symbol"].head(
                self.cfg["universe"]["top_n"]).tolist()
        res.finished_ms = now_ms()
        return res


class KlineCache:
    """Rolling local cache of completed 1-minute Binance klines (incremental refresh)."""
    KEEP_BARS = 10_300   # 7 days + buffer (per-coin AR features use a 7-day window)

    def __init__(self, bn: Binance, directory: Path) -> None:
        self.bn = bn
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, symbol: str, server_now: int) -> pd.DataFrame:
        path = self.dir / f"{symbol}.parquet"
        old = pd.read_parquet(path) if path.exists() else None
        if old is not None and not old.empty and int(old["open_time"].max()) > server_now - self.KEEP_BARS * MIN:
            start = int(old["open_time"].max())
        else:
            old = None
            start = server_now - self.KEEP_BARS * MIN
        new = self.bn.klines_range(symbol, "1m", start, server_now)
        df = new if old is None else pd.concat([old, new], ignore_index=True)
        df = df.drop_duplicates("open_time", keep="last").sort_values("open_time")
        # Only bars that have CLOSED by server_now are used (no partial/forming bar).
        df = df[df["open_time"] + MIN <= server_now]
        df = df.tail(self.KEEP_BARS).reset_index(drop=True)
        cols = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]
        df = df[cols]
        df.to_parquet(path, index=False)
        return df
