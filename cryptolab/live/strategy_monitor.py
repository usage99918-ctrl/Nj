"""Live signals for the three externally proposed strategies + order-book collection.

* Every minute: order-book snapshot (spread, bid/ask depth within 50 bps) for the current forecast
  universe -> rolling 24 h series used by S3-book (the order-book version of Liquidity Recovery).
* Every 5 minutes: S1 (ETH/BTC), S2 (breakout retest) and S3-book run on CLOSED bars only and emit
  new signals as immutable `signal` records (id = strategy:symbol:bar time, so restarts never
  duplicate). Only bars after the monitor started are eligible (no replay of history).
* Signals older than 24 h get an immutable `signal_outcome` record with forward returns at
  60 / 240 / 1440 minutes (last 1-minute close at or before each horizon).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cryptolab.features import to_grid
from cryptolab.live.context import book_metrics
from cryptolab.live.scanner import KlineCache
from cryptolab.storage.db import DuplicateRecord
from cryptolab.strategies import friend as F
from cryptolab.util.timeutil import MIN, now_ms, to_iso

log = logging.getLogger(__name__)
HORIZONS = (60, 240, 1440)


class StrategyMonitor:
    name = "strategy_signals"

    def __init__(self, cfg, store, bn, cache: KlineCache) -> None:
        self.cfg, self.store, self.bn, self.cache = cfg, store, bn, cache
        run_dir = cfg.path("data_dir") / "run"
        self.long_cache = KlineCache(bn, run_dir / "klines_long", keep_bars=46_500)  # S1 needs 31 days
        self.books_dir = run_dir / "books"
        self.books_dir.mkdir(parents=True, exist_ok=True)
        self.books: dict[str, list] = {}
        self.started_ms = now_ms()
        self.last_minute = None
        self.last_detect = None
        self.last_verify = 0

    # ------------------------------------------------------------------ inputs
    def universe(self) -> list[str]:
        r = self.store.conn.execute("SELECT payload FROM scans ORDER BY seq DESC LIMIT 1").fetchone()
        return json.loads(r[0]).get("binance_universe", [])[:25] if r else []

    def _book_path(self, sym: str) -> Path:
        return self.books_dir / f"{sym}.parquet"

    def collect_books(self, minute: int) -> None:
        for sym in self.universe():
            try:
                m = book_metrics(self.bn.depth(sym, 100))
            except Exception as e:
                log.debug("book %s: %s", sym, e)
                continue
            if not m.get("ok"):
                continue
            rows = self.books.setdefault(sym, [])
            rows.append({"t": minute, "spread": m["spread_bps"] / 1e4, "bid_depth_50": m["bid_depth_50bps_usd"],
                         "ask_depth_50": m["ask_depth_50bps_usd"], "imbalance_10": m["imbalance_10bps"]})
        if minute % (10 * MIN) == 0:  # persist the last 24 h so restarts keep the series
            for sym, rows in self.books.items():
                p = self._book_path(sym)
                old = pd.read_parquet(p) if p.exists() else pd.DataFrame()
                df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("t")
                df[df["t"] >= minute - 1440 * MIN].to_parquet(p, index=False)
                self.books[sym] = []

    def book_series(self, sym: str, index) -> tuple[pd.Series, pd.Series] | None:
        p = self._book_path(sym)
        parts = [pd.read_parquet(p)] if p.exists() else []
        if self.books.get(sym):
            parts.append(pd.DataFrame(self.books[sym]))
        if not parts:
            return None
        b = pd.concat(parts, ignore_index=True).drop_duplicates("t").set_index("t").sort_index()
        b = b.reindex(index)
        return b["spread"], b["bid_depth_50"]

    # ---------------------------------------------------------------- signals
    def _emit(self, strategy: str, sym: str, r: dict, stop, take, hold: int, validation: str, extra: dict) -> None:
        t = int(r["t"])
        sid = f"{strategy}:{sym}:{t}"
        rec = {"signal_id": sid, "strategy": strategy, "symbol": sym, "signal_ms": t, "signal_utc": to_iso(t),
               "detected_ms": now_ms(), "detected_utc": to_iso(now_ms()), "direction": int(r["direction"]),
               "side": "LONG" if r["direction"] > 0 else "SHORT (theoretical on spot)",
               "bar_close_price": float(r["price"]), "stop": None if stop is None else float(stop),
               "take": None if take is None else float(take), "max_hold_min": hold,
               "validation_status": validation, "details": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                            for k, v in {**r, **extra}.items() if k != "t"}}
        try:
            self.store.record("signal", rec)
            log.info("SIGNAL %s %s %s @ %s", strategy, sym, rec["side"], rec["bar_close_price"])
        except DuplicateRecord:
            pass

    def detect(self, server_now: int) -> None:
        since = max(self.started_ms - 10 * MIN, server_now - 30 * MIN)
        # S1 on the ETH/BTC spot pair
        try:
            ratio = to_grid(self.long_cache.get("ETHBTC", server_now))
            s1 = F.s1_signals(ratio, since_ms=since)
            sd4 = np.log(ratio["close"]).diff().rolling(1440 * 7, min_periods=1440).std() * math.sqrt(240)
            for r in s1.to_dict("records"):
                s = float(sd4.get(int(r["t"]), np.nan))
                px, d = float(r["price"]), int(r["direction"])
                stop = px * (1 - 1.5 * s * d) if np.isfinite(s) else None
                take = px * (1 + 2.0 * s * d) if np.isfinite(s) else None
                self._emit("S1_rs_reset_ethbtc", "ETHBTC", r, stop, take, 1440,
                           "backtest mixed (negative train, positive validation/test, n<=64); unproven", {})
        except Exception as e:
            log.warning("S1 detection failed: %s", e)
        for sym in self.universe():
            try:
                g = to_grid(self.cache.get(sym, server_now))
            except Exception as e:
                log.debug("klines %s: %s", sym, e)
                continue
            try:
                for r in F.s2_signals(g, since_ms=since).to_dict("records"):
                    px, d = float(r["price"]), int(r["direction"])
                    take = px + F.S2["rr"] * (px - r["stop"]) if d > 0 else px - F.S2["rr"] * (r["stop"] - px)
                    self._emit("S2_breakout_retest", sym, r, r["stop"], take, F.S2["max_hold_min"],
                               "backtest negative after costs on validation and test", {})
            except Exception as e:
                log.warning("S2 %s failed: %s", sym, e)
            try:
                bs = self.book_series(sym, g.index)
                if bs is not None and bs[0].notna().sum() >= 150:
                    for r in F.s3_signals(g, since_ms=since, spread=bs[0], bid_depth=bs[1]).to_dict("records"):
                        self._emit("S3_liquidity_recovery_book", sym, r, r["stop"], r["target"], F.S3["max_hold_min"],
                                   "UNVALIDATED: order-book version has no historical test (forward data only)", {})
            except Exception as e:
                log.warning("S3 %s failed: %s", sym, e)

    # ------------------------------------------------------------ verification
    def verify(self, server_now: int) -> None:
        q = ("SELECT s.payload FROM signals s LEFT JOIN signal_outcomes o ON o.signal_id = s.signal_id "
             "WHERE o.signal_id IS NULL AND s.signal_ms < ?")
        for (payload,) in self.store.conn.execute(q, (server_now - (1440 + 2) * MIN,)).fetchall():
            sig = json.loads(payload)
            out = {"signal_id": sig["signal_id"], "strategy": sig["strategy"], "symbol": sig["symbol"],
                   "verified_ms": server_now, "verified_utc": to_iso(server_now)}
            for h in HORIZONS:
                tt = sig["signal_ms"] + h * MIN
                try:
                    k = self.bn.klines(sig["symbol"], "1m", tt - 2 * MIN, tt - 1, 5)
                    k = k[k["open_time"] + MIN <= tt]
                    if not k.empty:
                        px = float(k["close"].iloc[-1])
                        out[f"r_{h}_pct"] = sig["direction"] * (px / sig["bar_close_price"] - 1) * 100
                except Exception as e:
                    out[f"r_{h}_error"] = str(e)[:80]
            try:
                self.store.record("signal_outcome", out)
            except DuplicateRecord:
                pass

    def tick(self, server_now: int) -> None:
        minute = (server_now // MIN) * MIN
        if minute == self.last_minute:
            return
        self.last_minute = minute
        self.collect_books(minute)
        if (minute // MIN) % 5 == 0 and self.last_detect != minute:
            self.last_detect = minute
            self.detect(server_now)
        if server_now - self.last_verify > 10 * MIN:
            self.last_verify = server_now
            self.verify(server_now)
