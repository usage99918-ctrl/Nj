"""Verification of expired forecasts against the SAME exchange and pair.

Primary method: the last Binance aggregated trade at or before the exact expiry instant,
searched within `max_trade_gap_s` -> status VERIFIED (exact-timestamp trade lookup).
Fallbacks (flagged APPROXIMATE): last trade within `fallback_trade_gap_s`; then the close
of the 1-minute candle ending at or before expiry (timing difference recorded).
If no price can be found 6 hours after expiry the forecast is recorded UNVERIFIABLE.
Outcomes are immutable records; nothing about the original forecast is changed.
"""
from __future__ import annotations

import logging
import math

from cryptolab.util.timeutil import MIN, now_ms, to_iso

log = logging.getLogger(__name__)
UNVERIFIABLE_AFTER_MS = 6 * 3600_000


def score(fc: dict, p1: float) -> dict:
    p0 = fc["p0"]
    ret = p1 / p0 - 1
    actual = "UP" if p1 > p0 else ("DOWN" if p1 < p0 else "FLAT")
    correct = actual == fc["direction"]          # FLAT is never correct (pre-registered rule)
    up = 1.0 if p1 > p0 else 0.0
    rt = fc["costs"]["round_trip"]
    borrow = fc["costs"]["short_borrow_for_horizon"]
    sgn = 1 if fc["direction"] == "UP" else -1
    return {
        "p1": p1, "actual_direction": actual, "correct": bool(correct),
        "actual_return_pct": ret * 100,
        "pred_error_abs": abs(fc["pred_price"] - p1), "pred_error_pct": (fc["pred_price"] / p1 - 1) * 100,
        "abs_pred_error_pct": abs(fc["pred_price"] / p1 - 1) * 100,
        "no_change_error_pct": abs(p0 / p1 - 1) * 100,
        "in_pi50": bool(fc["pi50"][0] <= p1 <= fc["pi50"][1]),
        "in_pi80": bool(fc["pi80"][0] <= p1 <= fc["pi80"][1]),
        "brier": (fc["p_up"] - up) ** 2,
        # Paper trading (spot, long-only): trade only when the frozen forecast said so
        "long_taken": bool(fc["long_signal"]),
        "long_gross_pct": ret * 100 if fc["long_signal"] else 0.0,
        "long_net_pct": (ret - rt) * 100 if fc["long_signal"] else 0.0,
        # Hypothetical always-trade direction bet (shorts are NOT executable on spot)
        "dir_gross_pct": sgn * ret * 100,
        "dir_net_pct": (sgn * ret - rt - (borrow if sgn < 0 else 0.0)) * 100,
        "short_is_theoretical": sgn < 0,
    }


class Verifier:
    def __init__(self, cfg, store, bn) -> None:
        self.cfg, self.store, self.bn = cfg, store, bn
        v = cfg["verification"]
        self.gap_ms = v["max_trade_gap_s"] * 1000
        self.fallback_ms = v["fallback_trade_gap_s"] * 1000
        self.delay_ms = v["min_delay_after_expiry_s"] * 1000

    def verify_due(self) -> list[dict]:
        try:
            self.bn.server_time()
        except Exception as e:
            log.warning("server time failed: %s", e)
        server_now = self.bn.clock.now()
        done = []
        for fc in self.store.pending_outcomes(due_before_ms=server_now - self.delay_ms):
            try:
                out = self.verify_one(fc, server_now)
            except Exception as e:
                log.warning("verify %s failed (will retry): %s", fc["forecast_id"], e)
                continue
            if out is not None:
                done.append(out)
        return done

    def verify_one(self, fc: dict, server_now: int) -> dict | None:
        sym, exp_ms = fc["symbol"], fc["expiry_ms"]
        status, method, trade = None, None, None
        t = self.bn.last_trade_at_or_before(sym, exp_ms, self.gap_ms)
        if t is not None:
            status, method, trade = "VERIFIED", "aggTrade at/before expiry (<= %ds)" % (self.gap_ms // 1000), t
        else:
            t = self.bn.last_trade_at_or_before(sym, exp_ms, self.fallback_ms)
            if t is not None:
                status, method, trade = "APPROXIMATE", "aggTrade at/before expiry (wider %ds window)" % (
                    self.fallback_ms // 1000), t
        p1, p1_ms = (trade["price"], trade["ts_ms"]) if trade else (None, None)
        if p1 is None:
            k = self.bn.klines(sym, "1m", exp_ms - 2 * MIN, exp_ms, 5)
            k = k[k["open_time"] + MIN <= exp_ms]
            if not k.empty:
                p1, p1_ms = float(k["close"].iloc[-1]), int(k["open_time"].iloc[-1]) + MIN
                status, method = "APPROXIMATE", "1m candle close at/before expiry (no trade found)"
        if p1 is None:
            if server_now - exp_ms < UNVERIFIABLE_AFTER_MS:
                return None
            rec = {"forecast_id": fc["forecast_id"], "experiment": fc["experiment"], "status": "UNVERIFIABLE",
                   "verified_ms": server_now, "verified_utc": to_iso(server_now),
                   "reason": "no trade or candle found for the expiry instant"}
            self.store.record("outcome", rec)
            return rec
        rec = {"forecast_id": fc["forecast_id"], "experiment": fc["experiment"], "status": status,
               "verified_ms": server_now, "verified_utc": to_iso(server_now), "method": method,
               "exchange": fc["exchange"], "symbol": sym, "expiry_ms": exp_ms, "expiry_utc": fc["expiry_utc"],
               "p1_trade_ms": p1_ms, "p1_trade_utc": to_iso(p1_ms), "timing_gap_ms": exp_ms - p1_ms,
               "trade_id": trade["id"] if trade else None, **score(fc, p1)}
        self.store.record("outcome", rec)
        log.info("VERIFIED %s %s pred=%s actual=%s correct=%s err=%.3f%%", fc["forecast_id"], sym, fc["direction"],
                 rec["actual_direction"], rec["correct"], rec["abs_pred_error_pct"])
        return rec
