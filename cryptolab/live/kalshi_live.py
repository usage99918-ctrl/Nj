"""Live Kalshi components (read-only).

KXRunner: the two arms of the controlled test, run every 15-minute Kalshi window for BTC and ETH:
  KX_<asset>_without and KX_<asset>_with share the SAME base features and reference price for a
  window (paired); WITH additionally uses the Kalshi YES mid at window open + 2 min. If no usable
  Kalshi quote exists, WITH records a gap (it never silently falls back to WITHOUT).
KalshiCollector: every 5 minutes, snapshot of the current 15-minute market and the nearest hourly
  above/below ladder (implied P(index above spot) at ONE expiry) for BTC and ETH.
Kalshi prices are contract prices (implied probabilities of BRTI-settled events), not spot prices.
"""
from __future__ import annotations

import hashlib
import logging
import math

import cloudpickle
import numpy as np

from cryptolab.backtest.stats import sigmoid
from cryptolab.backtest.kalshi_test import BASE_FEATURES, DECISION_OFFSET, WINDOW, logit
from cryptolab.exchanges.kalshi import DIRECTION_SERIES, Kalshi, ladder_summary
from cryptolab.features import reference_series, symbol_features, to_grid
from cryptolab.live.forecaster import ModelAbstained, extend_grid
from cryptolab.util.timeutil import MIN, now_ms, to_iso

log = logging.getLogger(__name__)
SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


class KXRunner:
    def __init__(self, cfg, store, bn, cache, http) -> None:
        self.cfg, self.store, self.bn, self.cache = cfg, store, bn, cache
        self.kalshi = Kalshi(http)
        self._model = None
        self._ctx: dict = {}

    def model(self) -> dict:
        if self._model is None:
            p = self.cfg.path("research_dir") / "models" / "KX_v1.pkl"
            blob = p.read_bytes()
            sha = hashlib.sha256(blob).hexdigest()
            rec = self.store.rows("model_versions", "experiment=?", ("KX_BTC_with",))
            if not rec or rec[-1]["model_sha256"] != sha:
                raise RuntimeError("KX model file hash mismatch: refusing to forecast")
            self._model = dict(cloudpickle.loads(blob), sha256=sha)
        return self._model

    def _context(self, asset: str, slot: int) -> dict:
        """Features, Kalshi quote and reference trade for (asset, window) -- computed ONCE for both arms."""
        key = (asset, slot)
        if key in self._ctx:
            return self._ctx[key]
        self.bn.server_time()
        server_now = self.bn.clock.now()
        t_common = (server_now // MIN) * MIN
        raws = {a: self.cache.get(SYM[a], server_now) for a in ("BTC", "ETH")}
        grids = {a: extend_grid(to_grid(raws[a]), t_common) for a in raws}
        refs = {a: reference_series(grids[a].reset_index().rename(columns={"index": "t"}).assign(
            open_time=lambda d: d["t"] - MIN)[["open_time", "open", "high", "low", "close", "volume", "quote_volume",
                                                "trades", "taker_buy_quote"]]) for a in grids}
        f = symbol_features(grids[asset], refs)
        row = f.iloc[-1]
        open_px = float(f["close"].get(slot, np.nan))
        z_open = math.log(float(row["close"]) / open_px) / (float(row["sig1m"]) * math.sqrt(max((t_common - slot) / MIN, 1)))
        m = self.model()
        base = float(m["base"].score(np.array([[row.get(c, np.nan) for c in BASE_FEATURES]], dtype=float))[0])
        quote = None
        try:
            mk = [x for x in self.kalshi.markets(DIRECTION_SERIES[asset], status="open", max_pages=1)
                  if x["open_ms"] == slot]
            if mk:
                q = mk[0]
                usable = (q["yes_bid"] or 0) > 0 and (q["yes_ask"] or 1) < 1 and q["yes_ask"] > q["yes_bid"] and \
                    (q["yes_ask"] - q["yes_bid"]) <= 0.10
                quote = {k: q[k] for k in ("ticker", "yes_bid", "yes_ask", "yes_mid", "floor_strike", "open_ms", "close_ms",
                                           "volume", "open_interest", "settlement_rule")}
                quote.update(usable=bool(usable), fetched_ms=now_ms(),
                             note="Kalshi YES price = implied P(BRTI 60-s average at window close >= at window open)")
        except Exception as e:
            quote = {"usable": False, "error": str(e)[:120]}
        ref = self.bn.latest_trade(SYM[asset])
        origin = int(self.bn.clock.now())
        ctx = {"base": base, "z_open": z_open, "quote": quote, "p0": float(ref["price"]), "ref_ms": ref["ts_ms"],
               "origin": origin, "sig": float(row["sig1m"]), "features_t_ms": t_common, "open_px": open_px}
        self._ctx = {k: v for k, v in self._ctx.items() if k[1] >= slot - WINDOW}
        self._ctx[key] = ctx
        return ctx

    def run(self, exp_key: str, slot: int, test_number: int) -> dict:
        exp = self.cfg["experiments"][exp_key]
        asset, arm, name = exp["asset"], exp["arm"], exp["name"]
        c = self._context(asset, slot)
        m = self.model()
        x = [1.0, c["base"], float(np.clip(c["z_open"], -10, 10))]
        if arm == "with":
            q = c["quote"]
            if not q or not q.get("usable"):
                raise ModelAbstained("no usable Kalshi quote for this window (WITH arm cannot forecast)")
            x.append(float(logit(q["yes_mid"])))
        p_up = float(sigmoid(np.dot(m["weights"][arm], x)))
        expiry = slot + WINDOW
        horizon_s = (expiry - c["origin"]) / 1000
        sig_h = c["sig"] * math.sqrt(max(horizon_s / 60, 1))
        costs = self.cfg["costs"]
        rt = 2 * (costs["taker_fee"]["binance"] + costs["backtest_half_spread_top10"] + costs["slippage_per_side"])
        rec = {"forecast_id": f"{exp_key}-{test_number:06d}", "experiment": name, "test_number": test_number,
               "slot_ms": slot, "slot_utc": to_iso(slot), "horizon_min": round(horizon_s / 60, 2),
               "exchange": "binance", "symbol": SYM[asset], "base": asset, "quote": "USDT", "pair": f"{asset}/USDT",
               "origin_ms": c["origin"], "origin_utc": to_iso(c["origin"]), "ref_trade_ms": c["ref_ms"],
               "ref_trade_utc": to_iso(c["ref_ms"]), "p0": c["p0"], "expiry_ms": expiry, "expiry_utc": to_iso(expiry),
               "pred_price": c["p0"], "direction": "UP" if p_up > 0.5 else "DOWN", "expected_move_pct": 0.0,
               "p_up": p_up, "p_direction": max(p_up, 1 - p_up),
               "probability_status": "Logistic calibration on historical Kalshi windows (see kalshi_ab_test.json)",
               "pi50": [c["p0"] * math.exp(-0.674 * sig_h), c["p0"] * math.exp(0.674 * sig_h)],
               "pi80": [c["p0"] * math.exp(-1.2816 * sig_h), c["p0"] * math.exp(1.2816 * sig_h)],
               "model_version": f"{name}-v1", "strategy": f"kalshi_ab_{arm}", "model_sha256": m["sha256"],
               "costs": {"round_trip": rt, "short_borrow_for_horizon": 0.0, "fee_per_side": costs["taker_fee"]["binance"],
                         "half_spread_measured": costs["backtest_half_spread_top10"],
                         "slippage_per_side": costs["slippage_per_side"]},
               "expected_net_long_pct": -rt * 100, "long_signal": False, "theoretical_short_signal": False,
               "opportunity_label": "NO - research forecast for the Kalshi with/without test (not a trading signal)",
               "n_examined": 1, "kalshi_arm": arm, "kalshi": c["quote"],
               "inputs": {"base_score": c["base"], "z_open": c["z_open"], "window_open_px": c["open_px"],
                          "features_t_ms": c["features_t_ms"]},
               "evidence_for": [], "evidence_against": [], "features": {},
               "scan_counts": {"markets_discovered": 0}, "created_local_ms": now_ms()}
        self.store.record("forecast", rec)
        log.info("FROZEN %s %s P(up)=%.3f kalshi=%s", rec["forecast_id"], asset, p_up,
                 (c["quote"] or {}).get("yes_mid"))
        return rec


class KalshiCollector:
    name = "kalshi_collector"

    def __init__(self, cfg, store, http, coinbase) -> None:
        self.cfg, self.store, self.k, self.cb = cfg, store, Kalshi(http), coinbase
        self.last = None

    def tick(self, server_now: int) -> None:
        slot = (server_now // (5 * MIN)) * (5 * MIN)
        if self.last == slot:
            return
        self.last = slot
        for asset in ("BTC", "ETH"):
            try:
                spot = self.cb.ticker(f"{asset}-USD").last
            except Exception:
                spot = None
            snap = {"snap_id": f"{asset}:{slot}", "asset": asset, "ts_ms": now_ms(), "utc": to_iso(now_ms()),
                    "spot_reference": {"source": "Coinbase spot (proxy for the BRTI settlement index)", "price": spot}}
            try:
                d = self.k.current_direction_market(asset, server_now)
                snap["direction_15m"] = None if d is None else {
                    k: d[k] for k in ("ticker", "yes_bid", "yes_ask", "yes_mid", "spread", "floor_strike", "open_ms",
                                      "close_ms", "volume", "open_interest", "yes_bid_size", "yes_ask_size")}
                lad = self.k.ladder(asset, server_now)
                snap["hourly_ladder"] = ladder_summary(lad, spot) if (lad and spot) else {"available": False}
                snap["note"] = ("Kalshi prices are implied probabilities of BRTI-settled contracts at their own expiry "
                                "times; they are not spot prices and are not comparable across expiries.")
            except Exception as e:
                snap["error"] = str(e)[:160]
            try:
                self.store.record("kalshi", snap)
            except Exception as e:
                log.debug("kalshi snapshot: %s", e)
