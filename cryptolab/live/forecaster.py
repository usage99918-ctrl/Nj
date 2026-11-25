"""Live forecasting cycle for one experiment slot: scan -> features -> score -> select ->
context -> freeze an immutable forecast record."""
from __future__ import annotations

import hashlib
import logging
import math
import time
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from cryptolab.backtest.engine import select_top
from cryptolab.backtest.stats import sigmoid
from cryptolab.backtest.research import git_commit
from cryptolab.features import MIN_HISTORY_BARS, MODEL_FEATURES, cross_sectional, reference_series, symbol_features, to_grid
from cryptolab.live.context import book_metrics, cross_exchange, derivatives_context, evidence, fear_greed
from cryptolab.live.scanner import KlineCache, Scanner
from cryptolab.util.timeutil import MIN, now_ms, to_iso

log = logging.getLogger(__name__)

FLAT_RULE = ("Direction is scored against the last Binance trade at or before the expiry instant versus the "
             "reference trade price. If the two prices are identical the outcome is FLAT and counts as an "
             "INCORRECT direction for both UP and DOWN forecasts (rule fixed before testing).")
PROB_STATUS = ("Calibrated on walk-forward out-of-sample backtest forecasts of this exact model; NOT yet validated "
               "on forward data (forward calibration is reported on the dashboard as outcomes accumulate).")


class LiveModel:
    def __init__(self, path: Path) -> None:
        blob = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(blob).hexdigest()
        d = cloudpickle.loads(blob)
        self.strategy = d["strategy"]
        self.cal = d["calibration"]
        self.h = int(d["horizon"])
        self.version = d["version"]


def extend_grid(g: pd.DataFrame, t_end: int) -> pd.DataFrame:
    """Extend a close-time grid to t_end with no-trade minutes (price carried forward)."""
    if g.index[-1] >= t_end:
        return g.loc[:t_end]
    extra = np.arange(g.index[-1] + MIN, t_end + MIN, MIN, dtype="int64")
    e = pd.DataFrame(index=extra, columns=g.columns)
    e["close"] = g["close"].iloc[-1]
    for c in ("open", "high", "low"):
        e[c] = e["close"]
    for c in ("volume", "quote_volume", "trades", "taker_buy_quote"):
        e[c] = 0.0
    e["missing"] = 0
    return pd.concat([g, e.astype(g.dtypes.to_dict())])


def build_live_panel(cache: KlineCache, universe: list[str], server_now: int) -> tuple[pd.DataFrame, dict]:
    t_common = (server_now // MIN) * MIN  # close time of the most recent fully closed bar
    diag = {"t_common_ms": t_common, "skipped": {}}
    ref = {}
    for k in ("BTC", "ETH"):
        raw = cache.get(f"{k}USDT", server_now)
        g = extend_grid(to_grid(raw), t_common)
        ref[k] = reference_series(g.reset_index().rename(columns={"index": "t"}).assign(
            open_time=lambda d: d["t"] - MIN)[["open_time", "open", "high", "low", "close", "volume", "quote_volume",
                                                "trades", "taker_buy_quote"]])
    rows = []
    for rank, sym in enumerate(universe, start=1):
        try:
            raw = cache.get(sym, server_now)
        except Exception as e:
            diag["skipped"][sym] = f"kline fetch failed: {str(e)[:80]}"
            continue
        if len(raw) < MIN_HISTORY_BARS:
            diag["skipped"][sym] = f"insufficient history ({len(raw)} bars)"
            continue
        last_close = int(raw["open_time"].iloc[-1]) + MIN
        if t_common - last_close > 10 * MIN:
            diag["skipped"][sym] = "stale (no bar for >10 min)"
            continue
        g = extend_grid(to_grid(raw), t_common)
        f = symbol_features(g, ref)
        row = f.iloc[-1].to_dict()
        row.update(t=int(f.index[-1]), symbol=sym, universe_rank=rank, last_bar_close_ms=last_close)
        rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel, diag
    panel = panel[panel["t"] == t_common].copy()
    need = ["r_1440", "vol_1440", "sig1m", "beta_btc"]
    bad = panel[need].isna().any(axis=1)
    for s in panel.loc[bad, "symbol"]:
        diag["skipped"][s] = "incomplete features"
    panel = panel[~bad].reset_index(drop=True)
    panel = cross_sectional(panel)
    return panel, diag


class Forecaster:
    def __init__(self, cfg, store, bn, others: dict, scanner: Scanner, cache: KlineCache) -> None:
        self.cfg, self.store, self.bn, self.others, self.scanner, self.cache = cfg, store, bn, others, scanner, cache
        self.models: dict[str, LiveModel] = {}

    def model(self, exp_key: str) -> LiveModel:
        if exp_key not in self.models:
            name = self.cfg["experiments"][exp_key]["name"]
            path = self.cfg.path("research_dir") / "models" / f"{name}_v1.pkl"
            m = LiveModel(path)
            # Integrity: the model file must match the version frozen in the ledger
            rec = self.store.rows("model_versions", "experiment=?", (name,))
            if not rec:
                raise RuntimeError(f"no model_version record for {name}")
            if rec[-1]["model_sha256"] != m.sha256:
                raise RuntimeError(f"model file hash mismatch for {name}: refusing to forecast")
            self.models[exp_key] = m
        return self.models[exp_key]

    def backtest_record(self, exp_key: str) -> dict:
        name = self.cfg["experiments"][exp_key]["name"]
        for r in self.store.rows("preregistrations", "prereg_id=?", (f"{name}-test-result-v1",)):
            m = r.get("test_metrics_selected") or {}
            return {"backtest_test_accuracy": m.get("accuracy"), "backtest_n": m.get("n_forecasts")}
        return {}

    def run(self, exp_key: str, slot_ms: int) -> dict:
        exp = self.cfg["experiments"][exp_key]
        name, h = exp["name"], int(exp["horizon_min"])
        model = self.model(exp_key)
        assert model.h == h
        t0 = time.time()
        self.bn.server_time()
        scan = self.scanner.scan()
        scan_id = f"{name}:{slot_ms}"
        server_now = self.bn.clock.now()
        panel, diag = build_live_panel(self.cache, scan.binance_universe, server_now)
        if panel.empty:
            raise RuntimeError(f"no analyzable markets (universe {len(scan.binance_universe)}): {diag['skipped']}")
        panel["sig_h"] = panel["sig1m"].astype("float64") * math.sqrt(h)
        z = np.asarray(model.strategy.predict(panel, h), dtype="float64")
        panel["z"] = z
        idx = select_top(panel["t"].values, z, panel["universe_rank"].values)
        scan_rec = {"scan_id": scan_id, "experiment": name, "slot_ms": slot_ms, **scan.summary(),
                    "markets_analyzed": int(len(panel)), "analysis_skipped": diag["skipped"],
                    "features_t_ms": diag["t_common_ms"]}
        if len(idx) == 0:
            self.store.record("scan", scan_rec)
            raise ModelAbstained("model produced no non-zero signal for any coin")
        i = int(idx[0])
        row = panel.iloc[i].to_dict()
        sym = row["symbol"]
        base = sym[:-4]
        cal = model.cal
        zi = float(row["z"])
        sig = float(row["sig_h"])
        mu = cal["b"] * zi * sig
        direction = 1 if zi > 0 else -1
        p_dir = float(sigmoid(cal["a_dir"] + cal["b_dir"] * abs(mu / sig)))
        p_up = p_dir if direction > 0 else 1 - p_dir

        # ---- context gathered BEFORE the reference price (so nothing after origin is used)
        book = book_metrics(self.bn.depth(sym, 100), self.cfg["costs"]["paper_notional_usd"])
        okx = self.others.get("okx")
        deriv = {}
        okx_tickers = None
        if okx is not None:
            try:
                deriv = derivatives_context(okx, base)
            except Exception as e:
                deriv = {"available": False, "error": str(e)[:120]}
            try:
                okx_tickers = okx.tickers("SPOT")
            except Exception:
                okx_tickers = None
        now_l = time.time()
        if now_l - getattr(self, "_fng_t", 0) > 3600:
            self._fng, self._fng_t = fear_greed(self.bn.http), now_l
        sentiment = self._fng
        kraken_t = None
        if self.others.get("kraken") is not None:
            try:
                kraken_t = self.others["kraken"].tickers()
            except Exception:
                now_l = time.time()
        if now_l - getattr(self, "_fng_t", 0) > 3600:
            self._fng, self._fng_t = fear_greed(self.bn.http), now_l
        sentiment = self._fng
        kraken_t = None
        # ---- reference price: last trade at or before the freeze instant
        ref = self.bn.latest_trade(sym)
        origin_ms = int(self.bn.clock.now())
        if ref is None or ref["ts_ms"] > origin_ms + 1000:
            raise RuntimeError("no valid reference trade")
        p0 = float(ref["price"])
        xex = cross_exchange(base, p0, self.others.get("coinbase"), kraken_t, okx_tickers)
        expiry_ms = origin_ms + h * MIN
        pred = p0 * math.exp(mu)
        q = cal
        pi = {k: p0 * math.exp(mu + q[k] * sig) for k in ("q10", "q25", "q75", "q90")}
        costs = self.cfg["costs"]
        fee = costs["taker_fee"]["binance"]
        half_spread = (book.get("spread_bps", 0) or 0) / 2 / 1e4 if book.get("ok") else costs["backtest_half_spread_other"]
        impact = max(book.get("buy_impact_bps") or 0, book.get("sell_impact_bps") or 0) / 1e4
        slippage = max(costs["slippage_per_side"], impact - half_spread)
        rt = 2 * (fee + half_spread + slippage)
        borrow = costs["short_borrow_cost_per_hour"] * h / 60
        exp_net_long = (math.exp(mu) - 1) - rt
        long_signal = direction > 0 and mu > rt
        short_theoretical = direction < 0 and (-mu) > rt + borrow
        if long_signal:
            label = "YES - hypothetical spot LONG (paper only): expected move exceeds estimated round-trip costs"
        elif short_theoretical:
            label = ("THEORETICAL SHORT ONLY - not executable on spot (needs margin/borrow); spot action = stay in cash")
        else:
            label = "NO - expected move is smaller than estimated round-trip costs; spot action = stay in cash"
        bt = self.backtest_record(exp_key)
        sup, con, neu = evidence(row, direction, book, deriv, xex,
                                 {"strategy": model.strategy.name, "z": zi, "n": int(len(panel)),
                                  "hypothesis": model.strategy.hypothesis, **bt}, sentiment)
        test_no = self.store.next_test_number(name)
        feats = {k: (None if (isinstance(row.get(k), float) and not math.isfinite(row.get(k))) else row.get(k))
                 for k in MODEL_FEATURES if k in row}
        top = panel.assign(absz=panel["z"].abs()).sort_values("absz", ascending=False).head(5)
        rec = {
            "forecast_id": f"{exp_key}-{test_no:06d}", "experiment": name, "test_number": test_no,
            "slot_ms": slot_ms, "slot_utc": to_iso(slot_ms), "horizon_min": h,
            "exchange": "binance", "symbol": sym, "base": base, "quote": "USDT", "pair": f"{base}/USDT",
            "origin_ms": origin_ms, "origin_utc": to_iso(origin_ms),
            "ref_trade_ms": ref["ts_ms"], "ref_trade_utc": to_iso(ref["ts_ms"]), "ref_trade_id": ref["id"], "p0": p0,
            "expiry_ms": expiry_ms, "expiry_utc": to_iso(expiry_ms),
            "pred_price": pred, "direction": "UP" if direction > 0 else "DOWN",
            "expected_move_pct": (math.exp(mu) - 1) * 100, "mu_log": mu, "sigma_log": sig, "z": zi,
            "p_up": p_up, "p_direction": p_dir, "probability_status": PROB_STATUS,
            "pi50": [pi["q25"], pi["q75"]], "pi80": [pi["q10"], pi["q90"]],
            "model_version": model.version, "strategy": model.strategy.name, "strategy_family": model.strategy.family,
            "strategy_hypothesis": model.strategy.hypothesis, "model_sha256": model.sha256,
            "costs": {"fee_per_side": fee, "half_spread_measured": half_spread, "slippage_per_side": slippage,
                      "round_trip": rt, "short_borrow_for_horizon": borrow,
                      "assumptions": "Binance retail taker fee 0.10%/side (assumption); half-spread from the live "
                                     "order book; slippage = max(1 bp, book-walk impact for $1,000 beyond half-spread)."},
            "expected_net_long_pct": exp_net_long * 100,
            "long_signal": bool(long_signal), "theoretical_short_signal": bool(short_theoretical),
            "opportunity_label": label,
            "n_examined": int(len(panel)), "scan_id": scan_id,
            "scan_counts": {"markets_discovered": scan.markets_discovered,
                            "markets_quality_passed": scan.markets_quality_passed,
                            "unique_assets_passed": scan.unique_assets_passed,
                            "binance_universe": len(scan.binance_universe), "markets_analyzed": int(len(panel))},
            "top_candidates": [{"symbol": r.symbol, "z": float(r.z)} for r in top.itertuples()],
            "features": feats,
            "data_timestamps": {"features_bar_close_ms": diag["t_common_ms"], "last_bar_close_ms": row["last_bar_close_ms"],
                                "book_fetched_ms": book.get("fetched_ms"), "server_clock_offset_ms": self.bn.clock.offset_ms},
            "evidence_for": sup, "evidence_against": con, "evidence_neutral": neu,
            "context": {"order_book": book, "derivatives": deriv, "cross_exchange": xex, "sentiment": sentiment},
            "rules": {"flat": FLAT_RULE, "verification": "Last Binance aggTrade at or before expiry_ms (same pair)."},
            "cycle_seconds": round(time.time() - t0, 2), "created_local_ms": now_ms(), "code_commit": git_commit(),
        }
        self.store.record("scan", scan_rec)
        self.store.record("forecast", rec)
        log.info("FROZEN %s %s %s p0=%.8g pred=%.8g P(dir)=%.3f expiry=%s", rec["forecast_id"], sym, rec["direction"],
                 p0, pred, p_dir, rec["expiry_utc"])
        return rec


class ModelAbstained(Exception):
    pass
