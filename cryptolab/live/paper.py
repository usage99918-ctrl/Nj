"""Paper-trading engine (simulation only -- never connects to an exchange account).

Portfolios
  gated            trades only strategies listed in [paper].approved_strategies. A strategy is
                   approved only after passing its pre-registered out-of-sample test net of costs;
                   none has, so this portfolio stays in cash and logs why.
  shadow:<name>    trades every signal of one strategy, purely to measure it.
Risk rules (all portfolios): position size = min(max_position_pct x equity, max_position_usd);
max open positions; one position per symbol; stop-loss / take-profit from the signal (defaults
otherwise); max holding time; daily loss limit (no new entries for the rest of the UTC day once
realized P&L <= -limit); kill switch (file data/run/KILL_SWITCH: closes all positions, blocks entries).
Execution model: fill = first Binance trade at least `execution_delay_s` after detection, worsened by
the measured half-spread (or the configured default) plus slippage; taker fee on both sides.
Exits are checked on closed 1-minute bars; if stop and target fall in the same minute the stop is
assumed first. Spot shorts are theoretical (gated portfolio never shorts).
Every closed position becomes an immutable `trade` record; entries/halts are ledger events.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from cryptolab.live.supervisor import record_event, run_dir
from cryptolab.storage.db import DuplicateRecord
from cryptolab.util.timeutil import MIN, now_ms, to_iso

log = logging.getLogger(__name__)


def kill_switch_file(cfg):
    return run_dir(cfg) / "KILL_SWITCH"


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


class PaperEngine:
    name = "paper_trading"

    def __init__(self, cfg, store, bn) -> None:
        self.cfg, self.store, self.bn = cfg, store, bn
        self.p = cfg["paper"]
        self.costs = cfg["costs"]
        st, _ = store.get_runtime("paper_state", None)
        self.state = st or {"portfolios": {}, "last_signal_seq": 0, "kill": False}
        for name in ["gated"] + [f"shadow:{s}" for s in self.p["shadow_strategies"]]:
            self.state["portfolios"].setdefault(name, {
                "equity": self.p["starting_equity_usd"], "day": None, "day_start_equity": self.p["starting_equity_usd"],
                "realized_today": 0.0, "halted_day": None, "positions": {}, "ignored": {}, "trades": 0})

    def save(self) -> None:
        self.store.set_runtime("paper_state", self.state)

    # ------------------------------------------------------------------ helpers
    def _accepts(self, pname: str, strategy: str) -> tuple[bool, str]:
        if pname == "gated":
            if strategy not in self.p["approved_strategies"]:
                return False, "strategy not approved (no out-of-sample edge after costs)"
            return True, ""
        return pname == f"shadow:{strategy}", "not this shadow portfolio"

    def _fill(self, symbol: str, after_ms: int):
        tr = self.bn.agg_trades(symbol, start_ms=after_ms, end_ms=after_ms + 60_000, limit=1)
        return (tr[0]["price"], tr[0]["ts_ms"]) if tr else (None, None)

    def _close(self, pname: str, pf: dict, sym: str, pos: dict, exit_px: float, exit_ms: int, reason: str) -> None:
        hs, slip, fee = pos["half_spread"], self.costs["slippage_per_side"], self.costs["taker_fee"]["binance"]
        d = pos["direction"]
        eff_exit = exit_px * (1 - d * (hs + slip))
        gross = d * (eff_exit - pos["entry_px_eff"]) * pos["qty"]
        fees = fee * (pos["entry_px_eff"] * pos["qty"] + eff_exit * pos["qty"])
        pnl = gross - fees
        pf["equity"] += pnl
        pf["realized_today"] += pnl
        pf["trades"] += 1
        rec = {"trade_id": f"{pname}:{pos['signal_id']}", "portfolio": pname, "strategy": pos["strategy"],
               "signal_id": pos["signal_id"], "symbol": sym, "direction": d, "theoretical_short": d < 0,
               "entry_ms": pos["entry_ms"], "entry_utc": to_iso(pos["entry_ms"]), "entry_trade_px": pos["entry_trade_px"],
               "entry_px_eff": pos["entry_px_eff"], "qty": pos["qty"], "notional_usd": pos["notional"],
               "stop": pos["stop"], "take": pos["take"], "exit_ms": exit_ms, "exit_utc": to_iso(exit_ms),
               "exit_px": exit_px, "exit_px_eff": eff_exit, "exit_reason": reason, "fees_usd": fees,
               "pnl_usd": pnl, "return_pct": pnl / pos["notional"] * 100, "equity_after": pf["equity"],
               "assumptions": {"fee_per_side": fee, "half_spread": hs, "slippage_per_side": slip,
                               "execution_delay_s": self.p["execution_delay_s"], "same_minute_rule": "stop first"}}
        try:
            self.store.record("trade", rec)
        except DuplicateRecord:
            pass
        log.info("PAPER %s closed %s %s pnl=%.2f (%s)", pname, sym, "LONG" if d > 0 else "SHORT", pnl, reason)

    # ------------------------------------------------------------------- tick
    def manage(self, pname: str, pf: dict, server_now: int, kill: bool) -> None:
        for sym, pos in list(pf["positions"].items()):
            if kill:
                tr = self.bn.latest_trade(sym)
                self._close(pname, pf, sym, pos, tr["price"], tr["ts_ms"], "kill switch")
                del pf["positions"][sym]
                continue
            start = pos.get("checked_until", pos["entry_ms"] - pos["entry_ms"] % MIN + MIN)
            if server_now < start + MIN:                 # no newly closed minute since the last check
                continue
            k = self.bn.klines(sym, "1m", start, server_now, 1000)
            k = k[(k["open_time"] >= start - MIN + 1) & (k["open_time"] + MIN <= server_now)]
            d = pos["direction"]
            closed = False
            for r in k.itertuples():
                bar_end = int(r.open_time) + MIN
                hit_s = pos["stop"] is not None and ((r.low <= pos["stop"]) if d > 0 else (r.high >= pos["stop"]))
                hit_t = pos["take"] is not None and ((r.high >= pos["take"]) if d > 0 else (r.low <= pos["take"]))
                if hit_s:
                    self._close(pname, pf, sym, pos, pos["stop"], bar_end, "stop" + ("+target same minute" if hit_t else ""))
                    closed = True
                    break
                if hit_t:
                    self._close(pname, pf, sym, pos, pos["take"], bar_end, "take-profit")
                    closed = True
                    break
                if bar_end >= pos["entry_ms"] + pos["max_hold_min"] * MIN:
                    self._close(pname, pf, sym, pos, float(r.close), bar_end, "max holding time")
                    closed = True
                    break
                pos["checked_until"] = bar_end
            if closed:
                del pf["positions"][sym]

    def enter(self, pname: str, pf: dict, sig: dict, server_now: int) -> None:
        ok, why = self._accepts(pname, sig["strategy"])
        if not ok:
            if pname == "gated":
                pf["ignored"][why] = pf["ignored"].get(why, 0) + 1
            return
        reason = None
        if pf["halted_day"] == _day(server_now):
            reason = "daily loss limit reached"
        elif len(pf["positions"]) >= self.p["max_open_positions"]:
            reason = "max open positions"
        elif sig["symbol"] in pf["positions"]:
            reason = "already holding this symbol"
        elif pname == "gated" and sig["direction"] < 0:
            reason = "spot short not executable"
        if reason:
            pf["ignored"][reason] = pf["ignored"].get(reason, 0) + 1
            return
        fill_after = sig["detected_ms"] + self.p["execution_delay_s"] * 1000
        if fill_after > server_now - 1000:          # fill moment not reached yet: retry next tick
            pf.setdefault("pending", {})[sig["signal_id"]] = sig
            return
        px, ts = self._fill(sig["symbol"], fill_after)
        if px is None:
            if server_now - fill_after < 5 * MIN:
                pf.setdefault("pending", {})[sig["signal_id"]] = sig
            else:
                pf["ignored"]["no fill price"] = pf["ignored"].get("no fill price", 0) + 1
            return
        pf.setdefault("pending", {}).pop(sig["signal_id"], None)
        hs = self.costs["backtest_half_spread_other"]
        d = sig["direction"]
        eff = px * (1 + d * (hs + self.costs["slippage_per_side"]))
        notional = min(self.p["max_position_pct"] * pf["equity"], self.p["max_position_usd"])
        stop = sig["stop"] if sig["stop"] else px * (1 - d * self.p["default_stop_pct"] / 100)
        take = sig["take"] if sig["take"] else px * (1 + d * self.p["default_take_pct"] / 100)
        pf["positions"][sig["symbol"]] = {"signal_id": sig["signal_id"], "strategy": sig["strategy"], "direction": d,
                                          "entry_ms": ts, "entry_trade_px": px, "entry_px_eff": eff,
                                          "qty": notional / eff, "notional": notional, "stop": stop, "take": take,
                                          "max_hold_min": sig["max_hold_min"], "half_spread": hs}
        record_event(self.cfg, "paper_entry", portfolio=pname, signal_id=sig["signal_id"], symbol=sig["symbol"],
                     direction=d, fill_px=px, fill_ms=ts, notional_usd=notional, stop=stop, take=take)

    def tick(self, server_now: int) -> None:
        kill = kill_switch_file(self.cfg).exists()
        if kill != self.state.get("kill", False):
            record_event(self.cfg, "kill_switch_engaged" if kill else "kill_switch_released")
            self.state["kill"] = kill
        rows = self.store.conn.execute("SELECT seq, payload FROM signals WHERE seq > ? ORDER BY seq",
                                       (self.state["last_signal_seq"],)).fetchall()
        for pname, pf in self.state["portfolios"].items():
            day = _day(server_now)
            if pf["day"] != day:
                pf["day"], pf["day_start_equity"], pf["realized_today"] = day, pf["equity"], 0.0
            self.manage(pname, pf, server_now, kill)
            limit = -self.p["daily_loss_limit_pct"] / 100 * pf["day_start_equity"]
            if pf["realized_today"] <= limit and pf["halted_day"] != day:
                pf["halted_day"] = day
                record_event(self.cfg, "daily_loss_limit_hit", portfolio=pname, realized_today=pf["realized_today"])
            if not kill:
                for sig in list(pf.get("pending", {}).values()):
                    pf["pending"].pop(sig["signal_id"], None)
                    self.enter(pname, pf, sig, server_now)
                for _, payload in rows:
                    self.enter(pname, pf, json.loads(payload), server_now)
        if rows:
            self.state["last_signal_seq"] = rows[-1][0]
        self.save()
