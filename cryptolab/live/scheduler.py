"""Worker loop for all live experiments.

Slot experiments (one forecast per slot; DB-enforced, so restarts never duplicate):
  A_10min, B_60min                      v1 models, each with its own independent market scan
  G2_5min .. G2_1440min                 generation-2 models sharing one scan per tick
  * A slot is attempted only within `max_start_delay_min` of its start, never back-filled.
  * Slots that pass without a forecast become immutable GAP records (with the reason).
Event monitors (strategy signals, Kalshi arms, paper trading) run every tick, isolated so that
one failing component never stops the others.
Operational evidence written to the ledger: worker start, network outage start/end, hourly
checkpoints (counts + continuity), and (via the supervisor) crashes and restarts.
Modes: `worker` (continuous, single instance via a file lock; normally started by the
supervisor `cryptolab run`) and `run-once` (one tick, for cron / GitHub Actions).
"""
from __future__ import annotations

import logging
import os
import platform
import signal
import time
import traceback
from pathlib import Path

from cryptolab.exchanges.binance import Binance
from cryptolab.exchanges.others import OKX, Bitfinex, Bitstamp, Coinbase, Gate, Kraken, KuCoin, MEXC
from cryptolab.live.forecaster import Forecaster, ModelAbstained
from cryptolab.live.scanner import KlineCache, Scanner
from cryptolab.live.supervisor import record_event, stop_file, write_heartbeat
from cryptolab.live.verifier import Verifier
from cryptolab.storage.db import DuplicateRecord, open_store
from cryptolab.util.http import HttpClient
from cryptolab.util.platform import FileLock, LockError, pid_alive
from cryptolab.util.timeutil import HOUR, MIN, now_ms, to_iso

log = logging.getLogger(__name__)

RunLock = FileLock  # backward-compatible name


class LiveSystem:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        hc = cfg["http"]
        self.http = HttpClient(hc["timeout_s"], hc["max_retries"], hc["backoff_base_s"], hc["min_interval_s"])
        ex = cfg["exchanges"]
        self.bn = Binance(self.http, ex["binance_base"], ex["binance_bulk_base"])
        self.others = {
            "coinbase": Coinbase(self.http, ex["coinbase_base"]), "kraken": Kraken(self.http, ex["kraken_base"]),
            "okx": OKX(self.http, ex["okx_base"]),
        }
        discovery = [Kraken(self.http, ex["kraken_base"]), OKX(self.http, ex["okx_base"]),
                     KuCoin(self.http, ex["kucoin_base"]), Gate(self.http, ex["gate_base"]),
                     MEXC(self.http, ex["mexc_base"]), Bitfinex(self.http, ex["bitfinex_base"]),
                     Bitstamp(self.http, ex["bitstamp_base"])]
        run_dir = cfg.path("data_dir") / "run"
        self.store = open_store(cfg)
        self.scanner = Scanner(cfg, self.bn, discovery, run_dir)
        self.cache = KlineCache(self.bn, run_dir / "klines")
        self.forecaster = Forecaster(cfg, self.store, self.bn, self.others, self.scanner, self.cache)
        from cryptolab.live.kalshi_live import KXRunner
        self.kx = KXRunner(cfg, self.store, self.bn, self.cache, self.http)
        self.verifier = Verifier(cfg, self.store, self.bn)
        self.last_attempt: dict[str, float] = {}
        self.monitors: list = []          # objects with .name and .tick(server_now)
        self.outage_since: int | None = None
        self.outage_error = ""
        self.started_ms = now_ms()
        self.last_checkpoint_hour: int | None = None
        self.ticks = 0

    # ------------------------------------------------------------------ slots
    def _exp(self, key):
        return self.cfg["experiments"][key]

    def slot_keys(self) -> list[str]:
        keys = list(self.cfg["experiments"].keys())
        return sorted([k for k in keys if k.startswith("G")], key=lambda k: self._exp(k)["horizon_min"]) + \
            [k for k in keys if not k.startswith("G")]

    def model_ready(self, key: str) -> bool:
        name = self._exp(key)["name"]
        mf = self._exp(key).get("model_file", f"{name}_v1.pkl")
        return (self.cfg.path("research_dir") / "models" / mf).exists() and bool(
            self.store.rows("model_versions", "experiment=?", (name,)))

    def last_recorded_slot(self, name: str) -> int | None:
        return self.store.scalar("SELECT MAX(s) FROM (SELECT MAX(slot_ms) s FROM forecasts WHERE experiment=? "
                                 "UNION ALL SELECT MAX(slot_ms) s FROM gaps WHERE experiment=?)", (name, name))

    def record_gap(self, name: str, slot_ms: int, reason: str) -> None:
        try:
            self.store.record("gap", {"gap_id": f"{name}:{slot_ms}", "experiment": name, "slot_ms": slot_ms,
                                      "slot_utc": to_iso(slot_ms), "reason": reason, "recorded_ms": now_ms()})
            log.warning("GAP %s %s: %s", name, to_iso(slot_ms), reason)
        except DuplicateRecord:
            pass

    def due_slot(self, key: str, server_now: int) -> int | None:
        """Record gaps for passed slots and return the current slot if a forecast is due."""
        exp = self._exp(key)
        name, cadence = exp["name"], int(exp["cadence_min"]) * MIN
        if not self.model_ready(key):  # experiment not started until its pre-registered model exists
            return None
        deadline = int(exp["max_start_delay_min"]) * MIN
        cur = (server_now // cadence) * cadence
        if server_now - cur < int(exp.get("offset_min", 0)) * MIN:  # decision only after the slot offset
            cur -= cadence  # still inside the previous slot's window for gap bookkeeping
        last = self.last_recorded_slot(name)
        if last is not None:
            err = self.store.get_runtime(f"last_error:{name}")[0]
            s = last + cadence
            while s < cur:  # slots that passed without a forecast
                self.record_gap(name, s, f"no forecast in slot window ({err or 'scheduler not running'})")
                s += cadence
        if self.store.slot_taken(name, cur):
            return None
        if server_now - cur < int(exp.get("offset_min", 0)) * MIN:
            return None
        if server_now - cur > deadline:
            if last is not None:  # experiment already started: missed slot
                err = self.store.get_runtime(f"last_error:{name}")[0]
                self.record_gap(name, cur, f"slot start missed by > {exp['max_start_delay_min']} min "
                                           f"({err or 'scheduler not running'})")
            return None
        if time.time() - self.last_attempt.get(key, 0) < 20:  # retry spacing after failures
            return None
        return cur

    def run_slot(self, key: str, slot: int, snap: dict | None = None) -> dict | None:
        name = self._exp(key)["name"]
        self.last_attempt[key] = time.time()
        try:
            if self._exp(key).get("type") == "kalshi":
                rec = self.kx.run(key, slot, self.store.next_test_number(name))
            else:
                rec = self.forecaster.run(key, slot, snap)
            self.store.set_runtime(f"last_error:{name}", None)
            return rec
        except ModelAbstained as e:
            self.record_gap(name, slot, f"model abstained: {e}")
        except DuplicateRecord:
            log.info("slot %s already recorded (concurrent run)", to_iso(slot))
        except Exception as e:
            log.error("cycle %s failed: %s\n%s", name, e, traceback.format_exc(limit=3))
            self.store.set_runtime(f"last_error:{name}", f"{type(e).__name__}: {str(e)[:160]}")
        return None

    def tick_experiment(self, key: str, server_now: int) -> dict | None:  # used by tests / legacy callers
        slot = self.due_slot(key, server_now)
        return self.run_slot(key, slot) if slot is not None else None

    # ------------------------------------------------------------ connectivity
    def check_connectivity(self) -> bool:
        try:
            self.bn.server_time()
            ok = True
        except Exception as e:
            ok, self.outage_error = False, f"{type(e).__name__}: {str(e)[:120]}"
        if not ok and self.outage_since is None:
            self.outage_since = now_ms()
            record_event(self.cfg, "network_outage_start", error=self.outage_error)
            for k in self.cfg["experiments"]:
                self.store.set_runtime(f"last_error:{self._exp(k)['name']}", f"network outage: {self.outage_error}")
        elif ok and self.outage_since is not None:
            dur = (now_ms() - self.outage_since) / 1000
            record_event(self.cfg, "network_outage_end", duration_s=round(dur, 1), error=self.outage_error)
            self.outage_since = None
        return ok

    def checkpoint(self) -> None:
        """Hourly proof-of-operation record (counts + continuity) in the hash-chained ledger."""
        hour = now_ms() // HOUR
        if self.last_checkpoint_hour == hour:
            return
        self.last_checkpoint_hour = hour
        exps = {}
        for k in self.cfg["experiments"]:
            name = self._exp(k)["name"]
            n_fc = self.store.scalar("SELECT COUNT(*) FROM forecasts WHERE experiment=?", (name,))
            n_out = self.store.scalar("SELECT COUNT(*) FROM outcomes WHERE experiment=?", (name,))
            n_gap = self.store.scalar("SELECT COUNT(*) FROM gaps WHERE experiment=?", (name,))
            exps[name] = {"forecasts": n_fc, "outcomes": n_out, "gaps": n_gap}
        seq, head = self.store.ledger.head
        record_event(self.cfg, "checkpoint", worker_uptime_s=round((now_ms() - self.started_ms) / 1000),
                     ticks=self.ticks, experiments=exps, ledger_seq_before=seq, ledger_head_before=head,
                     monitors=[m.name for m in self.monitors], http_failures=dict(self.http.stats.failures))

    # -------------------------------------------------------------------- tick
    def _beat(self) -> None:
        try:
            self.heartbeat()
        except Exception as e:
            log.debug("heartbeat: %s", e)

    def tick(self) -> dict:
        out = {"verified": 0, "forecasts": [], "online": True}
        self.ticks += 1
        self._beat()  # heartbeats between stages so a long tick is never mistaken for a hang
        if not self.check_connectivity():
            out["online"] = False
            return out
        try:
            out["verified"] = len(self.verifier.verify_due())
        except Exception as e:
            log.error("verification pass failed: %s", e)
        server_now = self.bn.clock.now()
        due = {k: self.due_slot(k, server_now) for k in self.slot_keys()}
        g_due = [k for k in due if k.startswith("G") and due[k] is not None]
        if g_due:
            try:
                snap = self.forecaster.snapshot()
            except Exception as e:
                snap = None
                for k in g_due:
                    self.last_attempt[k] = time.time()
                    self.store.set_runtime(f"last_error:{self._exp(k)['name']}", f"snapshot failed: {str(e)[:150]}")
                log.error("G2 snapshot failed: %s", e)
            if snap is not None:
                for k in g_due:
                    rec = self.run_slot(k, due[k], snap)
                    if rec:
                        out["forecasts"].append(rec["forecast_id"])
                    self._beat()
        for k in [k for k in due if not k.startswith("G") and due[k] is not None]:
            rec = self.run_slot(k, due[k])
            if rec:
                out["forecasts"].append(rec["forecast_id"])
            self._beat()
        for m in self.monitors:
            self._beat()
            try:
                m.tick(self.bn.clock.now())
            except Exception as e:
                log.error("monitor %s failed: %s\n%s", getattr(m, "name", m), e, traceback.format_exc(limit=2))
        try:
            self.checkpoint()
        except Exception as e:
            log.error("checkpoint failed: %s", e)
        return out

    def heartbeat(self) -> None:
        info = {"pid": os.getpid(), "online": self.outage_since is None, "ticks": self.ticks,
                "started_utc": to_iso(self.started_ms), "http_requests": dict(self.http.stats.requests),
                "http_failures": dict(self.http.stats.failures), "http_last_error": dict(self.http.stats.last_error)}
        write_heartbeat(self.cfg, info)
        self.store.set_runtime("heartbeat", info)


def attach_monitors(sysm: LiveSystem) -> None:
    """Register optional event monitors (each isolated; missing modules are skipped)."""
    try:
        from cryptolab.live.monitors import build_monitors
        sysm.monitors.extend(build_monitors(sysm))
    except ImportError:
        pass


def run_worker(cfg, poll_s: float = 5.0) -> None:
    lock = FileLock(cfg.path("data_dir") / "run" / "cryptolab.lock")
    lock.acquire()
    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True

    for sig in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _stop)
    pf = cfg.path("data_dir") / "run" / "daemon.pid"
    pf.write_text(str(os.getpid()))
    sysm = LiveSystem(cfg)
    attach_monitors(sysm)
    sysm.store.set_runtime("daemon", {"state": "running", "pid": os.getpid(), "started_utc": to_iso(now_ms())})
    record_event(cfg, "worker_running", monitors=[m.name for m in sysm.monitors], platform=platform.platform())
    log.info("worker started pid=%d monitors=%s", os.getpid(), [m.name for m in sysm.monitors])
    try:
        while not stop["flag"] and not stop_file(cfg).exists():
            try:
                sysm.tick()
            except Exception as e:
                log.error("tick failed: %s", e)
            try:
                sysm.heartbeat()
            except Exception as e:
                log.error("heartbeat failed: %s", e)
            for _ in range(int(poll_s * 10)):
                if stop["flag"] or stop_file(cfg).exists():
                    break
                time.sleep(0.1)
    finally:
        sysm.store.set_runtime("daemon", {"state": "stopped", "pid": os.getpid(), "stopped_utc": to_iso(now_ms())})
        record_event(cfg, "worker_stopped", reason="stop requested" if (stop["flag"] or stop_file(cfg).exists()) else "exit")
        pf.unlink(missing_ok=True)
        lock.release()
        log.info("worker stopped")


run_daemon = run_worker  # backward-compatible name


def run_once(cfg) -> dict:
    lock = FileLock(cfg.path("data_dir") / "run" / "cryptolab.lock")
    lock.acquire()
    try:
        sysm = LiveSystem(cfg)
        attach_monitors(sysm)
        out = sysm.tick()
        sysm.heartbeat()
        return out
    finally:
        lock.release()


def daemon_status(cfg) -> dict:
    from cryptolab.live.supervisor import status as sup_status
    st = sup_status(cfg)
    pf = cfg.path("data_dir") / "run" / "daemon.pid"
    pid = int(pf.read_text()) if pf.exists() else None
    store = open_store(cfg)
    hb, hb_ms = store.get_runtime("heartbeat")
    d, _ = store.get_runtime("daemon")
    age = (now_ms() - hb_ms) / 1000 if hb_ms else None
    return {"pid": pid, "process_alive": pid_alive(pid), "daemon": d, "last_heartbeat_age_s": age, "heartbeat": hb,
            "supervisor": {k: st[k] for k in ("supervisor_pid", "supervisor_alive", "healthy")}}


def stop_daemon(cfg, timeout_s: float = 90) -> bool:
    from cryptolab.live.supervisor import stop
    return stop(cfg, timeout_s)
