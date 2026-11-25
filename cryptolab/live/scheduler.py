"""Scheduling for the two independent live experiments.

Slots: experiment A has a slot every 10 minutes, B every 60 minutes (UTC-aligned).
  * A slot gets at most ONE forecast (DB unique constraint) -> restarts never duplicate.
  * A slot is attempted only within `max_start_delay_min` of its start; it is never
    back-filled later. Slots that pass without a forecast are recorded as immutable GAP
    records (with the reason), so missing tests stay visible in the results.
  * Gaps are recorded only after an experiment's first slot (its start).
Modes: `daemon` (continuous loop, single instance via a file lock) and `run-once`
(one tick: verify due outcomes, forecast the current slot if due, record gaps) for
cron / GitHub Actions style schedulers. Both take the same lock.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import time
import traceback
from pathlib import Path

from cryptolab.exchanges.binance import Binance
from cryptolab.exchanges.others import OKX, Bitfinex, Bitstamp, Coinbase, Gate, Kraken, KuCoin, MEXC
from cryptolab.live.forecaster import Forecaster, ModelAbstained
from cryptolab.live.scanner import KlineCache, Scanner
from cryptolab.live.verifier import Verifier
from cryptolab.storage.db import DuplicateRecord, open_store
from cryptolab.util.http import HttpClient
from cryptolab.util.timeutil import MIN, now_ms, to_iso

log = logging.getLogger(__name__)


class LockError(RuntimeError):
    pass


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = None

    def acquire(self) -> None:
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockError(f"another cryptolab process holds {self.path}")
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(str(os.getpid()))
        self.fh.flush()

    def release(self) -> None:
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


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
        self.verifier = Verifier(cfg, self.store, self.bn)
        self.last_attempt: dict[str, float] = {}

    # ------------------------------------------------------------------ slots
    def _exp(self, key):
        return self.cfg["experiments"][key]

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

    def model_ready(self, key: str) -> bool:
        name = self._exp(key)["name"]
        return (self.cfg.path("research_dir") / "models" / f"{name}_v1.pkl").exists() and bool(
            self.store.rows("model_versions", "experiment=?", (name,)))

    def tick_experiment(self, key: str, server_now: int) -> dict | None:
        exp = self._exp(key)
        name, cadence = exp["name"], int(exp["cadence_min"]) * MIN
        if not self.model_ready(key):  # experiment not started until its pre-registered model exists
            return None
        deadline = int(exp["max_start_delay_min"]) * MIN
        cur = (server_now // cadence) * cadence
        last = self.last_recorded_slot(name)
        if last is not None:
            err = self.store.get_runtime(f"last_error:{name}")[0]
            s = last + cadence
            while s < cur:  # slots that passed without a forecast
                self.record_gap(name, s, f"no forecast in slot window ({err or 'scheduler not running'})")
                s += cadence
        if self.store.slot_taken(name, cur):
            return None
        if server_now - cur > deadline:
            if last is not None:  # experiment already started: missed slot
                err = self.store.get_runtime(f"last_error:{name}")[0]
                self.record_gap(name, cur, f"slot start missed by > {exp['max_start_delay_min']} min "
                                           f"({err or 'scheduler not running'})")
            return None
        if time.time() - self.last_attempt.get(key, 0) < 20:  # retry spacing after failures
            return None
        self.last_attempt[key] = time.time()
        try:
            rec = self.forecaster.run(key, cur)
            self.store.set_runtime(f"last_error:{name}", None)
            return rec
        except ModelAbstained as e:
            self.record_gap(name, cur, f"model abstained: {e}")
        except DuplicateRecord:
            log.info("slot %s already recorded (concurrent run)", to_iso(cur))
        except Exception as e:
            log.error("cycle %s failed: %s\n%s", name, e, traceback.format_exc(limit=3))
            self.store.set_runtime(f"last_error:{name}", f"{type(e).__name__}: {str(e)[:160]}")
        return None

    def tick(self) -> dict:
        out = {"verified": 0, "forecasts": []}
        try:
            out["verified"] = len(self.verifier.verify_due())
        except Exception as e:
            log.error("verification pass failed: %s", e)
        server_now = self.bn.clock.now()
        for key in ("A", "B"):
            rec = self.tick_experiment(key, server_now)
            if rec:
                out["forecasts"].append(rec["forecast_id"])
        self.store.set_runtime("heartbeat", {"pid": os.getpid(), "utc": to_iso(now_ms()),
                                             "http_requests": dict(self.http.stats.requests),
                                             "http_failures": dict(self.http.stats.failures),
                                             "http_last_error": dict(self.http.stats.last_error)})
        return out


def _pidfile(cfg) -> Path:
    return cfg.path("data_dir") / "run" / "daemon.pid"


def run_daemon(cfg, poll_s: float = 5.0) -> None:
    lock = RunLock(cfg.path("data_dir") / "run" / "cryptolab.lock")
    lock.acquire()
    stop = {"flag": False}

    def _stop(signum, frame):
        log.info("signal %s received: stopping after current step", signum)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    pf = _pidfile(cfg)
    pf.write_text(str(os.getpid()))
    sysm = LiveSystem(cfg)
    sysm.store.set_runtime("daemon", {"state": "running", "pid": os.getpid(), "started_utc": to_iso(now_ms())})
    log.info("daemon started pid=%d", os.getpid())
    try:
        while not stop["flag"]:
            try:
                sysm.tick()
            except Exception as e:
                log.error("tick failed: %s", e)
            for _ in range(int(poll_s * 10)):
                if stop["flag"]:
                    break
                time.sleep(0.1)
    finally:
        sysm.store.set_runtime("daemon", {"state": "stopped", "pid": os.getpid(), "stopped_utc": to_iso(now_ms())})
        pf.unlink(missing_ok=True)
        lock.release()
        log.info("daemon stopped")


def run_once(cfg) -> dict:
    lock = RunLock(cfg.path("data_dir") / "run" / "cryptolab.lock")
    lock.acquire()
    try:
        sysm = LiveSystem(cfg)
        return sysm.tick()
    finally:
        lock.release()


def daemon_status(cfg) -> dict:
    pf = _pidfile(cfg)
    pid = int(pf.read_text()) if pf.exists() else None
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
    store = open_store(cfg)
    hb, hb_ms = store.get_runtime("heartbeat")
    d, _ = store.get_runtime("daemon")
    age = (now_ms() - hb_ms) / 1000 if hb_ms else None
    return {"pid": pid, "process_alive": alive, "daemon": d, "last_heartbeat_age_s": age, "heartbeat": hb}


def stop_daemon(cfg, timeout_s: float = 60) -> bool:
    pf = _pidfile(cfg)
    if not pf.exists():
        return False
    pid = int(pf.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pf.unlink(missing_ok=True)
        return False
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.5)
    return False
