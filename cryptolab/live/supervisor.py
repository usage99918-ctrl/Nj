"""Supervisor: keeps the worker running and proves it.

  python -m cryptolab run     foreground supervisor (use this under Task Scheduler / systemd / Docker)
  python -m cryptolab start   start the supervisor in the background
  python -m cryptolab stop    graceful stop (stop file; works on every OS)

The supervisor starts the worker as a child process and restarts it after a crash (exponential
backoff 5 s -> 5 min) or when its heartbeat is older than `hang_after_s` (hung). Every start,
crash, restart and stop is written to the hash-chained ledger as an `event` record, so the
operating history can be audited instead of claimed. OS-level autostart (Task Scheduler,
systemd, Docker restart policy) covers crashes of the supervisor itself and reboots.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from cryptolab.util.platform import FileLock, LockError, pid_alive, spawn_detached, terminate
from cryptolab.util.timeutil import now_ms, to_iso

log = logging.getLogger(__name__)


def run_dir(cfg) -> Path:
    d = cfg.path("data_dir") / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stop_file(cfg) -> Path:
    return run_dir(cfg) / "STOP"


def heartbeat_file(cfg) -> Path:
    return run_dir(cfg) / "heartbeat.json"


def read_heartbeat(cfg) -> dict:
    try:
        return json.loads(heartbeat_file(cfg).read_text())
    except Exception:
        return {}


def write_heartbeat(cfg, info: dict) -> None:
    p = heartbeat_file(cfg)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(info, ts_ms=now_ms(), utc=to_iso(now_ms()))))
    os.replace(tmp, p)


def record_event(cfg, kind: str, **data) -> None:
    from cryptolab.storage.db import open_store
    try:
        s = open_store(cfg)
        ts = now_ms()
        s.record("event", {"event_id": f"{kind}:{ts}:{os.getpid()}", "kind": kind, "ts_ms": ts, "utc": to_iso(ts),
                           "pid": os.getpid(), "platform": platform.system(), **data})
        s.close()
    except Exception as e:  # never let bookkeeping kill the supervisor
        log.error("could not record event %s: %s", kind, e)


def run_supervisor(cfg, hang_after_s: int = 300, worker_args: list[str] | None = None) -> int:
    lock = FileLock(run_dir(cfg) / "supervisor.lock")
    try:
        lock.acquire()
    except LockError as e:
        print(str(e))
        return 1
    stop_file(cfg).unlink(missing_ok=True)
    (run_dir(cfg) / "supervisor.pid").write_text(str(os.getpid()))
    stopping = {"flag": False}

    def _stop(*_):
        stopping["flag"] = True

    import threading
    if threading.current_thread() is threading.main_thread():
        for sig in ("SIGTERM", "SIGINT", "SIGBREAK"):
            if hasattr(signal, sig):
                signal.signal(getattr(signal, sig), _stop)
    worker_args = worker_args or ["-m", "cryptolab", "worker"]
    record_event(cfg, "supervisor_start", python=sys.version.split()[0])
    backoff, child, started = 5, None, 0.0
    try:
        while not stopping["flag"] and not stop_file(cfg).exists():
            if child is None or child.poll() is not None:
                if child is not None:
                    code = child.returncode
                    up = time.time() - started
                    record_event(cfg, "worker_exit", returncode=code, uptime_s=round(up, 1))
                    backoff = 5 if up > 600 else min(backoff * 2, 300)
                    log.warning("worker exited (code %s) after %.0fs; restarting in %ss", code, up, backoff)
                    for _ in range(backoff):
                        if stopping["flag"] or stop_file(cfg).exists():
                            break
                        time.sleep(1)
                    if stopping["flag"] or stop_file(cfg).exists():
                        break
                log_path = cfg.path("log_dir") / "worker.out"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                child = subprocess.Popen([sys.executable] + worker_args, cwd=str(cfg.root),
                                         stdout=open(log_path, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                started = time.time()
                record_event(cfg, "worker_start", worker_pid=child.pid)
            else:
                hb = read_heartbeat(cfg)
                age = (now_ms() - hb.get("ts_ms", 0)) / 1000 if hb else None
                if time.time() - started > hang_after_s and (age is None or age > hang_after_s):
                    record_event(cfg, "worker_hung", heartbeat_age_s=age, worker_pid=child.pid)
                    terminate(child.pid)
                    try:
                        child.wait(30)
                    except subprocess.TimeoutExpired:
                        child.kill()
            time.sleep(1)
    finally:
        if child is not None and child.poll() is None:
            stop_file(cfg).touch()          # worker sees the stop file and exits cleanly
            try:
                child.wait(60)
            except subprocess.TimeoutExpired:
                terminate(child.pid)
        record_event(cfg, "supervisor_stop")
        (run_dir(cfg) / "supervisor.pid").unlink(missing_ok=True)
        stop_file(cfg).unlink(missing_ok=True)
        lock.release()
    return 0


def start_background(cfg) -> str:
    st = status(cfg)
    if st["supervisor_alive"]:
        return f"already running (supervisor pid {st['supervisor_pid']})"
    stop_file(cfg).unlink(missing_ok=True)
    p = spawn_detached(["-m", "cryptolab", "run"], cfg.path("log_dir") / "supervisor.out", cfg.root)
    return f"started supervisor pid {p.pid}; logs in {cfg.path('log_dir')}"


def stop(cfg, timeout_s: int = 90) -> bool:
    stop_file(cfg).touch()
    pf = run_dir(cfg) / "supervisor.pid"
    pid = int(pf.read_text()) if pf.exists() else None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not pid_alive(pid):
            return True
        time.sleep(1)
    return False


def status(cfg) -> dict:
    pf = run_dir(cfg) / "supervisor.pid"
    pid = int(pf.read_text()) if pf.exists() else None
    hb = read_heartbeat(cfg)
    age = round((now_ms() - hb["ts_ms"]) / 1000, 1) if hb.get("ts_ms") else None
    return {"supervisor_pid": pid, "supervisor_alive": pid_alive(pid), "worker_pid": hb.get("pid"),
            "worker_alive": pid_alive(hb.get("pid")), "heartbeat_age_s": age,
            "healthy": bool(age is not None and age < 120), "heartbeat": hb,
            "stop_requested": stop_file(cfg).exists()}
