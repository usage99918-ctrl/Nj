"""Command-line interface.

  python -m cryptolab collect          download history + build point-in-time universe
  python -m cryptolab build-panels     compute research features/targets
  python -m cryptolab research         walk-forward backtest, pre-registration, test, live model
  python -m cryptolab scan             one multi-exchange market scan (prints counts)
  python -m cryptolab run              supervisor in the foreground (use under Task Scheduler/systemd/Docker)
  python -m cryptolab start | stop     start the supervisor in the background / stop everything gracefully
  python -m cryptolab run-once         one scheduler tick (verify + forecast due slots + gaps)
  python -m cryptolab daemon start|stop|status|run   (legacy aliases)
  python -m cryptolab verify           verify expired forecasts now
  python -m cryptolab dashboard        build reports/dashboard.html + CSV/JSON exports
  python -m cryptolab audit            verify ledger hash chain and DB consistency
  python -m cryptolab rebuild-db       rebuild the SQLite DB from the ledger
  python -m cryptolab status           experiment + worker status summary
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings

from cryptolab.config import load_config
from cryptolab.util.log import setup_logging


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cryptolab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to a TOML config (default config/default.toml + config/local.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    sub.add_parser("build-panels")
    r = sub.add_parser("research")
    r.add_argument("--horizons", default="60,10")
    sub.add_parser("scan")
    sub.add_parser("run-once")
    sub.add_parser("run")
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("worker")
    sub.add_parser("g2-research")
    sub.add_parser("health")
    ks = sub.add_parser("kill-switch")
    ks.add_argument("state", choices=["on", "off", "status"])
    d = sub.add_parser("daemon")
    d.add_argument("action", choices=["start", "stop", "status", "run"])
    sub.add_parser("verify")
    sub.add_parser("dashboard")
    sub.add_parser("audit")
    sub.add_parser("rebuild-db")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    cfg = load_config(a.config)
    setup_logging(cfg.path("log_dir"))

    if a.cmd == "collect":
        from cryptolab.data.history import collect
        from cryptolab.exchanges.binance import Binance
        from cryptolab.util.http import HttpClient
        _print(collect(cfg, Binance(HttpClient(min_interval_s=0.03), cfg["exchanges"]["binance_base"],
                                    cfg["exchanges"]["binance_bulk_base"]))["union_size"])
    elif a.cmd == "build-panels":
        from cryptolab.backtest.dataset import build_panels
        _print({k: str(v) for k, v in build_panels(cfg).items()})
    elif a.cmd == "research":
        from cryptolab.backtest.research import run_research
        from cryptolab.storage.db import open_store
        hs = tuple(int(x) for x in a.horizons.split(","))
        res = run_research(cfg, open_store(cfg), hs)
        _print({h: {"selected": v["selected"], "test_verdict": v["test_verdict"]} for h, v in res.items()})
    elif a.cmd == "scan":
        from cryptolab.live.scheduler import LiveSystem
        s = LiveSystem(cfg)
        res = s.scanner.scan()
        _print(res.summary())
    elif a.cmd == "run-once":
        from cryptolab.live.scheduler import run_once
        _print(run_once(cfg))
    elif a.cmd == "run":
        from cryptolab.live.supervisor import run_supervisor
        return run_supervisor(cfg)
    elif a.cmd == "start":
        from cryptolab.live.supervisor import start_background
        print(start_background(cfg))
    elif a.cmd == "stop":
        from cryptolab.live.supervisor import stop
        print("stopped" if stop(cfg) else "stop requested; still shutting down")
    elif a.cmd == "worker":
        from cryptolab.live.scheduler import run_worker
        run_worker(cfg)
    elif a.cmd == "health":
        from cryptolab.live.supervisor import status as sup_status
        st = sup_status(cfg)
        print(json.dumps({k: st[k] for k in ("healthy", "heartbeat_age_s", "supervisor_alive", "worker_alive")}))
        return 0 if st["healthy"] else 1
    elif a.cmd == "kill-switch":
        from cryptolab.live.paper import kill_switch_file
        f = kill_switch_file(cfg)
        if a.state == "on":
            f.write_text(f"engaged {__import__('datetime').datetime.utcnow().isoformat()}Z")
        elif a.state == "off":
            f.unlink(missing_ok=True)
        print(f"paper-trading kill switch: {'ON (no new entries; open positions are closed)' if f.exists() else 'off'}")
    elif a.cmd == "g2-research":
        from cryptolab.backtest.g2 import run_g2
        from cryptolab.storage.db import open_store
        _print(run_g2(cfg, open_store(cfg)))
    elif a.cmd == "verify":
        from cryptolab.live.scheduler import LiveSystem
        _print({"verified": len(LiveSystem(cfg).verifier.verify_due())})
    elif a.cmd == "daemon":
        from cryptolab.live import scheduler as sch
        if a.action == "run":
            sch.run_worker(cfg)
        elif a.action == "start":
            from cryptolab.live.supervisor import start_background
            print(start_background(cfg))
        elif a.action == "stop":
            print("stopped" if sch.stop_daemon(cfg) else "not running (or did not stop in time)")
        else:
            _print(sch.daemon_status(cfg))
    elif a.cmd == "dashboard":
        from cryptolab.report.dashboard import build_all
        _print(build_all(cfg))
    elif a.cmd == "audit":
        from cryptolab.storage.db import open_store
        res = open_store(cfg).audit()
        _print(res)
        return 0 if res["ok"] else 2
    elif a.cmd == "rebuild-db":
        db = cfg.path("db_file")
        for suffix in ("", "-wal", "-shm"):
            p = db.with_name(db.name + suffix)
            if p.exists():
                p.rename(p.with_name(p.name + ".bak"))
        from cryptolab.storage.db import open_store
        s = open_store(cfg)
        _print({"rebuilt": True, "audit_ok": s.audit()["ok"]})
    elif a.cmd == "status":
        from cryptolab.live.scheduler import daemon_status
        from cryptolab.report.metrics import experiment_summary
        from cryptolab.storage.db import open_store
        store = open_store(cfg)
        from cryptolab.report.metrics import continuity
        _print({"daemon": daemon_status(cfg),
                "experiments": {k: experiment_summary(store, v["name"]) for k, v in cfg["experiments"].items()},
                "continuity": continuity(store, cfg)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
