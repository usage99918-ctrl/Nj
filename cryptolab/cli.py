"""Command-line interface.

  python -m cryptolab collect          download history + build point-in-time universe
  python -m cryptolab build-panels     compute research features/targets
  python -m cryptolab research         walk-forward backtest, pre-registration, test, live model
  python -m cryptolab scan             one multi-exchange market scan (prints counts)
  python -m cryptolab run-once         one scheduler tick (verify + forecast due slots + gaps)
  python -m cryptolab daemon start|stop|status|run
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
    elif a.cmd == "verify":
        from cryptolab.live.scheduler import LiveSystem
        _print({"verified": len(LiveSystem(cfg).verifier.verify_due())})
    elif a.cmd == "daemon":
        from cryptolab.live import scheduler as sch
        if a.action == "run":
            sch.run_daemon(cfg)
        elif a.action == "start":
            st = sch.daemon_status(cfg)
            if st["process_alive"]:
                print(f"already running (pid {st['pid']})")
                return 0
            log_dir = cfg.path("log_dir")
            log_dir.mkdir(parents=True, exist_ok=True)
            out = open(log_dir / "daemon.out", "a")
            p = subprocess.Popen([sys.executable, "-m", "cryptolab", "daemon", "run"], stdout=out, stderr=out,
                                 stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(cfg.root))
            print(f"daemon started (pid {p.pid}); logs: {log_dir / 'daemon.out'}")
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
        _print({"daemon": daemon_status(cfg),
                "experiments": {k: experiment_summary(store, cfg["experiments"][k]["name"]) for k in ("A", "B")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
