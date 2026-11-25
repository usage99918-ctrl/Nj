"""`python -m cryptolab doctor` -- self-check of every live model on THIS computer's Python.

It exercises the same code path the worker uses (load frozen model -> verify its SHA-256 against
the ledger -> build a live feature panel -> predict -> select the coin -> probability layers),
but it RECORDS NOTHING: the database is opened read-only and klines go to a temporary cache.
Safe to run while the bot is running. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

DOCTOR_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"]


def _ro_model_versions(db_file: Path) -> dict:
    """Latest model_version payload per experiment, read-only (never writes, never reconciles)."""
    out: dict = {}
    if not db_file.exists():
        return out
    con = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        for (payload,) in con.execute("SELECT payload FROM model_versions ORDER BY seq"):
            d = json.loads(payload)
            out[d.get("experiment")] = d
    finally:
        con.close()
    return out


def _versions() -> dict:
    v = {"python": sys.version.split()[0], "platform": f"{platform.system()} {platform.release()}"}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "cloudpickle", "pyarrow"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception as e:  # pragma: no cover - reported, not fatal here
            v[mod] = f"MISSING ({type(e).__name__})"
    return v


def _live_panel(cfg, tmp: Path):
    from cryptolab.exchanges.binance import Binance
    from cryptolab.live.forecaster import build_live_panel
    from cryptolab.live.scanner import KlineCache
    from cryptolab.util.http import HttpClient
    hc, ex = cfg["http"], cfg["exchanges"]
    bn = Binance(HttpClient(hc["timeout_s"], hc["max_retries"], hc["backoff_base_s"], hc["min_interval_s"]),
                 ex["binance_base"], ex["binance_bulk_base"])
    # Seed the temporary cache from the worker's cache (read-only copy) to save downloads.
    live_cache = cfg.path("data_dir") / "run" / "klines"
    for s in DOCTOR_SYMBOLS:
        src = live_cache / f"{s}.parquet"
        if src.exists():
            try:
                shutil.copy2(src, tmp / src.name)
            except OSError:
                pass
    bn.server_time()
    now = bn.clock.now()
    panel, diag = build_live_panel(KlineCache(bn, tmp), DOCTOR_SYMBOLS, now)
    return panel, diag


def synthetic_panel(n: int = 12, seed: int = 7):
    """Deterministic fake feature panel (all model features) -- exercises shapes/code paths offline only."""
    import pandas as pd
    from cryptolab.features import MODEL_FEATURES
    rng = np.random.default_rng(seed)
    cols = {}
    for c in sorted(set(MODEL_FEATURES) | {"sig1m", "rel_vol_10", "rel_vol_60", "r_10", "r_60", "rsi_60"}):
        if c.startswith(("rsi_",)):
            v = rng.uniform(20, 80, n)
        elif c.startswith(("rel_vol", "vol_ratio", "btc_vol_ratio")):
            v = rng.lognormal(0, 0.5, n)
        elif c in ("sig1m",) or c.startswith(("vol_", "disp_")):
            v = rng.uniform(5e-4, 3e-3, n)
        elif c.startswith(("pos_", "bb_pctb", "breadth", "xs_rank")):
            v = rng.uniform(0, 1, n)
        elif c == "hour":
            v = rng.integers(0, 24, n).astype(float)
        else:
            v = rng.normal(0, 0.003, n)
        cols[c] = v
    p = pd.DataFrame(cols)
    p["t"] = 1_700_000_000_000
    p["symbol"] = [f"SYN{i}USDT" for i in range(n)]
    p["universe_rank"] = np.arange(1, n + 1)
    return p


def run_doctor(cfg, offline: bool = False) -> dict:
    from cryptolab.backtest.engine import select_top
    from cryptolab.live.forecaster import LiveModel
    from cryptolab.strategies.portable import _is_by_value

    t0 = time.time()
    rep: dict = {"versions": _versions(), "checks": [], "ok": True}

    def check(name: str, ok: bool, detail: str = "") -> None:
        rep["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        rep["ok"] = rep["ok"] and bool(ok)

    mv = _ro_model_versions(cfg.path("db_file"))
    check("database readable (read-only)", bool(mv), f"{len(mv)} model versions in {cfg.path('db_file')}")

    panel, panel_note = None, "offline mode"
    if offline:
        panel, panel_note = synthetic_panel(), "OFFLINE: synthetic panel (checks code paths and shapes, not live data)"
        check("synthetic feature panel (offline mode)", True, panel_note)
    tmp = Path(tempfile.mkdtemp(prefix="cryptolab-doctor-"))
    try:
        if not offline:
            try:
                panel, diag = _live_panel(cfg, tmp)
                panel_note = f"live panel: {len(panel)} coins at {diag['t_common_ms']}; skipped {diag['skipped']}"
                check("live market data + feature panel", len(panel) >= 3, panel_note)
            except Exception as e:
                panel_note = f"live panel failed: {type(e).__name__}: {str(e)[:160]}"
                check("live market data + feature panel", False, panel_note)
                panel = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    models_dir = cfg.path("research_dir") / "models"
    for key, exp in cfg["experiments"].items():
        name, etype = exp["name"], exp.get("type", "forecast")
        if etype == "kalshi":
            continue  # checked once below (one model file shared by four arms)
        if etype in ("e_touch", "f_double"):
            from cryptolab.live.opportunity import doctor_check
            for c in doctor_check(cfg, key, exp, mv):
                check(**c)
            continue
        path = models_dir / exp.get("model_file", f"{name}_v1.pkl")
        if not path.exists():
            check(f"{name}: model file", False, f"missing {path}")
            continue
        try:
            m = LiveModel(path)
            reg = mv.get(name, {})
            check(f"{name}: model hash matches ledger", reg.get("model_sha256") == m.sha256,
                  f"file {m.sha256[:16]}..., ledger {str(reg.get('model_sha256'))[:16]}...")
            rb = m.portability["rules_rebuilt"]
            check(f"{name}: portable on Python {rep['versions']['python']}", True,
                  (f"rule '{rb[0]['rule']}' rebuilt from library (bytecode identical: {rb[0]['bytecode_identical']})"
                   if rb else f"{m.strategy.kind} '{m.strategy.name}'; guarded stored functions: {m.portability['guarded']}"))
            if panel is None:
                check(f"{name}: predicts", False, f"not run ({panel_note})")
                continue
            p = panel.copy()
            p["sig_h"] = p["sig1m"].astype("float64") * math.sqrt(m.h)
            z = np.asarray(m.strategy.predict(p, m.h), dtype="float64")
            ok = z.shape == (len(p),) and bool(np.all(np.isfinite(z)))
            idx = select_top(p["t"].values, z, p["universe_rank"].values)
            detail = f"{len(p)} scores, selected {p['symbol'].iloc[int(idx[0])] if len(idx) else 'none (abstain)'}"
            if m.layers is not None and len(idx):
                i = int(idx[0])
                mu = m.cal["b"] * z[i] * float(p["sig_h"].iloc[i])
                three = [float(v[0]) for v in m.layers.three_class([mu], [float(p["sig_h"].iloc[i])])]
                ok = ok and abs(sum(three) - 1) < 1e-6
                detail += f"; P(up/down/sideways)={[round(x, 3) for x in three]}"
            check(f"{name}: predicts", ok, detail)
        except Exception as e:
            check(f"{name}: loads and predicts", False,
                  f"{type(e).__name__}: {str(e)[:200]} | {traceback.format_exc(limit=2)[-300:]}")

    kx = [e for e in cfg["experiments"].values() if e.get("type") == "kalshi"]
    if kx:
        import cloudpickle
        path = models_dir / kx[0].get("model_file", "KX_v1.pkl")
        try:
            blob = path.read_bytes()
            sha = hashlib.sha256(blob).hexdigest()
            d = cloudpickle.loads(blob)
            reg = mv.get("KX_BTC_with", {})
            check("KX (Kalshi arms): model hash matches ledger", reg.get("model_sha256") == sha, sha[:16] + "...")
            stored = [k for k, v in vars(d["base"]).items() if _is_by_value(v)]
            s = d["base"].score(np.zeros((1, len(d["features"]))))
            check("KX (Kalshi arms): loads and scores", not stored and bool(np.all(np.isfinite(s))),
                  f"no stored bytecode: {not stored}; score(0)={float(s[0]):.4f}")
        except Exception as e:
            check("KX (Kalshi arms): loads and scores", False, f"{type(e).__name__}: {str(e)[:200]}")
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def print_report(rep: dict) -> None:
    v = rep["versions"]
    print(f"cryptolab doctor  |  Python {v['python']} on {v['platform']}  |  " +
          ", ".join(f"{k} {v[k]}" for k in ("numpy", "pandas", "sklearn", "lightgbm", "cloudpickle")))
    for c in rep["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}" + (f"  -- {c['detail']}" if c["detail"] else ""))
    n_fail = sum(not c["ok"] for c in rep["checks"])
    print(f"RESULT: {'ALL CHECKS PASSED' if rep['ok'] else f'{n_fail} CHECK(S) FAILED'}  ({rep['seconds']} s; nothing was recorded)")
