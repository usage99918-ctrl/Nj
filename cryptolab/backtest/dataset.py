"""Build the point-in-time research panel: one row per (decision time t, universe coin)."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cryptolab.features import add_targets, cross_sectional, reference_series, symbol_features, to_grid
from cryptolab.util.timeutil import DAY, parse_date_ms

log = logging.getLogger(__name__)


def _month_of(ms: np.ndarray) -> np.ndarray:
    return pd.to_datetime(ms, unit="ms", utc=True).strftime("%Y-%m").values


def split_of(t_ms: np.ndarray, bt: dict) -> np.ndarray:
    start = parse_date_ms(bt["start"])
    tr_end = parse_date_ms(bt["train_end"]) + DAY
    va_end = parse_date_ms(bt["validation_end"]) + DAY
    te_end = parse_date_ms(bt["test_end"]) + DAY
    out = np.full(len(t_ms), "", dtype=object)
    out[(t_ms >= start) & (t_ms < tr_end)] = "train"
    out[(t_ms >= tr_end) & (t_ms < va_end)] = "validation"
    out[(t_ms >= va_end) & (t_ms < te_end)] = "test"
    return out


def _one_symbol(args) -> pd.DataFrame | None:
    sym, path, ref_paths, months, horizons, bt, ranks = args
    df = pd.read_parquet(path)
    if df.empty:
        return None
    g = to_grid(df)
    ref = {k: pd.read_pickle(p) for k, p in ref_paths.items()}
    f = symbol_features(g, ref)
    f = add_targets(f, horizons)
    f["missing"] = g["missing"]
    t = f.index.values
    mon = _month_of(t)
    in_uni = np.isin(mon, list(months))
    out = []
    for h in horizons:
        on_grid = (t % (h * 60_000)) == 0
        sel = f[in_uni & on_grid].copy()
        if sel.empty:
            continue
        sel["horizon"] = h
        out.append(sel)
    if not out:
        return None
    res = pd.concat(out)
    res["symbol"] = sym
    res.index.name = "t"
    res = res.reset_index()
    res["month"] = _month_of(res["t"].values)
    res["universe_rank"] = [ranks.get((m, sym), 99) for m in res["month"]]
    res["split"] = split_of(res["t"].values, bt)
    res = res[res["split"] != ""]
    # float32 for compactness except prices/targets
    for c in res.columns:
        if res[c].dtype == "float64" and not (c == "close" or c.startswith("y_")):
            res[c] = res[c].astype("float32")
    return res


def build_panels(cfg, horizons=(10, 60), workers: int = 4) -> dict[int, Path]:
    raw = cfg.path("raw_dir") / "binance_1m"
    research = cfg.path("research_dir")
    cache = cfg.path("data_dir") / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    meta = json.loads((research / "pit_universe.json").read_text())
    uni = meta["universe_by_month"]
    bt = cfg["backtest"]
    sym_months: dict[str, set] = {}
    ranks = {}
    for m, syms in uni.items():
        for i, s in enumerate(syms):
            sym_months.setdefault(s, set()).add(m)
            ranks[(m, s)] = i + 1
    ref_paths = {}
    for k in ("BTC", "ETH"):
        r = reference_series(pd.read_parquet(raw / f"{k}USDT.parquet"))
        p = cache / f"ref_{k}.pkl"
        r.to_pickle(p)
        ref_paths[k] = p
    jobs = []
    for s, months in sorted(sym_months.items()):
        p = raw / f"{s}.parquet"
        if not p.exists():
            log.warning("no minute history for %s; excluded (recorded as data gap)", s)
            continue
        jobs.append((s, p, ref_paths, months, list(horizons), bt, ranks))
    frames = []
    with ProcessPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(_one_symbol, jobs)):
            if res is not None:
                frames.append(res)
            if (i + 1) % 10 == 0:
                log.info("features %d/%d symbols", i + 1, len(jobs))
    panel = pd.concat(frames, ignore_index=True)
    out = {}
    for h in horizons:
        p = panel[panel["horizon"] == h].copy()
        p = p.sort_values(["t", "symbol"]).reset_index(drop=True)
        p = cross_sectional(p)
        p["sig_h"] = (p["sig1m"].astype("float64") * np.sqrt(h))
        p["y"] = p[f"y_{h}"]
        p["y_std"] = p["y"] / p["sig_h"]
        # Require a complete feature history and a defined target
        need = ["r_1440", "vol_1440", "sig_h", "beta_btc", "y"]
        before = len(p)
        p = p.dropna(subset=need)
        p = p[(p["sig_h"] > 0) & (p["missing_60"] < 30)]
        log.info("h=%d rows %d -> %d after completeness filter", h, before, len(p))
        path = cache / f"panel_h{h}.parquet"
        p.to_parquet(path, index=False)
        out[h] = path
    return out
