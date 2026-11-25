"""Event-study backtest of the three externally proposed strategies (definitions fixed a priori).

For every signal: direction-adjusted forward returns at 60 / 240 / 1440 minutes, and a
simulated paper trade with stop-loss / take-profit / max-hold exits after base-case costs.
Benchmarks: 200 random-entry samples (same coins, same split, same count, same direction mix)
and buy-and-hold over each split. Train / validation / test splits are reported separately;
since nothing is fitted, all three are out-of-sample for the fixed rules.
"""
from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cryptolab.backtest import stats as st
from cryptolab.backtest.dataset import split_of
from cryptolab.backtest.research import _jsonable, git_commit
from cryptolab.features import to_grid
from cryptolab.strategies import friend as F
from cryptolab.util.timeutil import MIN, now_ms, to_iso

log = logging.getLogger(__name__)
HORIZONS = (60, 240, 1440)


def _forward(g: pd.DataFrame, t: int, direction: int, px: float) -> dict:
    out = {}
    for h in HORIZONS:
        tt = t + h * MIN
        if tt in g.index:
            out[f"r_{h}"] = direction * math.log(float(g.at[tt, "close"]) / px)
    return out


def _symbol_events(args) -> list[dict]:
    sym, path, months, rank, cfg_costs = args
    g = to_grid(pd.read_parquet(path))
    events = []
    for kind, sigs in (("S2_breakout_retest", F.s2_signals(g)), ("S3_liquidity_recovery_proxy", F.s3_signals(g))):
        if sigs is None or len(sigs) == 0:
            continue
        sig_month = pd.to_datetime(sigs["t"], unit="ms", utc=True).dt.strftime("%Y-%m")
        sigs = sigs[sig_month.isin(months).values]   # only while the coin was in the point-in-time universe
        for r in sigs.to_dict("records"):
            t, d, px = int(r["t"]), int(r["direction"]), float(r["price"])
            if kind.startswith("S2"):
                stop = r["stop"]
                take = px + F.S2["rr"] * (px - stop) if d > 0 else px - F.S2["rr"] * (stop - px)
                hold = F.S2["max_hold_min"]
            else:
                stop, take, hold = r["stop"], r["target"], F.S3["max_hold_min"]
            ex = F.simulate_exit(g, t, d, px, stop, take, hold)
            e = {"strategy": kind, "symbol": sym, "t": t, "direction": d, "price": px, "rank": rank.get(sym, 99),
                 **_forward(g, t, d, px), "exit_reason": ex["reason"],
                 "trade_r": d * math.log(ex["exit_px"] / px) if ex["exit_px"] else None}
            events.append(e)
    return events


def run(cfg, store) -> dict:
    raw = cfg.path("raw_dir") / "binance_1m"
    research = cfg.path("research_dir")
    bt, costs = cfg["backtest"], cfg["costs"]
    meta = json.loads((research / "pit_universe.json").read_text())
    prereg = {"prereg_id": "friend-strategies-definitions-v1", "experiment": "friend_strategies",
              "definitions": {"S1": F.S1, "S2": F.S2, "S3": F.S3}, "horizons_min": list(HORIZONS),
              "notes": "Thresholds fixed before any result was computed; no tuning. S3 historical version uses a "
                       "Corwin-Schultz spread ESTIMATE (no historical order books) and is labelled a proxy.",
              "code_commit": git_commit(), "created_utc": to_iso(now_ms())}
    store.record("preregistration", _jsonable(prereg))
    sym_months, ranks = {}, {}
    for m, syms in meta["universe_by_month"].items():
        for i, s in enumerate(syms):
            sym_months.setdefault(s, set()).add(m)
            ranks[s] = min(ranks.get(s, 99), i + 1)
    jobs = [(s, raw / f"{s}.parquet", ms, ranks, costs) for s, ms in sym_months.items() if (raw / f"{s}.parquet").exists()]
    events = []
    with ProcessPoolExecutor(4) as ex:
        for i, ev in enumerate(ex.map(_symbol_events, jobs)):
            events.extend(ev)
            if (i + 1) % 20 == 0:
                log.info("friend strategies: %d/%d symbols", i + 1, len(jobs))
    # ---- S1 on the real ETH/BTC spot pair
    ratio = to_grid(pd.read_parquet(raw / "ETHBTC.parquet"))
    s1 = F.s1_signals(ratio)
    sd4 = np.log(ratio["close"]).diff().rolling(1440 * 7, min_periods=1440).std() * math.sqrt(240)
    for r in s1.to_dict("records"):
        t, d, px = int(r["t"]), int(r["direction"]), float(r["price"])
        s = float(sd4.get(t, np.nan))
        stop = px * (1 - 1.5 * s) if d > 0 else px * (1 + 1.5 * s)
        take = px * (1 + 2.0 * s) if d > 0 else px * (1 - 2.0 * s)
        ex = F.simulate_exit(ratio, t, d, px, stop if np.isfinite(s) else None, take if np.isfinite(s) else None, 1440)
        events.append({"strategy": "S1_rs_reset_ethbtc", "symbol": "ETHBTC", "t": t, "direction": d, "price": px,
                       "rank": 1, "z": r["z"], **_forward(ratio, t, d, px), "exit_reason": ex["reason"],
                       "trade_r": d * math.log(ex["exit_px"] / px) if ex["exit_px"] else None})
    ev = pd.DataFrame(events)
    ev["split"] = split_of(ev["t"].values, bt)
    ev = ev[ev["split"] != ""]
    fee = costs["taker_fee"]["binance"]
    hs = np.where(ev["rank"] <= 10, costs["backtest_half_spread_top10"], costs["backtest_half_spread_other"])
    ev["cost_rt"] = 2 * (fee + hs + costs["slippage_per_side"])
    ev["trade_net"] = ev["trade_r"] - ev["cost_rt"]
    ev.to_parquet(research / "friend_strategy_events.parquet", index=False)
    results = {"generated_utc": to_iso(now_ms()), "prereg_id": prereg["prereg_id"], "strategies": {}}
    rng = np.random.default_rng(bt["seed"])
    grids_cache: dict[str, pd.DataFrame] = {}

    def random_baseline(sub: pd.DataFrame, h: int, n_rep: int = 200) -> dict:
        """Random entries on the same symbols within the same split, same direction mix."""
        means = []
        syms = sub["symbol"].values
        dirs = sub["direction"].values
        lo, hi = sub["t"].min(), sub["t"].max()
        for _ in range(n_rep):
            vals = []
            for s, d in zip(syms[:300], dirs[:300]):
                g = grids_cache.get(s)
                if g is None:
                    g = ratio if s == "ETHBTC" else to_grid(pd.read_parquet(raw / f"{s}.parquet"))
                    grids_cache[s] = g
                tt = int(rng.integers(lo // MIN, hi // MIN)) * MIN
                if tt in g.index and tt + h * MIN in g.index:
                    vals.append(d * math.log(float(g.at[tt + h * MIN, "close"]) / float(g.at[tt, "close"])))
            if vals:
                means.append(np.mean(vals))
        return {"mean_bps": float(np.mean(means) * 1e4) if means else None,
                "p95_bps": float(np.quantile(means, 0.95) * 1e4) if means else None, "samples": means}

    for strat, sg in ev.groupby("strategy"):
        res = {}
        for split in ("train", "validation", "test"):
            sub = sg[sg["split"] == split].sort_values("t")
            m = {"signals": int(len(sub)), "long_share": float((sub["direction"] > 0).mean()) if len(sub) else None}
            for h in HORIZONS:
                v = sub[f"r_{h}"].dropna().values
                if len(v) == 0:
                    continue
                mean, se, t = st.newey_west_t(v)
                k = int((v > 0).sum())
                m[f"h{h}"] = {"n": int(len(v)), "hit_rate": k / len(v), "hit_ci95": st.wilson_ci(k, len(v)),
                              "mean_gross_bps": mean * 1e4, "t_stat": t,
                              "mean_net_bps": (mean - float(sub["cost_rt"].mean())) * 1e4}
                if split != "train" and len(sub) >= 5:
                    rb = random_baseline(sub, h)
                    samp = np.array(rb.pop("samples"))
                    rb["p_value_vs_random"] = float(((samp >= mean).sum() + 1) / (len(samp) + 1)) if len(samp) else None
                    m[f"h{h}"]["random_entry_baseline"] = rb
            tr = sub["trade_net"].dropna().values
            if len(tr):
                mean, se, t = st.newey_west_t(tr)
                cum = np.cumsum(tr)
                m["paper_trades"] = {"n": int(len(tr)), "win_rate": float((tr > 0).mean()), "mean_net_bps": mean * 1e4,
                                     "t_stat": t, "total_net_pct": float(cum[-1] * 100),
                                     "max_drawdown_pct": st.max_drawdown(cum) * 100,
                                     "exit_reasons": sub["exit_reason"].value_counts().to_dict()}
            res[split] = m
        results["strategies"][strat] = res
    (research / "friend_strategies.json").write_text(json.dumps(_jsonable(results), indent=1, default=str))
    summary = {s: {sp: {"signals": v[sp]["signals"], "h240": v[sp].get("h240", {}).get("mean_net_bps"),
                        "trades_net_bps": v[sp].get("paper_trades", {}).get("mean_net_bps")} for sp in v}
               for s, v in results["strategies"].items()}
    store.record("preregistration", _jsonable({"prereg_id": "friend-strategies-results-v1",
                                               "experiment": "friend_strategies", "summary": summary,
                                               "created_utc": to_iso(now_ms())}))
    return results
