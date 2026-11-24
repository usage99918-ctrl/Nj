"""Research pipeline: walk-forward backtest of all strategies, pre-registered model
selection on the validation period, a single untouched test evaluation, and fitting of
the live model. Every decision point is written to the hash-chained ledger in order.

Pre-registered selection rule (fixed before any validation/test metric was computed):
  * candidates: all strategies except the four constant/random baselines;
  * must have validation coverage >= 0.90 and >= 300 validation forecasts;
  * choose the highest validation BALANCED accuracy (immune to always-up/down drift);
    tie-break by validation mean cross-sectional IC;
  * the choice is frozen in the ledger BEFORE the test months are even simulated.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from cryptolab.backtest import stats as st
from cryptolab.backtest.engine import WalkForward, summarize
from cryptolab.storage.ledger import canonical
from cryptolab.strategies.library import all_strategies
from cryptolab.util.timeutil import now_ms, to_iso

log = logging.getLogger(__name__)

EXCLUDED_FROM_SELECTION = {"baseline_random", "baseline_always_up", "baseline_always_down", "baseline_no_change"}
SELECTION_RULE = ("Highest validation balanced accuracy among non-baseline strategies with validation coverage "
                  ">= 0.90 and >= 300 forecasts; tie-break by validation mean IC.")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def split_summaries(sel: dict, ics: dict, slots: pd.DataFrame, split: str, strategies) -> dict:
    n_slots = int((slots["split"] == split).sum())
    split_ts = set(slots.loc[slots["split"] == split, "t"])
    out = {}
    for s in strategies:
        fc = sel[s.name]
        fc = fc[fc["split"] == split] if len(fc) else fc
        ic = ics[s.name]
        ic = ic[ic.index.isin(split_ts)] if len(ic) else ic
        m = summarize(fc, ic, n_slots)
        m.update(family=s.family, kind=s.kind, hypothesis=s.hypothesis)
        out[s.name] = m
    return out


def multiple_testing(sel: dict, slots: pd.DataFrame, split: str, summaries: dict, seed: int) -> dict:
    names = [n for n in summaries if n not in EXCLUDED_FROM_SELECTION and summaries[n].get("n_forecasts", 0) > 0]
    p_acc = {n: summaries[n].get("p_acc_gt_50", 1.0) for n in names}
    p_dir = {n: summaries[n].get("p_gross_dir_gt_0", 1.0) for n in names}
    ts = slots.loc[slots["split"] == split, "t"].sort_values().values
    pos = {t: i for i, t in enumerate(ts)}
    D_acc = np.zeros((len(ts), len(names)))
    D_net = np.zeros((len(ts), len(names)))
    for j, n in enumerate(names):
        fc = sel[n]
        fc = fc[fc["split"] == split]
        ii = fc["t"].map(pos).values
        D_acc[ii, j] = fc["correct"].values.astype(float) - 0.5
        D_net[ii, j] = fc["net_ls"].values
    rc_acc = st.reality_check(D_acc, n_boot=1000, seed=seed)
    rc_net = st.reality_check(D_net, n_boot=1000, seed=seed + 1)
    srs = []
    for j in range(len(names)):
        col = D_net[:, j]
        srs.append(col.mean() / col.std() if col.std() > 0 else np.nan)
    return {
        "n_strategies_tested": len(names),
        "holm_p_accuracy": st.holm(p_acc), "bh_q_accuracy": st.benjamini_hochberg(p_acc),
        "holm_p_gross_dir": st.holm(p_dir),
        "reality_check_accuracy": {**rc_acc, "best": names[rc_acc["best_index"]]},
        "reality_check_net_long_short": {**rc_net, "best": names[rc_net["best_index"]]},
        "_sr_trials": srs, "_names": names,
    }


def select(summaries: dict) -> tuple[str, list]:
    elig = [(n, m) for n, m in summaries.items()
            if n not in EXCLUDED_FROM_SELECTION and m.get("coverage", 0) >= 0.90 and m.get("n_forecasts", 0) >= 300]
    elig.sort(key=lambda kv: (kv[1].get("balanced_accuracy", 0), kv[1].get("ic_mean", -1)), reverse=True)
    return elig[0][0], [n for n, _ in elig[:10]]


def buy_and_hold(panel: pd.DataFrame, split: str) -> dict:
    p = panel[panel["split"] == split]
    btc = p[p["symbol"] == "BTCUSDT"].sort_values("t")["y"].sum()
    ew = p.groupby("t")["y"].mean().sum()
    return {"btc_buy_hold_pct": float(np.expm1(btc) * 100), "equal_weight_universe_pct": float(np.expm1(ew) * 100)}


def breakdowns(fc: pd.DataFrame) -> dict:
    fc = fc.copy()
    fc["regime"] = np.where(fc["btc_trend"] > 0, "BTC 24h up", "BTC 24h down") + " / " + \
        np.where(fc["btc_vol_ratio"] > 1, "vol rising", "vol calm")
    by_reg = fc.groupby("regime").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                      gross_dir_bps=("gross_dir", lambda v: v.mean() * 1e4),
                                      net_long_bps=("net_long", lambda v: v.mean() * 1e4)).reset_index()
    by_sym = fc.groupby("symbol").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                      gross_dir_bps=("gross_dir", lambda v: v.mean() * 1e4)).reset_index()
    by_sym = by_sym.sort_values("n", ascending=False).head(25)
    return {"by_regime": by_reg.to_dict("records"), "by_symbol_top25": by_sym.to_dict("records"),
            "reliability": st.reliability_bins(fc["p_up"].values, (fc["y"] > 0).astype(float).values)}


def equity_curves(sel: dict, names: list[str], max_points: int = 400) -> dict:
    out = {}
    for n in names:
        fc = sel[n].sort_values("t")
        if fc.empty:
            continue
        step = max(1, len(fc) // max_points)
        cum_long = np.cumsum(fc["net_long"].values)
        cum_ls = np.cumsum(fc["net_ls"].values)
        cum_gross = np.cumsum(fc["gross_dir"].values)
        idx = np.arange(0, len(fc), step)
        out[n] = {"t": fc["t"].values[idx].tolist(), "split": fc["split"].values[idx].tolist(),
                  "net_long_pct": (cum_long[idx] * 100).round(3).tolist(),
                  "net_ls_pct": (cum_ls[idx] * 100).round(3).tolist(),
                  "gross_dir_pct": (cum_gross[idx] * 100).round(3).tolist()}
    return out


def run_horizon(cfg, store, h: int, exp_key: str) -> dict:
    bt, costs = cfg["backtest"], cfg["costs"]
    research = cfg.path("research_dir")
    cache = cfg.path("data_dir") / "cache"
    panel = pd.read_parquet(cache / f"panel_h{h}.parquet")
    strategies = all_strategies(h, bt["seed"])
    exp_name = cfg["experiments"][exp_key]["name"]
    log.info("h=%d: %d rows, %d decision times, %d strategies", h, len(panel), panel["t"].nunique(), len(strategies))
    slots = panel.groupby("t").agg(split=("split", "first")).reset_index()
    wf = WalkForward(panel, strategies, h, bt, costs)
    months = sorted(panel["month"].unique())
    oos = months[2:]
    val_end = bt["validation_end"][:7]
    phase1 = [m for m in oos if m <= val_end]
    phase2 = [m for m in oos if m > val_end]

    # ---------------- phase 1: train (walk-forward OOS) + validation
    wf.run_months(phase1)
    sel, ics, _ = wf.results()
    train_sum = split_summaries(sel, ics, slots, "train", strategies)
    val_sum = split_summaries(sel, ics, slots, "validation", strategies)
    mt_val = multiple_testing(sel, slots, "validation", val_sum, bt["seed"])
    chosen, top10 = select(val_sum)
    j = mt_val["_names"].index(chosen)
    dsr = st.deflated_sharpe(mt_val["_sr_trials"][j], int(val_sum[chosen]["n_forecasts"]), np.array(mt_val["_sr_trials"]))
    prereg = {
        "prereg_id": f"{exp_name}-selection-v1",
        "experiment": exp_name, "horizon_min": h,
        "selection_rule": SELECTION_RULE,
        "selected_strategy": chosen, "runner_ups": top10,
        "validation_metrics_selected": val_sum[chosen],
        "validation_holm_p_accuracy_selected": mt_val["holm_p_accuracy"].get(chosen),
        "validation_reality_check_accuracy_p": mt_val["reality_check_accuracy"]["p_value"],
        "validation_deflated_sharpe_selected": dsr,
        "n_strategies_tested": mt_val["n_strategies_tested"],
        "test_period": [bt["validation_end"], bt["test_end"]],
        "test_hypotheses": ["H1: test directional accuracy > 50% (one-sided binomial, alpha=0.05)",
                            "H2: test mean gross directional return > 0 (Newey-West t, alpha=0.05)",
                            "H3: test mean long-only return after base-case costs > 0"],
        "code_commit": git_commit(),
        "created_utc": to_iso(now_ms()),
    }
    store.record("preregistration", _jsonable(prereg))
    log.info("h=%d PRE-REGISTERED selection: %s (val BA %.4f)", h, chosen, val_sum[chosen]["balanced_accuracy"])

    # ---------------- phase 2: untouched test period
    wf.run_months(phase2)
    sel, ics, cal_log = wf.results()
    test_sum = split_summaries(sel, ics, slots, "test", strategies)
    mt_test = multiple_testing(sel, slots, "test", test_sum, bt["seed"] + 7)
    ts = test_sum[chosen]
    verdict = {
        "H1_accuracy_gt_50": bool(ts.get("p_acc_gt_50", 1) < 0.05),
        "H2_gross_dir_gt_0": bool(ts.get("p_gross_dir_gt_0", 1) < 0.05),
        "H3_long_net_gt_0": bool(ts.get("long_t", -9) > 1.645),
        "p_accuracy": ts.get("p_acc_gt_50"), "p_gross_dir": ts.get("p_gross_dir_gt_0"), "long_t": ts.get("long_t"),
    }
    sel_fc = sel[chosen]
    vt = sel_fc[sel_fc["split"].isin(["validation", "test"])]

    # ---------------- live model: refit on all history, calibrate on all OOS predictions
    strat = next(s for s in strategies if s.name == chosen)
    if strat.model is not None:
        strat.fit(panel, h)
    cal = wf.final_calibration(chosen)
    models_dir = research / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    version = f"{exp_name}-v1-{chosen}"
    blob = cloudpickle.dumps({"strategy": strat, "calibration": cal.to_dict(), "horizon": h, "version": version})
    mpath = models_dir / f"{exp_name}_v1.pkl"
    mpath.write_bytes(blob)
    spec = {"version": version, "experiment": exp_name, "horizon_min": h, "strategy": chosen,
            "family": strat.family, "hypothesis": strat.hypothesis, "kind": strat.kind,
            "calibration": cal.to_dict(), "trained_through": bt["test_end"], "model_file": mpath.name,
            "model_sha256": hashlib.sha256(blob).hexdigest(), "selection_prereg": prereg["prereg_id"],
            "code_commit": git_commit(), "created_utc": to_iso(now_ms())}
    store.record("model_version", _jsonable(spec))

    keep = sorted(set([chosen] + top10[:5] + [f"mom_{h}", f"rev_{h}", "baseline_random", "baseline_always_up",
                                               "baseline_always_down"]))
    result = {
        "experiment": exp_name, "horizon_min": h,
        "rows": int(len(panel)), "decision_times": int(panel["t"].nunique()),
        "coins_in_universe_union": int(panel["symbol"].nunique()),
        "splits": {k: {"from": to_iso(int(panel.loc[panel["split"] == k, "t"].min())),
                       "to": to_iso(int(panel.loc[panel["split"] == k, "t"].max())),
                       "decision_times": int((slots["split"] == k).sum())}
                   for k in ("train", "validation", "test")},
        "n_strategies": len([s for s in strategies if not s.name.startswith("baseline_")]),
        "n_baselines": len([s for s in strategies if s.name.startswith("baseline_")]) + 2,
        "selected": chosen, "prereg": prereg, "test_verdict": verdict,
        "summaries": {"train": train_sum, "validation": val_sum, "test": test_sum},
        "multiple_testing": {"validation": {k: v for k, v in mt_val.items() if not k.startswith("_")},
                             "test": {k: v for k, v in mt_test.items() if not k.startswith("_")}},
        "buy_and_hold": {k: buy_and_hold(panel, k) for k in ("train", "validation", "test")},
        "selected_breakdowns_val_test": breakdowns(vt) if len(vt) else {},
        "equity_curves": equity_curves(sel, keep),
        "calibration_log_selected": cal_log[chosen],
        "live_model": spec,
    }
    result = _jsonable(result)
    (research / f"backtest_{exp_name}.json").write_text(json.dumps(result, indent=1, default=str))
    store.record("preregistration", {"prereg_id": f"{exp_name}-test-result-v1", "experiment": exp_name,
                                     "selected_strategy": chosen, "test_metrics_selected": _jsonable(ts),
                                     "verdict": verdict, "created_utc": to_iso(now_ms())})
    # Compact per-forecast records of the selected strategy for audit/dashboard
    sel_fc.to_parquet(research / f"backtest_{exp_name}_selected_forecasts.parquet", index=False)
    return result


def run_research(cfg, store, horizons=(10, 60)) -> dict:
    out = {}
    for h in horizons:
        key = "A" if h == 10 else "B"
        t0 = time.time()
        out[h] = run_horizon(cfg, store, h, key)
        log.info("h=%d research finished in %.0f s", h, time.time() - t0)
    return out
