"""Forward-experiment metrics computed from the immutable forecast/outcome/gap records.

Benchmarks are evaluated on exactly the same forecast windows (same coin, same origin
and expiry), so differences are attributable to direction calls, not market periods.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from cryptolab.backtest import stats as st


def _join(store, name: str) -> tuple[list[dict], dict, list[dict]]:
    fcs = store.rows("forecasts", "experiment=?", (name,))
    outs = {o["forecast_id"]: o for o in store.rows("outcomes", "experiment=?", (name,))}
    gaps = store.rows("gaps", "experiment=?", (name,))
    return fcs, outs, gaps


def _rate(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": [None, None], "p_vs_50": None}
    lo, hi = st.wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi], "p_vs_50": st.binom_p_greater(k, n)}


def experiment_summary(store, name: str, seed: int = 7) -> dict:
    fcs, outs, gaps = _join(store, name)
    scored = [(f, outs[f["forecast_id"]]) for f in fcs
              if f["forecast_id"] in outs and outs[f["forecast_id"]]["status"] in ("VERIFIED", "APPROXIMATE")]
    status = Counter(outs[f["forecast_id"]]["status"] if f["forecast_id"] in outs else "PENDING" for f in fcs)
    res = {
        "experiment": name, "forecasts": len(fcs), "gaps": len(gaps),
        "gap_reasons": dict(Counter(g["reason"].split(" (")[0] for g in gaps)),
        "status_counts": dict(status), "scored": len(scored),
        "first_forecast_utc": fcs[0]["origin_utc"] if fcs else None,
        "last_forecast_utc": fcs[-1]["origin_utc"] if fcs else None,
        "model_versions": sorted({f["model_version"] for f in fcs}),
    }
    if not scored:
        return res
    corr = np.array([o["correct"] for _, o in scored], dtype=bool)
    actual_up = np.array([o["actual_direction"] == "UP" for _, o in scored])
    actual_dn = np.array([o["actual_direction"] == "DOWN" for _, o in scored])
    pred_up = np.array([f["direction"] == "UP" for f, _ in scored])
    res["accuracy"] = _rate(int(corr.sum()), len(corr))
    res["flat_outcomes"] = int(sum(o["actual_direction"] == "FLAT" for _, o in scored))
    res["approximate_outcomes"] = int(sum(o["status"] == "APPROXIMATE" for _, o in scored))
    tpr = (pred_up & actual_up).sum() / max(actual_up.sum(), 1)
    tnr = (~pred_up & actual_dn).sum() / max(actual_dn.sum(), 1)
    res["balanced_accuracy"] = float(0.5 * (tpr + tnr)) if actual_up.any() and actual_dn.any() else None
    res["frac_pred_up"] = float(pred_up.mean())
    res["mean_abs_pred_error_pct"] = float(np.mean([o["abs_pred_error_pct"] for _, o in scored]))
    res["mean_no_change_error_pct"] = float(np.mean([o["no_change_error_pct"] for _, o in scored]))
    res["error_ratio_vs_no_change"] = (res["mean_abs_pred_error_pct"] / res["mean_no_change_error_pct"]
                                       if res["mean_no_change_error_pct"] > 0 else None)
    res["coverage_pi50"] = _rate(int(sum(o["in_pi50"] for _, o in scored)), len(scored))
    res["coverage_pi80"] = _rate(int(sum(o["in_pi80"] for _, o in scored)), len(scored))
    brier = np.array([o["brier"] for _, o in scored])
    res["brier"] = float(brier.mean())
    res["brier_skill_vs_0.5"] = float(1 - brier.mean() / 0.25)
    p_up = np.array([f["p_up"] for f, _ in scored])
    res["reliability"] = st.reliability_bins(p_up, actual_up.astype(float), bins=5)
    res["mean_p_direction"] = float(np.mean([f["p_direction"] for f, _ in scored]))
    w, l = st.streaks(corr)
    res["max_win_streak"], res["max_loss_streak"] = int(w), int(l)

    # ---- paper trading (spot, long-only, fixed notional per trade)
    long_net = np.array([o["long_net_pct"] for _, o in scored]) / 100
    taken = np.array([o["long_taken"] for _, o in scored])
    dir_net = np.array([o["dir_net_pct"] for _, o in scored]) / 100
    dir_gross = np.array([o["dir_gross_pct"] for _, o in scored]) / 100
    res["paper_long_only"] = {
        "trades": int(taken.sum()), "total_net_pct_of_notional": float(long_net.sum() * 100),
        "win_rate": float((long_net[taken] > 0).mean()) if taken.any() else None,
        "max_drawdown_pct_of_notional": st.max_drawdown(np.cumsum(long_net)) * 100,
        "note": "Spot long-only: positions only when the frozen forecast was UP with expected move > costs.",
    }
    res["hypothetical_direction_bets"] = {
        "gross_total_pct": float(dir_gross.sum() * 100), "net_total_pct": float(dir_net.sum() * 100),
        "mean_net_bps": float(dir_net.mean() * 1e4),
        "max_drawdown_net_pct": st.max_drawdown(np.cumsum(dir_net)) * 100,
        "short_count": int((~pred_up).sum()),
        "note": "Trades every forecast in its predicted direction. Shorts are theoretical (margin/borrow) "
                "and NOT executable on spot.",
    }
    m, se, t = st.newey_west_t(dir_gross)
    res["gross_direction_return"] = {"mean_bps": m * 1e4 if np.isfinite(m) else None, "t_stat": t,
                                     "p_gt_0": st.p_from_t(t)}

    # ---- benchmarks on identical windows
    rng = np.random.default_rng(seed)
    rand_up = rng.random(len(scored)) < 0.5
    feats_r = []
    for f, _ in scored:
        h = f["horizon_min"]
        feats_r.append((f.get("features") or {}).get(f"r_{h}"))
    mom_dir = np.array([None if r is None else (r > 0) for r in feats_r], dtype=object)
    has_r = np.array([r is not None and r != 0 for r in feats_r])
    rets = np.array([o["actual_return_pct"] for _, o in scored]) / 100
    rt = np.array([f["costs"]["round_trip"] for f, _ in scored])
    bm = {
        "random_direction(seeded)": _rate(int(((rand_up & actual_up) | (~rand_up & actual_dn)).sum()), len(scored)),
        "always_up": _rate(int(actual_up.sum()), len(scored)),
        "always_down": _rate(int(actual_dn.sum()), len(scored)),
    }
    if has_r.any():
        md = np.array([bool(x) for x in mom_dir[has_r]])
        au, ad = actual_up[has_r], actual_dn[has_r]
        bm["simple_momentum"] = _rate(int(((md & au) | (~md & ad)).sum()), int(has_r.sum()))
        bm["simple_mean_reversion"] = _rate(int(((~md & au) | (md & ad)).sum()), int(has_r.sum()))
    res["benchmarks_accuracy"] = bm
    res["benchmarks_returns"] = {
        "buy_and_hold_same_windows_gross_pct": float(rets.sum() * 100),
        "buy_and_hold_same_windows_net_pct": float((rets - rt).sum() * 100),
        "stay_in_cash_pct": 0.0,
        "no_change_forecast_mean_abs_error_pct": res["mean_no_change_error_pct"],
    }

    # ---- by asset and by market condition
    by_sym: dict[str, list] = {}
    by_reg: dict[str, list] = {}
    for (f, o), c in zip(scored, corr):
        by_sym.setdefault(f["symbol"], []).append(c)
        ft = f.get("features") or {}
        reg = ("BTC 24h up" if (ft.get("btc_trend") or 0) > 0 else "BTC 24h down") + " / " + \
              ("vol rising" if (ft.get("btc_vol_ratio") or 0) > 1 else "vol calm")
        by_reg.setdefault(reg, []).append(c)
    res["by_asset"] = sorted(({"symbol": k, "n": len(v), "accuracy": float(np.mean(v))} for k, v in by_sym.items()),
                             key=lambda d: -d["n"])
    res["by_market_condition"] = [{"regime": k, "n": len(v), "accuracy": float(np.mean(v))} for k, v in by_reg.items()]
    res["sample_size_note"] = _sample_note(len(scored))
    return res


def _sample_note(n: int) -> str:
    if n < 30:
        return f"n={n}: far too small for any conclusion; results are anecdotal."
    hw = 1.96 * math.sqrt(0.25 / n) * 100
    return (f"n={n}: a 95% interval on accuracy is roughly +/-{hw:.1f} percentage points; "
            f"an edge smaller than that cannot be distinguished from chance.")


def equity_series(store, name: str) -> dict:
    fcs, outs, _ = _join(store, name)
    xs, long_cum, dir_cum, acc_cum = [], [], [], []
    lc = dc = 0.0
    k = n = 0
    for f in fcs:
        o = outs.get(f["forecast_id"])
        if not o or o["status"] not in ("VERIFIED", "APPROXIMATE"):
            continue
        lc += o["long_net_pct"]
        dc += o["dir_net_pct"]
        n += 1
        k += int(o["correct"])
        xs.append(f["test_number"])
        long_cum.append(round(lc, 4))
        dir_cum.append(round(dc, 4))
        acc_cum.append(round(k / n * 100, 2))
    return {"test_number": xs, "long_net_cum_pct": long_cum, "dir_net_cum_pct": dir_cum, "accuracy_cum_pct": acc_cum}
