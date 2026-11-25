"""Generation-2 research: multi-horizon experiments with calibrated probability layers.

For every horizon h in {5, 10, 60, 240, 1440} minutes:
  * point forecast + direction + intervals exactly as in A/B (shared engine), and
  * P(UP), P(DOWN), P(SIDEWAYS) where SIDEWAYS means |log return| <= tau_h,
    tau_h = max(0.25%, 0.25% * sqrt(h/60))  (pre-registered, see docs/PREREGISTRATION_G2.md),
  * P(price touches +x%) and P(touches -x%) at ANY time within the window for pre-registered x.
All probability layers are fit walk-forward on the strategy's own PAST out-of-sample
selected forecasts and evaluated on later months only.

Model choice: 5/240/1440 min use the same pre-registered rule as A/B (validation balanced
accuracy) with the minimum validation sample scaled to min(300, 0.9 x validation slots).
10/60 min reuse the A/B v1 selections (no new selection) so G2 adds probability layers only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time

import cloudpickle
import numpy as np
import pandas as pd

from cryptolab.backtest import stats as st
from cryptolab.backtest.engine import WalkForward, summarize
from cryptolab.backtest.research import (EXCLUDED_FROM_SELECTION, SELECTION_RULE, _jsonable, breakdowns, buy_and_hold,
                                         equity_curves, git_commit, multiple_testing, select, split_summaries)
from cryptolab.strategies.library import all_strategies
from cryptolab.util.timeutil import DAY, now_ms, parse_date_ms, to_iso

log = logging.getLogger(__name__)


def sideways_tau(h: int) -> float:
    return max(0.0025, 0.0025 * math.sqrt(h / 60))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Newton-IRLS logistic regression with intercept (column 0 of X must be ones)."""
    w = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):
        p = st.sigmoid(X @ w)
        g = X.T @ (p - y) + R @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + R + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


class ProbLayers:
    """3-class and touch probabilities from (mu, sigma) of a forecast."""

    def __init__(self, h: int, targets_pct: list[float]):
        self.h, self.tau = h, sideways_tau(h)
        self.targets = [x / 100 for x in targets_pct]
        self.uq = None
        self.w_up = self.w_dn = None
        self.n = 0

    def fit(self, mu, sig, y, ymax, ymin) -> "ProbLayers":
        mu, sig, y = np.asarray(mu, float), np.asarray(sig, float), np.asarray(y, float)
        ok = np.isfinite(mu) & np.isfinite(sig) & (sig > 0) & np.isfinite(y)
        u = (y[ok] - mu[ok]) / sig[ok]
        self.n = int(ok.sum())
        self.uq = np.quantile(u, np.linspace(0, 1, 201)) if len(u) >= 50 else None
        ymax, ymin = np.asarray(ymax, float), np.asarray(ymin, float)
        rows_u, rows_d, out_u, out_d = [], [], [], []
        ok2 = ok & np.isfinite(ymax) & np.isfinite(ymin)
        for x in self.targets:
            lx = np.log(x / sig[ok2])
            m = mu[ok2] / sig[ok2]
            rows_u.append(np.column_stack([np.ones(ok2.sum()), lx, m]))
            rows_d.append(np.column_stack([np.ones(ok2.sum()), lx, -m]))
            out_u.append((ymax[ok2] >= np.log(1 + x)).astype(float))
            out_d.append((ymin[ok2] <= np.log(1 - x)).astype(float))
        if ok2.sum() >= 50:
            self.w_up = fit_logistic(np.vstack(rows_u), np.concatenate(out_u))
            self.w_dn = fit_logistic(np.vstack(rows_d), np.concatenate(out_d))
        return self

    def _cdf(self, z):
        if self.uq is None:  # normal fallback (documented)
            from scipy.stats import norm
            return norm.cdf(z)
        grid = np.linspace(0, 1, len(self.uq))
        return np.interp(z, self.uq, grid, left=0.0, right=1.0)

    def three_class(self, mu, sig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu, sig = np.asarray(mu, float), np.asarray(sig, float)
        p_up = 1 - self._cdf((self.tau - mu) / sig)
        p_dn = self._cdf((-self.tau - mu) / sig)
        p_side = np.clip(1 - p_up - p_dn, 0, 1)
        return p_up, p_dn, p_side

    def touch(self, mu, sig, x: float, side: str) -> np.ndarray:
        w = self.w_up if side == "up" else self.w_dn
        mu, sig = np.asarray(mu, float), np.asarray(sig, float)
        if w is None:  # reflection-principle fallback for driftless diffusion
            from scipy.stats import norm
            return np.clip(2 * (1 - norm.cdf(np.log(1 + x) / sig)), 0, 1)
        m = mu / sig if side == "up" else -mu / sig
        return st.sigmoid(w[0] + w[1] * np.log(x / sig) + w[2] * m)

    def to_dict(self) -> dict:
        return {"h": self.h, "tau": self.tau, "targets": self.targets, "n": self.n,
                "uq": None if self.uq is None else [float(v) for v in self.uq],
                "w_up": None if self.w_up is None else [float(v) for v in self.w_up],
                "w_dn": None if self.w_dn is None else [float(v) for v in self.w_dn]}

    @classmethod
    def from_dict(cls, d: dict) -> "ProbLayers":
        p = cls(d["h"], [x * 100 for x in d["targets"]])
        p.tau, p.n = d["tau"], d["n"]
        p.uq = None if d["uq"] is None else np.array(d["uq"])
        p.w_up = None if d["w_up"] is None else np.array(d["w_up"])
        p.w_dn = None if d["w_dn"] is None else np.array(d["w_dn"])
        return p


def _three_class_outcome(y, tau):
    return np.where(y > tau, 0, np.where(y < -tau, 1, 2))  # 0=UP, 1=DOWN, 2=SIDEWAYS


def evaluate_layers(fc: pd.DataFrame, h: int, targets_pct: list[float], bt: dict) -> tuple[pd.DataFrame, dict]:
    """Walk-forward fit/evaluate of the probability layers on selected forecasts (by month)."""
    months = sorted(fc["month"].unique())
    embargo = bt["embargo_days"] * DAY
    tau = sideways_tau(h)
    out_rows = []
    for M in months[1:]:
        cutoff = parse_date_ms(M + "-01") - embargo - h * 60_000
        pool = fc[fc["t"] < cutoff]
        cur = fc[fc["month"] == M]
        if len(pool) < 50 or cur.empty:
            continue
        L = ProbLayers(h, targets_pct).fit(pool["mu"], pool["sig"], pool["y"], pool["ymax"], pool["ymin"])
        pu, pd_, ps = L.three_class(cur["mu"], cur["sig"])
        # baselines: climatology (pool class frequencies) and "no signal" (mu = 0, same sigma/F)
        oc_pool = _three_class_outcome(pool["y"].values, tau)
        clim = np.bincount(oc_pool, minlength=3) / len(oc_pool)
        nu, nd, ns = L.three_class(np.zeros(len(cur)), cur["sig"])
        r = cur[["t", "split", "month", "symbol", "y", "ymax", "ymin", "mu", "sig"]].copy()
        r["p_up3"], r["p_dn3"], r["p_side3"] = pu, pd_, ps
        r["n_up3"], r["n_dn3"], r["n_side3"] = nu, nd, ns
        r["c_up3"], r["c_dn3"], r["c_side3"] = clim[0], clim[1], clim[2]
        for x in L.targets:
            r[f"tu_{x}"] = L.touch(cur["mu"], cur["sig"], x, "up")
            r[f"td_{x}"] = L.touch(cur["mu"], cur["sig"], x, "dn")
            r[f"ctu_{x}"] = float((pool["ymax"] >= np.log(1 + x)).mean())
            r[f"ctd_{x}"] = float((pool["ymin"] <= np.log(1 - x)).mean())
        out_rows.append(r)
    ev = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    return ev, _layer_metrics(ev, h, targets_pct)


def _brier3(P, oc):
    O = np.eye(3)[oc]
    return float(((P - O) ** 2).sum(axis=1).mean())


def _rps(P, oc):
    # ordered classes DOWN < SIDEWAYS < UP
    order = [1, 2, 0]
    Pc = np.cumsum(P[:, order], axis=1)
    O = np.eye(3)[oc][:, order]
    Oc = np.cumsum(O, axis=1)
    return float((((Pc - Oc) ** 2)[:, :2]).sum(axis=1).mean() / 2)


def _layer_metrics(ev: pd.DataFrame, h: int, targets_pct: list[float]) -> dict:
    out = {}
    if ev.empty:
        return out
    tau = sideways_tau(h)
    for split in ("validation", "test"):
        e = ev[ev["split"] == split]
        if e.empty:
            continue
        oc = _three_class_outcome(e["y"].values, tau)
        P = e[["p_up3", "p_dn3", "p_side3"]].values
        N = e[["n_up3", "n_dn3", "n_side3"]].values
        C = e[["c_up3", "c_dn3", "c_side3"]].values
        m = {"n": int(len(e)), "sideways_tau_pct": tau * 100,
             "outcome_freq": {k: float((oc == i).mean()) for i, k in enumerate(["UP", "DOWN", "SIDEWAYS"])},
             "brier3_model": _brier3(P, oc), "brier3_no_signal": _brier3(N, oc), "brier3_climatology": _brier3(C, oc),
             "rps_model": _rps(P, oc), "rps_climatology": _rps(C, oc),
             "argmax_accuracy": float((P.argmax(axis=1) == oc).mean()),
             "reliability_up": st.reliability_bins(e["p_up3"].values, (oc == 0).astype(float), 5),
             "reliability_sideways": st.reliability_bins(e["p_side3"].values, (oc == 2).astype(float), 5),
             "touch": {}}
        for xp in targets_pct:
            x = xp / 100
            hu = (e["ymax"] >= np.log(1 + x)).astype(float).values
            hd = (e["ymin"] <= np.log(1 - x)).astype(float).values
            m["touch"][f"+{xp}%"] = {"freq": float(hu.mean()), "mean_p": float(e[f"tu_{x}"].mean()),
                                     "brier": float(((e[f"tu_{x}"] - hu) ** 2).mean()),
                                     "brier_climatology": float(((e[f"ctu_{x}"] - hu) ** 2).mean())}
            m["touch"][f"-{xp}%"] = {"freq": float(hd.mean()), "mean_p": float(e[f"td_{x}"].mean()),
                                     "brier": float(((e[f"td_{x}"] - hd) ** 2).mean()),
                                     "brier_climatology": float(((e[f"ctd_{x}"] - hd) ** 2).mean())}
        out[split] = m
    return out


def run_g2_horizon(cfg, store, exp_key: str) -> dict:
    exp = cfg["experiments"][exp_key]
    name, h = exp["name"], int(exp["horizon_min"])
    targets = list(exp["touch_targets_pct"])
    bt, costs = cfg["backtest"], cfg["costs"]
    research = cfg.path("research_dir")
    panel = pd.read_parquet(cfg.path("data_dir") / "cache" / f"panel_h{h}.parquet")
    slots = panel.groupby("t").agg(split=("split", "first")).reset_index()
    strategies = all_strategies(h, bt["seed"])
    fixed = None
    if exp.get("fixed_strategy_from"):
        src = store.rows("model_versions", "experiment=?", (exp["fixed_strategy_from"],))
        fixed = src[-1]["strategy"]
        keep = {fixed, f"mom_{h}", f"rev_{h}"} | EXCLUDED_FROM_SELECTION
        strategies = [s for s in strategies if s.name in keep]
    log.info("G2 %s: %d rows, %d strategies, fixed=%s", name, len(panel), len(strategies), fixed)
    wf = WalkForward(panel, strategies, h, bt, costs)
    months = sorted(panel["month"].unique())
    oos = months[2:]
    val_end = bt["validation_end"][:7]
    wf.run_months([m for m in oos if m <= val_end])
    sel, ics, _ = wf.results()
    val_sum = split_summaries(sel, ics, slots, "validation", strategies)
    n_val_slots = int((slots["split"] == "validation").sum())
    min_n = min(300, int(0.9 * n_val_slots))
    if fixed:
        chosen, top = fixed, [fixed]
        rule = f"Fixed: inherits the pre-registered v1 selection of {exp['fixed_strategy_from']} ({fixed}); G2 adds probability layers only."
        mt_val = {"n_strategies_tested": 0, "reality_check_accuracy": {"p_value": None}}
    else:
        mt_val = multiple_testing(sel, slots, "validation", val_sum, bt["seed"])
        chosen, top = select(val_sum, min_n=min_n)
        rule = SELECTION_RULE.replace(">= 300 forecasts", f">= {min_n} forecasts (min(300, 0.9 x validation slots))")
    prereg = {"prereg_id": f"{name}-selection-v1", "experiment": name, "horizon_min": h, "selection_rule": rule,
              "selected_strategy": chosen, "runner_ups": top[:10], "validation_metrics_selected": val_sum[chosen],
              "validation_reality_check_accuracy_p": mt_val.get("reality_check_accuracy", {}).get("p_value"),
              "n_strategies_tested": mt_val.get("n_strategies_tested"),
              "sideways_definition": f"|log return| <= {sideways_tau(h) * 100:.3f}%",
              "touch_targets_pct": targets,
              "test_hypotheses": ["H1: test accuracy > 50% (binomial, alpha 0.05)",
                                  "H2: test gross directional return > 0 (Newey-West, alpha 0.05)",
                                  "H3: test long-only net return > 0",
                                  "H4: test 3-class Brier score of the model < climatology Brier",
                                  "H5: test touch-probability Brier < climatology Brier for every target"],
              "code_commit": git_commit(), "created_utc": to_iso(now_ms())}
    store.record("preregistration", _jsonable(prereg))
    wf.run_months([m for m in oos if m > val_end])
    sel, ics, cal_log = wf.results()
    test_sum = split_summaries(sel, ics, slots, "test", strategies)
    ts = test_sum[chosen]
    ev, layer_m = evaluate_layers(sel[chosen], h, targets, bt)
    lt = layer_m.get("test", {})
    verdict = {
        "H1_accuracy_gt_50": bool(ts.get("p_acc_gt_50", 1) < 0.05),
        "H2_gross_dir_gt_0": bool(ts.get("p_gross_dir_gt_0", 1) < 0.05),
        "H3_long_net_gt_0": bool(ts.get("long_t", -9) > 1.645),
        "H4_three_class_beats_climatology": bool(lt and lt["brier3_model"] < lt["brier3_climatology"]),
        "H5_touch_beats_climatology": bool(lt and all(v["brier"] < v["brier_climatology"] for v in lt["touch"].values())),
    }
    strat = next(s for s in strategies if s.name == chosen)
    if strat.model is not None:
        strat.fit(panel, h)
    cal = wf.final_calibration(chosen)
    allfc = sel[chosen]
    layers = ProbLayers(h, targets).fit(allfc["mu"], allfc["sig"], allfc["y"], allfc["ymax"], allfc["ymin"])
    version = f"{name}-v1-{chosen}"
    blob = cloudpickle.dumps({"strategy": strat, "calibration": cal.to_dict(), "prob_layers": layers.to_dict(),
                              "horizon": h, "version": version})
    mdir = research / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{name}_v1.pkl").write_bytes(blob)
    spec = {"version": version, "experiment": name, "horizon_min": h, "strategy": chosen, "family": strat.family,
            "hypothesis": strat.hypothesis, "kind": strat.kind, "calibration": cal.to_dict(),
            "prob_layers_summary": {"tau": layers.tau, "targets": layers.targets, "n": layers.n},
            "trained_through": bt["test_end"], "model_file": f"{name}_v1.pkl",
            "model_sha256": hashlib.sha256(blob).hexdigest(), "selection_prereg": prereg["prereg_id"],
            "code_commit": git_commit(), "created_utc": to_iso(now_ms())}
    store.record("model_version", _jsonable(spec))
    keep_eq = [n for n in {chosen, f"mom_{h}", f"rev_{h}", "baseline_random"} if n in sel and len(sel[n])]
    result = _jsonable({
        "experiment": name, "horizon_min": h, "rows": int(len(panel)), "decision_times": int(panel["t"].nunique()),
        "coins_in_universe_union": int(panel["symbol"].nunique()),
        "splits": {k: {"from": to_iso(int(panel.loc[panel["split"] == k, "t"].min())),
                       "to": to_iso(int(panel.loc[panel["split"] == k, "t"].max())),
                       "decision_times": int((slots["split"] == k).sum())} for k in ("train", "validation", "test")},
        "n_strategies": len([s for s in strategies if not s.name.startswith("baseline_")]),
        "selected": chosen, "fixed_from": exp.get("fixed_strategy_from"), "prereg": prereg, "test_verdict": verdict,
        "summaries": {"validation": val_sum, "test": test_sum},
        "probability_layers": layer_m,
        "multiple_testing": {"validation": {k: v for k, v in mt_val.items() if not str(k).startswith("_")}},
        "buy_and_hold": {k: buy_and_hold(panel, k) for k in ("validation", "test")},
        "selected_breakdowns_val_test": breakdowns(allfc[allfc["split"].isin(["validation", "test"])]),
        "equity_curves": equity_curves(sel, keep_eq), "calibration_log_selected": cal_log[chosen], "live_model": spec,
    })
    (research / f"backtest_{name}.json").write_text(json.dumps(result, indent=1, default=str))
    store.record("preregistration", {"prereg_id": f"{name}-test-result-v1", "experiment": name,
                                     "selected_strategy": chosen, "test_metrics_selected": _jsonable(ts),
                                     "probability_layers_test": _jsonable(lt), "verdict": verdict,
                                     "created_utc": to_iso(now_ms())})
    return result


def run_g2(cfg, store, keys=("G1440", "G240", "G60", "G10", "G5")) -> dict:
    out = {}
    for k in keys:
        t0 = time.time()
        r = run_g2_horizon(cfg, store, k)
        out[k] = {"selected": r["selected"], "verdict": r["test_verdict"]}
        log.info("G2 %s done in %.0fs: %s", k, time.time() - t0, out[k])
    return out
