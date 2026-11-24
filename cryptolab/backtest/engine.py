"""Walk-forward evaluation engine that simulates the live forecasting procedure.

For each horizon h and each month M (from the 3rd month on):
  1. training rows = decisions with t < start(M) - embargo - h   (strictly past data)
  2. model strategies are refit on the training rows; all strategies score month M
  3. calibration (b, P(direction correct), interval quantiles) is fit on the strategy's
     own PAST out-of-sample predictions (expanding window)
  4. at each decision time the coin with the largest |z| is selected and its frozen
     forecast is scored against the realized price h minutes later.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cryptolab.backtest import stats as st
from cryptolab.strategies.library import Strategy
from cryptolab.util.timeutil import DAY, parse_date_ms

log = logging.getLogger(__name__)

Q = (0.10, 0.25, 0.75, 0.90)


@dataclass
class Calibration:
    b: float = 0.0
    a_dir: float = 0.0     # P(direction correct) = sigmoid(a_dir + b_dir * |b z|)
    b_dir: float = 0.0
    q10: float = -1.28
    q25: float = -0.674
    q75: float = 0.674
    q90: float = 1.28
    n_pool: int = 0
    n_sel_pool: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def select_top(t: np.ndarray, z: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Index of the max-|z| row per decision time (ties -> most liquid). Abstain if max |z| == 0."""
    absz = np.abs(np.nan_to_num(z, nan=0.0))
    order = np.lexsort((rank, -absz, t))
    ts = t[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = ts[1:] != ts[:-1]
    idx = order[first]
    return idx[absz[idx] > 0]


def fit_calibration(z: np.ndarray, y_std: np.ndarray, t: np.ndarray, rank: np.ndarray,
                    sig: np.ndarray, y: np.ndarray) -> Calibration:
    ok = np.isfinite(z) & np.isfinite(y_std)
    z, y_std, t, rank, sig, y = z[ok], np.clip(y_std[ok], -10, 10), t[ok], rank[ok], sig[ok], y[ok]
    c = Calibration(n_pool=int(len(z)))
    den = float(np.sum(z * z))
    if den > 0:
        c.b = max(0.0, float(np.sum(z * y_std) / den))
    sel = select_top(t, z, rank)
    c.n_sel_pool = int(len(sel))
    if len(sel) >= 50:
        zs, ys, ss = z[sel], y[sel], sig[sel]
        x = np.abs(c.b * zs)
        correct = (np.sign(zs) * ys > 0).astype(float)
        c.a_dir, c.b_dir = st.fit_logistic_1d(x, correct)
        u = (ys - c.b * zs * ss) / ss
        u = u[np.isfinite(u)]
        if len(u) >= 50:
            c.q10, c.q25, c.q75, c.q90 = [float(v) for v in np.quantile(u, Q)]
    return c


def apply_forecast(rows: pd.DataFrame, z: np.ndarray, cal: Calibration, costs: dict, h: int) -> pd.DataFrame:
    """Turn selected rows into frozen forecasts + realized scoring (vectorized)."""
    sig = rows["sig_h"].values.astype("float64")
    y = rows["y"].values.astype("float64")
    d = np.sign(z)
    mu = cal.b * z * sig
    p_dir = st.sigmoid(cal.a_dir + cal.b_dir * np.abs(mu / sig))
    p_up = np.where(d > 0, p_dir, 1 - p_dir)
    out = pd.DataFrame({
        "t": rows["t"].values, "symbol": rows["symbol"].values, "rank": rows["universe_rank"].values,
        "split": rows["split"].values, "month": rows["month"].values, "z": z, "dir": d, "mu": mu, "sig": sig,
        "p_dir": p_dir, "p_up": p_up, "y": y,
        "lo50": mu + cal.q25 * sig, "hi50": mu + cal.q75 * sig, "lo80": mu + cal.q10 * sig, "hi80": mu + cal.q90 * sig,
        "btc_trend": rows["btc_trend"].values, "btc_vol_ratio": rows["btc_vol_ratio"].values,
    })
    out["flat"] = out["y"] == 0
    out["correct"] = (out["dir"] * out["y"] > 0)
    hs = np.where(out["rank"] <= 10, costs["backtest_half_spread_top10"], costs["backtest_half_spread_other"])
    slip = costs["slippage_per_side"]
    fee = costs["taker_fee"]["binance"]
    out["cost_rt"] = 2 * (fee + hs + slip)
    out["cost_rt_low"] = 2 * (costs["low_cost_fee"] + hs + slip)
    take_long = (out["dir"] > 0) & (out["mu"] > out["cost_rt"])
    out["long_taken"] = take_long
    out["net_long"] = np.where(take_long, out["y"] - out["cost_rt"], 0.0)
    take_long_low = (out["dir"] > 0) & (out["mu"] > out["cost_rt_low"])
    out["net_long_low"] = np.where(take_long_low, out["y"] - out["cost_rt_low"], 0.0)
    borrow = costs["short_borrow_cost_per_hour"] * h / 60
    out["gross_dir"] = out["dir"] * out["y"]
    out["net_ls"] = out["gross_dir"] - out["cost_rt"] - np.where(out["dir"] < 0, borrow, 0.0)
    out["net_ls_low"] = out["gross_dir"] - out["cost_rt_low"] - np.where(out["dir"] < 0, borrow, 0.0)
    out["in50"] = (out["y"] >= out["lo50"]) & (out["y"] <= out["hi50"])
    out["in80"] = (out["y"] >= out["lo80"]) & (out["y"] <= out["hi80"])
    out["brier"] = (out["p_up"] - (out["y"] > 0).astype(float)) ** 2
    out["abs_err"] = np.abs(out["mu"] - out["y"])      # log-return error ~ % price error
    out["abs_err_rw"] = np.abs(out["y"])               # no-change (random walk) forecast error
    return out


def ic_by_time(t: np.ndarray, z: np.ndarray, y: np.ndarray) -> pd.Series:
    """Cross-sectional Spearman rank correlation between score and realized return per time."""
    df = pd.DataFrame({"t": t, "z": z, "y": y}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    g = df.groupby("t")
    df["rz"] = g["z"].rank()
    df["ry"] = g["y"].rank()
    df["rzc"] = df["rz"] - df.groupby("t")["rz"].transform("mean")
    df["ryc"] = df["ry"] - df.groupby("t")["ry"].transform("mean")
    num = (df["rzc"] * df["ryc"]).groupby(df["t"]).sum()
    den = np.sqrt((df["rzc"] ** 2).groupby(df["t"]).sum() * (df["ryc"] ** 2).groupby(df["t"]).sum())
    ic = num / den.replace(0, np.nan)
    return ic.dropna()


class WalkForward:
    """Stateful walk-forward runner so the test months can be run only AFTER pre-registration."""

    def __init__(self, panel: pd.DataFrame, strategies: list[Strategy], h: int, bt: dict, costs: dict) -> None:
        self.panel, self.strategies, self.h, self.bt, self.costs = panel, strategies, h, bt, costs
        self.months = sorted(panel["month"].unique())
        self.embargo = bt["embargo_days"] * DAY
        self.t_all = panel["t"].values
        self.y_std = panel["y_std"].values.astype("float64")
        self.y_all = panel["y"].values.astype("float64")
        self.sig_all = panel["sig_h"].values.astype("float64")
        self.rank_all = panel["universe_rank"].values
        self.month_all = panel["month"].values
        self.rule_z = {s.name: s.predict(panel, h) for s in strategies if s.model is None}
        self.pool: dict[str, list] = {s.name: [] for s in strategies}
        self.selected: dict[str, list] = {s.name: [] for s in strategies}
        self.ics: dict[str, list] = {s.name: [] for s in strategies}
        self.cal_log: dict[str, list] = {s.name: [] for s in strategies}
        self.done_months: list[str] = []

    def run_months(self, months: list[str]) -> None:
        h, panel = self.h, self.panel
        for M in months:
            t0 = time.time()
            first = not self.done_months
            m_start = parse_date_ms(M + "-01")
            cutoff = m_start - self.embargo - h * 60_000
            train_mask = self.t_all < cutoff
            month_mask = self.month_all == M
            train = panel[train_mask]
            mrows = panel[month_mask]
            m_idx = np.where(month_mask)[0]
            for s in self.strategies:
                if s.model is not None:
                    s.fit(train, h)
                    z_m = s.predict(mrows, h)
                else:
                    z_m = self.rule_z[s.name][month_mask]
                # Calibration pool = this strategy's own past out-of-sample predictions. For the
                # first OOS month only, use the training rows (rules are parameter-free there;
                # model predictions are in-sample -- documented limitation, train split only).
                if first:
                    z_tr = self.rule_z[s.name][train_mask] if s.model is None else s.predict(train, h)
                    pool = [(np.where(train_mask)[0], z_tr)]
                else:
                    pool = self.pool[s.name]
                pidx = np.concatenate([p[0] for p in pool])
                pz = np.concatenate([p[1] for p in pool])
                usable = self.t_all[pidx] < cutoff
                cal = fit_calibration(pz[usable], self.y_std[pidx][usable], self.t_all[pidx][usable],
                                      self.rank_all[pidx][usable], self.sig_all[pidx][usable],
                                      self.y_all[pidx][usable])
                self.cal_log[s.name].append(dict(month=M, **cal.to_dict()))
                sel = select_top(self.t_all[m_idx], z_m, self.rank_all[m_idx])
                if len(sel):
                    fc = apply_forecast(panel.iloc[m_idx[sel]], z_m[sel], cal, self.costs, h)
                    fc["n_universe"] = panel["n_universe"].values[m_idx[sel]]
                    self.selected[s.name].append(fc)
                self.ics[s.name].append(ic_by_time(self.t_all[m_idx], z_m, self.y_all[m_idx]))
                if first and s.model is None:
                    self.pool[s.name].append(pool[0])  # rules keep parameter-free training history
                self.pool[s.name].append((m_idx, z_m))
            self.done_months.append(M)
            log.info("h=%d month %s done in %.0fs (train rows %d)", h, M, time.time() - t0, int(train_mask.sum()))

    def results(self) -> tuple[dict, dict, dict]:
        sel_out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in self.selected.items()}
        ic_out = {k: (pd.concat(v) if v else pd.Series(dtype=float)) for k, v in self.ics.items()}
        return sel_out, ic_out, self.cal_log

    def final_calibration(self, name: str) -> Calibration:
        """Calibration from ALL out-of-sample history (used for the live model)."""
        pool = self.pool[name]
        pidx = np.concatenate([p[0] for p in pool])
        pz = np.concatenate([p[1] for p in pool])
        return fit_calibration(pz, self.y_std[pidx], self.t_all[pidx], self.rank_all[pidx], self.sig_all[pidx],
                               self.y_all[pidx])


def decision_slots(panel: pd.DataFrame) -> pd.DataFrame:
    """All decision times per split (for coverage)."""
    return panel.groupby("t").agg(split=("split", "first")).reset_index()


def summarize(fc: pd.DataFrame, ic: pd.Series, n_slots: int) -> dict:
    """Metrics for one strategy on one split."""
    n = len(fc)
    out = {"n_slots": int(n_slots), "n_forecasts": int(n), "coverage": n / n_slots if n_slots else float("nan")}
    if n == 0:
        return out
    k = int(fc["correct"].sum())
    out["accuracy"] = k / n
    out["acc_ci95"] = st.wilson_ci(k, n)
    out["p_acc_gt_50"] = st.binom_p_greater(k, n)
    up, dn = fc["y"] > 0, fc["y"] < 0
    tpr = ((fc["dir"] > 0) & up).sum() / max(up.sum(), 1)
    tnr = ((fc["dir"] < 0) & dn).sum() / max(dn.sum(), 1)
    out["balanced_accuracy"] = float(0.5 * (tpr + tnr))
    out["frac_pred_up"] = float((fc["dir"] > 0).mean())
    out["majority_baseline_same_candidates"] = float(max(up.mean(), dn.mean()))
    out["n_flat"] = int(fc["flat"].sum())
    m, se, t = st.newey_west_t(fc["gross_dir"].values)
    out["gross_dir_mean_bps"], out["gross_dir_t"] = m * 1e4, t
    out["p_gross_dir_gt_0"] = st.p_from_t(t)
    out["brier"] = float(fc["brier"].mean())
    out["brier_skill_vs_0.5"] = 1 - out["brier"] / 0.25
    out["mean_p_dir"] = float(fc["p_dir"].mean())
    out["mae_pct"] = float(fc["abs_err"].mean() * 100)
    out["mae_rw_pct"] = float(fc["abs_err_rw"].mean() * 100)
    out["mae_ratio_vs_no_change"] = out["mae_pct"] / out["mae_rw_pct"] if out["mae_rw_pct"] > 0 else float("nan")
    out["cover50"] = float(fc["in50"].mean())
    out["cover80"] = float(fc["in80"].mean())
    for col, key in (("net_long", "long"), ("net_long_low", "long_lowcost"), ("net_ls", "ls"), ("net_ls_low", "ls_lowcost")):
        v = fc[col].values
        cum = np.cumsum(v)
        m, se, t = st.newey_west_t(v)
        out[f"{key}_mean_bps"] = m * 1e4
        out[f"{key}_t"] = t
        out[f"{key}_total_pct"] = float(cum[-1] * 100)
        out[f"{key}_maxdd_pct"] = st.max_drawdown(cum) * 100
    out["long_trades"] = int(fc["long_taken"].sum())
    lt = fc[fc["long_taken"]]
    out["long_win_rate"] = float((lt["net_long"] > 0).mean()) if len(lt) else float("nan")
    w, l = st.streaks(fc["correct"].values)
    out["max_win_streak"], out["max_loss_streak"] = w, l
    if len(ic):
        out["ic_mean"] = float(ic.mean())
        out["ic_t"] = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 2 and ic.std() > 0 else float("nan")
        out["ic_frac_pos"] = float((ic > 0).mean())
    return out
