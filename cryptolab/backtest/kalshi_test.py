"""Controlled test: does Kalshi's 15-minute BTC/ETH market improve our direction forecasts?

Design (fixed in code before any result was computed; see docs/PREREGISTRATION_KALSHI.md):
  decision time t = window open + 2 min (Kalshi KXBTC15M / KXETH15M windows, UTC quarter hours)
  target: Binance spot (BTCUSDT / ETHUSDT) last trade at window close vs at t   (UP if higher)
  Stage 1 (shared base model): L2 logistic regression on Binance-only features, trained ONLY on
          windows before the first Kalshi window (no overlap with any Kalshi data).
  Stage 2 (on Kalshi windows; first 60% chronologically = train, last 40% = untouched test):
      WITHOUT: logistic( base score, move since window open )
      WITH:    logistic( base score, move since window open, logit(Kalshi YES mid at t) )
  Kalshi quote usable only if two-sided, 0 < bid < ask < 1 and spread <= 0.10; windows without a
  usable quote are excluded from BOTH arms (paired comparison).
  Test statistics: accuracy (McNemar exact test), Brier and log loss (paired bootstrap 95% CI of
  the difference), gross directional return per window (paired t-test). Kalshi is adopted only if
  WITH beats WITHOUT on the pre-registered primary metric (Brier, CI excluding zero).
Kalshi YES prices are prediction-market contract prices for a BRTI-settled event, not spot
prices; they enter only as a probability feature.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math

import cloudpickle
import numpy as np
import pandas as pd
from scipy import stats as sps

from cryptolab.backtest import stats as st
from cryptolab.backtest.g2 import fit_logistic
from cryptolab.backtest.research import _jsonable, git_commit
from cryptolab.features import MODEL_FEATURES, reference_series, symbol_features, to_grid
from cryptolab.util.timeutil import MIN, now_ms, to_iso

log = logging.getLogger(__name__)
ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
BASE_FEATURES = [c for c in MODEL_FEATURES if not c.startswith(("xs_", "breadth", "mkt_", "disp"))]
DECISION_OFFSET = 2 * MIN
WINDOW = 15 * MIN


class BaseModel:
    """Standardize + L2 logistic regression; produces a logit score."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.med = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.sc = np.where((q3 - q1) > 1e-12, (q3 - q1) / 1.349, 1.0)
        Z = self._z(X)
        self.w = fit_logistic(np.column_stack([np.ones(len(Z)), Z]), y, l2=50.0)
        return self

    def _z(self, X):
        Z = (X - self.med) / self.sc
        return np.clip(np.where(np.isfinite(Z), Z, 0.0), -8, 8)

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), self._z(X)]) @ self.w


def logit(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


def asset_frame(raw_1m: pd.DataFrame, btc_ref: pd.DataFrame, eth_ref: pd.DataFrame) -> pd.DataFrame:
    g = to_grid(raw_1m)
    f = symbol_features(g, {"BTC": btc_ref, "ETH": eth_ref})
    lc = np.log(f["close"])
    t = f.index.values
    dec = (t - DECISION_OFFSET) % WINDOW == 0
    out = f.loc[dec, BASE_FEATURES + ["close", "sig1m"]].copy()
    tt = out.index.values
    close = f["close"]
    out["open_close"] = close.reindex(tt - DECISION_OFFSET).values         # price at window open
    out["end_close"] = close.reindex(tt - DECISION_OFFSET + WINDOW).values  # price at window close (target only)
    out["z_open"] = np.log(out["close"] / out["open_close"]) / (out["sig1m"] * math.sqrt(2))
    out["y_up"] = (out["end_close"] > out["close"]).astype(float)
    out["y_ret"] = np.log(out["end_close"] / out["close"])
    out["window_open_ms"] = tt - DECISION_OFFSET
    return out.dropna(subset=["open_close", "end_close", "sig1m"])


def kalshi_quotes(path) -> pd.DataFrame:
    k = pd.read_parquet(path)
    k = k[k["minute_end_ms"] == k["open_ms"] + DECISION_OFFSET].copy()
    ok = (k["yes_bid"] > 0) & (k["yes_ask"] < 1) & (k["yes_ask"] > k["yes_bid"]) & ((k["yes_ask"] - k["yes_bid"]) <= 0.10)
    k["usable"] = ok
    k["p_kalshi"] = (k["yes_bid"] + k["yes_ask"]) / 2
    return k[["open_ms", "close_ms", "ticker", "p_kalshi", "usable", "yes_bid", "yes_ask", "result"]].rename(
        columns={"open_ms": "window_open_ms"})


def _metrics(p, y, ret):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pred = p > 0.5
    return {"n": int(len(y)), "accuracy": float((pred == (y > 0.5)).mean()),
            "brier": float(((p - y) ** 2).mean()), "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "gross_dir_bps": float((np.where(pred, 1, -1) * ret).mean() * 1e4),
            "reliability": st.reliability_bins(p, y, 5)}


def run(cfg, store, bn) -> dict:
    raw = cfg.path("raw_dir")
    research = cfg.path("research_dir")
    frames = {}
    for a, sym in ASSETS.items():
        d = pd.read_parquet(raw / "binance_1m" / f"{sym}.parquet")
        start = int(d["open_time"].max()) + MIN
        extra = bn.klines_range(sym, "1m", start, bn.server_time() - MIN)  # bring history up to now
        frames[a] = pd.concat([d, extra[d.columns]], ignore_index=True) if len(extra) else d
    btc_ref, eth_ref = reference_series(frames["BTC"]), reference_series(frames["ETH"])
    data = []
    for a in ASSETS:
        f = asset_frame(frames[a], btc_ref, eth_ref)
        f["asset"] = a
        q = kalshi_quotes(raw / "kalshi" / f"{a}_15m.parquet")
        f = f.merge(q, on="window_open_ms", how="left")
        data.append(f)
    df = pd.concat(data, ignore_index=True).sort_values(["window_open_ms", "asset"]).reset_index(drop=True)
    k0 = int(df.loc[df["p_kalshi"].notna(), "window_open_ms"].min())
    kal = df[df["usable"].fillna(False).astype(bool)].copy()
    cut_idx = int(len(kal["window_open_ms"].unique()) * 0.6)
    split_ms = int(np.sort(kal["window_open_ms"].unique())[cut_idx])
    prereg = {"prereg_id": "kalshi-15m-with-vs-without-v1", "experiment": "kalshi_ab",
              "design": __doc__.strip()[:1500], "base_model_train_until": to_iso(k0 - 86_400_000),
              "kalshi_first_window": to_iso(k0), "stage2_train": [to_iso(k0), to_iso(split_ms)],
              "test_from": to_iso(split_ms), "primary_metric": "Brier score difference (WITH - WITHOUT) < 0 with 95% CI < 0",
              "n_windows_with_usable_quote": int(len(kal)), "code_commit": git_commit(), "created_utc": to_iso(now_ms())}
    store.record("preregistration", _jsonable(prereg))
    # ---- stage 1: base model strictly before any Kalshi data
    base_tr = df[df["window_open_ms"] < k0 - 86_400_000].dropna(subset=BASE_FEATURES[:5])
    base = BaseModel().fit(base_tr[BASE_FEATURES].values.astype(float), base_tr["y_up"].values)
    kal["base"] = base.score(kal[BASE_FEATURES].values.astype(float))
    kal["lk"] = logit(kal["p_kalshi"])
    kal["zo"] = kal["z_open"].clip(-10, 10)
    tr, te = kal[kal["window_open_ms"] < split_ms], kal[kal["window_open_ms"] >= split_ms]

    def design(d, with_k, kalshi_only=False):
        cols = [np.ones(len(d))]
        if not kalshi_only:
            cols += [d["base"].values, d["zo"].values]
        if with_k or kalshi_only:
            cols.append(d["lk"].values)
        return np.column_stack(cols)

    arms = {"without": (False, False), "with": (True, False), "kalshi_only": (False, True)}
    W, P = {}, {}
    for name, (wk, ko) in arms.items():
        W[name] = fit_logistic(design(tr, wk, ko), tr["y_up"].values, l2=1.0)
        P[name] = st.sigmoid(design(te, wk, ko) @ W[name])
    y, ret = te["y_up"].values, te["y_ret"].values
    res = {name: _metrics(P[name], y, ret) for name in arms}
    # paired comparisons WITH vs WITHOUT
    pw, po = P["with"], P["without"]
    cw, co = (pw > 0.5) == (y > 0.5), (po > 0.5) == (y > 0.5)
    b, c = int((cw & ~co).sum()), int((~cw & co).sum())
    mcnemar_p = float(sps.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
    rng = np.random.default_rng(cfg["backtest"]["seed"])
    db = (pw - y) ** 2 - (po - y) ** 2
    dl = -(y * np.log(np.clip(pw, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pw, 1e-6, 1))) + \
        (y * np.log(np.clip(po, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - po, 1e-6, 1)))
    boots_b, boots_l = [], []
    n = len(y)
    for _ in range(2000):
        i = rng.integers(0, n, n)
        boots_b.append(db[i].mean())
        boots_l.append(dl[i].mean())
    gw, go = np.where(pw > 0.5, 1, -1) * ret, np.where(po > 0.5, 1, -1) * ret
    tt = sps.ttest_rel(gw, go) if n > 2 else None
    comparison = {"n_test_windows": n, "mcnemar_with_better": b, "mcnemar_without_better": c, "mcnemar_p": mcnemar_p,
                  "brier_diff_with_minus_without": float(db.mean()),
                  "brier_diff_ci95": [float(np.quantile(boots_b, 0.025)), float(np.quantile(boots_b, 0.975))],
                  "logloss_diff_with_minus_without": float(dl.mean()),
                  "logloss_diff_ci95": [float(np.quantile(boots_l, 0.025)), float(np.quantile(boots_l, 0.975))],
                  "gross_dir_diff_bps": float((gw - go).mean() * 1e4),
                  "gross_dir_diff_p": float(tt.pvalue) if tt is not None else None,
                  "kalshi_coefficient": float(W["with"][-1])}
    comparison["adopt_kalshi"] = bool(comparison["brier_diff_ci95"][1] < 0)
    costs = cfg["costs"]
    rt = 2 * (costs["taker_fee"]["binance"] + costs["backtest_half_spread_top10"] + costs["slippage_per_side"])
    for name in arms:  # spot long-only paper rule: long if P(up) - 0.5 implies an edge larger than costs
        p = P[name]
        exp_move = (2 * p - 1) * np.abs(ret).mean()   # crude expected move from probability (documented)
        take = (p > 0.5) & (exp_move > rt)
        res[name]["long_trades"] = int(take.sum())
        res[name]["long_net_pct_total"] = float(((ret - rt) * take).sum() * 100)
    out = {"prereg": prereg, "test_metrics": res, "comparison": comparison,
           "coefficients": {k: [float(v) for v in W[k]] for k in W},
           "n_train_windows": int(len(tr)), "n_test_windows": int(len(te)),
           "excluded_unusable_quote": int(df["p_kalshi"].notna().sum() - len(kal)),
           "test_from": to_iso(split_ms), "test_to": to_iso(int(te["window_open_ms"].max()) + WINDOW) if len(te) else None}
    blob = cloudpickle.dumps({"base": base, "features": BASE_FEATURES, "weights": W, "version": "KX-v1"})
    (research / "models").mkdir(parents=True, exist_ok=True)
    (research / "models" / "KX_v1.pkl").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    for a in ASSETS:
        for arm in ("without", "with"):
            store.record("model_version", {"version": f"KX_{a}_{arm}-v1", "experiment": f"KX_{a}_{arm}",
                                           "horizon_min": 13, "strategy": f"kalshi_ab_{arm}", "model_file": "KX_v1.pkl",
                                           "model_sha256": sha, "selection_prereg": prereg["prereg_id"],
                                           "created_utc": to_iso(now_ms())})
    (research / "kalshi_ab_test.json").write_text(json.dumps(_jsonable(out), indent=1, default=str))
    store.record("preregistration", _jsonable({"prereg_id": "kalshi-15m-with-vs-without-results-v1",
                                               "experiment": "kalshi_ab", "comparison": comparison,
                                               "test_metrics": {k: {m: v for m, v in r.items() if m != "reliability"}
                                                                for k, r in res.items()},
                                               "created_utc": to_iso(now_ms())}))
    return out
