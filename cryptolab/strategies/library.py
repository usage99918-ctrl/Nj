"""Strategy library.

Every strategy maps the panel (rows = coin x decision time) to a dimensionless score
z (positive = bullish). The common pipeline then
  * predicts expected log return  mu = b * z * sigma_h  (b fit walk-forward, b >= 0),
  * picks, at each decision time, the coin with the largest |z| (strongest evidence),
  * forecasts direction sign(z), a calibrated P(direction correct) and 50%/80%
    prediction intervals.
Rule strategies have no fitted parameters besides b. Model strategies are refit
walk-forward each month on data strictly before the month (minus an embargo).

Each strategy is a distinct, stated hypothesis -- not a grid of near-duplicates.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _sd(p: pd.DataFrame, L: int) -> pd.Series:
    """Expected std of an L-minute log return for each row (from the blended 1m vol)."""
    return p["sig1m"].astype("float64") * np.sqrt(L)


def _clipz(z) -> np.ndarray:
    z = np.asarray(z, dtype="float64")
    z = np.where(np.isfinite(z), z, 0.0)
    return np.clip(z, -10, 10)


@dataclass
class Strategy:
    name: str
    family: str
    hypothesis: str
    score: Optional[Callable[[pd.DataFrame], np.ndarray]] = None  # rule: panel -> z
    model: Optional[object] = None                                 # model strategies
    kind: str = "rule"           # rule | model | baseline
    baseline: bool = False
    params: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame, h: int) -> None:
        if self.model is not None:
            self.model.fit(train, h)

    def predict(self, df: pd.DataFrame, h: int) -> np.ndarray:
        if self.model is not None:
            return _clipz(self.model.predict(df, h))
        return _clipz(self.score(df))


# =========================================================================== rules
def rules(h: int) -> list[Strategy]:
    S: list[Strategy] = []

    def add(name, family, hyp, fn, baseline=False):
        S.append(Strategy(name, family, hyp, score=fn, baseline=baseline))

    # 1. Momentum / trend following ------------------------------------------------
    for L in (5, 10, 30, 60, 240, 1440):
        add(f"mom_{L}", "momentum", f"Past {L}-min return continues (time-series momentum).",
            lambda p, L=L: p[f"r_{L}"] / _sd(p, L), baseline=(L == h))
    for fast, slow in ((10, 60), (30, 240), (60, 1440)):
        add(f"ma_cross_{fast}_{slow}", "momentum", f"MA({fast}) above MA({slow}) signals uptrend.",
            lambda p, f=fast, s=slow: p[f"ma_x_{f}_{s}"] / _sd(p, s))
    add("macd_hist", "momentum", "MACD histogram (5m-equivalent 12/26/9) sign persists.",
        lambda p: p["macd_hist"] / _sd(p, 60))
    for hl in (60, 240):
        add(f"ewm_drift_{hl}", "momentum", f"EWMA drift (half-life {hl}m) persists.",
            lambda p, hl=hl: p[f"ewm_drift_{hl}"] / p["sig1m"] * np.sqrt(hl))
    add("trend_vote", "momentum", "Agreement of 10/60/240/1440-min trend signs.",
        lambda p: (np.sign(p["r_10"]) + np.sign(p["r_60"]) + np.sign(p["r_240"]) + np.sign(p["r_1440"])) / 2)

    # 2. Mean reversion ------------------------------------------------------------
    for L in (1, 5, 10, 30, 60, 240):
        add(f"rev_{L}", "mean_reversion", f"Past {L}-min return reverses (liquidity provision).",
            lambda p, L=L: -p[f"r_{L}"] / _sd(p, L), baseline=(L == h))
    for L in (30, 60, 240, 1440):
        add(f"rev_ma_{L}", "mean_reversion", f"Price reverts toward its {L}-min moving average.",
            lambda p, L=L: -p[f"dev_ma_{L}"] / _sd(p, max(L // 2, 1)))
    for n in (14, 60):
        add(f"rsi_rev_{n}", "mean_reversion", f"RSI({n}) overbought/oversold reverts.",
            lambda p, n=n: (50 - p[f"rsi_{n}"]) / 10)
    add("rsi_extreme_14", "mean_reversion", "Only extreme RSI(14) (<25 or >75) reverts.",
        lambda p: np.where((p["rsi_14"] < 25) | (p["rsi_14"] > 75), (50 - p["rsi_14"]) / 10, 0.0))
    for L in (60, 240):
        add(f"bb_rev_{L}", "mean_reversion", f"Bollinger %B({L}) outside bands reverts.",
            lambda p, L=L: -(p[f"bb_pctb_{L}"] - 0.5) * 4)

    # 3. Breakout / support-resistance ---------------------------------------------
    for N in (60, 240, 1440):
        add(f"brk_follow_{N}", "breakout", f"Break of the {N}-min high/low continues.",
            lambda p, N=N: p[f"brk_{N}"] / p["sig1m"])
        add(f"brk_fade_{N}", "breakout", f"Break of the {N}-min high/low fails (false breakout).",
            lambda p, N=N: -p[f"brk_{N}"] / p["sig1m"])
    for N in (240, 1440):
        add(f"range_follow_{N}", "breakout", f"Price near the top of its {N}-min range keeps rising.",
            lambda p, N=N: (p[f"pos_{N}"] - 0.5) * 4)
        add(f"range_fade_{N}", "breakout", f"{N}-min range high/low act as resistance/support.",
            lambda p, N=N: -(p[f"pos_{N}"] - 0.5) * 4)

    # 4. Relative volume -------------------------------------------------------------
    for L in (10, 60):
        add(f"rv_mom_{L}", "rel_volume", f"{L}-min moves on high relative volume continue.",
            lambda p, L=L: p[f"r_{L}"] / _sd(p, L) * np.log1p(p[f"rel_vol_{L}"].clip(0, 50)))
        add(f"rv_rev_{L}", "rel_volume", f"{L}-min moves on high relative volume overshoot and revert.",
            lambda p, L=L: -p[f"r_{L}"] / _sd(p, L) * np.log1p(p[f"rel_vol_{L}"].clip(0, 50)))
    add("vol_spike_rev", "rel_volume", "Volume spikes (rel. volume >3x) mark exhaustion: fade the 10-min move.",
        lambda p: np.where(p["rel_vol_10"] > 3, -p["r_10"] / _sd(p, 10), 0.0))
    add("quiet_drift", "rel_volume", "Low-volume 60-min drifts persist (no informed opposition).",
        lambda p: np.where(p["rel_vol_60"] < 0.7, p["r_60"] / _sd(p, 60), 0.0))

    # 5. Order-flow imbalance (taker-buy ratio = historical proxy for book imbalance) --
    for L in (10, 30, 60):
        add(f"ofi_follow_{L}", "order_flow", f"Aggressive buying over {L} min predicts further rise.",
            lambda p, L=L: p[f"tbr_{L}"] / 0.05)
        add(f"ofi_fade_{L}", "order_flow", f"Aggressive buying over {L} min exhausts and reverses.",
            lambda p, L=L: -p[f"tbr_{L}"] / 0.05)
    add("ofi_absorption", "order_flow", "Price falling despite taker buying = absorption (bullish), and vice versa.",
        lambda p: p["tbr_30"] / 0.05 - p["r_30"] / _sd(p, 30))

    # 6. Volatility-based ------------------------------------------------------------
    add("volreg_switch", "volatility", "High short/long vol ratio -> reversal; calm -> momentum.",
        lambda p: np.where(p["vol_ratio"] > 1.3, -p["r_10"] / _sd(p, 10), p["r_60"] / _sd(p, 60)))
    add("squeeze_breakout", "volatility", "Breakouts after volatility compression continue.",
        lambda p: np.where(p["vol_ratio"] < 0.7, p["brk_60"] / p["sig1m"], 0.0))
    add("low_vol_anomaly", "volatility", "Cross-sectionally low-volatility coins outperform.",
        lambda p: -p["xs_rank_vol_60"] * 4)
    add("hv_reversal", "volatility", "Reversal strength scales with volatility expansion.",
        lambda p: -p["r_10"] / _sd(p, 10) * p["vol_ratio"].clip(0, 5))

    # 7. Statistical time-series -----------------------------------------------------
    add("ar1_adaptive", "time_series", f"Per-coin AR(1) of {h}-min returns estimated over the past 7 days.",
        lambda p: p[f"ac_{h}"] * p[f"r_{h}"] / _sd(p, h) * 5)

    # 12. Regime-gated rules ---------------------------------------------------------
    add("regime_gated_mom", "regime", "Momentum only works when BTC's 24h trend is up.",
        lambda p: np.where(p["btc_trend"] > 0, p["r_60"] / _sd(p, 60), 0.0))
    add("regime_gated_rev", "regime", "Short-term reversal only in high-volatility BTC regimes.",
        lambda p: np.where(p["btc_vol_ratio"] > 1.2, -p["r_10"] / _sd(p, 10), 0.0))
    add("breadth_regime", "regime", "One-sided markets (breadth>70% or <30%) trend; mixed markets revert.",
        lambda p: np.where((p["breadth_60"] > 0.7) | (p["breadth_60"] < 0.3),
                           np.sign(p["mkt_r_60"]) * p["beta_btc"].clip(0, 3), -p["r_10"] / _sd(p, 10)))

    # 13. Cross-asset / lead-lag -----------------------------------------------------
    for L in (5, 10, 30, 60):
        add(f"btc_lead_{L}", "cross_asset", f"Alts catch up to BTC's {L}-min move (beta-adjusted lead-lag).",
            lambda p, L=L: (p["beta_btc"] * p[f"btc_r_{L}"] - p[f"r_{L}"]) / _sd(p, L))
    for L in (10, 60):
        add(f"eth_lead_{L}", "cross_asset", f"Coins catch up to ETH's {L}-min move.",
            lambda p, L=L: (p[f"eth_r_{L}"] - p[f"r_{L}"]) / _sd(p, L))
    for L in (10, 60, 240):
        add(f"resid_rev_{L}", "cross_asset", f"Idiosyncratic (BTC-beta residual) {L}-min move reverts.",
            lambda p, L=L: -p[f"resid_r_{L}"] / _sd(p, L))
    add("resid_mom_240", "cross_asset", "Idiosyncratic 4h outperformance persists (news diffusion).",
        lambda p: p["resid_r_240"] / _sd(p, 240))
    add("btc_beta_mom", "cross_asset", "BTC's 60-min trend transmits to high-beta coins.",
        lambda p: p["beta_btc"].clip(0, 3) * p["btc_r_60"] / _sd(p, 60))
    add("xs_rel_rev_60", "cross_asset", "Coins that out/under-performed the universe over 60 min revert.",
        lambda p: -p["xs_rel_r_60"] / _sd(p, 60))
    add("xs_rel_mom_60", "cross_asset", "Cross-sectional 60-min relative strength persists.",
        lambda p: p["xs_rel_r_60"] / _sd(p, 60))

    # 14. Original experimental hypotheses ------------------------------------------
    add("liq_shock_rev", "experimental",
        "Forced-liquidation overshoot: >3-sigma 10-min move on >2x volume with one-sided taker flow reverts.",
        lambda p: np.where((np.abs(p["r_10"] / _sd(p, 10)) > 3) & (p["rel_vol_10"] > 2)
                           & (np.sign(p["tbr_10"]) == np.sign(p["r_10"])), -p["r_10"] / _sd(p, 10), 0.0))
    add("exhaustion", "experimental", "A 4h trend with fading volume (rel. vol <0.8) is exhausted and reverses.",
        lambda p: np.where(p["rel_vol_60"] < 0.8, -p["r_240"] / _sd(p, 240), 0.0))
    add("dispersion_rev", "experimental", "Reversal is stronger when cross-sectional dispersion is high.",
        lambda p: -p["r_10"] / _sd(p, 10) * (p["disp_60"] / p["disp_60"].median()).clip(0, 4))
    add("breadth_divergence", "experimental", "A coin moving against market breadth reverts toward the crowd.",
        lambda p: np.where(((p["breadth_60"] < 0.3) & (p["r_60"] > 0)) | ((p["breadth_60"] > 0.7) & (p["r_60"] < 0)),
                           -p["r_60"] / _sd(p, 60), 0.0))
    add("calm_drift", "experimental", "In compressed volatility, the EWMA drift persists.",
        lambda p: np.where(p["vol_ratio"] < 0.8, p["ewm_drift_240"] / p["sig1m"] * np.sqrt(240), 0.0))
    add("us_session_mom", "experimental", "Momentum persists only in the high-activity 13-21 UTC session.",
        lambda p: np.where((p["hour"] >= 13) & (p["hour"] < 21), p["r_60"] / _sd(p, 60), 0.0))
    return S


# ======================================================================= baselines
class _RandomZ:
    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, train, h):
        pass

    def predict(self, df, h):
        rng = np.random.default_rng(self.seed + int(df["t"].iloc[0] // 60_000) % 100_000 if len(df) else 0)
        return rng.standard_normal(len(df))


class _ConstZ:
    def __init__(self, v: float):
        self.v = v

    def fit(self, train, h):
        pass

    def predict(self, df, h):
        return np.full(len(df), self.v)


def baselines(seed: int) -> list[Strategy]:
    return [
        Strategy("baseline_random", "baseline", "Random direction, random coin.", model=_RandomZ(seed),
                 kind="baseline", baseline=True),
        Strategy("baseline_always_up", "baseline", "Always predict UP (most liquid coin).", model=_ConstZ(1.0),
                 kind="baseline", baseline=True),
        Strategy("baseline_always_down", "baseline", "Always predict DOWN (most liquid coin).", model=_ConstZ(-1.0),
                 kind="baseline", baseline=True),
        Strategy("baseline_no_change", "baseline", "Price unchanged (random walk); direction undefined.",
                 model=_ConstZ(0.0), kind="baseline", baseline=True),
    ]


def all_strategies(h: int, seed: int) -> list[Strategy]:
    from cryptolab.strategies.models import model_strategies
    rs = rules(h)
    return rs + model_strategies(h, seed, rs) + baselines(seed)
