"""Feature engineering shared by the backtest and the live forecaster.

Time convention (no look-ahead):
  * Input: 1-minute klines keyed by `open_time`. A bar is only known once it has
    closed, so every series is re-indexed by bar CLOSE time t = open_time + 60s.
  * A feature value at time t uses only bars whose close time is <= t.
  * The target for horizon h is log(close[t+h] / close[t]): the last traded price at
    t+h versus at t -- the same definition the live verifier uses.
The same functions run on the live feed, so backtest and live features are identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MS = 60_000

MOM_WINDOWS = [1, 5, 10, 30, 60, 240, 1440]
VOL_WINDOWS = [30, 60, 240, 1440]
RANGE_WINDOWS = [60, 240, 1440]
BETA_WINDOW = 1440
# Minimum 1-minute history (bars) for every feature to be defined.
MIN_HISTORY_BARS = 1440 + 10

FEATURE_COLUMNS: list[str] = []  # populated at import (see _feature_names)


def to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Re-index 1m klines on a complete close-time grid; forward-fill price, zero volume."""
    if df.empty:
        return df
    d = df.copy()
    d["t"] = d["open_time"].astype("int64") + MIN_MS
    d = d.drop_duplicates("t").set_index("t").sort_index()
    full = np.arange(d.index[0], d.index[-1] + MIN_MS, MIN_MS, dtype="int64")
    missing = len(full) - len(d)
    d = d.reindex(full)
    d["missing"] = d["close"].isna().astype("int8")
    d["close"] = d["close"].ffill()
    for c in ("open", "high", "low"):
        d[c] = d[c].fillna(d["close"])
    for c in ("volume", "quote_volume", "trades", "taker_buy_quote"):
        if c in d:
            d[c] = d[c].fillna(0.0)
    d.attrs["missing_minutes"] = int(missing)
    return d


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(dn > 0, 100.0).where(up.notna())


def symbol_features(g: pd.DataFrame, ref: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Per-symbol time-series features on the 1-minute close-time grid."""
    c = g["close"]
    lc = np.log(c)
    lr = lc.diff()
    f = pd.DataFrame(index=g.index)
    f["close"] = c
    for L in MOM_WINDOWS:
        f[f"r_{L}"] = lc - lc.shift(L)
    for L in VOL_WINDOWS:
        f[f"vol_{L}"] = lr.rolling(L, min_periods=max(10, L // 2)).std()
    # Volatility forecast per minute (blend of horizons) used to scale targets/forecasts
    f["sig1m"] = np.sqrt(0.5 * f["vol_60"] ** 2 + 0.3 * f["vol_240"] ** 2 + 0.2 * f["vol_1440"] ** 2)
    f["vol_ratio"] = f["vol_60"] / f["vol_1440"]
    tr = pd.concat([g["high"] - g["low"], (g["high"] - c.shift()).abs(), (g["low"] - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_60"] = (tr / c).rolling(60, min_periods=30).mean()
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_60"] = _rsi(c, 60)
    ema_f, ema_s = c.ewm(span=60, adjust=False).mean(), c.ewm(span=130, adjust=False).mean()
    macd = (ema_f - ema_s) / c
    sig = macd.ewm(span=45, adjust=False).mean()
    f["macd"] = macd
    f["macd_hist"] = macd - sig
    for L in (60, 240):
        ma = c.rolling(L, min_periods=L // 2).mean()
        sd = c.rolling(L, min_periods=L // 2).std()
        f[f"bb_pctb_{L}"] = (c - (ma - 2 * sd)) / (4 * sd)
        f[f"bb_width_{L}"] = 4 * sd / ma
    for L in (30, 60, 240, 1440):
        f[f"dev_ma_{L}"] = lc - np.log(c.rolling(L, min_periods=L // 2).mean())
    ma = {L: c.rolling(L, min_periods=L // 2).mean() for L in (10, 30, 60, 240, 1440)}
    f["ma_x_10_60"] = np.log(ma[10] / ma[60])
    f["ma_x_30_240"] = np.log(ma[30] / ma[240])
    f["ma_x_60_1440"] = np.log(ma[60] / ma[1440])
    qv = g["quote_volume"]
    qv_day = qv.rolling(1440, min_periods=720).sum()
    for L in (10, 60):
        f[f"rel_vol_{L}"] = qv.rolling(L).sum() / (qv_day * L / 1440)
        tb = g["taker_buy_quote"].rolling(L).sum()
        f[f"tbr_{L}"] = (tb / qv.rolling(L).sum()).fillna(0.5) - 0.5
    tb30 = g["taker_buy_quote"].rolling(30).sum()
    f["tbr_30"] = (tb30 / qv.rolling(30).sum()).fillna(0.5) - 0.5
    tr_day = g["trades"].rolling(1440, min_periods=720).sum()
    f["trades_rel_10"] = g["trades"].rolling(10).sum() / (tr_day * 10 / 1440)
    f["qv_day_usd"] = qv_day
    for N in RANGE_WINDOWS:
        hh = g["high"].rolling(N, min_periods=N // 2).max()
        ll = g["low"].rolling(N, min_periods=N // 2).min()
        f[f"pos_{N}"] = (c - ll) / (hh - ll)
        prev_hh, prev_ll = hh.shift(1), ll.shift(1)
        up = np.log(c / prev_hh)
        dn = np.log(c / prev_ll)
        f[f"brk_{N}"] = np.where(up > 0, up, np.where(dn < 0, dn, 0.0))
    # Per-coin adaptive AR(1) coefficient of 10/60-min returns over the past 7 days
    for h in (10, 60):
        rh = lc - lc.shift(h)
        f[f"ac_{h}"] = rh.rolling(7 * 1440, min_periods=2 * 1440).corr(rh.shift(h))
    for hl in (60, 240):
        f[f"ewm_drift_{hl}"] = lr.ewm(halflife=hl, adjust=False, min_periods=hl).mean()
    # Cross-asset features (BTC/ETH reference series on the same grid)
    if ref:
        btc = ref.get("BTC")
        eth = ref.get("ETH")
        if btc is not None:
            b = btc.reindex(f.index)
            blr = np.log(b["close"]).diff()
            for L in (5, 10, 30, 60, 240):
                f[f"btc_r_{L}"] = b[f"r_{L}"] if f"r_{L}" in b else np.nan
            cov = lr.rolling(BETA_WINDOW, min_periods=720).cov(blr)
            var = blr.rolling(BETA_WINDOW, min_periods=720).var()
            f["beta_btc"] = cov / var
            f["corr_btc"] = lr.rolling(BETA_WINDOW, min_periods=720).corr(blr)
            for L in (10, 60, 240):
                f[f"resid_r_{L}"] = f[f"r_{L}"] - f["beta_btc"] * f[f"btc_r_{L}"]
            f["btc_vol_ratio"] = b["vol_60"] / b["vol_1440"]
            f["btc_trend"] = b["r_1440"]
            f["btc_vol_1440"] = b["vol_1440"]
        if eth is not None:
            e = eth.reindex(f.index)
            for L in (10, 60):
                f[f"eth_r_{L}"] = e[f"r_{L}"]
    hours = (f.index // 3_600_000) % 24
    f["hod_sin"] = np.sin(2 * np.pi * hours / 24)
    f["hod_cos"] = np.cos(2 * np.pi * hours / 24)
    f["hour"] = hours.astype("int16")
    f["missing_60"] = g["missing"].rolling(60, min_periods=1).sum() if "missing" in g else 0
    return f


def reference_series(df1m: pd.DataFrame) -> pd.DataFrame:
    """Minimal features for BTC/ETH used as cross-asset references."""
    g = to_grid(df1m)
    lc = np.log(g["close"])
    lr = lc.diff()
    r = pd.DataFrame(index=g.index)
    r["close"] = g["close"]
    for L in (5, 10, 30, 60, 240, 1440):
        r[f"r_{L}"] = lc - lc.shift(L)
    r["vol_60"] = lr.rolling(60, min_periods=30).std()
    r["vol_1440"] = lr.rolling(1440, min_periods=720).std()
    return r


def add_targets(f: pd.DataFrame, horizons=(10, 60), highs: pd.Series | None = None,
                lows: pd.Series | None = None) -> pd.DataFrame:
    """y_h = log(close[t+h]/close[t]); with highs/lows also the path extremes over (t, t+h]:
    ymax_h = log(max high / close[t]), ymin_h = log(min low / close[t])  (targets only, never features)."""
    lc = np.log(f["close"])
    for h in horizons:
        f[f"y_{h}"] = lc.shift(-h) - lc
        if highs is not None and lows is not None:
            f[f"ymax_{h}"] = np.log(highs.rolling(h, min_periods=h).max().shift(-h)) - lc
            f[f"ymin_{h}"] = np.log(lows.rolling(h, min_periods=h).min().shift(-h)) - lc
    return f


def cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Features computed across the universe at each decision time `t` (uses only time-t data)."""
    g = panel.groupby("t", sort=False)
    panel["n_universe"] = g["symbol"].transform("count")
    panel["breadth_60"] = g["r_60"].transform(lambda s: (s > 0).mean())
    panel["mkt_r_60"] = g["r_60"].transform("mean")
    panel["mkt_r_10"] = g["r_10"].transform("mean")
    panel["disp_60"] = g["r_60"].transform("std")
    for c in ("r_10", "r_60", "r_240", "rel_vol_60", "vol_60", "tbr_60", "resid_r_60"):
        if c in panel:
            panel[f"xs_rank_{c}"] = g[c].rank(pct=True) - 0.5
    panel["xs_rel_r_60"] = panel["r_60"] - panel["mkt_r_60"]
    panel["xs_rel_r_10"] = panel["r_10"] - panel["mkt_r_10"]
    return panel


# Explicit ALLOWLIST of model inputs. Every entry is computed only from data at or before
# the decision time. (A blacklist previously let the target column `y` through -- caught by
# the smoke test that showed 100% accuracy; see docs/METHODOLOGY.md "Leakage audit".)
MODEL_FEATURES = (
    [f"r_{L}" for L in MOM_WINDOWS] + [f"vol_{L}" for L in VOL_WINDOWS] +
    ["sig1m", "vol_ratio", "atr_60", "rsi_14", "rsi_60", "macd", "macd_hist",
     "bb_pctb_60", "bb_width_60", "bb_pctb_240", "bb_width_240",
     "dev_ma_30", "dev_ma_60", "dev_ma_240", "dev_ma_1440", "ma_x_10_60", "ma_x_30_240", "ma_x_60_1440",
     "rel_vol_10", "rel_vol_60", "tbr_10", "tbr_30", "tbr_60", "trades_rel_10"] +
    [f"pos_{N}" for N in RANGE_WINDOWS] + [f"brk_{N}" for N in RANGE_WINDOWS] +
    ["ac_10", "ac_60", "ewm_drift_60", "ewm_drift_240",
     "btc_r_5", "btc_r_10", "btc_r_30", "btc_r_60", "btc_r_240", "beta_btc", "corr_btc",
     "resid_r_10", "resid_r_60", "resid_r_240", "btc_vol_ratio", "btc_trend", "btc_vol_1440",
     "eth_r_10", "eth_r_60", "hod_sin", "hod_cos",
     "breadth_60", "mkt_r_60", "mkt_r_10", "disp_60",
     "xs_rank_r_10", "xs_rank_r_60", "xs_rank_r_240", "xs_rank_rel_vol_60", "xs_rank_vol_60", "xs_rank_tbr_60",
     "xs_rank_resid_r_60", "xs_rel_r_60", "xs_rel_r_10"]
)
FORBIDDEN = {"y", "y_std", "y_10", "y_60", "close", "t", "symbol", "split", "month", "ymax", "ymin"}


def model_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Allowlisted feature columns present in `panel`; raises if a target could leak in."""
    cols = [c for c in MODEL_FEATURES if c in panel.columns]
    leak = [c for c in cols if c in FORBIDDEN or c.startswith(("y_", "ymax", "ymin"))]
    if leak:
        raise RuntimeError(f"target leakage: {leak}")
    return cols
