"""The three externally proposed ("friend's") strategies, defined BEFORE any testing.

All thresholds below are fixed a priori (docs/PREREGISTRATION_STRATEGIES.md); nothing is tuned.
Each detector takes 1-minute bars on the close-time grid (see features.to_grid) and returns
signal events using only data at or before the signal bar's close.

S1 Relative-Strength Reset (ETH vs BTC, traded on the ETH/BTC pair)
   divergence D = 24h change of log(ETH/BTC); z = D / rolling 30-day std of D
   "unusually large": |z| >= 2; "beginning to reverse": the 60-min change of log(ETH/BTC) has
   the opposite sign of D and |change| >= 0.25 x its typical 60-min size.
   Signal: expect ETH/BTC to move back (LONG ETHBTC after ETH underperformed; SHORT after it
   outperformed -- a short is theoretical on spot). Checked every 5 min; 4h cooldown.
S2 Breakout Retest (any universe coin, 5-minute bars)
   compression: 4h realized vol < 0.6 x 7-day realized vol on the bar before the breakout;
   breakout: 5-min close > highest high of the prior 4h (48 bars) -> level L;
   retest within 2h (24 bars): a bar's low <= L*(1+0.15%) and close >= L*(1-0.10%), with no
   close < L*(1-0.30%) since the breakout (failed breakout), and that bar's volume >= 1.2 x the
   average volume of the 48 bars before the breakout. Signal LONG at that bar's close.
   Mirror image (breakdown, retest from below) gives a theoretical SHORT.
   Paper exits: stop L*(1-0.5%), take-profit 2x risk, max hold 4h.
S3 Liquidity Recovery (any universe coin)
   sharp drop: 30-min return <= min(-2.5%, -3 x expected 30-min sigma);
   recovery within 60 min: price retraced >= 30% of the drop from its low, taker-buy share of
   the last 10 minutes >= 55%, and spread back to <= 1.2 x its pre-drop level.
   Historical version (S3-proxy): spread ESTIMATED from 1-minute highs/lows (Corwin-Schultz);
   no order-book data exists historically, so this version is NOT an order-book validation.
   Live version (S3-book): real Binance order-book spread and bid depth (+-50 bps) must both
   recover; it can only be evaluated on forward data.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN = 60_000

S1 = {"div_window": 1440, "z_window": 30 * 1440, "z_min": 2.0, "rev_window": 60, "rev_frac": 0.25,
      "check_every": 5, "cooldown": 240}
S2 = {"bar": 5, "range_bars": 48, "retest_bars": 24, "vol_short": 48, "vol_long": 7 * 288, "compress": 0.6,
      "touch_tol": 0.0015, "hold_tol": 0.0010, "fail_tol": 0.0030, "vol_mult": 1.2, "stop": 0.005, "rr": 2.0,
      "max_hold_min": 240}
S3 = {"drop_window": 30, "drop_min": 0.025, "drop_sigma": 3.0, "recover_window": 60, "retrace": 0.30,
      "taker_min": 0.55, "spread_mult": 1.2, "depth_mult": 1.0, "stop_below_low": 0.003, "max_hold_min": 240,
      "cooldown": 240}


def _cooldown(times: np.ndarray, gap_ms: int) -> np.ndarray:
    keep, last = [], -10**18
    for i, t in enumerate(times):
        if t - last >= gap_ms:
            keep.append(i)
            last = t
    return np.array(keep, dtype=int)


# ----------------------------------------------------------------------------- S1
def s1_signals(ratio_grid: pd.DataFrame, since_ms: int | None = None) -> pd.DataFrame:
    """ratio_grid: ETHBTC 1-minute close-time grid (or ETH/BTC synthetic) with column 'close'."""
    p = S1
    lr = np.log(ratio_grid["close"].astype(float))
    D = lr - lr.shift(p["div_window"])
    sd = D.rolling(p["z_window"], min_periods=p["z_window"] // 3).std()
    z = D / sd
    r60 = lr - lr.shift(p["rev_window"])
    sd60 = r60.rolling(p["z_window"], min_periods=p["z_window"] // 3).std()
    t = ratio_grid.index.values
    grid = (t % (p["check_every"] * MIN)) == 0
    rev = (np.sign(r60) == -np.sign(D)) & (r60.abs() >= p["rev_frac"] * sd60)
    cond = grid & (z.abs() >= p["z_min"]) & rev
    if since_ms is not None:
        cond &= t > since_ms
    idx = np.where(cond.values)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["t", "direction", "z", "D", "r60", "price"])
    idx = idx[_cooldown(t[idx], p["cooldown"] * MIN)]
    return pd.DataFrame({"t": t[idx], "direction": -np.sign(D.values[idx]).astype(int), "z": z.values[idx],
                         "D": D.values[idx], "r60": r60.values[idx], "price": ratio_grid["close"].values[idx]})


# ----------------------------------------------------------------------------- S2
def resample_5m(g: pd.DataFrame) -> pd.DataFrame:
    """5-minute bars keyed by close time from a 1-minute close-time grid."""
    key = ((g.index.values - 1) // (5 * MIN) + 1) * (5 * MIN)
    agg = g.groupby(key).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                             close=("close", "last"), volume=("quote_volume", "sum"), n=("close", "size"))
    return agg[agg["n"] == 5].drop(columns="n")


def s2_signals(g1: pd.DataFrame, since_ms: int | None = None) -> pd.DataFrame:
    p = S2
    b = resample_5m(g1)
    if len(b) < p["vol_long"] + p["range_bars"] + 2:
        return pd.DataFrame(columns=["t", "direction", "level", "stop", "price"])
    lr = np.log(b["close"]).diff()
    vs = lr.rolling(p["vol_short"]).std()
    vl = lr.rolling(p["vol_long"], min_periods=p["vol_long"] // 2).std()
    compressed = (vs < p["compress"] * vl).shift(1, fill_value=False)
    hh = b["high"].rolling(p["range_bars"]).max().shift(1)
    ll = b["low"].rolling(p["range_bars"]).min().shift(1)
    avgv = b["volume"].rolling(p["range_bars"]).mean().shift(1)
    up_brk = compressed & (b["close"] > hh)
    dn_brk = compressed & (b["close"] < ll)
    t = b.index.values
    hi, lo, cl, vol = b["high"].values, b["low"].values, b["close"].values, b["volume"].values
    out = []
    for direction, brk, lvl in ((1, up_brk.values, hh.values), (-1, dn_brk.values, ll.values)):
        for i in np.where(brk)[0]:
            L, v0 = lvl[i], avgv.values[i]
            if not np.isfinite(L) or not np.isfinite(v0):
                continue
            for j in range(i + 1, min(i + 1 + p["retest_bars"], len(b))):
                if direction > 0:
                    if cl[j] < L * (1 - p["fail_tol"]):
                        break  # failed breakout
                    touched = lo[j] <= L * (1 + p["touch_tol"]) and cl[j] >= L * (1 - p["hold_tol"])
                else:
                    if cl[j] > L * (1 + p["fail_tol"]):
                        break
                    touched = hi[j] >= L * (1 - p["touch_tol"]) and cl[j] <= L * (1 + p["hold_tol"])
                if touched and vol[j] >= p["vol_mult"] * v0:
                    stop = L * (1 - p["stop"]) if direction > 0 else L * (1 + p["stop"])
                    out.append({"t": int(t[j]), "direction": direction, "level": float(L), "stop": float(stop),
                                "price": float(cl[j]), "breakout_t": int(t[i])})
                    break
    df = pd.DataFrame(out, columns=["t", "direction", "level", "stop", "price", "breakout_t"])
    if since_ms is not None and not df.empty:
        df = df[df["t"] > since_ms]
    return df.sort_values("t").drop_duplicates("t").reset_index(drop=True)


# ----------------------------------------------------------------------------- S3
def corwin_schultz(high: pd.Series, low: pd.Series, window: int = 10) -> pd.Series:
    """Corwin-Schultz (2012) high-low bid-ask spread estimate (fraction), averaged over `window` bars."""
    lh = np.log(high / low) ** 2
    beta = lh + lh.shift(1)
    h2, l2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1), pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0).rolling(window, min_periods=window // 2).mean()


def s3_signals(g1: pd.DataFrame, since_ms: int | None = None, spread: pd.Series | None = None,
               bid_depth: pd.Series | None = None) -> pd.DataFrame:
    """Proxy version when spread/bid_depth are None (spread estimated from highs/lows).
    Live order-book version passes measured spread (fraction) and bid depth series on the grid."""
    p = S3
    c = g1["close"].astype(float)
    lc = np.log(c)
    lr = lc.diff()
    sig30 = lr.rolling(1440, min_periods=720).std() * math.sqrt(p["drop_window"])
    r30 = lc - lc.shift(p["drop_window"])
    thresh = np.minimum(-p["drop_min"], -p["drop_sigma"] * sig30)
    drop = r30 <= thresh
    spr = spread if spread is not None else corwin_schultz(g1["high"], g1["low"])
    qv = g1["quote_volume"].rolling(10).sum()
    taker = g1["taker_buy_quote"].rolling(10).sum() / qv.replace(0, np.nan)
    t = c.index.values
    cv, spv, tk = c.values, spr.values, taker.values
    dv = bid_depth.values if bid_depth is not None else None
    lows = g1["low"].values
    out, last = [], -10**18
    for i in np.where(drop.values)[0]:
        if t[i] - last < p["cooldown"] * MIN:
            continue
        top = cv[i - p["drop_window"]] if i >= p["drop_window"] else np.nan
        pre_lo = max(0, i - p["drop_window"] - 120)
        pre_spread = np.nanmedian(spv[pre_lo:i - p["drop_window"]]) if i - p["drop_window"] > pre_lo else np.nan
        pre_depth = np.nanmedian(dv[pre_lo:i - p["drop_window"]]) if dv is not None and i - p["drop_window"] > pre_lo else np.nan
        if not (np.isfinite(top) and np.isfinite(pre_spread)):
            continue
        low = lows[i]
        for j in range(i + 1, min(i + 1 + p["recover_window"], len(cv))):
            low = min(low, lows[j])
            depth = top - low
            if depth <= 0:
                continue
            retr = (cv[j] - low) / depth
            ok = (retr >= p["retrace"] and np.isfinite(tk[j]) and tk[j] >= p["taker_min"]
                  and np.isfinite(spv[j]) and spv[j] <= p["spread_mult"] * max(pre_spread, 1e-6))
            if ok and dv is not None:
                ok = np.isfinite(dv[j]) and np.isfinite(pre_depth) and dv[j] >= p["depth_mult"] * pre_depth
            if ok:
                out.append({"t": int(t[j]), "direction": 1, "drop_pct": float((low / top - 1) * 100),
                            "retrace": float(retr), "taker_share": float(tk[j]), "spread_now": float(spv[j]),
                            "spread_pre": float(pre_spread), "low": float(low), "stop": float(low * (1 - p["stop_below_low"])),
                            "target": float(top), "price": float(cv[j]),
                            "spread_source": "order book" if spread is not None else "Corwin-Schultz estimate (proxy)"})
                last = t[i]
                break
    df = pd.DataFrame(out)
    if since_ms is not None and not df.empty:
        df = df[df["t"] > since_ms]
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ trade simulation
def simulate_exit(g1: pd.DataFrame, entry_t: int, direction: int, entry_px: float, stop: float | None,
                  take: float | None, max_hold_min: int) -> dict:
    """Walk 1-minute bars after entry. If stop and target are hit in the same minute, assume the
    STOP was hit first (conservative). Exit at the horizon close otherwise."""
    seg = g1.loc[entry_t + MIN: entry_t + max_hold_min * MIN]
    if seg.empty:
        return {"exit_t": None, "exit_px": None, "reason": "no data"}
    lo, hi, t = seg["low"].values, seg["high"].values, seg.index.values
    n = len(seg)
    hs = (lo <= stop) if (stop is not None and direction > 0) else ((hi >= stop) if stop is not None else np.zeros(n, bool))
    ht = (hi >= take) if (take is not None and direction > 0) else ((lo <= take) if take is not None else np.zeros(n, bool))
    i_s = int(np.argmax(hs)) if hs.any() else n
    i_t = int(np.argmax(ht)) if ht.any() else n
    if i_s < n and i_s <= i_t:
        return {"exit_t": int(t[i_s]), "exit_px": float(stop), "reason": "stop" + ("+target same bar" if i_s == i_t else "")}
    if i_t < n:
        return {"exit_t": int(t[i_t]), "exit_px": float(take), "reason": "target"}
    return {"exit_t": int(t[-1]), "exit_px": float(seg["close"].values[-1]), "reason": "time"}
