"""Leakage, split and statistics tests for the research engine."""
import numpy as np
import pandas as pd
import pytest

from cryptolab.backtest import stats as st
from cryptolab.backtest.engine import WalkForward, fit_calibration, select_top
from cryptolab.features import (FORBIDDEN, MODEL_FEATURES, add_targets, cross_sectional, model_feature_columns,
                                reference_series, symbol_features, to_grid)
from cryptolab.strategies.library import Strategy, rules


def _synthetic_klines(n=4000, seed=0, start=1_767_225_600_000):
    rng = np.random.default_rng(seed)
    lr = rng.normal(0, 0.001, n)
    close = 100 * np.exp(np.cumsum(lr))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0005, n))
    qv = rng.uniform(1e4, 5e4, n)
    return pd.DataFrame({"open_time": start + np.arange(n) * 60_000, "open": open_, "high": high, "low": low,
                         "close": close, "volume": qv / close, "quote_volume": qv,
                         "trades": rng.integers(10, 100, n), "taker_buy_quote": qv * rng.uniform(0.3, 0.7, n)})


def test_features_have_no_lookahead():
    """Features at time t must be identical whether or not data after t exists."""
    k = _synthetic_klines(12_000, seed=1)
    btc = _synthetic_klines(12_000, seed=2)
    eth = _synthetic_klines(12_000, seed=3)
    cut = 11_000  # truncate: the last available bar has index cut-1
    full_ref = {"BTC": reference_series(btc), "ETH": reference_series(eth)}
    trunc_ref = {"BTC": reference_series(btc.iloc[:cut]), "ETH": reference_series(eth.iloc[:cut])}
    f_full = symbol_features(to_grid(k), full_ref)
    f_trunc = symbol_features(to_grid(k.iloc[:cut]), trunc_ref)
    t = f_trunc.index[-1]
    cols = [c for c in MODEL_FEATURES if c in f_full.columns and not c.startswith(("xs_", "breadth", "mkt_", "disp"))]
    a, b = f_full.loc[t, cols].astype(float), f_trunc.loc[t, cols].astype(float)
    both_nan = a.isna() & b.isna()
    diff = (a - b).abs()[~both_nan]
    assert len(cols) > 50
    assert diff.max() < 1e-9, diff.sort_values().tail()


def test_target_is_future_and_aligned():
    k = _synthetic_klines(500)
    f = add_targets(symbol_features(to_grid(k)), horizons=(10,))
    t = f.index[100]
    expected = np.log(f["close"].loc[t + 10 * 60_000] / f["close"].loc[t])
    assert abs(f.loc[t, "y_10"] - expected) < 1e-12
    # bar close-time indexing: feature time t uses the bar that OPENED at t - 60s
    assert f.index[0] == k["open_time"].iloc[0] + 60_000


def test_model_features_never_include_targets():
    assert not (set(MODEL_FEATURES) & FORBIDDEN)
    assert not any(c.startswith("y_") for c in MODEL_FEATURES)
    df = pd.DataFrame({c: [0.0] for c in MODEL_FEATURES + ["y", "y_std", "y_10", "close"]})
    cols = model_feature_columns(df)
    assert "y" not in cols and "y_std" not in cols and "close" not in cols


def test_select_top_picks_max_abs_and_ties_to_liquidity():
    t = np.array([1, 1, 1, 2, 2])
    z = np.array([0.5, -2.0, 2.0, 0.0, 0.0])
    rank = np.array([1, 3, 2, 1, 2])
    idx = select_top(t, z, rank)
    assert list(idx) == [2]  # t=1: |2.0| tie broken by rank 2 < 3; t=2: all zero -> abstain


def _panel(n_t=600, n_c=8, seed=0, signal=0.0):
    """Synthetic decision panel spanning 4 months with an optional planted signal in r_10."""
    rng = np.random.default_rng(seed)
    t0 = 1_767_225_600_000  # 2026-01-01
    ts = t0 + np.arange(n_t) * 4 * 3600_000  # every 4h -> ~100 days
    rows = []
    for t in ts:
        r10 = rng.normal(0, 1, n_c)
        y = signal * r10 + rng.normal(0, 1, n_c)
        for c in range(n_c):
            rows.append({"t": t, "symbol": f"C{c}", "r_10": r10[c] * 0.001, "sig1m": 0.001 / np.sqrt(10),
                         "y": y[c] * 0.001, "universe_rank": c + 1})
    p = pd.DataFrame(rows)
    p["sig_h"] = p["sig1m"] * np.sqrt(10)
    p["y_std"] = p["y"] / p["sig_h"]
    p["month"] = pd.to_datetime(p["t"], unit="ms", utc=True).dt.strftime("%Y-%m")
    p["split"] = "train"
    p["btc_trend"] = 0.0
    p["btc_vol_ratio"] = 1.0
    p["n_universe"] = n_c
    return p


class _Spy:
    """Model that records the max training timestamp it was fit on."""

    def __init__(self):
        self.max_train_t = []

    def fit(self, train, h):
        self.max_train_t.append(int(train["t"].max()))

    def predict(self, df, h):
        return df["r_10"].values * 1000


def test_walk_forward_respects_embargo_and_finds_planted_signal():
    p = _panel(signal=0.3)
    spy = _Spy()
    strat = Strategy("spy", "test", "spy", model=spy, kind="model")
    bt = {"embargo_days": 1}
    costs = {"backtest_half_spread_top10": 0, "backtest_half_spread_other": 0, "slippage_per_side": 0,
             "taker_fee": {"binance": 0}, "low_cost_fee": 0, "short_borrow_cost_per_hour": 0}
    wf = WalkForward(p, [strat], 10, bt, costs)
    months = sorted(p["month"].unique())
    wf.run_months(months[2:])
    from cryptolab.util.timeutil import parse_date_ms
    for M, mx in zip(months[2:], spy.max_train_t):
        assert mx < parse_date_ms(M + "-01") - 86_400_000 - 10 * 60_000  # strictly before month - embargo - h
    sel, _, _ = wf.results()
    acc = sel["spy"]["correct"].mean()
    assert acc > 0.6  # planted signal is detected


def test_shuffled_target_gives_chance_accuracy():
    p = _panel(signal=0.0, seed=5, n_t=700)
    strat = Strategy("mom", "test", "r_10 follows", score=lambda d: d["r_10"] / 0.001)
    costs = {"backtest_half_spread_top10": 0, "backtest_half_spread_other": 0, "slippage_per_side": 0,
             "taker_fee": {"binance": 0}, "low_cost_fee": 0, "short_borrow_cost_per_hour": 0}
    wf = WalkForward(p, [strat], 10, {"embargo_days": 1}, costs)
    wf.run_months(sorted(p["month"].unique())[2:])
    sel, _, _ = wf.results()
    acc = sel["mom"]["correct"].mean()
    n = len(sel["mom"])
    assert abs(acc - 0.5) < 4 * np.sqrt(0.25 / n)


def test_calibration_slope_nonnegative_and_quantiles_ordered():
    rng = np.random.default_rng(0)
    n = 5000
    z = rng.normal(size=n)
    y_std = -0.1 * z + rng.normal(size=n)  # hypothesis wrong -> b clipped at 0
    t = np.repeat(np.arange(n // 5), 5)
    cal = fit_calibration(z, y_std, t, np.tile(np.arange(5), n // 5), np.ones(n), y_std)
    assert cal.b == 0.0
    assert cal.q10 < cal.q25 < cal.q75 < cal.q90


def test_rule_library_is_distinct_and_large():
    names = [s.name for s in rules(10)]
    assert len(names) == len(set(names)) >= 75


def test_stats_helpers():
    lo, hi = st.wilson_ci(55, 100)
    assert 0.45 < lo < 0.46 and 0.64 < hi < 0.65
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] >= adj["c"]
    m, se, t = st.newey_west_t(np.r_[np.ones(50), -np.ones(50)])
    assert abs(m) < 1e-12
    rng = np.random.default_rng(1)
    D = rng.normal(0, 1, (500, 20))  # 20 strategies with zero true edge
    assert st.reality_check(D, n_boot=300, seed=2)["p_value"] > 0.05
    a, b = st.fit_logistic_1d(rng.normal(size=2000), rng.integers(0, 2, 2000))
    assert abs(a) < 0.2 and abs(b) < 0.2
