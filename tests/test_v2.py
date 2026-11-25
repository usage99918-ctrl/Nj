"""Kalshi ladder maths, G2 probability layers, G2 outcome scoring, dashboard auth."""
import math
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from cryptolab.backtest.g2 import ProbLayers, sideways_tau
from cryptolab.exchanges.kalshi import ladder_summary
from cryptolab.live.verifier import score_g2


def test_kalshi_ladder_interpolates_one_expiry_and_skips_one_sided_quotes():
    lad = [{"floor_strike": k, "yes_bid": p - 0.01, "yes_ask": p + 0.01, "yes_mid": p, "spread": 0.02,
            "close_ms": 1, "event_ticker": "E", "volume": 10} for k, p in ((99, 0.9), (100, 0.6), (101, 0.3), (102, 0.1))]
    lad.append({"floor_strike": 103, "yes_bid": 0.0, "yes_ask": 0.05, "yes_mid": 0.025, "spread": 0.05,
                "close_ms": 1, "event_ticker": "E", "volume": 1})  # one-sided -> ignored
    s = ladder_summary(lad, 100.5)
    assert s["available"] and s["n_strikes_quoted"] == 4
    assert s["p_above_spot"] == pytest.approx(0.45)
    assert 100 < s["implied_median"] < 101


def test_sideways_threshold_definition():
    assert sideways_tau(5) == pytest.approx(0.0025) and sideways_tau(60) == pytest.approx(0.0025)
    assert sideways_tau(240) == pytest.approx(0.005) and sideways_tau(1440) == pytest.approx(0.0025 * math.sqrt(24))


def test_probability_layers_are_calibrated_on_synthetic_random_walk():
    rng = np.random.default_rng(0)
    n, sig = 20_000, 0.004
    paths = rng.normal(0, sig / math.sqrt(20), (n, 20)).cumsum(axis=1)
    y, ymax, ymin = paths[:, -1], paths.max(axis=1), paths.min(axis=1)
    mu = np.zeros(n)
    L = ProbLayers(60, [0.5]).fit(mu[:15000], np.full(15000, sig), y[:15000], ymax[:15000], ymin[:15000])
    pu, pd, ps = L.three_class(mu[15000:], np.full(5000, sig))
    assert np.allclose(pu + pd + ps, 1)
    real_up = (y[15000:] > L.tau).mean()
    assert abs(pu.mean() - real_up) < 0.03
    pt = L.touch(mu[15000:], np.full(5000, sig), 0.005, "up")
    assert abs(pt.mean() - (ymax[15000:] >= math.log(1.005)).mean()) < 0.03


def test_g2_outcome_scoring():
    fc = {"p0": 100.0, "three_class": {"p_up": 0.5, "p_down": 0.3, "p_sideways": 0.2, "sideways_tau_pct": 0.25},
          "touch": {"+0.5%": {"price": 100.5, "p": 0.4}, "-0.5%": {"price": 99.5, "p": 0.3}}}
    r = score_g2(fc, 100.1, hi=100.6, lo=99.8)
    assert r["actual_class3"] == "SIDEWAYS" and r["brier3"] == pytest.approx(0.25 + 0.09 + 0.64)
    assert r["touch_results"]["+0.5%"]["hit"] and not r["touch_results"]["-0.5%"]["hit"]


def test_dashboard_server_requires_token(tmp_path):
    from cryptolab.config import load_config
    from cryptolab.report import server
    cfg = load_config()
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    tok = server.get_token(cfg)
    t = threading.Thread(target=server.serve, args=(cfg, "127.0.0.1", 8811), daemon=True)
    t.start()
    time.sleep(1.5)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen("http://127.0.0.1:8811/api/status?token=wrong", timeout=10)
    assert e.value.code == 401
    assert len(tok) >= 24
