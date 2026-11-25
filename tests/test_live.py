"""Live-path tests: verification scoring, quality filters, scheduler slots/gaps (no network)."""
import types

import pytest

from cryptolab.config import load_config
from cryptolab.exchanges.base import Ticker, is_excluded_base
from cryptolab.live.scanner import quality_check
from cryptolab.live.verifier import score
from cryptolab.storage.db import Store

Q = {"min_quote_volume_24h_usd": 5e6, "max_spread_bps": 10, "max_staleness_s": 180, "max_tick_bps": 5,
     "usd_quotes": ["USDT", "USD", "USDC"]}


def _fc(direction="UP", long_signal=False):
    return {"p0": 100.0, "direction": direction, "pred_price": 100.05, "pi50": [99.9, 100.1], "pi80": [99.8, 100.3],
            "p_up": 0.55 if direction == "UP" else 0.45, "long_signal": long_signal,
            "costs": {"round_trip": 0.0024, "short_borrow_for_horizon": 0.00002}}


def test_flat_outcome_is_incorrect_for_both_directions():
    for d in ("UP", "DOWN"):
        r = score(_fc(d), 100.0)
        assert r["actual_direction"] == "FLAT" and r["correct"] is False


def test_scoring_errors_intervals_and_costs():
    r = score(_fc("UP", long_signal=True), 100.2)
    assert r["correct"] and r["actual_direction"] == "UP"
    assert not r["in_pi50"] and r["in_pi80"]
    assert r["long_net_pct"] == pytest.approx((0.002 - 0.0024) * 100)
    assert r["brier"] == pytest.approx((0.55 - 1) ** 2)
    d = score(_fc("DOWN"), 99.0)
    assert d["correct"] and d["short_is_theoretical"] and d["long_taken"] is False and d["long_net_pct"] == 0.0
    assert d["dir_net_pct"] == pytest.approx((0.01 - 0.0024 - 0.00002) * 100)


def _t(**kw):
    base = dict(exchange="x", symbol="ABCUSDT", base="ABC", quote="USDT", last=10.0, bid=9.999, ask=10.001,
                quote_volume_24h=1e7, ts_ms=1_000_000, tick_size=0.001)
    base.update(kw)
    return Ticker(**base)


def test_quality_checks():
    now = 1_000_000 + 1000
    assert quality_check(_t(), Q, now) is None
    assert quality_check(_t(quote="BTC"), Q, now) == "non-USD quote"
    assert quality_check(_t(quote_volume_24h=1e5), Q, now) == "low 24h volume"
    assert quality_check(_t(bid=9.9, ask=10.1), Q, now) == "wide spread"
    assert quality_check(_t(), Q, now + 10 * 60_000) == "stale"
    assert quality_check(_t(tick_size=0.01), Q, now) == "coarse tick"
    assert quality_check(_t(base="USDC"), Q, now) == "excluded asset type"
    assert quality_check(_t(extra={"price_banded": True}), Q, now).startswith("price-banded")
    assert is_excluded_base("BTCUP") and is_excluded_base("ETH3L") and not is_excluded_base("JUP")
    assert is_excluded_base("NVDAB") and not is_excluded_base("BNB")


class _FakeForecaster:
    def __init__(self, store, fail=False):
        self.store, self.fail, self.calls = store, fail, 0

    def run(self, key, slot, snap=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("exchange down")
        n = self.store.next_test_number("A_10min")
        rec = {"forecast_id": f"A-{n:06d}", "experiment": "A_10min", "test_number": n, "slot_ms": slot,
               "origin_ms": slot + 1000, "expiry_ms": slot + 601_000}
        self.store.record("forecast", rec)
        return rec


def _system(tmp_path, fail=False):
    from cryptolab.live.scheduler import LiveSystem
    cfg = load_config()
    sysm = LiveSystem.__new__(LiveSystem)
    sysm.cfg = cfg
    sysm.store = Store(tmp_path / "db.sqlite", tmp_path / "ledger")
    sysm.forecaster = _FakeForecaster(sysm.store, fail)
    sysm.last_attempt = {}
    sysm.model_ready = lambda key: True
    return sysm


def test_scheduler_one_forecast_per_slot_and_gaps(tmp_path):
    s = _system(tmp_path)
    slot = 1_795_550_400_000  # aligned to 10 min
    assert s.tick_experiment("A", slot + 30_000)["forecast_id"] == "A-000001"
    s.last_attempt = {}
    assert s.tick_experiment("A", slot + 90_000) is None  # same slot: no duplicate
    s.last_attempt = {}
    # scheduler offline for 3 slots, back 1 min into the 4th
    rec = s.tick_experiment("A", slot + 4 * 600_000 + 60_000)
    assert rec["slot_ms"] == slot + 4 * 600_000
    gaps = s.store.rows("gaps")
    assert [g["slot_ms"] for g in gaps] == [slot + 600_000, slot + 1_200_000, slot + 1_800_000]
    s.last_attempt = {}
    # late start (> max_start_delay) records a gap instead of a late forecast
    assert s.tick_experiment("A", slot + 5 * 600_000 + 6 * 60_000) is None
    assert s.store.rows("gaps")[-1]["slot_ms"] == slot + 5 * 600_000
    assert s.store.audit()["ok"]


def test_scheduler_failure_becomes_gap_with_reason(tmp_path):
    s = _system(tmp_path)
    slot = 1_795_550_400_000
    s.tick_experiment("A", slot + 10_000)            # starts the experiment
    s.forecaster.fail = True
    s.last_attempt = {}
    assert s.tick_experiment("A", slot + 600_000 + 10_000) is None   # fails inside its window
    s.last_attempt = {}
    s.tick_experiment("A", slot + 1_200_000 + 10_000)               # next slot: previous one becomes a gap
    g = s.store.rows("gaps")[0]
    assert g["slot_ms"] == slot + 600_000 and "exchange down" in g["reason"]
