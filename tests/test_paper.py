"""Paper engine: gating, delayed fills, stop exits with costs, daily loss limit, kill switch."""
import pandas as pd

from cryptolab.config import load_config
from cryptolab.live import paper as P
from cryptolab.storage.db import Store

T0 = 1_795_550_400_000  # minute aligned


class FakeBN:
    def __init__(self):
        self.px = 100.0
        self.bars = []  # (open_time, high, low, close)

    def agg_trades(self, symbol, start_ms=None, end_ms=None, limit=1):
        return [{"price": self.px, "ts_ms": start_ms + 10, "id": 1, "qty": 1, "buyer_maker": False}]

    def latest_trade(self, symbol):
        return {"price": self.px, "ts_ms": T0 + 10 * 60_000}

    def klines(self, symbol, interval, start, end, limit):
        rows = [b for b in self.bars if start <= b[0] <= end]
        return pd.DataFrame({"open_time": [b[0] for b in rows], "high": [b[1] for b in rows],
                             "low": [b[2] for b in rows], "close": [b[3] for b in rows]})


def _setup(tmp_path, approved=()):
    cfg = load_config()
    for k, v in {"data_dir": "data", "db_file": "data/db.sqlite", "ledger_dir": "data/ledger"}.items():
        cfg["paths"][k] = str(tmp_path / v)
    cfg["paper"]["approved_strategies"] = list(approved)
    store = Store(tmp_path / "data/db.sqlite", tmp_path / "data/ledger")
    return cfg, store


def _signal(store, sid="S2_breakout_retest:SOLUSDT:1", direction=1, stop=99.0, take=102.0, t=T0):
    store.record("signal", {"signal_id": sid, "strategy": "S2_breakout_retest", "symbol": "SOLUSDT", "signal_ms": t,
                            "detected_ms": t + 1000, "direction": direction, "bar_close_price": 100.0,
                            "stop": stop, "take": take, "max_hold_min": 240})


def test_gated_stays_in_cash_and_shadow_trades_with_costs(tmp_path):
    cfg, store = _setup(tmp_path)
    bn = FakeBN()
    eng = P.PaperEngine(cfg, store, bn)
    _signal(store)
    eng.tick(T0 + 1500)                       # fill time not reached -> pending, no entry yet
    assert not eng.state["portfolios"]["shadow:S2_breakout_retest"]["positions"]
    eng.tick(T0 + 10_000)                     # fill after the execution delay
    sh = eng.state["portfolios"]["shadow:S2_breakout_retest"]
    assert "SOLUSDT" in sh["positions"]
    assert not eng.state["portfolios"]["gated"]["positions"]
    assert eng.state["portfolios"]["gated"]["ignored"]  # logged why it stayed in cash
    bn.bars = [(T0 + 60_000, 100.5, 98.5, 99.0)]       # stop hit in the first full minute
    eng.tick(T0 + 3 * 60_000)
    tr = store.rows("paper_trades")
    assert len(tr) == 1 and tr[0]["exit_reason"].startswith("stop") and tr[0]["pnl_usd"] < -10
    assert tr[0]["fees_usd"] > 0 and not sh["positions"]


def test_daily_loss_limit_and_kill_switch(tmp_path):
    cfg, store = _setup(tmp_path)
    cfg["paper"]["daily_loss_limit_pct"] = 0.1
    bn = FakeBN()
    eng = P.PaperEngine(cfg, store, bn)
    _signal(store)
    eng.tick(T0 + 10_000)
    bn.bars = [(T0 + 60_000, 100.5, 98.5, 99.0)]
    eng.tick(T0 + 3 * 60_000)                 # loss > 0.1% of equity -> halted for the day
    sh = eng.state["portfolios"]["shadow:S2_breakout_retest"]
    assert sh["halted_day"] is not None
    _signal(store, sid="S2_breakout_retest:SOLUSDT:2", t=T0 + 5 * 60_000)
    eng.tick(T0 + 6 * 60_000)
    assert not sh["positions"] and sh["ignored"].get("daily loss limit reached") == 1
    # kill switch closes open positions and blocks entries
    cfg2, store2 = _setup(tmp_path / "k")
    eng2 = P.PaperEngine(cfg2, store2, FakeBN())
    _signal(store2)
    eng2.tick(T0 + 10_000)
    assert eng2.state["portfolios"]["shadow:S2_breakout_retest"]["positions"]
    P.kill_switch_file(cfg2).touch()
    eng2.tick(T0 + 20_000)
    assert not eng2.state["portfolios"]["shadow:S2_breakout_retest"]["positions"]
    assert store2.rows("paper_trades")[0]["exit_reason"] == "kill switch"
    kinds = [e["kind"] for e in store2.rows("events")]
    assert "kill_switch_engaged" in kinds
