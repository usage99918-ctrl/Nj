"""Frozen live models must run identically on any Python version (PC runs 3.12, models were frozen on 3.13)."""
import sys
import types

import cloudpickle
import numpy as np
import pytest

from cryptolab.config import load_config
from cryptolab.strategies.library import Strategy, rules
from cryptolab.strategies.models import FeatureModel
from cryptolab.strategies.portable import FrozenModelError, make_portable

LIVE_RULES = [("rsi_rev_60", 60), ("rv_rev_10", 240), ("quiet_drift", 1440)]


def _panel(n=40, seed=1):
    from cryptolab.live.doctor import synthetic_panel
    return synthetic_panel(n, seed)


def _lib(name, h):
    return {s.name: s for s in rules(h)}[name]


def _foreign(fn):
    """Simulate a function pickled by another Python version: same names/constants/defaults, alien bytecode."""
    code = fn.__code__.replace(co_code=bytes(len(fn.__code__.co_code)))
    return types.FunctionType(code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)


@pytest.mark.parametrize("name,h", LIVE_RULES)
def test_rule_with_foreign_bytecode_is_rebuilt_from_library(name, h):
    frozen = cloudpickle.loads(cloudpickle.dumps(_lib(name, h)))
    frozen.score = _foreign(frozen.score)
    rep = make_portable(frozen, h)
    assert [r["rule"] for r in rep["rules_rebuilt"]] == [name]
    assert rep["rules_rebuilt"][0]["bytecode_identical"] is False
    assert frozen.score.__code__ == _lib(name, h).score.__code__       # alien bytecode is never executed
    p = _panel()
    np.testing.assert_array_equal(frozen.predict(p, h), _lib(name, h).predict(p, h))


def test_changed_rule_definition_is_refused():
    frozen = cloudpickle.loads(cloudpickle.dumps(_lib("rsi_rev_60", 60)))
    f = frozen.score
    frozen.score = types.FunctionType(_foreign(f).__code__, f.__globals__, f.__name__, (14,), f.__closure__)
    with pytest.raises(FrozenModelError, match="differs"):
        make_portable(frozen, 60)


def test_unknown_rule_is_refused():
    s = Strategy("not_a_library_rule", "x", "x", score=lambda p: p["r_10"])
    with pytest.raises(FrozenModelError, match="not in the strategy library"):
        make_portable(s, 60)


def test_stored_factory_is_guarded_not_executed():
    s = Strategy("m", "ml", "x", model=FeatureModel(lambda: 1 / 0), kind="model")
    rep = make_portable(s, 10)
    assert rep["guarded"] == ["strategy.model.make"]
    with pytest.raises(FrozenModelError):
        s.model.make()


def test_live_model_files_load_portably():
    cfg = load_config()
    from cryptolab.live.forecaster import LiveModel
    d = cfg.path("research_dir") / "models"
    files = {"B_60min": 60, "G2_60min": 60, "G2_240min": 240, "G2_1440min": 1440}
    if not all((d / f"{k}_v1.pkl").exists() for k in files):
        pytest.skip("live model files not present")
    p = _panel()
    for k, h in files.items():
        m = LiveModel(d / f"{k}_v1.pkl")
        rb = m.portability["rules_rebuilt"]
        assert len(rb) == 1 and m.h == h
        if sys.version_info[:2] == (3, 13):   # the version that froze the files: must be bit-identical
            assert rb[0]["bytecode_identical"] is True
        z = m.strategy.predict(p, h)
        assert z.shape == (len(p),) and np.all(np.isfinite(z))


def test_doctor_offline_passes_and_records_nothing():
    cfg = load_config()
    from cryptolab.live.doctor import run_doctor
    from cryptolab.storage.ledger import Ledger
    if not cfg.path("db_file").exists():
        pytest.skip("no database")
    before = Ledger(cfg.path("ledger_dir")).head
    rep = run_doctor(cfg, offline=True)
    assert rep["ok"], [c for c in rep["checks"] if not c["ok"]]
    assert Ledger(cfg.path("ledger_dir")).head == before
