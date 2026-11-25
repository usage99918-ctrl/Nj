import json
import sqlite3

import pytest

from cryptolab.storage.db import DuplicateRecord, Store
from cryptolab.storage.ledger import Ledger


def _fc(i, slot, expiry=2_000):
    return {"forecast_id": f"A-{i:06d}", "experiment": "A_10min", "test_number": i, "slot_ms": slot,
            "symbol": "BTCUSDT", "origin_ms": 1_000, "expiry_ms": expiry, "model_version": "v1", "p0": 100.0}


def test_forecast_is_immutable_and_unique(tmp_path):
    s = Store(tmp_path / "db.sqlite", tmp_path / "ledger")
    s.record("forecast", _fc(1, 600_000))
    with pytest.raises(sqlite3.IntegrityError):
        s.conn.execute("UPDATE forecasts SET payload='{}' WHERE forecast_id='A-000001'")
    with pytest.raises(sqlite3.IntegrityError):
        s.conn.execute("DELETE FROM forecasts")
    with pytest.raises(DuplicateRecord):  # same slot twice
        s.record("forecast", _fc(2, 600_000))
    with pytest.raises(DuplicateRecord):  # same test number twice
        s.record("forecast", _fc(1, 1_200_000))
    with pytest.raises(DuplicateRecord):  # gap for a slot that has a forecast
        s.record("gap", {"gap_id": "A_10min:600000", "experiment": "A_10min", "slot_ms": 600_000})
    assert s.audit()["ok"]


def test_outcome_cannot_precede_expiry(tmp_path):
    s = Store(tmp_path / "db.sqlite", tmp_path / "ledger")
    s.record("forecast", _fc(1, 600_000, expiry=5_000))
    with pytest.raises(DuplicateRecord):
        s.record("outcome", {"forecast_id": "A-000001", "experiment": "A_10min", "status": "VERIFIED",
                             "verified_ms": 4_999})
    s.record("outcome", {"forecast_id": "A-000001", "experiment": "A_10min", "status": "VERIFIED",
                         "verified_ms": 5_001})
    assert s.pending_outcomes() == []


def test_ledger_detects_tampering(tmp_path):
    s = Store(tmp_path / "db.sqlite", tmp_path / "ledger")
    for i in range(1, 4):
        s.record("forecast", _fc(i, i * 600_000))
    assert s.ledger.verify_chain()["ok"]
    shard = sorted((tmp_path / "ledger").glob("*.jsonl"))[0]
    lines = shard.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["data"]["p0"] = 101.0  # silent edit of a frozen forecast
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    shard.write_text("\n".join(lines) + "\n")
    res = Ledger(tmp_path / "ledger").verify_chain()
    assert not res["ok"] and any("altered" in e for e in res["errors"])


def test_rebuild_db_from_ledger(tmp_path):
    s = Store(tmp_path / "db.sqlite", tmp_path / "ledger")
    for i in range(1, 6):
        s.record("forecast", _fc(i, i * 600_000))
    s.close()
    (tmp_path / "db.sqlite").unlink()
    for p in tmp_path.glob("db.sqlite-*"):
        p.unlink()
    s2 = Store(tmp_path / "db.sqlite", tmp_path / "ledger")  # reconcile on open
    assert s2.scalar("SELECT COUNT(*) FROM forecasts") == 5
    assert s2.next_test_number("A_10min") == 6
    assert s2.audit()["ok"]


def test_two_writers_share_one_chain(tmp_path):
    """Two Store instances (e.g. daemon + research process) appending alternately keep one valid chain."""
    a = Store(tmp_path / "db.sqlite", tmp_path / "ledger")
    b = Store(tmp_path / "db.sqlite", tmp_path / "ledger")
    a.record("forecast", _fc(1, 600_000))
    b.record("preregistration", {"prereg_id": "x"})
    a.record("forecast", _fc(2, 1_200_000))
    b.record("preregistration", {"prereg_id": "y"})
    chain = Ledger(tmp_path / "ledger").verify_chain()
    assert chain["ok"] and chain["records"] == 4
    assert a.audit()["ok"] and b.audit()["ok"]
