"""SQLite store with immutability enforcement, mirrored to the hash-chained ledger.

Write protocol (crash-safe, no duplicates):
  BEGIN IMMEDIATE -> INSERT (UNIQUE constraints reject duplicates) -> append ledger line
  (fsync) -> COMMIT.  If the process dies after the ledger append but before COMMIT,
  `reconcile()` (run at every start) re-inserts the ledger record. The ledger is the
  source of truth; the DB can be rebuilt from it at any time.

Immutability: triggers abort every UPDATE/DELETE on research records.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from cryptolab.storage.ledger import Ledger, canonical, data_hash
from cryptolab.util.timeutil import now_ms

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# type -> (table, primary key field, extra indexed columns)
TABLES = {
    "model_version": ("model_versions", "version", ["experiment"]),
    "preregistration": ("preregistrations", "prereg_id", []),
    "scan": ("scans", "scan_id", ["experiment", "slot_ms"]),
    "forecast": ("forecasts", "forecast_id", ["experiment", "test_number", "slot_ms", "symbol", "origin_ms",
                                              "expiry_ms", "model_version"]),
    "gap": ("gaps", "gap_id", ["experiment", "slot_ms"]),
    "outcome": ("outcomes", "forecast_id", ["experiment", "status", "verified_ms"]),
}

DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runtime (key TEXT PRIMARY KEY, value TEXT, updated_ms INTEGER);

CREATE TABLE IF NOT EXISTS model_versions (
  version TEXT PRIMARY KEY, experiment TEXT, payload TEXT NOT NULL, hash TEXT NOT NULL, seq INTEGER);
CREATE TABLE IF NOT EXISTS preregistrations (
  prereg_id TEXT PRIMARY KEY, payload TEXT NOT NULL, hash TEXT NOT NULL, seq INTEGER);
CREATE TABLE IF NOT EXISTS scans (
  scan_id TEXT PRIMARY KEY, experiment TEXT, slot_ms INTEGER, payload TEXT NOT NULL, hash TEXT NOT NULL, seq INTEGER);
CREATE TABLE IF NOT EXISTS forecasts (
  forecast_id TEXT PRIMARY KEY, experiment TEXT NOT NULL, test_number INTEGER NOT NULL, slot_ms INTEGER NOT NULL,
  symbol TEXT, origin_ms INTEGER, expiry_ms INTEGER, model_version TEXT,
  payload TEXT NOT NULL, hash TEXT NOT NULL, seq INTEGER,
  UNIQUE(experiment, slot_ms), UNIQUE(experiment, test_number));
CREATE TABLE IF NOT EXISTS gaps (
  gap_id TEXT PRIMARY KEY, experiment TEXT NOT NULL, slot_ms INTEGER NOT NULL,
  payload TEXT NOT NULL, hash TEXT NOT NULL, seq INTEGER, UNIQUE(experiment, slot_ms));
CREATE TABLE IF NOT EXISTS outcomes (
  forecast_id TEXT PRIMARY KEY REFERENCES forecasts(forecast_id), experiment TEXT, status TEXT, verified_ms INTEGER,
  payload TEXT NOT NULL, hash TEXT NOT NULL, seq INTEGER);
CREATE INDEX IF NOT EXISTS ix_fc_exp ON forecasts(experiment, expiry_ms);
"""

_TRIGGER_TABLES = ["model_versions", "preregistrations", "scans", "forecasts", "gaps", "outcomes"]


def _triggers() -> str:
    parts = []
    for t in _TRIGGER_TABLES:
        parts.append(f"CREATE TRIGGER IF NOT EXISTS {t}_no_update BEFORE UPDATE ON {t} "
                     f"BEGIN SELECT RAISE(ABORT, '{t} records are immutable'); END;")
        parts.append(f"CREATE TRIGGER IF NOT EXISTS {t}_no_delete BEFORE DELETE ON {t} "
                     f"BEGIN SELECT RAISE(ABORT, '{t} records are immutable'); END;")
    # A slot holds either a forecast or a gap record, never both.
    parts.append("CREATE TRIGGER IF NOT EXISTS gaps_no_forecast BEFORE INSERT ON gaps "
                 "WHEN EXISTS (SELECT 1 FROM forecasts f WHERE f.experiment=NEW.experiment AND f.slot_ms=NEW.slot_ms) "
                 "BEGIN SELECT RAISE(ABORT, 'slot already has a forecast'); END;")
    parts.append("CREATE TRIGGER IF NOT EXISTS forecasts_no_gap BEFORE INSERT ON forecasts "
                 "WHEN EXISTS (SELECT 1 FROM gaps g WHERE g.experiment=NEW.experiment AND g.slot_ms=NEW.slot_ms) "
                 "BEGIN SELECT RAISE(ABORT, 'slot already recorded as a gap'); END;")
    # An outcome may only be recorded after the forecast's expiry time.
    parts.append("CREATE TRIGGER IF NOT EXISTS outcomes_after_expiry BEFORE INSERT ON outcomes "
                 "WHEN (SELECT expiry_ms FROM forecasts f WHERE f.forecast_id=NEW.forecast_id) IS NULL "
                 "OR NEW.verified_ms < (SELECT expiry_ms FROM forecasts f WHERE f.forecast_id=NEW.forecast_id) "
                 "BEGIN SELECT RAISE(ABORT, 'outcome before expiry or unknown forecast'); END;")
    return "\n".join(parts)


class DuplicateRecord(Exception):
    pass


class Store:
    def __init__(self, db_file: Path, ledger_dir: Path) -> None:
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(ledger_dir)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_file), timeout=30, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(DDL)
        self.conn.executescript(_triggers())
        self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.reconcile()

    # ------------------------------------------------------------------ writes
    def record(self, rtype: str, data: dict, ts_ms: Optional[int] = None) -> dict:
        """Insert an immutable record into the DB and the ledger atomically."""
        table, pk, cols = TABLES[rtype]
        ts_ms = ts_ms or now_ms()
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self._insert(table, pk, cols, data, seq=self.ledger.head[0] + 1)
                rec = self.ledger.append(rtype, data, ts_ms)
                self.conn.execute("COMMIT")
                return rec
            except sqlite3.IntegrityError as e:
                self.conn.execute("ROLLBACK")
                raise DuplicateRecord(f"{rtype} {data.get(pk)}: {e}") from e
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def _insert(self, table: str, pk: str, cols: list[str], data: dict, seq: int) -> None:
        payload = canonical(data)
        names = [pk] + cols + ["payload", "hash", "seq"]
        vals = [data.get(pk)] + [data.get(c) for c in cols] + [payload, data_hash(data), seq]
        self.conn.execute(f"INSERT INTO {table} ({','.join(names)}) VALUES ({','.join('?' * len(names))})", vals)

    # ---------------------------------------------------------------- recovery
    def reconcile(self) -> int:
        """Insert any ledger record missing from the DB (crash recovery / rebuild)."""
        added = 0
        with self._lock:
            for rec in self.ledger.records():
                rtype = rec["type"]
                if rtype not in TABLES:
                    continue
                table, pk, cols = TABLES[rtype]
                key = rec["data"].get(pk)
                row = self.conn.execute(f"SELECT 1 FROM {table} WHERE {pk}=?", (key,)).fetchone()
                if row is None:
                    try:
                        self.conn.execute("BEGIN IMMEDIATE")
                        self._insert(table, pk, cols, rec["data"], rec["seq"])
                        self.conn.execute("COMMIT")
                        added += 1
                    except sqlite3.IntegrityError as e:
                        self.conn.execute("ROLLBACK")
                        log.error("reconcile conflict for %s %s: %s", rtype, key, e)
        if added:
            log.warning("reconciled %d ledger records into the database", added)
        return added

    def audit(self) -> dict:
        chain = self.ledger.verify_chain()
        ledger_hashes = {}
        for rec in self.ledger.records():
            if rec["type"] in TABLES:
                table, pk, _ = TABLES[rec["type"]]
                ledger_hashes[(table, rec["data"].get(pk))] = data_hash(rec["data"])
        mismatched, missing_in_ledger, checked = [], [], 0
        for _rtype, (table, pk, _) in TABLES.items():
            for row in self.conn.execute(f"SELECT {pk} AS k, payload, hash FROM {table}"):
                checked += 1
                h = data_hash(json.loads(row["payload"]))
                lh = ledger_hashes.get((table, row["k"]))
                if lh is None:
                    missing_in_ledger.append(f"{table}:{row['k']}")
                elif lh != h or row["hash"] != h:
                    mismatched.append(f"{table}:{row['k']}")
        db_keys = set()
        for _rtype, (table, pk, _) in TABLES.items():
            db_keys |= {(table, r[0]) for r in self.conn.execute(f"SELECT {pk} FROM {table}")}
        missing_in_db = [f"{t}:{k}" for (t, k) in ledger_hashes if (t, k) not in db_keys]
        ok = chain["ok"] and not mismatched and not missing_in_ledger and not missing_in_db
        return {"ok": ok, "ledger": chain, "db_rows_checked": checked, "hash_mismatches": mismatched,
                "db_rows_missing_from_ledger": missing_in_ledger, "ledger_records_missing_from_db": missing_in_db}

    # ------------------------------------------------------------------- reads
    def rows(self, table: str, where: str = "", params: Iterable[Any] = ()) -> list[dict]:
        q = f"SELECT payload FROM {table} {('WHERE ' + where) if where else ''} ORDER BY seq"
        return [json.loads(r[0]) for r in self.conn.execute(q, tuple(params))]

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        r = self.conn.execute(sql, tuple(params)).fetchone()
        return r[0] if r else None

    def pending_outcomes(self, experiment: Optional[str] = None, due_before_ms: Optional[int] = None) -> list[dict]:
        q = "SELECT f.payload FROM forecasts f LEFT JOIN outcomes o ON o.forecast_id=f.forecast_id WHERE o.forecast_id IS NULL"
        params: list = []
        if experiment:
            q += " AND f.experiment=?"
            params.append(experiment)
        if due_before_ms is not None:
            q += " AND f.expiry_ms<=?"
            params.append(due_before_ms)
        q += " ORDER BY f.expiry_ms"
        return [json.loads(r[0]) for r in self.conn.execute(q, params)]

    def next_test_number(self, experiment: str) -> int:
        v = self.scalar("SELECT MAX(test_number) FROM forecasts WHERE experiment=?", (experiment,))
        return int(v or 0) + 1

    def slot_taken(self, experiment: str, slot_ms: int) -> bool:
        return bool(self.scalar("SELECT 1 FROM forecasts WHERE experiment=? AND slot_ms=? UNION ALL "
                                "SELECT 1 FROM gaps WHERE experiment=? AND slot_ms=?",
                                (experiment, slot_ms, experiment, slot_ms)))

    # ---------------------------------------------------------- runtime state
    def set_runtime(self, key: str, value: Any) -> None:
        with self._lock:
            self.conn.execute("INSERT INTO runtime(key, value, updated_ms) VALUES(?,?,?) "
                              "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ms=excluded.updated_ms",
                              (key, json.dumps(value), now_ms()))

    def get_runtime(self, key: str, default: Any = None) -> Any:
        r = self.conn.execute("SELECT value, updated_ms FROM runtime WHERE key=?", (key,)).fetchone()
        return (json.loads(r[0]), r[1]) if r else (default, None)

    def close(self) -> None:
        self.conn.close()


def open_store(cfg) -> Store:
    return Store(cfg.path("db_file"), cfg.path("ledger_dir"))
