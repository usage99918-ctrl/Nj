"""Append-only, hash-chained JSONL ledger (the system's source of truth).

Each line: {"seq", "type", "ts_ms", "data", "prev_hash", "hash"} where
hash = sha256(prev_hash + canonical_json({"seq","type","ts_ms","data"})).
Any edit, deletion or reordering of an earlier line breaks every later hash, so
silent modification is detectable with `cryptolab audit`.
Shards: one file per UTC day (small diffs when committed to git).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

GENESIS = "0" * 64


def _clean(o):
    """Make floats JSON-safe and deterministic (NaN/inf -> None)."""
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return o
    if isinstance(o, float):
        o = float(o)  # also normalizes numpy float subclasses
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (list, dict)):  # numpy scalar / 0-d array
        try:
            return _clean(o.item())
        except (ValueError, TypeError):
            pass
    if isinstance(o, int):
        return int(o)
    return o


def canonical(obj) -> str:
    return json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def data_hash(data: dict) -> str:
    return sha256(canonical(data))


class Ledger:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._seq, self._head = self._load_head()

    def _shards(self) -> list[Path]:
        return sorted(self.dir.glob("*.jsonl"))

    def _load_head(self) -> tuple[int, str]:
        for shard in reversed(self._shards()):
            last = None
            with open(shard, "rb") as fh:
                for line in fh:
                    if line.strip():
                        last = line
            if last is not None:
                rec = json.loads(last)
                return int(rec["seq"]), rec["hash"]
        return 0, GENESIS

    @property
    def head(self) -> tuple[int, str]:
        return self._seq, self._head

    def append(self, rtype: str, data: dict, ts_ms: int) -> dict:
        seq = self._seq + 1
        body = {"seq": seq, "type": rtype, "ts_ms": int(ts_ms), "data": _clean(data)}
        h = sha256(self._head + canonical(body))
        rec = dict(body, prev_hash=self._head, hash=h)
        day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        path = self.dir / f"{day}.jsonl"
        # Shard names must sort chronologically; never append to an older shard than the newest.
        shards = self._shards()
        if shards and shards[-1].name > path.name:
            path = shards[-1]
        line = canonical(rec) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._seq, self._head = seq, h
        return rec

    def records(self) -> Iterator[dict]:
        for shard in self._shards():
            with open(shard, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)

    def verify_chain(self) -> dict:
        prev, n, errors = GENESIS, 0, []
        expected_seq = 1
        for rec in self.records():
            body = {k: rec[k] for k in ("seq", "type", "ts_ms", "data")}
            h = sha256(prev + canonical(body))
            if rec.get("prev_hash") != prev:
                errors.append(f"seq {rec.get('seq')}: prev_hash mismatch")
            if rec.get("hash") != h:
                errors.append(f"seq {rec.get('seq')}: hash mismatch (record altered)")
            if rec.get("seq") != expected_seq:
                errors.append(f"seq {rec.get('seq')}: expected {expected_seq} (missing/reordered record)")
            prev = rec.get("hash", h)
            expected_seq = int(rec.get("seq", expected_seq)) + 1
            n += 1
            if len(errors) > 20:
                break
        return {"records": n, "ok": not errors, "errors": errors, "head": prev}
