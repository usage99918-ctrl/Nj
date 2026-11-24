"""Time helpers. All internal timestamps are integer milliseconds since the Unix epoch, UTC."""
from __future__ import annotations

import time
from datetime import datetime, timezone

MS = 1
SEC = 1000
MIN = 60 * SEC
HOUR = 60 * MIN
DAY = 24 * HOUR


def now_ms() -> int:
    return int(time.time() * 1000)


def to_iso(ms: int | float | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_date_ms(s: str) -> int:
    """'2026-01-01' or ISO datetime -> ms (UTC)."""
    if len(s) == 10:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def floor_to(ms: int, step_ms: int) -> int:
    return (ms // step_ms) * step_ms


class ExchangeClock:
    """Tracks the offset between the local clock and an exchange's server clock.

    offset_ms = server_time - local_time (estimated at the request midpoint).
    """

    def __init__(self) -> None:
        self.offset_ms = 0
        self.rtt_ms = None
        self.measured_at = None

    def update(self, local_before: int, local_after: int, server_ms: int) -> None:
        mid = (local_before + local_after) // 2
        self.offset_ms = server_ms - mid
        self.rtt_ms = local_after - local_before
        self.measured_at = local_after

    def now(self) -> int:
        return now_ms() + self.offset_ms
