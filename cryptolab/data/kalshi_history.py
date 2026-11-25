"""Download settled Kalshi 15-minute BTC/ETH direction markets and their 1-minute quote candles.

Output: data/raw/kalshi/{BTC,ETH}_15m.parquet with one row per (market, minute):
  ticker, open_ms, close_ms, floor_strike, result, minute_end_ms, yes_bid, yes_ask, last, volume
Only information timestamped at or before each candle's end is stored per row, so a
decision at time t can use exactly the rows with minute_end_ms <= t.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from cryptolab.exchanges.kalshi import DIRECTION_SERIES, Kalshi

log = logging.getLogger(__name__)


def download(kalshi: Kalshi, out_dir: Path, assets=("BTC", "ETH"), workers: int = 4) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for asset in assets:
        ser = DIRECTION_SERIES[asset]
        mk = [m for m in kalshi.markets(ser, status="settled", max_pages=200) if m["open_ms"] and m["close_ms"]]
        path = out_dir / f"{asset}_15m.parquet"
        have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ticker"])
        done = set(have["ticker"]) if not have.empty else set()
        todo = [m for m in mk if m["ticker"] not in done]
        log.info("kalshi %s: %d settled markets, %d to fetch", asset, len(mk), len(todo))

        def one(m):
            try:
                cs = kalshi.candlesticks(ser, m["ticker"], m["open_ms"] // 1000, m["close_ms"] // 1000, 1)
            except Exception as e:
                log.warning("candles %s: %s", m["ticker"], e)
                return []
            return [{"ticker": m["ticker"], "open_ms": m["open_ms"], "close_ms": m["close_ms"],
                     "floor_strike": m["floor_strike"], "result": m["result"],
                     "expiration_value": m["expiration_value"], "minute_end_ms": c["end_ms"],
                     "yes_bid": c["yes_bid"], "yes_ask": c["yes_ask"], "last": c["last"], "volume": c["volume"]}
                    for c in cs]

        rows = []
        with ThreadPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(one, todo)):
                rows.extend(r)
                if (i + 1) % 250 == 0:
                    log.info("kalshi %s candles %d/%d", asset, i + 1, len(todo))
        new = pd.DataFrame(rows)
        full = pd.concat([have, new], ignore_index=True) if not have.empty else new
        if not full.empty:
            full = full.drop_duplicates(["ticker", "minute_end_ms"]).sort_values(["open_ms", "minute_end_ms"])
            full.to_parquet(path, index=False)
        report[asset] = {"settled_markets": len(mk), "rows": int(len(full)),
                         "first_window_ms": int(full["open_ms"].min()) if len(full) else None,
                         "last_window_ms": int(full["close_ms"].max()) if len(full) else None}
    return report
