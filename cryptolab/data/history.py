"""Historical data collection and point-in-time (PIT) universe construction.

Survivorship-bias control: the candidate list is EVERY Binance spot USDT pair in
exchangeInfo, including delisted ones (status BREAK). Each month's universe is the
top-N by trailing 30-day quote volume computed only from data available before that
month began. 1-minute history is then downloaded only for (symbol, month) pairs that
were in the universe at the time (plus a warm-up month).
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cryptolab.exchanges.base import is_excluded_base
from cryptolab.exchanges.binance import Binance
from cryptolab.util.timeutil import DAY, parse_date_ms

log = logging.getLogger(__name__)

MIN_COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_quote"]


def eligible_usdt_symbols(bn: Binance) -> list[dict]:
    out = []
    for s in bn.symbols(quote="USDT", include_delisted=True):
        if is_excluded_base(s["base"]):
            continue
        out.append(s)
    return out


def download_daily_klines(bn: Binance, symbols: list[str], start: str, end: str, out_file: Path,
                          workers: int = 4) -> pd.DataFrame:
    start_ms, end_ms = parse_date_ms(start), parse_date_ms(end) + DAY - 1

    def one(sym: str) -> pd.DataFrame:
        df = bn.klines_range(sym, "1d", start_ms, end_ms)
        if df.empty:
            return df
        df["symbol"] = sym
        return df[["symbol", "open_time", "close", "quote_volume", "trades"]]

    frames = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, s): s for s in symbols}
        for i, f in enumerate(as_completed(futs)):
            try:
                df = f.result()
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                log.warning("daily klines %s failed: %s", futs[f], e)
            if (i + 1) % 100 == 0:
                log.info("daily klines %d/%d", i + 1, len(symbols))
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_file, index=False)
    return daily


PEG_MAX_DAILY_VOL = 0.003  # trailing daily log-return std below this => pegged asset, excluded


def rank_by_trailing_volume(daily: pd.DataFrame, asof_ms: int, n: int, lookback_days: int = 30,
                            min_days: int = 20, allowed: set[str] | None = None) -> list[str]:
    """Top-n symbols by quote volume summed over [asof - lookback, asof), using only past data.

    Requires >= min_days of data in the window, a bar on the day before `asof`
    (i.e. the market was alive when the universe was formed), an eligible base asset,
    and trailing daily volatility >= PEG_MAX_DAILY_VOL (excludes stablecoins/pegs that
    are not on the static exclusion list).
    """
    lo = asof_ms - lookback_days * DAY
    w = daily[(daily["open_time"] >= lo) & (daily["open_time"] < asof_ms)]
    if allowed is not None:
        w = w[w["symbol"].isin(allowed)]
    if w.empty:
        return []
    w = w.sort_values(["symbol", "open_time"])
    w = w.assign(lr=np.log(w["close"]).groupby(w["symbol"]).diff())
    g = w.groupby("symbol").agg(qv=("quote_volume", "sum"), days=("open_time", "count"),
                                last=("open_time", "max"), vol=("lr", "std"))
    g = g[(g["days"] >= min_days) & (g["last"] >= asof_ms - DAY) & (g["vol"] >= PEG_MAX_DAILY_VOL)]
    g = g[[is_excluded_base(_base_of(s)) is None for s in g.index]]
    return g.sort_values("qv", ascending=False).head(n).index.tolist()


def _base_of(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def month_starts(start: str, end: str) -> list[date]:
    d = datetime.strptime(start, "%Y-%m-%d").date().replace(day=1)
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    while d <= e:
        out.append(d)
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def build_pit_universe(daily: pd.DataFrame, start: str, end: str, n: int) -> dict[str, list[str]]:
    uni = {}
    for m in month_starts(start, end):
        asof = int(datetime(m.year, m.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
        uni[m.strftime("%Y-%m")] = rank_by_trailing_volume(daily, asof, n)
    return uni


def _prev_month(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    return f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"


def download_minute_history(bn: Binance, universe: dict[str, list[str]], raw_dir: Path, end: str,
                            workers: int = 6) -> dict:
    """Download 1m klines for every (symbol, month) in the universe plus its warm-up month."""
    need: dict[str, set[str]] = {}
    for ym, syms in universe.items():
        for s in syms:
            need.setdefault(s, set()).update({ym, _prev_month(ym)})
    for s in ("BTCUSDT", "ETHUSDT"):  # cross-asset reference series: always full history
        need.setdefault(s, set()).update(*[{ym, _prev_month(ym)} for ym in universe])
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    out_dir = raw_dir / "binance_1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    def fetch_month(sym: str, ym: str) -> pd.DataFrame | None:
        y, m = map(int, ym.split("-"))
        first = date(y, m, 1)
        if first > end_d:
            return None
        nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        if nxt - timedelta(days=1) <= end_d:
            df = bn.bulk_monthly_klines(sym, "1m", y, m)
            if df is not None:
                return df
        # current/partial month (or monthly file missing): stitch daily files
        frames, d = [], first
        while d < nxt and d <= end_d:
            dd = bn.bulk_daily_klines(sym, "1m", d.isoformat())
            if dd is not None:
                frames.append(dd)
            d += timedelta(days=1)
        return pd.concat(frames, ignore_index=True) if frames else None

    def one(sym: str) -> tuple[str, int, list[str]]:
        path = out_dir / f"{sym}.parquet"
        have = pd.read_parquet(path) if path.exists() else None
        have_months = set()
        if have is not None and not have.empty:
            have_months = set(pd.to_datetime(have["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m"))
        frames = [have] if have is not None else []
        missing = []
        for ym in sorted(need[sym]):
            if ym in have_months and not (ym == end[:7] and int(have["open_time"].max()) < parse_date_ms(end)):
                continue
            df = fetch_month(sym, ym)
            if df is None or df.empty:
                missing.append(ym)
                continue
            frames.append(df[MIN_COLS])
        if not frames:
            return sym, 0, missing
        full = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
        full = full[MIN_COLS].reset_index(drop=True)
        full.to_parquet(path, index=False)
        return sym, len(full), missing

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, s) for s in sorted(need)]
        for i, f in enumerate(as_completed(futs)):
            try:
                sym, n, missing = f.result()
                report[sym] = {"rows": n, "missing_months": missing}
                log.info("[%d/%d] %s rows=%d missing=%s", i + 1, len(futs), sym, n, missing)
            except Exception as e:
                log.exception("minute history failed: %s", e)
    return report


def collect(cfg, bn: Binance) -> dict:
    raw = cfg.path("raw_dir")
    research = cfg.path("research_dir")
    research.mkdir(parents=True, exist_ok=True)
    bt = cfg["backtest"]
    syms = eligible_usdt_symbols(bn)
    log.info("eligible USDT symbols (incl. delisted): %d", len(syms))
    hist_start = (datetime.strptime(bt["history_start"], "%Y-%m-%d").date() - timedelta(days=45)).isoformat()
    daily = download_daily_klines(bn, [s["symbol"] for s in syms], hist_start, bt["test_end"],
                                  raw / "binance_daily.parquet")
    universe = build_pit_universe(daily, bt["start"], bt["test_end"], cfg["universe"]["top_n"])
    status = {s["symbol"]: s["status"] for s in syms}
    union = sorted({s for v in universe.values() for s in v})
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_symbols_incl_delisted": len(syms),
        "candidates_with_daily_data": int(daily["symbol"].nunique()) if not daily.empty else 0,
        "universe_top_n": cfg["universe"]["top_n"],
        "universe_by_month": universe,
        "union_size": len(union),
        "union_currently_delisted": [s for s in union if status.get(s) != "TRADING"],
    }
    (research / "pit_universe.json").write_text(json.dumps(meta, indent=1))
    log.info("PIT universe union=%d; delisted-now in union: %s", len(union), meta["union_currently_delisted"])
    rep = download_minute_history(bn, universe, raw, bt["test_end"])
    meta["minute_history"] = rep
    (research / "pit_universe.json").write_text(json.dumps(meta, indent=1))
    return meta
