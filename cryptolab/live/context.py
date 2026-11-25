"""Market context for a forecast candidate: order book microstructure, derivatives data,
cross-exchange price confirmation, and human-readable supporting/contradicting evidence.

Context is recorded with every forecast. Order-book imbalance, funding, open interest and
liquidations are NOT model inputs (no free point-in-time history exists to backtest them);
they are logged so they can be studied on forward data, and shown as evidence.
"""
from __future__ import annotations

import logging
import statistics
from typing import Optional

from cryptolab.util.timeutil import now_ms

log = logging.getLogger(__name__)


def book_metrics(book: dict, notional_usd: float = 1000.0) -> dict:
    bids, asks = book["bids"], book["asks"]
    if not bids or not asks:
        return {"ok": False}
    bb, ba = bids[0][0], asks[0][0]
    mid = (bb + ba) / 2
    out = {"ok": True, "best_bid": bb, "best_ask": ba, "mid": mid, "spread_bps": (ba - bb) / mid * 1e4,
           "fetched_ms": book.get("fetched_ms")}
    for bps in (10, 50):
        bn = sum(p * q for p, q in bids if p >= mid * (1 - bps / 1e4))
        an = sum(p * q for p, q in asks if p <= mid * (1 + bps / 1e4))
        out[f"bid_depth_{bps}bps_usd"] = bn
        out[f"ask_depth_{bps}bps_usd"] = an
        out[f"imbalance_{bps}bps"] = (bn - an) / (bn + an) if (bn + an) > 0 else 0.0

    def walk(levels, side):
        remaining, cost, qty = notional_usd, 0.0, 0.0
        for p, q in levels:
            take = min(remaining, p * q)
            cost += take
            qty += take / p
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9 or qty == 0:
            return None
        avg = cost / qty
        return (avg / mid - 1) * 1e4 if side == "buy" else (1 - avg / mid) * 1e4

    out["buy_impact_bps"] = walk(asks, "buy")    # includes half-spread
    out["sell_impact_bps"] = walk(bids, "sell")
    return out


def derivatives_context(okx, base: str, oi_cache: Optional[dict] = None) -> dict:
    inst = f"{base}-USDT-SWAP"
    out = {"source": "okx", "inst_id": inst}
    fr = okx.funding_rate(inst)
    if fr is None:
        out["available"] = False
        return out
    out["available"] = True
    out["funding_rate_8h"] = fr["funding_rate"]
    try:
        oi = oi_cache if oi_cache is not None else okx.open_interest_all()
        if inst in oi:
            out["open_interest_usd"] = oi[inst]["oi_usd"]
    except Exception as e:
        out["oi_error"] = str(e)[:100]
    liq = okx.liquidations(f"{base}-USDT")
    cutoff = now_ms() - 3600_000
    recent = [x for x in liq if x["ts_ms"] >= cutoff]
    out["liquidations_1h_count"] = len(recent)
    # long positions are liquidated with sell orders; short positions with buy orders
    out["liquidations_1h_long_contracts"] = sum((x["sz"] or 0) for x in recent if x.get("pos_side") == "long")
    out["liquidations_1h_short_contracts"] = sum((x["sz"] or 0) for x in recent if x.get("pos_side") == "short")
    return out


def cross_exchange(base: str, binance_price: float, coinbase=None, kraken_tickers=None, okx_tickers=None) -> dict:
    prices = {}
    if coinbase is not None:
        for q in ("USD", "USDT"):
            try:
                t = coinbase.ticker(f"{base}-{q}")
                if t.last:
                    prices[f"coinbase:{base}-{q}"] = t.last
                    break
            except Exception:
                continue
    for name, tickers in (("kraken", kraken_tickers), ("okx", okx_tickers)):
        if not tickers:
            continue
        for t in tickers:
            if t.base.upper() == base.upper() and t.quote.upper() in ("USD", "USDT") and t.last:
                prices[f"{name}:{t.symbol}"] = t.last
                break
    out = {"prices": prices, "binance": binance_price}
    if prices:
        med = statistics.median(prices.values())
        out["median_other"] = med
        out["binance_premium_bps"] = (binance_price / med - 1) * 1e4
        out["divergence_flag"] = abs(out["binance_premium_bps"]) > 50
    return out


def fear_greed(http) -> dict:
    """Market-wide Crypto Fear & Greed index (alternative.me, daily). Context only."""
    try:
        d = http.get("https://api.alternative.me/fng/", params={"limit": 1})["data"][0]
        return {"available": True, "value": int(d["value"]), "classification": d["value_classification"],
                "timestamp_ms": int(d["timestamp"]) * 1000, "source": "alternative.me (daily)"}
    except Exception as e:
        return {"available": False, "error": str(e)[:100]}


def _lean(v, pos, neg):
    if v is None:
        return 0
    return 1 if pos(v) else (-1 if neg(v) else 0)


def evidence(row: dict, direction: int, book: dict, deriv: dict, xex: dict,
             model: dict | None = None, sentiment: dict | None = None) -> tuple[list, list, list]:
    """Conventional readings of each indicator. Returns (supporting, contradicting, neutral)."""
    items = []

    def add(name, value, text, lean):
        items.append({"indicator": name, "value": value, "reading": text, "lean": lean})

    if model:
        add(f"Model signal ({model['strategy']})", model["z"],
            f"z = {model['z']:+.2f}, strongest of {model['n']} coins; {model['hypothesis']}", direction)
        if model.get("backtest_test_accuracy") is not None:
            add("Model backtest record (untouched test period)", model["backtest_test_accuracy"],
                f"{model['backtest_test_accuracy'] * 100:.1f}% directional accuracy, n={model['backtest_n']}", 0)
    rsi60 = row.get("rsi_60")
    if rsi60 is not None:
        add("RSI(60, 1m)", rsi60, f"{rsi60:.1f} (mean-reversion view: <40 oversold, >60 overbought)",
            _lean(rsi60, lambda v: v < 40, lambda v: v > 60))
    r60 = row.get("r_60")
    add("60-min momentum", r60, f"{r60 * 100:+.2f}% over 60 min" if r60 is not None else "n/a",
        _lean(r60, lambda v: v > 0, lambda v: v < 0))
    r10 = row.get("r_10")
    add("10-min move (short-term reversal view)", r10, f"{r10 * 100:+.2f}% over 10 min; large moves tend to partly revert",
        _lean(r10, lambda v: v < -0.003, lambda v: v > 0.003))
    mx = row.get("ma_x_60_1440")
    add("Trend MA(60) vs MA(1440)", mx, "MA60 above 24h MA" if (mx or 0) > 0 else "MA60 below 24h MA",
        _lean(mx, lambda v: v > 0, lambda v: v < 0))
    rsi = row.get("rsi_14")
    add("RSI(14, 1m)", rsi, f"{rsi:.1f} ({'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'})"
        if rsi is not None else "n/a", _lean(rsi, lambda v: v < 30, lambda v: v > 70))
    mh = row.get("macd_hist")
    add("MACD histogram", mh, "positive" if (mh or 0) > 0 else "negative", _lean(mh, lambda v: v > 0, lambda v: v < 0))
    bb = row.get("bb_pctb_60")
    add("Bollinger %B(60)", bb, f"{bb:.2f} (below lower band <0, above upper >1)" if bb is not None else "n/a",
        _lean(bb, lambda v: v < 0, lambda v: v > 1))
    brk = row.get("brk_240")
    add("4h breakout", brk, "above prior 4h high" if (brk or 0) > 0 else "below prior 4h low" if (brk or 0) < 0
        else "inside 4h range", _lean(brk, lambda v: v > 0, lambda v: v < 0))
    tbr = row.get("tbr_60")
    add("Taker buy imbalance (60m)", tbr, f"{(tbr or 0) * 100:+.1f} pp vs 50%",
        _lean(tbr, lambda v: v > 0.05, lambda v: v < -0.05))
    rv = row.get("rel_vol_60")
    if rv is not None and r60 is not None and rv > 1.5:
        add("Relative volume confirms move", rv, f"{rv:.1f}x normal volume behind the 60-min move",
            1 if r60 > 0 else -1)
    btc = row.get("btc_r_60")
    add("BTC 60-min trend", btc, f"BTC {btc * 100:+.2f}% (beta {row.get('beta_btc', 0):.2f})" if btc is not None else "n/a",
        _lean(btc, lambda v: v > 0.002, lambda v: v < -0.002))
    if book and book.get("ok"):
        imb = book.get("imbalance_10bps")
        add("Order book imbalance (+/-10 bps)", imb, f"{imb:+.2f} (bid minus ask depth share)",
            _lean(imb, lambda v: v > 0.2, lambda v: v < -0.2))
    if deriv and deriv.get("available"):
        fr = deriv.get("funding_rate_8h")
        add("Perp funding rate (OKX, contrarian)", fr, f"{fr * 100:+.4f}% per 8h" if fr is not None else "n/a",
            _lean(fr, lambda v: v < -0.0001, lambda v: v > 0.0003))
    if deriv and deriv.get("available") and (deriv.get("liquidations_1h_count") or 0) >= 3:
        ll, sl = deriv.get("liquidations_1h_long_contracts") or 0, deriv.get("liquidations_1h_short_contracts") or 0
        add("Perp liquidations last hour (OKX, contrarian)", ll - sl,
            f"{deriv['liquidations_1h_count']} liquidations: longs {ll:,.0f} vs shorts {sl:,.0f} contracts "
            "(forced selling often overshoots)", _lean(ll - sl, lambda v: v > 0, lambda v: v < 0))
    if xex and xex.get("binance_premium_bps") is not None:
        pr = xex["binance_premium_bps"]
        add("Binance premium vs other exchanges", pr, f"{pr:+.1f} bps (premiums tend to converge)",
            _lean(pr, lambda v: v < -10, lambda v: v > 10))
    if sentiment and sentiment.get("available"):
        v = sentiment["value"]
        add("Fear & Greed index (daily, contrarian)", v, f"{v} ({sentiment['classification']})",
            _lean(v, lambda x: x <= 25, lambda x: x >= 75))
    sup = [i for i in items if i["lean"] == direction and direction != 0]
    con = [i for i in items if i["lean"] == -direction and direction != 0]
    neu = [i for i in items if i["lean"] == 0]
    return sup, con, neu
