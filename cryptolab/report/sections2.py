"""Dashboard sections added in the standalone-application upgrade (all figures from records)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np

from cryptolab.backtest import stats as st
from cryptolab.report.dashboard import E, _f, ci, pct, peq, pill, price, pval, utc
from cryptolab.report.metrics import kalshi_pairs
from cryptolab.util.timeutil import MIN, now_ms, to_iso


def _table(head: list[str], rows: list[list[str]], num_from: int = 1) -> str:
    th = "".join(f"<th{' class=num' if i >= num_from else ''}>{E(h)}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(r)) +
                   "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def status_section(cfg, sup: dict, cont: dict, kill: bool, store) -> str:
    ok = sup.get("healthy")
    head = pill("worker healthy", "good") if ok else pill("worker NOT running", "crit")
    ks = pill("paper kill switch ON", "crit") if kill else pill("kill switch off", "neutral")
    ev = cont.get("events", {})
    now = now_ms()
    rows = []
    for key, exp in cfg["experiments"].items():
        name = exp["name"]
        c = cont["experiments"].get(name, {})
        cad = int(exp["cadence_min"]) * MIN
        nxt = (now // cad + 1) * cad + int(exp.get("offset_min", 0)) * MIN
        err = store.get_runtime(f"last_error:{name}")[0]
        rows.append([f"<code>{E(name)}</code>", f"{exp['horizon_min']} min",
                     ("–" if not c.get("started") else f"{c['forecasts']}"),
                     ("–" if not c.get("started") else f"{c['gaps']}"),
                     ("not started" if not c.get("started") else f"{c['coverage_pct']}%"),
                     ("–" if not c.get("started") else f"{c['longest_gap_run_minutes']} min"),
                     utc(nxt)[11:16] + " UTC", E((err or "")[:60]) or "–"])
    return f"""
<section class="panel" aria-labelledby="st-h"><h2 id="st-h">System status and continuity</h2>
<p>{head} {ks} <span class="muted small">heartbeat {_f(sup.get('heartbeat_age_s'), 0, ' s ago')} · supervisor pid {sup.get('supervisor_pid') or '–'} · worker pid {sup.get('worker_pid') or '–'}</span></p>
<p class="small">Evidence from the ledger: first start {utc(ev.get('first_start_utc'))}; hourly checkpoints in {ev.get('hours_with_checkpoint', 0)} of {ev.get('hours_since_first_start', 0)} hours ({ev.get('checkpoint_coverage_pct', '–')}%);
{ev.get('crashes_restarted', 0)} worker crashes/hangs restarted automatically; {ev.get('network_outages', 0)} network outages ({_f(ev.get('network_outage_total_s'), 0)} s total).
Continuity is only claimed for periods covered by these records.</p>
{_table(['Experiment', 'Horizon', 'Forecasts', 'Missed slots', 'Slot coverage', 'Longest gap', 'Next slot', 'Last error'], rows, 2)}
</section>"""


def prices_section(prices: list[dict]) -> str:
    if not prices:
        return ""
    cells = "".join(f'<div class="tile"><div class="tile-label">{E(p["symbol"])}</div>'
                    f'<div class="tile-value small-value">{price(p["last"])}</div>'
                    f'<div class="tile-sub">{p["change_pct"]:+.2f}% 24h · {utc(p["ts_ms"])[11:19]}</div></div>' for p in prices)
    return f'<section aria-label="Live prices"><div class="tiles prices">{cells}</div><p class="muted small">Binance spot last trade at dashboard build time (not a forecast).</p></section>'


def experiments_overview(cfg, fwd: dict) -> str:
    rows = []
    for key, exp in cfg["experiments"].items():
        s = fwd.get(key) or {}
        a = s.get("accuracy") or {}
        tc, tb = s.get("three_class"), s.get("touch")
        pl = s.get("paper_long_only") or {}
        rows.append([f"<code>{E(exp['name'])}</code>", f"{exp['horizon_min']} min", str(s.get("forecasts", 0)),
                     str(a.get("n", 0)), (f"{a['k']}/{a['n']} = {pct(a['rate'])}" if a.get("n") else "–"),
                     ci(a.get("ci95")) if a.get("n") else "–", _f(s.get("brier"), 3),
                     _f(tc["brier3"], 3) if tc else "–",
                     _f(np.mean([v["brier"] for v in tb.values()]), 3) if tb else "–",
                     (f"{pct((s.get('coverage_pi50') or {}).get('rate'), 0)}/{pct((s.get('coverage_pi80') or {}).get('rate'), 0)}"
                      if a.get("n") else "–"),
                     (f"{pl.get('trades', 0)} · {_f(pl.get('total_net_pct_of_notional'), 2, '%')}" if a.get("n") else "–"),
                     str(s.get("gaps", 0))])
    return f"""
<section class="panel" aria-labelledby="ov-h"><h2 id="ov-h">Forward experiments (5 minutes to 24 hours)</h2>
<p class="small muted">Wins/losses use the pre-registered rule (unchanged price = loss). Brier: lower is better (0.25 = coin flip for
binary; 0.667 = uniform for 3-class). Touch Brier averages the touch targets. Paper = spot long-only trades taken by the frozen forecasts.</p>
{_table(['Experiment', 'Horizon', 'Frozen', 'Scored', 'Correct', '95% CI', 'Brier', '3-class Brier', 'Touch Brier', 'PI 50/80 cover', 'Paper trades · net', 'Gaps'], rows, 2)}
</section>"""


def latest_cards(cfg, store) -> str:
    cards = []
    outs = {o["forecast_id"]: o for o in store.rows("outcomes")}
    for key, exp in cfg["experiments"].items():
        if exp.get("type") == "kalshi":
            continue
        r = store.conn.execute("SELECT payload FROM forecasts WHERE experiment=? ORDER BY seq DESC LIMIT 1",
                               (exp["name"],)).fetchone()
        if not r:
            continue
        f = json.loads(r[0])
        o = outs.get(f["forecast_id"])
        res = ("pending" if not o else ("correct" if o.get("correct") else ("unverifiable" if o["status"] == "UNVERIFIABLE" else "wrong")))
        kind = {"pending": "neutral", "correct": "good", "wrong": "crit", "unverifiable": "warn"}[res]
        tc = f.get("three_class")
        tline = ""
        if tc:
            tline = (f"<div class='small'>3-class: up {pct(tc['p_up'], 0)} · down {pct(tc['p_down'], 0)} · sideways "
                     f"{pct(tc['p_sideways'], 0)} <span class='muted'>(sideways = ±{tc['sideways_tau_pct']:.2f}%)</span></div>")
        touch = ""
        if f.get("touch"):
            touch = "<div class='small'>Touch: " + " · ".join(f"{E(k)} {pct(v['p'], 0)}" for k, v in f["touch"].items()) + "</div>"
        cards.append(f"""<div class="mini"><div class="fhead"><b>{E(exp['name'])}</b><span class="muted small">{exp['horizon_min']} min</span></div>
<div><span class="sym-s">{E(f['pair'])}</span> <span class="dir dir-{f['direction'].lower()}">{'▲' if f['direction'] == 'UP' else '▼'} {f['direction']}</span>
P(correct) {pct(f['p_direction'])} {pill(res, kind)}</div>
<div class="small mono">{price(f['p0'])} → {price(f['pred_price'])} · expires {utc(f['expiry_utc'])[11:19]}</div>{tline}{touch}</div>""")
    return f'<section class="panel" aria-labelledby="lc-h"><h2 id="lc-h">Latest frozen forecast per experiment</h2><div class="minigrid">{"".join(cards) or "<p class=muted>None yet.</p>"}</div></section>'


def g2_backtests(cfg) -> str:
    rows = []
    for key, exp in cfg["experiments"].items():
        if not key.startswith("G"):
            continue
        p = cfg.path("research_dir") / f"backtest_{exp['name']}.json"
        if not p.exists():
            rows.append([f"<code>{E(exp['name'])}</code>", "research running", "", "", "", "", "", "", ""])
            continue
        r = json.loads(p.read_text())
        t = r["summaries"]["test"][r["selected"]]
        lt = r.get("probability_layers", {}).get("test", {})
        tch = lt.get("touch", {})
        tb = np.mean([v["brier"] for v in tch.values()]) if tch else None
        tbc = np.mean([v["brier_climatology"] for v in tch.values()]) if tch else None
        v = r["test_verdict"]
        flags = " ".join(pill(k.split("_")[0], "good" if ok else "crit") for k, ok in v.items())
        rows.append([f"<code>{E(exp['name'])}</code>", f"<code>{E(r['selected'])}</code>" + (" (from " + E(r['fixed_from']) + ")" if r.get("fixed_from") else ""),
                     f"{pct(t.get('accuracy'))} ({ci(t.get('acc_ci95'))})", str(t.get("n_forecasts")),
                     _f(t.get("gross_dir_mean_bps"), 1), _f(t.get("ls_mean_bps"), 1),
                     (f"{_f(lt.get('brier3_model'), 3)} / {_f(lt.get('brier3_no_signal'), 3)} / {_f(lt.get('brier3_climatology'), 3)}" if lt else "–"),
                     (f"{_f(tb, 3)} / {_f(tbc, 3)}" if tb is not None else "–"), flags])
    return f"""<section class="panel" aria-labelledby="g2b-h"><h2 id="g2b-h">Backtests by horizon (untouched test period Oct 1 – Nov 22)</h2>
<p class="small muted">H1 accuracy &gt; 50% · H2 gross return &gt; 0 · H3 long-only net &gt; 0 · H4 3-class beats climatology · H5 every touch target beats climatology.
3-class Brier shown as model / no-signal / climatology.</p>
{_table(['Experiment', 'Model', 'Test accuracy (95% CI)', 'n', 'Gross bps', 'Net bps (every call)', '3-class Brier', 'Touch Brier model/clim.', 'Hypotheses'], rows, 2)}</section>"""


def strategies_section(cfg, store) -> str:
    p = cfg.path("research_dir") / "friend_strategies.json"
    bt_rows = []
    if p.exists():
        r = json.loads(p.read_text())
        for name, v in r["strategies"].items():
            for sp in ("validation", "test"):
                m = v.get(sp, {})
                h = m.get("h240", {})
                pt = m.get("paper_trades", {})
                rb = h.get("random_entry_baseline", {})
                bt_rows.append([f"<code>{E(name)}</code>", sp, str(m.get("signals", 0)),
                                f"{pct(h.get('hit_rate'))}" if h else "–", _f(h.get("mean_gross_bps"), 1),
                                _f(h.get("mean_net_bps"), 1), pval(rb.get("p_value_vs_random")) if rb else "–",
                                f"{pt.get('n', 0)} · {pct(pt.get('win_rate'))}" if pt else "–",
                                f"{_f(pt.get('mean_net_bps'), 1)} (t {_f(pt.get('t_stat'), 2)})" if pt else "–"])
    sigs = store.rows("signals")
    souts = {o["signal_id"]: o for o in store.rows("signal_outcomes")}
    live = {}
    for s in sigs:
        d = live.setdefault(s["strategy"], {"n": 0, "verified": [], "last": None})
        d["n"] += 1
        d["last"] = s
        o = souts.get(s["signal_id"])
        if o and o.get("r_240_pct") is not None:
            d["verified"].append(o["r_240_pct"])
    live_rows = [[f"<code>{E(k)}</code>", str(v["n"]), str(len(v["verified"])),
                  _f(np.mean(v["verified"]), 3, "%") if v["verified"] else "–",
                  f"{E(v['last']['symbol'])} {E(v['last']['side'])} {utc(v['last']['signal_utc'])[5:16]}",
                  E(v["last"]["validation_status"][:60])] for k, v in live.items()]
    state = store.get_runtime("paper_state", {})[0] or {}
    trades = store.rows("paper_trades")
    port_rows = []
    for name, pf in (state.get("portfolios") or {}).items():
        tr = [t for t in trades if t["portfolio"] == name]
        pnl = np.array([t["pnl_usd"] for t in tr])
        cum = np.cumsum(pnl) if len(pnl) else np.array([0.0])
        port_rows.append([f"<code>{E(name)}</code>", f"${pf['equity']:,.2f}", str(len(tr)),
                          pct((pnl > 0).mean()) if len(pnl) else "–", f"${pnl.sum():,.2f}" if len(pnl) else "$0.00",
                          f"${st.max_drawdown(cum):,.2f}", str(len(pf.get("positions", {}))),
                          E(", ".join(f"{k}: {v}" for k, v in (pf.get("ignored") or {}).items())[:90]) or "–"])
    return f"""<section class="panel" aria-labelledby="fs-h"><h2 id="fs-h">Friend's strategies and paper trading</h2>
<p class="small">S1 Relative-Strength Reset (ETH/BTC), S2 Breakout Retest, S3 Liquidity Recovery. Thresholds were fixed before testing.
The historical S3 uses an <b>estimated</b> spread (no historical order books) and is not an order-book validation; live S3 uses real order books but has no track record yet.</p>
<h3>Backtest (4-hour horizon and stop/target paper trades, after costs)</h3>
{_table(['Strategy', 'Split', 'Signals', '4h hit rate', '4h gross bps', '4h net bps', 'p vs random entries', 'Paper trades · win', 'Mean net bps/trade'], bt_rows, 2)}
<h3>Live signals</h3>
{_table(['Strategy', 'Signals', '24h-verified', 'Mean 4h move (direction-adjusted)', 'Latest', 'Status'], live_rows, 1) if live_rows else '<p class="muted">No live signals yet.</p>'}
<h3>Paper portfolios ($10,000 each; position ≤10% of equity, max $2,000; daily loss limit 2%; kill switch)</h3>
<p class="small muted">The gated portfolio trades only strategies that passed an out-of-sample test after costs. None has, so it stays in cash.
Shadow portfolios trade every signal of one strategy to measure it (shorts there are theoretical).</p>
{_table(['Portfolio', 'Equity', 'Closed trades', 'Win rate', 'Net P&L', 'Max drawdown', 'Open', 'Signals not taken (why)'], port_rows, 1)}
</section>"""


def kalshi_section(cfg, store) -> str:
    p = cfg.path("research_dir") / "kalshi_ab_test.json"
    test_html = "<p class='muted'>Controlled test not run yet (history download in progress).</p>"
    if p.exists():
        r = json.loads(p.read_text())
        c, m = r["comparison"], r["test_metrics"]
        rows = [[k, str(v["n"]), pct(v["accuracy"]), _f(v["brier"], 4), _f(v["log_loss"], 4), _f(v["gross_dir_bps"], 2)]
                for k, v in m.items()]
        verdict = pill("Kalshi ADOPTED: improves Brier (CI below 0)", "good") if c["adopt_kalshi"] else \
            pill("Kalshi NOT adopted: no demonstrated improvement", "neutral")
        test_html = f"""<p class="small">Test windows {E(r['test_from'][:16])} → {E((r.get('test_to') or '')[:16])} UTC, {c['n_test_windows']} BTC+ETH windows
(train {r['n_train_windows']}; {r['excluded_unusable_quote']} windows excluded from both arms for unusable quotes). {verdict}</p>
{_table(['Arm', 'n', 'Accuracy', 'Brier', 'Log loss', 'Gross bps'], rows, 1)}
<p class="small">WITH − WITHOUT Brier: {_f(c['brier_diff_with_minus_without'], 5)} (95% CI {_f(c['brier_diff_ci95'][0], 5)} to {_f(c['brier_diff_ci95'][1], 5)});
log loss diff {_f(c['logloss_diff_with_minus_without'], 5)}; McNemar {peq(c['mcnemar_p'])} ({c['mcnemar_with_better']} vs {c['mcnemar_without_better']} discordant);
gross return difference {_f(c['gross_dir_diff_bps'], 2)} bps ({peq(c['gross_dir_diff_p'])}).</p>"""
    snaps = {}
    for s in store.conn.execute("SELECT payload FROM kalshi_snapshots ORDER BY seq DESC LIMIT 6").fetchall():
        d = json.loads(s[0])
        snaps.setdefault(d["asset"], d)
    srows = []
    for a, d in sorted(snaps.items()):
        dm = d.get("direction_15m") or {}
        lad = d.get("hourly_ladder") or {}
        srows.append([a, utc(d["utc"])[11:19], E(dm.get("ticker", "–")), pct(dm.get("yes_mid")), _f((dm.get("spread") or 0) * 100, 1, "¢"),
                      _f(dm.get("volume"), 0), price((d.get("spot_reference") or {}).get("price")),
                      pct(lad.get("p_above_spot")) if lad.get("available") else "–",
                      _f(lad.get("implied_median_vs_spot_pct"), 3, "%") if lad.get("available") else "–"])
    pairs = kalshi_pairs(store)
    prow = [[a, str(v.get("paired_windows", 0)), _f(v.get("brier_without"), 4), _f(v.get("brier_with"), 4),
             pct(v.get("accuracy_without")), pct(v.get("accuracy_with"))] for a, v in pairs.items()]
    return f"""<section class="panel" aria-labelledby="k-h"><h2 id="k-h">Kalshi prediction markets (read-only research source)</h2>
<p class="small">Kalshi prices are implied probabilities of contracts that settle on CF Benchmarks' index averaged over 60 seconds; they are not
spot prices, and contracts with different expiries are never mixed. No Kalshi account is used and no Kalshi trades are made.</p>
<h3>Controlled test: same base model with vs. without Kalshi (15-minute BTC/ETH direction)</h3>{test_html}
<h3>Live market snapshot</h3>
{_table(['Asset', 'Time', '15-min contract', 'Implied P(up)', 'Spread', 'Volume', 'Spot (Coinbase)', 'Hourly: P(above spot)', 'Implied median vs spot'], srows, 1) if srows else '<p class="muted">No snapshot yet.</p>'}
<h3>Live paired arms (forward data)</h3>{_table(['Asset', 'Paired windows', 'Brier without', 'Brier with', 'Accuracy without', 'Accuracy with'], prow, 1)}
</section>"""


def news_section(cfg) -> str:
    p = cfg.path("data_dir") / "run" / "news.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    items = "".join(f"<li><a href='{E(n['link'])}' rel='noopener' target='_blank'>{E(n['title'])}</a> "
                    f"<span class='muted small'>{E(n['source'])} · {utc(n['published_utc'])}</span></li>" for n in d.get("items", [])[:12])
    llm = ""
    sp = cfg.path("data_dir") / "run" / "llm_summary.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        llm = (f"<div class='aiblock'><b>AI summary</b> <span class='muted small'>({E(s.get('model', ''))}, {utc(s.get('generated_utc'))}; "
               f"generated from the headlines below and the system's own records; not used for any prediction)</span>"
               f"<p class='small'>{E(s.get('text', ''))}</p></div>")
    return f"""<section class="panel" aria-labelledby="n-h"><h2 id="n-h">Crypto news (context only)</h2>{llm}
<ul class="news">{items}</ul><p class="small muted">Fetched {utc(d.get('fetched_utc'))} from public RSS feeds. News is not a model input.</p></section>"""
