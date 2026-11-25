"""Static research dashboard (self-contained HTML, no external data calls).

Writes:
  reports/dashboard.html           standalone document (open locally / GitHub Pages)
  reports/dashboard_fragment.html  same page without the document skeleton (for hosting
                                   services that add their own <html>/<head>/<body>)
  reports/exports/*.csv|json       full record exports
Charts are server-rendered SVG (readable without JavaScript); a small script adds a
crosshair + tooltip layer.
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from cryptolab.report.export import export_all
from cryptolab.report.metrics import equity_series, experiment_summary
from cryptolab.storage.db import open_store
from cryptolab.util.timeutil import now_ms, to_iso

E = html.escape


# --------------------------------------------------------------------------- formatting
def _f(x, d=1, suffix=""):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:,.{d}f}{suffix}"


def pct(x, d=1):
    return _f(None if x is None else x * 100, d, "%")


def price(x):
    if x is None:
        return "–"
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:.6g}"


def pval(p):
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def peq(p):
    """'p = 0.013' / 'p < 0.001' for use in sentences."""
    v = pval(p)
    return "p " + (v if v.startswith("<") else "= " + v)


def ci(c):
    if not c or c[0] is None:
        return "–"
    return f"{c[0] * 100:.1f}–{c[1] * 100:.1f}%"


def utc(ms_or_iso):
    if ms_or_iso is None:
        return "–"
    if isinstance(ms_or_iso, (int, float)):
        ms_or_iso = to_iso(int(ms_or_iso))
    return ms_or_iso.replace("T", " ").replace(".000Z", "Z")[:19] + " UTC"


def pill(text, kind="neutral"):
    icon = {"good": "✓", "warn": "!", "crit": "✕", "neutral": "•", "info": "i"}.get(kind, "•")
    return f'<span class="pill pill-{kind}"><span aria-hidden="true">{icon}</span> {E(text)}</span>'


# --------------------------------------------------------------------------- charts
def _nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=mag * 10)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


def line_chart(cid, xs, x_labels, series, y_fmt="{:.1f}%", ref_y=None, ref_label=None, markers=None,
               height=250, y_title=""):
    """series: list of dicts {name, var, ys}. One y-axis only."""
    W, H = 760, height
    L, R, T, B = 56, 128, 14, 30
    pw, ph = W - L - R, H - T - B
    vals = [y for s in series for y in s["ys"] if y is not None]
    if ref_y is not None:
        vals.append(ref_y)
    if not vals or len(xs) < 2:
        return '<p class="muted">Not enough data points to chart yet.</p>'
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 1
    ticks = _nice_ticks(lo - pad, hi + pad)
    y0, y1 = ticks[0], ticks[-1]
    xmin, xmax = xs[0], xs[-1]
    sx = lambda x: L + (x - xmin) / ((xmax - xmin) or 1) * pw
    sy = lambda y: T + (1 - (y - y0) / ((y1 - y0) or 1)) * ph
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart-svg" role="img" aria-labelledby="{cid}-t" '
             f'data-left="{L}" data-width="{pw}" data-top="{T}" data-height="{ph}">']
    for t in ticks:
        y = sy(t)
        parts.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{L - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{E(y_fmt.format(t))}</text>')
    nlab = 5
    for i in range(nlab):
        j = round(i * (len(xs) - 1) / (nlab - 1))
        x = sx(xs[j])
        anchor = "start" if i == 0 else ("end" if i == nlab - 1 else "middle")
        parts.append(f'<text x="{x:.1f}" y="{H - 8}" class="tick" text-anchor="{anchor}">{E(x_labels[j])}</text>')
    if ref_y is not None and y0 <= ref_y <= y1:
        y = sy(ref_y)
        parts.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>')
        if ref_label:
            parts.append(f'<text x="{L + pw + 6}" y="{y + 4:.1f}" class="tick">{E(ref_label)}</text>')
    for mx, mlabel in (markers or []):
        if xmin <= mx <= xmax:
            x = sx(mx)
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T + ph}" class="axis"/>')
            parts.append(f'<text x="{x + 4:.1f}" y="{T + 12}" class="tick">{E(mlabel)}</text>')
    parts.append(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" class="axis"/>')
    ends = []
    for s in series:
        pts = [(sx(x), sy(y)) for x, y in zip(xs, s["ys"]) if y is not None]
        if len(pts) < 2:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<path d="{d}" class="line" style="stroke:var({s["var"]})"/>')
        ex, ey = pts[-1]
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" class="enddot" style="fill:var({s["var"]})"/>')
        ends.append((ey, s["name"], ex))
    ends.sort()
    last = -99
    for ey, name, ex in ends:  # direct end labels only where they don't collide (legend covers the rest)
        if ey - last >= 15:
            parts.append(f'<text x="{ex + 8:.1f}" y="{ey + 4:.1f}" class="endlabel">{E(name)}</text>')
            last = ey
    parts.append(f'<line class="xhair" x1="0" x2="0" y1="{T}" y2="{T + ph}" visibility="hidden"/>')
    parts.append("</svg>")
    data = {"x": x_labels, "series": [{"name": s["name"], "var": s["var"],
                                       "v": [None if y is None else y_fmt.format(y) for y in s["ys"]]} for s in series]}
    legend = ""
    if len(series) >= 2:
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background:var({s["var"]})"></i>{E(s["name"])}</span>' for s in series) + "</div>"
    cap = f'<figcaption class="small muted">{E(y_title)}</figcaption>' if y_title else ""
    return (f'<figure class="chart" id="{cid}">{cap}{legend}<div class="chart-box">{"".join(parts)}'
            f'<div class="tip" hidden></div></div>'
            f'<script type="application/json" class="chart-data">{json.dumps(data)}</script></figure>')


# --------------------------------------------------------------------------- sections
def _load_research(cfg) -> dict:
    out = {}
    for key in ("A", "B"):
        name = cfg["experiments"][key]["name"]
        p = cfg.path("research_dir") / f"backtest_{name}.json"
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


def _latest_scan(store, name):
    rows = store.rows("scans", "experiment=?", (name,))
    return rows[-1] if rows else None


def verdict_section(research: dict, fwd: dict, cfg) -> str:
    items = []
    for key in ("A", "B"):
        r = research.get(key)
        h = cfg["experiments"][key]["horizon_min"]
        if not r:
            items.append(f"<li><b>{h}-minute backtest:</b> not run yet.</li>")
            continue
        sel = r["selected"]
        ts = r["summaries"]["test"][sel]
        v = r["test_verdict"]
        rc = r["multiple_testing"]["validation"]["reality_check_accuracy"]["p_value"]
        edge = v["H1_accuracy_gt_50"] or v["H2_gross_dir_gt_0"]
        if edge and v["H3_long_net_gt_0"]:
            line = "a statistically significant directional edge that also survived base-case costs on the untouched test period."
        elif edge:
            line = ("some statistically significant directional signal on the untouched test period, but it did "
                    "<b>not</b> survive realistic trading costs.")
        else:
            line = "<b>no statistically significant edge</b> on the untouched test period."
        items.append(
            f"<li><b>{h}-minute backtest</b> (pre-registered model <code>{E(sel)}</code>, chosen from "
            f"{r['n_strategies']} strategies on validation data): {line} Test accuracy {pct(ts.get('accuracy'))} "
            f"(95% CI {ci(ts.get('acc_ci95'))}, n={ts.get('n_forecasts')}, {peq(ts.get('p_acc_gt_50'))}); "
            f"validation reality-check {peq(rc)}.</li>")
    for key in ("A", "B"):
        s = fwd.get(key) or {}
        h = cfg["experiments"][key]["horizon_min"]
        acc = s.get("accuracy")
        if not acc or not acc.get("n"):
            nf = s.get("forecasts", 0)
            items.append(f"<li><b>{h}-minute forward test:</b> {nf} forecast{'s' if nf != 1 else ''} frozen, none scored yet.</li>")
        else:
            items.append(f"<li><b>{h}-minute forward test:</b> {acc['k']}/{acc['n']} directions correct "
                         f"({pct(acc['rate'])}, 95% CI {ci(acc['ci95'])}). {E(s.get('sample_size_note', ''))}</li>")
    return ('<section class="panel verdict" aria-labelledby="verdict-h"><h2 id="verdict-h">Bottom line</h2>'
            f'<ul>{"".join(items)}</ul><p class="muted small">Backtests are never counted as forward predictions. '
            'A predictive advantage is only claimed when the pre-registered test hypotheses pass; '
            'small forward samples are reported but not interpreted as skill.</p></section>')


def tiles_section(research, fwd, scans) -> str:
    sA, sB = scans.get("A"), scans.get("B")
    s = sA or sB
    n_strat = sum(r.get("n_strategies", 0) for r in research.values())
    tiles = []
    if s:
        tiles.append(("Markets screened (last scan)", f"{s['markets_discovered']:,}",
                      f"{len(s['per_exchange'])} exchanges · {s['markets_quality_passed']:,} passed quality checks · "
                      f"{s['markets_analyzed']} analyzed for prediction"))
    else:
        tiles.append(("Markets screened", "–", "no live scan recorded yet"))
    import glob
    full = 0
    for fp in glob.glob(str(Path(__file__).resolve().parents[2] / "data" / "research" / "backtest_*.json")):
        try:
            d = json.loads(Path(fp).read_text())
            if not d.get("fixed_from"):
                full += d.get("n_strategies", 0)
        except Exception:
            pass
    tiles.append(("Strategy backtests", f"{full or n_strat}",
                  "strategy × horizon walk-forward backtests (100–101 configurations at 10m, 60m, 5m, 4h, 24h) "
                  "+ 3 friend strategies + baselines"))
    for key, label in (("A", "10-min forecasts"), ("B", "60-min forecasts")):
        f = fwd.get(key) or {}
        acc = f.get("accuracy") or {}
        sub = (f"{acc.get('k', 0)} correct / {acc.get('n', 0)} scored · {f.get('status_counts', {}).get('PENDING', 0)} pending"
               f" · {f.get('gaps', 0)} gaps")
        tiles.append((label, f"{f.get('forecasts', 0)}", sub))
    return '<section class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-label">{E(a)}</div><div class="tile-value">{E(b)}</div>'
        f'<div class="tile-sub">{E(c)}</div></div>' for a, b, c in tiles) + "</section>"


def _forecast_card(fc: dict, out: dict | None) -> str:
    if not fc:
        return '<p class="muted">No forecast frozen yet.</p>'
    d = fc["direction"]
    status = pill("pending — expires " + utc(fc["expiry_utc"]), "neutral")
    if out:
        if out["status"] == "UNVERIFIABLE":
            status = pill("unverifiable", "warn")
        else:
            status = pill(f"{'correct' if out['correct'] else 'wrong'}: actual {out['actual_direction']} "
                          f"{out['actual_return_pct']:+.3f}% → {price(out['p1'])}", "good" if out["correct"] else "crit")
            if out["status"] == "APPROXIMATE":
                status += " " + pill("approximate verification", "warn")
    ev_for = "".join(f"<li>{E(i['indicator'])}: {E(str(i['reading']))}</li>" for i in fc["evidence_for"]) or "<li>none</li>"
    ev_ag = "".join(f"<li>{E(i['indicator'])}: {E(str(i['reading']))}</li>" for i in fc["evidence_against"]) or "<li>none</li>"
    c = fc["costs"]
    return f"""
<div class="fcard">
  <div class="fhead"><span class="sym">{E(fc['pair'])}</span><span class="dir dir-{d.lower()}">{'▲' if d == 'UP' else '▼'} {d}</span>
  <span class="muted">test #{fc['test_number']} · {E(fc['exchange'])}</span></div>
  <div class="status-row">{status}</div>
  <dl class="kv">
    <dt>Frozen at</dt><dd class="mono">{utc(fc['origin_utc'])}</dd>
    <dt>Expires</dt><dd class="mono">{utc(fc['expiry_utc'])}</dd>
    <dt>Reference trade</dt><dd class="mono">{price(fc['p0'])} @ {utc(fc['ref_trade_utc'])}</dd>
    <dt>Predicted price</dt><dd class="mono">{price(fc['pred_price'])} ({fc['expected_move_pct']:+.3f}%)</dd>
    <dt>P(direction correct)</dt><dd class="mono">{pct(fc['p_direction'])} <span class="muted small">(backtest-calibrated, forward-unvalidated)</span></dd>
    <dt>50% range</dt><dd class="mono">{price(fc['pi50'][0])} – {price(fc['pi50'][1])}</dd>
    <dt>80% range</dt><dd class="mono">{price(fc['pi80'][0])} – {price(fc['pi80'][1])}</dd>
    <dt>Round-trip cost</dt><dd class="mono">{c['round_trip'] * 100:.3f}% <span class="muted small">(fee {c['fee_per_side'] * 100:.2f}%/side, half-spread {c['half_spread_measured'] * 1e4:.2f} bp, slippage {c['slippage_per_side'] * 1e4:.2f} bp)</span></dd>
    <dt>Expected net (long)</dt><dd class="mono">{fc['expected_net_long_pct']:+.3f}%</dd>
    <dt>Trading opportunity?</dt><dd>{E(fc['opportunity_label'])}</dd>
    <dt>Coins examined</dt><dd class="mono">{fc['n_examined']} analyzed · {fc['scan_counts']['markets_discovered']:,} markets screened</dd>
    <dt>Model</dt><dd class="mono">{E(fc['model_version'])}</dd>
  </dl>
  <div class="evidence">
    <div><h4>Evidence for</h4><ul>{ev_for}</ul></div>
    <div><h4>Evidence against</h4><ul>{ev_ag}</ul></div>
  </div>
</div>"""


def experiment_section(key, cfg, store, fwd, research) -> str:
    exp = cfg["experiments"][key]
    name, h = exp["name"], exp["horizon_min"]
    s = fwd.get(key) or {}
    fcs = store.rows("forecasts", "experiment=?", (name,))
    outs = {o["forecast_id"]: o for o in store.rows("outcomes", "experiment=?", (name,))}
    latest = fcs[-1] if fcs else None
    acc = s.get("accuracy") or {}
    bms = s.get("benchmarks_accuracy") or {}
    rows = []
    if acc.get("n"):
        rows.append(("This model", acc))
        for k, v in bms.items():
            rows.append((k.replace("_", " "), v))
    bm_table = ""
    if rows:
        bm_table = ('<div class="tablewrap"><table><thead><tr><th>Direction rule (same windows)</th><th>Correct</th>'
                    '<th>Accuracy</th><th>95% CI</th><th>p vs 50%</th></tr></thead><tbody>' +
                    "".join(f"<tr><td>{E(n)}</td><td class='num'>{v['k']}/{v['n']}</td><td class='num'>{pct(v['rate'])}</td>"
                            f"<td class='num'>{ci(v['ci95'])}</td><td class='num'>{pval(v['p_vs_50'])}</td></tr>"
                            for n, v in rows) + "</tbody></table></div>")
    metrics = ""
    if acc.get("n"):
        pl = s["paper_long_only"]
        hd = s["hypothetical_direction_bets"]
        metrics = f"""
<dl class="kv compact">
  <dt>Mean abs. price error</dt><dd class="num">{_f(s['mean_abs_pred_error_pct'], 3, '%')} vs no-change {_f(s['mean_no_change_error_pct'], 3, '%')} (ratio {_f(s['error_ratio_vs_no_change'], 3)})</dd>
  <dt>50% / 80% interval coverage</dt><dd class="num">{pct(s['coverage_pi50']['rate'])} / {pct(s['coverage_pi80']['rate'])}</dd>
  <dt>Brier score (skill vs 0.5)</dt><dd class="num">{_f(s['brier'], 4)} ({_f(s['brier_skill_vs_0.5'] * 100, 1, '%')})</dd>
  <dt>Balanced accuracy</dt><dd class="num">{pct(s.get('balanced_accuracy'))}</dd>
  <dt>Paper spot long-only</dt><dd class="num">{pl['trades']} trades · net {_f(pl['total_net_pct_of_notional'], 3, '%')} of notional · max DD {_f(pl['max_drawdown_pct_of_notional'], 3, '%')}</dd>
  <dt>Hypothetical direction bets</dt><dd class="num">gross {_f(hd['gross_total_pct'], 3, '%')} · net {_f(hd['net_total_pct'], 3, '%')} ({hd['short_count']} theoretical shorts)</dd>
  <dt>Buy &amp; hold same windows</dt><dd class="num">gross {_f(s['benchmarks_returns']['buy_and_hold_same_windows_gross_pct'], 3, '%')} · net {_f(s['benchmarks_returns']['buy_and_hold_same_windows_net_pct'], 3, '%')}</dd>
  <dt>Win / loss streaks</dt><dd class="num">{s['max_win_streak']} / {s['max_loss_streak']}</dd>
  <dt>Flat / approximate outcomes</dt><dd class="num">{s['flat_outcomes']} / {s['approximate_outcomes']}</dd>
</dl>"""
    eq = equity_series(store, name)
    chart = ""
    if len(eq["test_number"]) >= 2:
        chart = line_chart(f"fwd-{key}", eq["test_number"], [f"#{x}" for x in eq["test_number"]],
                           [{"name": "Cumulative accuracy", "var": "--s1", "ys": eq["accuracy_cum_pct"]}],
                           y_fmt="{:.0f}%", ref_y=50, ref_label="50% chance", height=200)
    recent = []
    for f in reversed(fcs[-12:]):
        o = outs.get(f["forecast_id"])
        if o and o["status"] != "UNVERIFIABLE":
            res = pill("correct", "good") if o["correct"] else pill("wrong", "crit")
            act = f"{price(o['p1'])} ({o['actual_return_pct']:+.3f}%)"
        elif o:
            res, act = pill("unverifiable", "warn"), "–"
        else:
            res, act = pill("pending", "neutral"), "–"
        recent.append(f"<tr><td class='num'>{f['test_number']}</td><td class='mono'>{utc(f['origin_utc'])[11:]}</td>"
                      f"<td>{E(f['pair'])}</td><td>{f['direction']}</td><td class='num'>{price(f['p0'])}</td>"
                      f"<td class='num'>{price(f['pred_price'])}</td><td class='num'>{pct(f['p_direction'])}</td>"
                      f"<td class='num'>{act}</td><td>{res}</td></tr>")
    recent_table = ('<div class="tablewrap"><table><thead><tr><th>#</th><th>Frozen (UTC)</th><th>Pair</th><th>Call</th>'
                    '<th>Price</th><th>Predicted</th><th>P(dir)</th><th>Actual</th><th>Result</th></tr></thead><tbody>'
                    + "".join(recent) + "</tbody></table></div>") if recent else ""
    gaps = s.get("gap_reasons") or {}
    gap_txt = ", ".join(f"{v}× {E(k)}" for k, v in gaps.items()) or "none"
    model = (research.get(key) or {}).get("live_model", {})
    return f"""
<section class="panel exp" aria-labelledby="exp-{key}">
  <h2 id="exp-{key}">Experiment {key} · {h}-minute forecasts</h2>
  <p class="muted small">Model <code>{E(model.get('version', '–'))}</code> — {E(model.get('hypothesis', ''))}
  Started {utc(s.get('first_forecast_utc'))}. Gaps (slots without a forecast): {gap_txt}.</p>
  <h3>Latest frozen forecast</h3>
  {_forecast_card(latest, outs.get(latest['forecast_id']) if latest else None)}
  <h3>Forward results</h3>
  {metrics or '<p class="muted">No forecast has expired and been verified yet.</p>'}
  {bm_table}
  {chart}
  <h3>Recent forecasts</h3>
  {recent_table or '<p class="muted">None yet.</p>'}
</section>"""


def backtest_section(key, r) -> str:
    if not r:
        return ""
    h = r["horizon_min"]
    sel = r["selected"]
    S = r["summaries"]
    base = [sel, f"mom_{h}", f"rev_{h}", "baseline_random", "baseline_always_up", "baseline_always_down"]
    rows = []
    for n in base:
        if n not in S["test"]:
            continue
        v, t = S["validation"][n], S["test"][n]
        tag = " (selected)" if n == sel else (" (simple momentum)" if n == f"mom_{h}" else
                                              " (simple mean reversion)" if n == f"rev_{h}" else "")
        rows.append(f"<tr{' class=sel' if n == sel else ''}><td><code>{E(n)}</code>{E(tag)}</td>"
                    f"<td class='num'>{pct(v.get('balanced_accuracy'))}</td><td class='num'>{pct(t.get('accuracy'))}</td>"
                    f"<td class='num'>{ci(t.get('acc_ci95'))}</td><td class='num'>{pval(t.get('p_acc_gt_50'))}</td>"
                    f"<td class='num'>{_f(t.get('gross_dir_mean_bps'), 2)}</td><td class='num'>{_f(t.get('ls_mean_bps'), 1)}</td>"
                    f"<td class='num'>{t.get('long_trades', 0)}</td><td class='num'>{_f(t.get('long_total_pct'), 2, '%')}</td>"
                    f"<td class='num'>{_f(t.get('ic_mean'), 4)}</td><td class='num'>{_f(t.get('mae_ratio_vs_no_change'), 3)}</td>"
                    f"<td class='num'>{pct(t.get('cover50'), 0)}/{pct(t.get('cover80'), 0)}</td></tr>")
    comp = ('<div class="tablewrap"><table><thead><tr><th>Strategy</th><th>Val. bal. acc.</th><th>Test acc.</th>'
            '<th>Test 95% CI</th><th>p vs 50%</th><th>Gross bps/fcst</th><th>Net L/S bps</th><th>Long trades</th>'
            '<th>Long net total</th><th>Test IC</th><th>Error vs no-change</th><th>PI 50/80 cover</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")
    ranked = sorted(((n, m) for n, m in S["validation"].items() if not n.startswith("baseline_")),
                    key=lambda kv: kv[1].get("balanced_accuracy") or 0, reverse=True)
    top = ranked[:8]
    worst = ranked[-5:]
    hp = r["multiple_testing"]["validation"]["holm_p_accuracy"]

    def lrow(n, m):
        t = S["test"].get(n, {})
        elig = "" if (m.get("coverage", 0) >= 0.9 and m.get("n_forecasts", 0) >= 300) else " " + pill("ineligible: coverage", "neutral")
        return (f"<tr{' class=sel' if n == sel else ''}><td><code>{E(n)}</code>{elig}</td><td>{E(m.get('family', ''))}</td>"
                f"<td class='num'>{pct(m.get('balanced_accuracy'))}</td><td class='num'>{pct(m.get('accuracy'))}</td>"
                f"<td class='num'>{pval(hp.get(n))}</td><td class='num'>{pct(t.get('balanced_accuracy'))}</td>"
                f"<td class='num'>{pct(t.get('accuracy'))}</td><td class='num'>{_f(m.get('coverage', 0) * 100, 0, '%')}</td></tr>")
    lead = ('<div class="tablewrap"><table><thead><tr><th>Strategy</th><th>Family</th><th>Val. bal. acc.</th>'
            '<th>Val. acc.</th><th>Holm p (val.)</th><th>Test bal. acc.</th><th>Test acc.</th><th>Coverage</th></tr>'
            '</thead><tbody>' + "".join(lrow(n, m) for n, m in top) +
            '<tr class="sep"><td colspan="8">… lowest five …</td></tr>' + "".join(lrow(n, m) for n, m in worst) +
            "</tbody></table></div>")
    mt = r["multiple_testing"]
    dsr = r["prereg"]["validation_deflated_sharpe_selected"]
    eqc = r.get("equity_curves", {})
    chart = ""
    names = [(sel, "--s1", f"{sel} (selected)"), (f"mom_{h}", "--s2", f"mom_{h} (simple momentum)"),
             (f"rev_{h}", "--s3", f"rev_{h} (simple reversal)"), ("baseline_random", "--s4", "random")]
    avail = [(n, v, lab) for n, v, lab in names if n in eqc]
    if avail:
        base_t = eqc[avail[0][0]]["t"]
        series = []
        for n, v, lab in avail:
            m = dict(zip(eqc[n]["t"], eqc[n]["gross_dir_pct"]))
            series.append({"name": lab, "var": v, "ys": [m.get(t) for t in base_t]})
        xl = [datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for t in base_t]
        markers = []
        for split, lab in (("validation", "validation →"), ("test", "test →")):
            ts = [t for t, sp in zip(base_t, eqc[avail[0][0]]["split"]) if sp == split]
            if ts:
                markers.append((ts[0], lab))
        chart = line_chart(f"bt-{key}", base_t, xl, series, y_fmt="{:.0f}%", markers=markers, height=260,
                           y_title="cumulative gross directional return (before costs)")
    bh = r["buy_and_hold"]
    reg = r.get("selected_breakdowns_val_test", {}).get("by_regime", [])
    reg_rows = "".join(f"<tr><td>{E(x['regime'])}</td><td class='num'>{x['n']}</td><td class='num'>{pct(x['accuracy'])}</td>"
                       f"<td class='num'>{_f(x['gross_dir_bps'], 2)}</td></tr>" for x in reg)
    sp = r["splits"]
    return f"""
<section class="panel" aria-labelledby="bt-{key}-h">
  <h2 id="bt-{key}-h">Backtest · {h}-minute horizon</h2>
  <p class="small">Walk-forward on Binance 1-minute data, point-in-time top-30 universe ({r['coins_in_universe_union']} coins over time incl. delisted),
  {r['decision_times']:,} decision times. Train {sp['train']['from'][:10]} → {sp['train']['to'][:10]},
  validation {sp['validation']['from'][:10]} → {sp['validation']['to'][:10]},
  <b>untouched test</b> {sp['test']['from'][:10]} → {sp['test']['to'][:10]}. {r['n_strategies']} strategies + baselines;
  each decision picks the coin with the strongest signal and scores its direction {h} minutes later.</p>
  <p class="small">Pre-registered selection: <code>{E(sel)}</code> — {E(r['prereg']['selection_rule'])}
  Validation multiple-testing: White reality-check {peq(mt['validation']['reality_check_accuracy']['p_value'])}
  (best: <code>{E(mt['validation']['reality_check_accuracy']['best'])}</code>), Holm-adjusted {peq(mt['validation']['holm_p_accuracy'].get(sel))}
  for the selected model; deflated-Sharpe probability of its after-cost returns {pct(dsr.get('dsr_prob'))}.
  Test verdict: {pill('accuracy > 50%: ' + ('pass' if r['test_verdict']['H1_accuracy_gt_50'] else 'fail'), 'good' if r['test_verdict']['H1_accuracy_gt_50'] else 'crit')}
  {pill('gross return > 0: ' + ('pass' if r['test_verdict']['H2_gross_dir_gt_0'] else 'fail'), 'good' if r['test_verdict']['H2_gross_dir_gt_0'] else 'crit')}
  {pill('net long-only > 0: ' + ('pass' if r['test_verdict']['H3_long_net_gt_0'] else 'fail'), 'good' if r['test_verdict']['H3_long_net_gt_0'] else 'crit')}</p>
  {chart}
  <p class="small"><b>Costs change the picture.</b> The curves above are before costs. With base-case costs
  (about {_f(r['summaries']['test'][sel].get('gross_dir_mean_bps', 0) - r['summaries']['test'][sel].get('ls_mean_bps', 0), 0)} bps per
  round trip) the selected model loses {_f(-r['summaries']['test'][sel].get('ls_mean_bps', 0), 1)} bps per forecast on the
  test period if every call is traded; its expected moves never exceeded costs, so the spot long-only rule took
  {r['summaries']['test'][sel].get('long_trades', 0)} trades.</p>
  <h3>Selected model vs. baselines on the test period</h3>
  {comp}
  <p class="small muted">Buy &amp; hold over the test period: BTC {_f(bh['test']['btc_buy_hold_pct'], 1, '%')},
  equal-weight universe {_f(bh['test']['equal_weight_universe_pct'], 1, '%')} (validation: BTC {_f(bh['validation']['btc_buy_hold_pct'], 1, '%')}).
  Net figures use Binance taker fees 0.10%/side plus spread and slippage assumptions.</p>
  <h3>Strategy leaderboard (ranked on validation, test shown for reference)</h3>
  {lead}
  <h3>Selected model by market condition (validation + test)</h3>
  <div class="tablewrap"><table><thead><tr><th>Regime</th><th>n</th><th>Accuracy</th><th>Gross bps</th></tr></thead><tbody>{reg_rows}</tbody></table></div>
</section>"""


def ops_section(cfg, store, scans, daemon, audit) -> str:
    rows = []
    s = scans.get("A") or scans.get("B")
    if s:
        for ex, v in s["per_exchange"].items():
            if v.get("ok"):
                rows.append(f"<tr><td>{E(ex)}</td><td>{pill('ok', 'good')}</td><td class='num'>{v['discovered']:,}</td>"
                            f"<td class='num'>{v['quality_passed']:,}</td><td class='num'>{v['seconds']}</td></tr>")
            else:
                rows.append(f"<tr><td>{E(ex)}</td><td>{pill('failed', 'crit')}</td><td colspan='3'>{E(v.get('error', ''))}</td></tr>")
    rej = ", ".join(f"{E(k)} {v:,}" for k, v in (s or {}).get("reject_reasons", {}).items())
    warn = (s or {}).get("warnings", [])
    hb = daemon.get("heartbeat") or {}
    alive = daemon.get("process_alive")
    age = daemon.get("last_heartbeat_age_s")
    wstate = pill("running", "good") if alive and age is not None and age < 120 else pill("not running", "warn")
    fails = hb.get("http_failures") or {}
    led = audit["ledger"]
    return f"""
<section class="panel" aria-labelledby="ops-h">
  <h2 id="ops-h">Workers, data quality and integrity</h2>
  <dl class="kv compact">
    <dt>Local worker</dt><dd>{wstate} <span class="muted small">pid {daemon.get('pid') or '–'}, last heartbeat {_f(age, 0, ' s ago') if age is not None else '–'}</span></dd>
    <dt>Ledger</dt><dd>{pill('hash chain intact', 'good') if audit['ok'] else pill('AUDIT FAILED', 'crit')} <span class="mono small">{led['records']} records · head {E(led['head'][:16])}…</span></dd>
    <dt>HTTP failures (this worker)</dt><dd class="mono small">{E(json.dumps(fails)) if fails else 'none'}</dd>
    <dt>Scan warnings</dt><dd class="small">{E('; '.join(warn)) if warn else 'none'}</dd>
    <dt>Quality rejections</dt><dd class="small">{rej or '–'}</dd>
  </dl>
  <div class="tablewrap"><table><thead><tr><th>Exchange</th><th>Status</th><th>Markets</th><th>Passed checks</th><th>Seconds</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>
</section>"""


METHOD = """
<section class="panel small" aria-labelledby="m-h">
  <h2 id="m-h">Method and limitations</h2>
  <ul>
    <li><b>Data.</b> Public, read-only exchange endpoints only. Forecasts and verification use Binance spot (public market-data mirror);
    Coinbase, Kraken, OKX, KuCoin, Gate.io, MEXC, Bitfinex and Bitstamp are scanned for counts, quality and cross-exchange price checks;
    OKX supplies perpetual funding, open interest and liquidations as context. No account, no keys, no orders.</li>
    <li><b>Frozen forecasts.</b> Each forecast is written once to SQLite (UPDATE/DELETE blocked by triggers) and to an append-only
    SHA-256 hash-chained ledger; restarts cannot duplicate a slot.</li>
    <li><b>Verification.</b> Last trade on the same pair at or before the exact expiry instant (Binance aggTrades). Wider windows or candle
    closes are flagged approximate. Identical start/end prices count as an incorrect direction.</li>
    <li><b>Probabilities</b> are calibrated on walk-forward backtest forecasts and are not yet validated on forward data.</li>
    <li><b>Shorts</b> are theoretical (margin/borrow); spot results are long-or-cash only.</li>
    <li><b>Not used as model inputs:</b> order-book imbalance, funding, open interest, liquidations, news, social sentiment and on-chain data
    (no free point-in-time history to backtest them). The first three are recorded with every forecast for future study.</li>
    <li><b>Limits.</b> Backtest costs are assumptions (retail taker fees); one exchange; ~11 months of history; regime changes can erase
    any historical pattern. Nothing here is investment advice.</li>
  </ul>
</section>"""

CSS = """
:root{
  /* Layout: audit-report column — verdict first, then experiments, backtests, integrity. */
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --border:rgba(11,11,11,0.10); --accent:#2a78d6; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#898781;
  --good:#0ca30c; --good-ink:#006300; --warn:#fab219; --crit:#d03b3b; --sel:rgba(42,120,214,0.08);
  --f-display:"Space Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-body:"Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono:"JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,0.10); --accent:#3987e5; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#898781;
  --good:#0ca30c; --good-ink:#0ca30c; --warn:#fab219; --crit:#e66767; --sel:rgba(57,135,229,0.14); color-scheme:dark; } }
:root[data-theme="dark"]{
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,0.10); --accent:#3987e5; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#898781;
  --good:#0ca30c; --good-ink:#0ca30c; --warn:#fab219; --crit:#e66767; --sel:rgba(57,135,229,0.14); color-scheme:dark; }
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:14px;line-height:1.5;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-block:20px 48px;padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px));display:grid;gap:16px}
header{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:8px 24px}
h1{font-family:var(--f-display);font-size:1.75rem;letter-spacing:-0.01em;margin:0;text-wrap:balance}
h2{font-family:var(--f-display);font-size:1.2rem;margin:0 0 8px;text-wrap:balance}
h3{font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-2);margin:18px 0 8px}
h4{font-size:.8rem;margin:0 0 4px;color:var(--ink-2)}
code,.mono{font-family:var(--f-mono);font-size:.86em}
.muted{color:var(--muted)} .small{font-size:.86rem} p{margin:6px 0;max-width:110ch}
.sub{color:var(--ink-2)}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px;min-width:0}
.verdict{border-left:3px solid var(--accent)} .verdict ul{margin:0;padding-left:18px;display:grid;gap:6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;min-width:0}
.tile-label{font-size:.78rem;color:var(--ink-2)} .tile-value{font-family:var(--f-display);font-size:1.9rem;font-weight:600;font-variant-numeric:tabular-nums}
.tile-sub{font-size:.8rem;color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px;align-items:start}
.tablewrap{overflow-x:auto;margin:6px 0}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis);padding:6px 8px;white-space:nowrap}
td{border-bottom:1px solid var(--grid);padding:5px 8px;white-space:nowrap} td.num,th.num{text-align:right}
tr.sel td{background:var(--sel)} tr.sep td{color:var(--muted);text-align:center;font-size:.8rem}
.kv{display:grid;grid-template-columns:minmax(120px,auto) 1fr;gap:4px 14px;margin:6px 0}
.kv dt{color:var(--ink-2)} .kv dd{margin:0;min-width:0;overflow-wrap:anywhere} .kv.compact{font-size:.86rem} .kv dd .muted,.kv dd .small{font-family:var(--f-body)}
.pill{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--border);border-radius:999px;padding:1px 8px;font-size:.78rem;white-space:nowrap}
.pill-good{border-color:var(--good);color:var(--good-ink)} .pill-crit{border-color:var(--crit);color:var(--crit)}
.pill-warn{border-color:var(--warn)} .pill-neutral{color:var(--ink-2)}
.fcard{border:1px solid var(--border);border-radius:8px;padding:12px;display:grid;gap:8px}
.fhead{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline}
.sym{font-family:var(--f-display);font-size:1.3rem;font-weight:600}
.dir{font-weight:700} .dir-up{color:var(--good-ink)} .dir-down{color:var(--crit)}
.evidence{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;font-size:.84rem}
.evidence ul{margin:0;padding-left:16px}
.chart{margin:10px 0} .chart-box{position:relative}
.chart-svg{width:100%;height:auto;display:block;overflow:visible}
.chart-svg .grid{stroke:var(--grid);stroke-width:1} .chart-svg .axis{stroke:var(--axis);stroke-width:1}
.chart-svg .ref{stroke:var(--axis);stroke-width:1}
.chart-svg .tick{fill:var(--muted);font-size:11px;font-family:var(--f-body)}
.chart-svg .endlabel{fill:var(--ink-2);font-size:11px;font-family:var(--f-body)}
.chart-svg .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart-svg .enddot{stroke:var(--surface);stroke-width:2}
.chart-svg .xhair{stroke:var(--ink-2);stroke-width:1}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.8rem;color:var(--ink-2);margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:6px;
  padding:6px 8px;font-size:.78rem;box-shadow:0 2px 8px rgba(0,0,0,.12);min-width:140px;z-index:2}
.tip b{font-variant-numeric:tabular-nums} .tip i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
footer{color:var(--muted);font-size:.8rem}
.minigrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:10px}
.mini{border:1px solid var(--border);border-radius:8px;padding:10px;display:grid;gap:4px;min-width:0}
.sym-s{font-family:var(--f-display);font-weight:600}
.tile-value.small-value{font-size:1.3rem}
.tiles.prices{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.aiblock{border-left:3px solid var(--accent);padding:4px 10px;margin:6px 0}
ul.news{margin:0;padding-left:18px;display:grid;gap:4px} ul.news a{color:var(--ink)}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}
"""

JS = """
(function(){
  document.querySelectorAll('figure.chart').forEach(function(fig){
    var svg = fig.querySelector('svg'), tip = fig.querySelector('.tip'), dataEl = fig.querySelector('.chart-data');
    if(!svg || !tip || !dataEl) return;
    var data = JSON.parse(dataEl.textContent), xh = svg.querySelector('.xhair');
    var L = +svg.dataset.left, W = +svg.dataset.width, n = data.x.length, vb = svg.viewBox.baseVal;
    function show(evt){
      var r = svg.getBoundingClientRect(), sx = vb.width / r.width;
      var x = (evt.clientX - r.left) * sx, i = Math.round((x - L) / W * (n - 1));
      if(i < 0 || i >= n){ hide(); return; }
      var px = L + i / (n - 1) * W;
      xh.setAttribute('x1', px); xh.setAttribute('x2', px); xh.setAttribute('visibility', 'visible');
      while(tip.firstChild) tip.removeChild(tip.firstChild);
      var head = document.createElement('div'); head.textContent = data.x[i]; tip.appendChild(head);
      data.series.forEach(function(s){
        if(s.v[i] == null) return;
        var row = document.createElement('div'), key = document.createElement('i'), val = document.createElement('b');
        key.style.background = 'var(' + s.var + ')'; val.textContent = s.v[i];
        row.appendChild(key); row.appendChild(val); row.appendChild(document.createTextNode(' ' + s.name));
        tip.appendChild(row);
      });
      tip.hidden = false;
      var left = px / sx + 12; if(left + tip.offsetWidth > r.width) left = px / sx - tip.offsetWidth - 12;
      tip.style.left = Math.max(0, left) + 'px'; tip.style.top = '8px';
    }
    function hide(){ tip.hidden = true; xh.setAttribute('visibility', 'hidden'); }
    svg.addEventListener('pointermove', show); svg.addEventListener('pointerleave', hide);
  });
})();
"""


def _live_prices(cfg, store) -> list[dict]:
    try:
        from cryptolab.util.http import HttpClient
        syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ETHBTC"]
        for r in store.conn.execute("SELECT DISTINCT symbol FROM forecasts ORDER BY seq DESC LIMIT 6").fetchall():
            if r[0] not in syms:
                syms.append(r[0])
        data = HttpClient(timeout_s=8, max_retries=1).get(f"{cfg['exchanges']['binance_base']}/api/v3/ticker/24hr",
                                                          params={"symbols": json.dumps(syms[:8], separators=(',', ':'))})
        return [{"symbol": d["symbol"], "last": float(d["lastPrice"]), "change_pct": float(d["priceChangePercent"]),
                 "ts_ms": int(d["closeTime"])} for d in data]
    except Exception:
        return []


def build_all(cfg) -> dict:
    from cryptolab.live.scheduler import daemon_status
    from cryptolab.live.supervisor import status as sup_status
    from cryptolab.live.paper import kill_switch_file
    from cryptolab.report import sections2 as S2
    from cryptolab.report.metrics import continuity
    store = open_store(cfg)
    research = _load_research(cfg)
    fwd = {k: experiment_summary(store, v["name"]) for k, v in cfg["experiments"].items()}
    scans = {k: _latest_scan(store, cfg["experiments"][k]["name"]) for k in ("A", "B")}
    try:
        daemon = daemon_status(cfg)
    except Exception:
        daemon = {}
    audit = store.audit()
    gen = to_iso(now_ms())
    body = f"""
<div class="wrap">
<header><div><h1>Claude Crypto Research Lab</h1>
<div class="sub small">Independent forecasting experiments on public crypto market data · paper trading only · generated {utc(gen)}</div></div>
<div>{pill('no real trading', 'info')} {pill('ledger intact' if audit['ok'] else 'ledger audit failed', 'good' if audit['ok'] else 'crit')}</div></header>
{verdict_section(research, fwd, cfg)}
{S2.prices_section(_live_prices(cfg, store))}
{S2.status_section(cfg, sup_status(cfg), continuity(store, cfg), kill_switch_file(cfg).exists(), store)}
{tiles_section(research, fwd, scans)}
{S2.experiments_overview(cfg, fwd)}
{S2.latest_cards(cfg, store)}
{S2.strategies_section(cfg, store)}
{S2.kalshi_section(cfg, store)}
{S2.g2_backtests(cfg)}
<div class="grid2">{experiment_section('A', cfg, store, fwd, research)}{experiment_section('B', cfg, store, fwd, research)}</div>
{backtest_section('A', research.get('A'))}
{backtest_section('B', research.get('B'))}
{ops_section(cfg, store, scans, daemon, audit)}
{S2.news_section(cfg)}
{METHOD}
<footer>Records: SQLite + hash-chained JSONL ledger (data/ledger). Exports: reports/exports. Generated by the cryptolab package; figures are recomputed from immutable records on every build.</footer>
</div>"""
    head = ('<title>Claude Crypto Research Lab</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700'
            '&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;600&display=swap">\n'
            f"<style>{CSS}</style>\n")
    out_dir = cfg.path("reports_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    fragment = head + body + f"\n<script>{JS}</script>\n"
    (out_dir / "dashboard_fragment.html").write_text(fragment, encoding="utf-8")
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "</head>\n<body>\n" + body + f"\n<script>{JS}</script>\n</body>\n</html>\n")
    (out_dir / "dashboard.html").write_text(full, encoding="utf-8")
    summary = {"generated_utc": gen, "forward": fwd,
               "backtest": {k: {"selected": v["selected"], "test_verdict": v["test_verdict"],
                                "test_selected": v["summaries"]["test"][v["selected"]]} for k, v in research.items()},
               "audit_ok": audit["ok"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    ex = export_all(store, out_dir / "exports")
    return {"dashboard": str(out_dir / "dashboard.html"), "exports": ex, "audit_ok": audit["ok"]}
