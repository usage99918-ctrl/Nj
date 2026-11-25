"""News headlines (public RSS) and an OPTIONAL AI summary.

News is context only: it is never a model input and never triggers a trade.

The AI summary is disabled by default ([llm].enabled = false). It is a paid service: it runs only
when you enable it AND provide Anthropic API credentials (e.g. ANTHROPIC_API_KEY), and it needs the
optional package from requirements-llm.txt. The model receives text only -- headlines with their
sources/timestamps and a summary of the system's own records -- and has no tools, so it cannot
place orders or change anything. Every summary is saved with its model name and timestamp.
"""
from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

from cryptolab.util.timeutil import now_ms, to_iso

log = logging.getLogger(__name__)
FEEDS = {"Cointelegraph": "https://cointelegraph.com/rss", "Decrypt": "https://decrypt.co/feed",
         "The Block": "https://www.theblock.co/rss.xml"}


def fetch_news(http, out_path: Path, max_items: int = 30) -> dict:
    items, errors = [], {}
    for source, url in FEEDS.items():
        try:
            raw = http.get(url, raw=True, timeout_s=20)
            root = ET.fromstring(raw)
            for it in root.iter("item"):
                title, link = (it.findtext("title") or "").strip(), (it.findtext("link") or "").strip()
                pub = it.findtext("pubDate")
                try:
                    ms = int(parsedate_to_datetime(pub).timestamp() * 1000) if pub else None
                except (TypeError, ValueError):
                    ms = None
                if title and link and ms:
                    items.append({"title": title, "link": link, "source": source, "published_utc": to_iso(ms), "ms": ms})
        except Exception as e:
            errors[source] = str(e)[:120]
    items = sorted({i["link"]: i for i in items}.values(), key=lambda i: -i["ms"])[:max_items]
    data = {"fetched_utc": to_iso(now_ms()), "items": items, "errors": errors, "feeds": FEEDS}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=1))
    return data


def _records_digest(cfg, store) -> str:
    from cryptolab.report.metrics import experiment_summary
    lines = []
    for key, exp in cfg["experiments"].items():
        s = experiment_summary(store, exp["name"])
        a = s.get("accuracy") or {}
        r = store.conn.execute("SELECT payload FROM forecasts WHERE experiment=? ORDER BY seq DESC LIMIT 1",
                               (exp["name"],)).fetchone()
        last = json.loads(r[0]) if r else None
        lines.append(f"- {exp['name']} ({exp['horizon_min']} min): {s.get('forecasts', 0)} forecasts, "
                     f"{a.get('k', 0)}/{a.get('n', 0)} correct" +
                     (f"; latest {last['pair']} {last['direction']} P(correct)={last['p_direction']:.2f}; "
                      f"trade decision: {last.get('opportunity_label', '')}" if last else ""))
    st, _ = store.get_runtime("paper_state", {})
    for name, pf in ((st or {}).get("portfolios") or {}).items():
        lines.append(f"- paper portfolio {name}: equity ${pf['equity']:.2f}, open {len(pf.get('positions', {}))}, "
                     f"signals not taken: {pf.get('ignored', {})}")
    return "\n".join(lines)


def llm_summary(cfg, store, news: dict, out_path: Path) -> dict | None:
    """Optional. Returns None (and does nothing) unless explicitly enabled and the SDK is installed."""
    conf = cfg.get("llm", {})
    if not conf.get("enabled"):
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("llm enabled but the optional 'anthropic' package is not installed (requirements-llm.txt)")
        return None
    heads = "\n".join(f"[{i + 1}] {n['title']} ({n['source']}, {n['published_utc']}) {n['link']}"
                      for i, n in enumerate(news.get("items", [])[:15]))
    prompt = (
        "You are writing a short briefing for the owner of a cryptocurrency research and PAPER-trading system. "
        "It never places real orders. Use only the material below; do not invent prices, statistics or news.\n\n"
        f"NEWS HEADLINES (cite as [n]):\n{heads}\n\nSYSTEM RECORDS:\n{_records_digest(cfg, store)}\n\n"
        "Write three short sections in plain text: 1) the most relevant news in 4-6 bullets, each citing its [n]; "
        "2) in plain language, what the latest forecasts say and why the system did or did not take paper trades; "
        "3) notable performance results or errors. Keep it under 300 words. No investment advice.")
    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / auth profile from the environment
    try:
        resp = client.beta.messages.create(
            model=conf.get("model", "claude-opus-5-5"), max_tokens=4000,
            output_config={"effort": conf.get("effort", "low")},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": prompt}])
    except anthropic.AuthenticationError:
        log.warning("llm summary: no valid Anthropic credentials; skipping")
        return None
    except anthropic.RateLimitError:
        log.warning("llm summary: rate limited; will retry next interval")
        return None
    except anthropic.APIStatusError as e:
        log.warning("llm summary: API error %s", e.status_code)
        return None
    except anthropic.APIConnectionError:
        log.warning("llm summary: network error")
        return None
    if resp.stop_reason == "refusal":
        log.warning("llm summary declined (%s)", getattr(resp.stop_details, "category", None))
        return None
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    out = {"text": text, "model": resp.model, "generated_utc": to_iso(now_ms()),
           "sources": [{"n": i + 1, "title": n["title"], "link": n["link"], "source": n["source"],
                        "published_utc": n["published_utc"]} for i, n in enumerate(news.get("items", [])[:15])],
           "note": "AI-generated from the listed headlines and the system's records; not used for predictions."}
    out_path.write_text(json.dumps(out, indent=1))
    return out


class NewsMonitor:
    name = "news"

    def __init__(self, cfg, store, http) -> None:
        self.cfg, self.store, self.http = cfg, store, http
        self.last_news = self.last_llm = 0.0

    def tick(self, server_now: int) -> None:
        run = self.cfg.path("data_dir") / "run"
        if time.time() - self.last_news > 30 * 60:
            self.last_news = time.time()
            news = fetch_news(self.http, run / "news.json")
            interval = self.cfg.get("llm", {}).get("interval_min", 60) * 60
            if self.cfg.get("llm", {}).get("enabled") and time.time() - self.last_llm > interval:
                self.last_llm = time.time()
                llm_summary(self.cfg, self.store, news, run / "llm_summary.json")
