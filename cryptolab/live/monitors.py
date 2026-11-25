"""Event monitors attached to the worker (each isolated: a failure in one never stops others)."""
from __future__ import annotations


def build_monitors(sysm) -> list:
    from cryptolab.live.kalshi_live import KalshiCollector
    from cryptolab.live.paper import PaperEngine
    from cryptolab.live.strategy_monitor import StrategyMonitor
    from cryptolab.report.news import NewsMonitor
    return [StrategyMonitor(sysm.cfg, sysm.store, sysm.bn, sysm.cache),
            PaperEngine(sysm.cfg, sysm.store, sysm.bn),
            KalshiCollector(sysm.cfg, sysm.store, sysm.http, sysm.others["coinbase"]),
            NewsMonitor(sysm.cfg, sysm.store, sysm.http)]
