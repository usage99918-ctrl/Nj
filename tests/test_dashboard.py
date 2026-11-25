"""The dashboard builds from records alone and shows forward results."""
import json

from cryptolab.config import load_config
from cryptolab.live.verifier import score
from cryptolab.report.dashboard import build_all
from cryptolab.storage.db import Store


def test_dashboard_builds_with_forward_records(tmp_path):
    cfg = load_config()
    for k in ("db_file", "ledger_dir", "reports_dir", "research_dir", "data_dir"):
        cfg["paths"][k] = str(tmp_path / k)
    (tmp_path / "research_dir").mkdir()
    store = Store(tmp_path / "db_file", tmp_path / "ledger_dir")
    fc = {"forecast_id": "A-000001", "experiment": "A_10min", "test_number": 1, "slot_ms": 1_795_550_400_000,
          "slot_utc": "2026-11-24T20:00:00.000Z", "horizon_min": 10, "exchange": "binance", "symbol": "SOLUSDT",
          "base": "SOL", "quote": "USDT", "pair": "SOL/USDT", "origin_ms": 1_795_550_401_000,
          "origin_utc": "2026-11-24T20:00:01.000Z", "ref_trade_ms": 1_795_550_400_900,
          "ref_trade_utc": "2026-11-24T20:00:00.900Z", "p0": 70.0, "expiry_ms": 1_795_551_001_000,
          "expiry_utc": "2026-11-24T20:10:01.000Z", "pred_price": 70.01, "direction": "UP", "expected_move_pct": 0.014,
          "p_up": 0.53, "p_direction": 0.53, "pi50": [69.9, 70.1], "pi80": [69.8, 70.2], "model_version": "A-v1-x",
          "strategy": "x", "costs": {"round_trip": 0.0024, "short_borrow_for_horizon": 0.0, "fee_per_side": 0.001,
                                     "half_spread_measured": 0.00005, "slippage_per_side": 0.0001},
          "expected_net_long_pct": -0.2, "long_signal": False, "theoretical_short_signal": False,
          "opportunity_label": "NO", "n_examined": 25, "scan_counts": {"markets_discovered": 11784},
          "evidence_for": [{"indicator": "RSI", "reading": "29"}], "evidence_against": [],
          "features": {"r_10": 0.001, "btc_trend": 0.01, "btc_vol_ratio": 0.9}}
    store.record("forecast", fc)
    store.record("outcome", {"forecast_id": "A-000001", "experiment": "A_10min", "status": "VERIFIED",
                             "verified_ms": 1_795_551_010_000, **score(fc, 70.05)})
    out = build_all(cfg)
    html = (tmp_path / "reports_dir" / "dashboard.html").read_text()
    assert out["audit_ok"]
    assert "Claude Crypto Research Lab" in html and "SOL/USDT" in html and "1/1" in html
    assert html.startswith("<!doctype html>")
    summary = json.loads((tmp_path / "reports_dir" / "summary.json").read_text())
    assert summary["forward"]["A"]["accuracy"]["k"] == 1
    assert (tmp_path / "reports_dir" / "exports" / "forecasts.csv").read_text().count("\n") == 2
