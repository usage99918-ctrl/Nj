"""CSV / JSON exports of every record (forecasts, outcomes, gaps, scans, model versions)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

FORECAST_COLS = ["forecast_id", "experiment", "test_number", "slot_utc", "horizon_min", "exchange", "pair", "symbol",
                 "origin_utc", "ref_trade_utc", "p0", "expiry_utc", "pred_price", "direction", "expected_move_pct",
                 "p_up", "p_direction", "pi50", "pi80", "model_version", "strategy", "n_examined",
                 "expected_net_long_pct", "long_signal", "theoretical_short_signal", "opportunity_label"]
OUTCOME_COLS = ["forecast_id", "experiment", "status", "method", "verified_utc", "p1", "p1_trade_utc", "timing_gap_ms",
                "actual_direction", "correct", "actual_return_pct", "abs_pred_error_pct", "no_change_error_pct",
                "in_pi50", "in_pi80", "brier", "long_taken", "long_net_pct", "dir_gross_pct", "dir_net_pct"]


def _csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([json.dumps(r.get(c)) if isinstance(r.get(c), (list, dict)) else r.get(c) for c in cols])


def export_all(store, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fcs = store.rows("forecasts")
    outs = store.rows("outcomes")
    gaps = store.rows("gaps")
    scans = store.rows("scans")
    models = store.rows("model_versions")
    prereg = store.rows("preregistrations")
    _csv(out_dir / "forecasts.csv", fcs, FORECAST_COLS)
    _csv(out_dir / "outcomes.csv", outs, OUTCOME_COLS)
    _csv(out_dir / "gaps.csv", gaps, ["gap_id", "experiment", "slot_utc", "reason"])
    _csv(out_dir / "scans.csv", scans, ["scan_id", "experiment", "markets_discovered", "markets_quality_passed",
                                        "unique_assets_passed", "markets_analyzed"])
    for name, rows in (("forecasts", fcs), ("outcomes", outs), ("gaps", gaps), ("model_versions", models),
                       ("preregistrations", prereg)):
        (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=1, default=str))
    return {"forecasts": len(fcs), "outcomes": len(outs), "gaps": len(gaps), "scans": len(scans),
            "dir": str(out_dir)}
