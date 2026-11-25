# Deployment and operation

The live experiments need something that runs every 10 minutes for days. A Claude Code cloud
session is **not** that: its container is reclaimed after inactivity, so the in-session daemon
only covers the hours the session is active. Use one of the options below for continuous
operation. Run exactly **one** writer at a time (the ledger is a single hash chain).

## Option 1 — GitHub Actions (free for public repositories)

`.github/workflows/live-experiments.yml` runs `cryptolab run-once` every 10 minutes, rebuilds
the dashboard and commits `data/ledger/` and `reports/` back to the branch.

1. Give the repository write access for Actions (Settings → Actions → General → Workflow
   permissions → Read and write).
2. Make the branch holding this code the **default branch** (GitHub only runs schedules there),
   or merge it into the default branch.
3. Run the `network-probe` workflow once (Actions tab → network-probe → Run workflow) and check
   that `data-api.binance.vision` returns 200 from GitHub's runners. If Binance is blocked there,
   use Option 2.
4. Stop any other running writer (`python -m cryptolab daemon stop`), push the latest
   `data/ledger/`, then enable the workflow.

Caveats: GitHub frequently starts scheduled runs 5–20 minutes late and may skip runs under
load, so Experiment A will show gaps (recorded honestly as gap records). The 60-minute
experiment is far less affected.

## Option 2 — any always-on machine (VPS, home server, Raspberry Pi)

```bash
git clone <repo> && cd <repo>
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m cryptolab audit            # rebuilds the SQLite DB from the ledger and checks the chain
python -m cryptolab daemon start     # background worker (single instance, file-locked)
python -m cryptolab daemon status
python -m cryptolab dashboard        # writes reports/dashboard.html
python -m cryptolab daemon stop
```

Run it under systemd (`Restart=always`) or cron (`*/10 * * * * cd /path && .venv/bin/python -m
cryptolab run-once`) for automatic restarts. Restarts are safe: the database rejects a second
forecast for the same slot, missed slots become gap records, and pending forecasts are verified
from historical trades whenever the worker comes back (up to 6 hours after expiry, then
UNVERIFIABLE). The machine needs outbound HTTPS to the exchange hosts listed in
`config/default.toml`. Binance's main domain is geo-restricted in some regions (e.g. the US);
the public mirror `data-api.binance.vision` usually remains available.

## Re-running the research

```bash
python -m cryptolab collect        # ~1 GB of 1-minute history, ~5 min
python -m cryptolab build-panels   # ~1 min
python -m cryptolab research       # ~30 min on 4 CPUs
```

Re-running research writes a new pre-registration; to keep the experiment honest, a new
selection must be versioned (v2) and evaluated only on forecasts made after it is frozen.

## Files

| Path | Content | In git |
|---|---|---|
| `data/ledger/*.jsonl` | append-only hash-chained records (source of truth) | yes |
| `data/cryptolab.sqlite` | working database, rebuilt from the ledger on demand | no |
| `data/research/` | universe, backtest results (JSON), live model files | yes |
| `data/raw/`, `data/cache/`, `data/run/` | downloaded history and caches (regenerable) | no |
| `reports/` | dashboard (HTML), summary JSON, CSV/JSON exports | yes |
| `logs/` | worker logs | no |
