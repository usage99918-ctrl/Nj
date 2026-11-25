# Claude Crypto Research Lab

A standalone cryptocurrency forecasting and **paper-trading** research application. It does the
following:
- scans spot markets on 8 exchanges;
- forecasts at 5-minute, 10-minute, 60-minute, 4-hour and 24-hour horizons, with UP/DOWN/SIDEWAYS
  and "touch ±x%" probabilities;
- tests three externally proposed trading strategies;
- runs a controlled test of whether Kalshi prediction-market prices improve forecasts;
- freezes every forecast in a tamper-evident ledger and verifies it against exchange trades at
  the exact expiry time;
- reports everything, including failures and missed cycles, on a dashboard you can open from
  your phone.

**Public, read-only market data only. No exchange or Kalshi account, no API keys, no real
orders, no paid services. Trading is simulated.** Nothing here is investment advice.

## Start here

| I want to… | Do this |
|---|---|
| Install it on my Windows PC (auto-start, auto-restart) | `powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1` (see [docs/INSTALL.md](docs/INSTALL.md)) |
| Run it on a Linux server / Raspberry Pi | `bash deploy/install_linux.sh` |
| Run it with Docker | `docker compose up -d` |
| See the dashboard (also from my phone) | `python -m cryptolab serve` (phone access: [docs/INSTALL.md](docs/INSTALL.md#phone-access-to-the-dashboard-secure)) |
| Check that it is really running | `python -m cryptolab status` (coverage, gaps, restarts, outages from the ledger) |
| Stop paper trading immediately | `python -m cryptolab kill-switch on` (or the button on the dashboard) |
| Read what the research found | [docs/RESULTS.md](docs/RESULTS.md) |

## Commands

```text
python -m cryptolab run                 supervisor in the foreground (what Task Scheduler / systemd / Docker run)
python -m cryptolab start | stop        start in the background / stop gracefully (works on Windows and Linux)
python -m cryptolab status | health     experiment + continuity report / exit code 0 if the worker is healthy
python -m cryptolab serve               token-protected dashboard on http://127.0.0.1:8787
python -m cryptolab dashboard           write reports/dashboard.html + CSV/JSON exports
python -m cryptolab kill-switch on|off  paper-trading emergency stop
python -m cryptolab audit               verify the hash-chained ledger and the database
python -m pytest                        run the test suite
```

Research, re-run only if you want to reproduce it: `collect`, `build-panels`, `research` (A/B),
`g2-research` (5m–24h).

## What runs

| Component | What it does |
|---|---|
| Experiments **A_10min, B_60min** | Original pre-registered experiments, unchanged |
| **G2_5min … G2_1440min** | New horizons with 3-class and touch probabilities (separate, so comparisons stay fair) |
| **KX_{BTC,ETH}_{without,with}** | Kalshi controlled test: identical model with/without Kalshi prices, paired windows |
| Strategy monitor | Friend's S1 Relative-Strength Reset, S2 Breakout Retest, S3 Liquidity Recovery (order-book version) |
| Paper engine | Gated portfolio (approved strategies only; currently none, so it holds cash) and one shadow portfolio per strategy. Rules: position limits, stop-loss/take-profit, daily loss limit, kill switch, fees/spread/slippage/delay |
| Kalshi collector | 15-minute BTC/ETH up/down markets and hourly price ladders, every 5 min |
| News | RSS headlines with sources and timestamps; optional AI summary (off by default) |
| Supervisor | Restarts the worker after crashes or hangs; every start, crash, outage and hourly checkpoint goes into the ledger |

## Layout

`cryptolab/` source:
- `exchanges/` clients;
- `live/` worker, supervisor, forecaster, verifier, monitors, paper engine;
- `backtest/` research;
- `strategies/` strategy library and the friend's strategies;
- `report/` dashboard, server, news;
- `storage/` immutable SQLite and ledger.

`config/default.toml` holds all parameters. `data/ledger/` holds the records (source of truth).
`data/research/` holds the backtest results and live models. `reports/` holds the dashboard and
exports. `deploy/` holds the installers. `docs/` has the pre-registrations, methodology, install
guide and results.
