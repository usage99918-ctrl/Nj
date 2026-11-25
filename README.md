# Claude Crypto Research Lab

An independent cryptocurrency forecasting and paper-trading research system. It scans spot
markets on 8 exchanges, backtests 100 strategy configurations per horizon with walk-forward
validation, freezes live 10-minute and 60-minute forecasts in a tamper-evident ledger, verifies
them against exchange trades at the exact expiry time, and reports the results against
baselines with explicit uncertainty.

**Public, read-only market data only. No exchange account, no API keys, no real orders.**
Nothing here is investment advice.

## Results so far

See `reports/dashboard.html` (regenerated from the immutable records by
`python -m cryptolab dashboard`) and `reports/summary.json`. Headline findings are written up in
[`docs/RESULTS.md`](docs/RESULTS.md).

## What is in the box

| Component | Where |
|---|---|
| Exchange clients with retries/backoff (Binance mirror, Coinbase, Kraken, OKX, KuCoin, Gate.io, MEXC, Bitfinex, Bitstamp) | `cryptolab/exchanges/` |
| Multi-exchange market scanner with quality checks and counts | `cryptolab/live/scanner.py` |
| Historical collector + point-in-time universe (incl. delisted coins) | `cryptolab/data/history.py` |
| Feature engine shared by backtest and live (no look-ahead, tested) | `cryptolab/features.py` |
| 78 rule strategies + 22 fitted models (ridge, logistic, LightGBM, extra-trees, AR, ensembles, regime switching) + baselines | `cryptolab/strategies/` |
| Walk-forward backtest that simulates the live selection procedure | `cryptolab/backtest/engine.py` |
| Statistics: Wilson CIs, Newey-West, Holm/BH, White reality check, deflated Sharpe | `cryptolab/backtest/stats.py` |
| Pre-registered model selection + untouched test evaluation | `cryptolab/backtest/research.py` |
| Live forecaster (frozen forecasts with probabilities, intervals, costs, evidence) | `cryptolab/live/forecaster.py` |
| Exact-timestamp verifier + paper trading | `cryptolab/live/verifier.py`, `cryptolab/report/metrics.py` |
| Scheduler: daemon (start/stop/status) and run-once mode, gap records, single-writer lock | `cryptolab/live/scheduler.py` |
| Immutable SQLite store + SHA-256 hash-chained ledger, audit and rebuild | `cryptolab/storage/` |
| Static dashboard + CSV/JSON exports | `cryptolab/report/` |
| Tests (leakage, embargo, immutability, tamper detection, scheduler, scoring) | `tests/` |
| GitHub Actions workflow for continuous operation | `.github/workflows/live-experiments.yml` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                      # 20 tests
python -m cryptolab scan              # one multi-exchange scan with counts
python -m cryptolab daemon start      # run both live experiments in the background
python -m cryptolab status            # worker + experiment status
python -m cryptolab dashboard         # reports/dashboard.html + exports
python -m cryptolab audit             # verify the ledger hash chain and DB consistency
python -m cryptolab daemon stop
```

Full research pipeline (≈40 minutes, ≈1 GB download): `collect` → `build-panels` → `research`.

## Documents

- [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) — rules fixed before testing
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — data, features, strategies, statistics, leakage audit
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — continuous operation (GitHub Actions or any server)
- [`docs/RESULTS.md`](docs/RESULTS.md) — what the backtests and forward tests show, limitations
