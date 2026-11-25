# Methodology

## Data sources (verified reachable 2026-11-24 from the build environment)

| Source | Used for | Notes |
|---|---|---|
| Binance spot (data-api.binance.vision, data.binance.vision) | universe, features, forecasts, verification (aggTrades), order book | api.binance.com is geo-blocked (HTTP 451) here; the official public market-data mirror is used |
| Coinbase Exchange public API | discovery counts, cross-exchange price check | the Coinbase *account* connector was deliberately **not** used |
| Kraken, OKX, KuCoin, Gate.io, MEXC, Bitfinex, Bitstamp | discovery counts, quality screening, cross-exchange checks | MEXC is slow (~20 s per call) |
| OKX public derivatives | funding rate, open interest, liquidations (context only) | Bybit and Binance futures are geo-blocked here |
| Not used | news, social sentiment, on-chain metrics | no free source with point-in-time history; GDELT was unreachable. Reported as a limitation, never fabricated |

## Market scan and quality checks

Every cycle scans all spot markets on 8 exchanges. A market passes quality checks only if it is
trading, USD/USDT/USDC-quoted, not a stablecoin/pegged/wrapped/leveraged/tokenized-equity asset,
has ≥ $5M 24h quote volume, spread ≤ 10 bps, a quote no older than 180 s, and tick size ≤ 5 bps
of price (coarse ticks make short-horizon outcomes mostly "flat"). Binance instruments with
tight exchange price bands (ask multiplier ≤ 1.2, bid multiplier ≥ 0.8) are excluded: on
2026-11-24 all 212 such symbols were stablecoins or tokenized stocks/ETFs.

The forecasting universe is the top 30 quality-passed Binance USDT pairs by trailing 30-day
volume (same ranking rule as the backtest). **Known difference:** the live spread/tick filters
could not be applied historically (no free historical order books), so the backtest universe
also contains liquid coins with coarse ticks (e.g. ADA, PEPE, TAO) that the live filter
excludes.

## Features (all computable from data at or before the decision time)

Returns over 1–1440 min; realized volatility (30–1440 min) and a blended volatility forecast;
ATR; RSI(14/60); MACD; Bollinger %B/width; distance to moving averages; MA crossovers;
relative volume; taker-buy imbalance (trade-side order-flow proxy); trade-count intensity;
range position and breakouts of 1h/4h/24h highs/lows (support/resistance proxies); per-coin
7-day AR(1) coefficients; EWMA drift; BTC/ETH returns, rolling beta and correlation to BTC,
BTC-residual returns; BTC trend and volatility regime; universe breadth, average return and
dispersion; cross-sectional ranks; hour of day.

**Leakage audit.** (1) `tests/test_research_integrity.py::test_features_have_no_lookahead`
recomputes every feature with all future data removed and requires identical values. (2) The
first smoke test of the ML models showed 100% accuracy: the feature blacklist let the target
column `y` through. It was replaced by an explicit allowlist with a hard guard
(`model_feature_columns` raises if any target-like column appears), and a test. (3) The walk-
forward engine is tested to never train on data later than month start − embargo − horizon,
and a shuffled-target test confirms chance-level accuracy.

## Strategy families (100 configurations per horizon)

momentum (13) · mean reversion (15) · breakout/support-resistance (10) · relative volume (6) ·
order-flow imbalance (7) · volatility-based (4) · statistical time series (3) · regression (6)
· classification (3) · tree-based ML (6) · ensembles (4) · regime detection/switching (4) ·
cross-asset lead-lag/residual (13) · original experimental hypotheses (6). Each is a stated
hypothesis (see `cryptolab/strategies/`), not a parameter sweep. Baselines: random direction,
always up, always down, no change, simple momentum, simple mean reversion, buy-and-hold.

## Statistics

Wilson intervals and exact binomial tests for accuracy; balanced accuracy; Newey-West HAC
t-statistics for returns; cross-sectional rank IC; Brier score and reliability bins; interval
coverage; Holm and Benjamini-Hochberg corrections across strategies; White's Reality Check
(stationary bootstrap) for the best of all strategies; deflated Sharpe ratio. Decisions are
spaced exactly one horizon apart, so consecutive backtest outcomes do not overlap.

## Verification of live forecasts

Reference price: the last Binance aggTrade at or before the moment the forecast is frozen.
Outcome: the last aggTrade at or before the exact expiry instant (searched within 60 s →
`VERIFIED`; within 600 s or via the 1-minute candle close → `APPROXIMATE`, timing gap recorded;
nothing after 6 h → `UNVERIFIABLE`). Forecast records are immutable; outcomes are separate
immutable records.

## Paper trading

Spot long-only: a hypothetical $1,000 buy only when the frozen forecast is UP **and** its
expected move exceeds the estimated round-trip cost (measured spread + fee + book-walk
slippage). Otherwise cash. A separate "direction bet" ledger trades every forecast in its
predicted direction; its shorts are labelled theoretical (they need margin/borrowing and are
not executable on spot).
