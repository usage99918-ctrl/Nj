# Results (backtests completed 2026-11-24; forward tests started 2026-11-25 05:13 UTC)

## Summary

**Neither horizon shows a tradable edge.** Both pre-registered models show a statistically
significant but very small directional signal on the untouched test period. Neither model is
profitable after realistic retail trading costs. Even in a low-cost scenario (0.02% fees per
side), trading every forecast still loses money. Under the rule that trades only when the
expected move exceeds costs, a spot trader would almost never trade. The forward experiments
have only just started, and their samples are far too small to say anything yet.

## Data actually used

- 684 eligible Binance USDT spot pairs considered, **including delisted ones**. 544 had data in the
  period. The monthly top-30 point-in-time universe used 83 distinct coins over time,
  including TON, which was delisted mid-period.
- 1-minute klines from 2025-12-01 to 2026-11-22: about 740 MB, 1.40 M coin-decisions at 10 min
  and 234 k at 60 min.
- Excluded from the universe: stablecoins and pegs (static list plus trailing volatility below
  0.3%), wrapped tokens, leveraged tokens and tokenized equities. The `U` stablecoin and four
  tokenized stocks were caught and removed before any model was fit.

## Backtest — Experiment A (10-minute horizon)

| | Validation (Jul–Sep) | **Test (Oct 1 – Nov 22, untouched)** |
|---|---|---|
| Pre-registered model | `lgbm_reg_huber` (LightGBM, Huber loss), chosen from 100 configurations | |
| Forecasts | 13,248 | 7,632 |
| Directional accuracy (flat = wrong) | 52.3% | **51.5%** (95% CI 50.4–52.7%, p = 0.004) |
| Accuracy on non-flat outcomes | 62.9% | 56.8% |
| Balanced accuracy | 62.9% | 56.4% |
| Gross directional return per forecast | +7.5 bps (t = 16.0) | **+3.1 bps** (t = 5.2) |
| Net per forecast if every call is traded (retail costs, about 25 bps round trip) | −17.2 bps | **−21.5 bps** |
| Same, at 0.02% fees per side | −1.2 bps | −5.5 bps |
| Spot long-only trades (expected move > costs) | 2 | 0 |
| Cross-sectional rank IC | 0.076 | 0.049 (t = 18.9) |
| 50% / 80% interval coverage | 53.9% / 81.9% | 46.9% / 78.4% |
| Price error vs a no-change forecast | 0.995× | 0.997× |

Pre-registered hypotheses on the test period: H1 (accuracy > 50%) **passed**, H2 (gross
return > 0) **passed**, H3 (long-only net > 0) **failed**.

Over all 100 strategies, White's reality check gives p = 0.001 on both validation and test, so
the best strategy beats chance after accounting for the search. The decay from validation to
test is clear: balanced accuracy fell from 62.9% to 56.4%.

## Backtest — Experiment B (60-minute horizon)

| | Validation | **Test (untouched)** |
|---|---|---|
| Pre-registered model | `rsi_rev_60` (RSI(60) mean reversion) | |
| Forecasts | 2,208 | 1,272 |
| Directional accuracy | 56.8% | **53.1%** (95% CI 50.4–55.9%, p = 0.013) |
| Balanced accuracy | 58.0% | 54.0% |
| Gross directional return per forecast | +12.8 bps (t = 4.4) | +1.8 bps (t = 0.57, not significant) |
| Net per forecast, every call traded | −11.8 bps | −22.8 bps |
| Spot long-only trades | 0 | 0 |
| 50% / 80% interval coverage | 48.9% / 79.3% | 52.5% / 80.0% |

H1 **passed**, H2 **failed**, H3 **failed**.

## What the research says about strategy families (test-period mean IC)

- **Short-term mean reversion is the most consistent effect.** Mean IC is +0.028 at 10 minutes
  and +0.022 at 60 minutes.
- **Momentum and trend-following is consistently negative** at these horizons (−0.020 and
  −0.022). Simple momentum lost to random direction at both horizons.
- **Regression, tree and ensemble models are the strongest** (+0.04 to +0.05 at 10 minutes).
  They mainly combine reversal signals.
- Breakout, relative-volume and order-flow (taker-imbalance) rules are near zero.
- The highest validation scores belonged to gated strategies (for example `liq_shock_rev`, a
  liquidation-overshoot hypothesis). They forecast in under 90% of slots, so they were
  ineligible under the pre-registered rule. Their test results were not significant.

## Benchmarks (test period)

| | 10-min accuracy | 60-min accuracy |
|---|---|---|
| Selected model | 51.5% | 53.1% |
| Random direction | 47.3% | 49.4% |
| Always up | 48.9% | 47.6% |
| Always down | 50.9% | 52.4% |
| Simple momentum | 44.4% | 46.8% |
| Simple mean reversion | 51.3% | 51.3% |

Plain accuracies below 50% for the trivial rules reflect the flat-is-wrong rule, which matters
most at 10 minutes.

Buy-and-hold over the test period: BTC −26.7%, equal-weight universe −43.3%. The
long-or-cash model stayed in cash, so it avoided that drawdown. That came from a lack of
signals, not from skill.

## Forward experiments

Live since 2026-11-25 05:13 UTC. See `reports/dashboard.html` and `reports/summary.json`. With
fewer than about 100 scored forecasts per experiment, the 95% interval on accuracy is wider
than ±10 percentage points. Results are reported but not interpreted.

## Main limitations

- The costs are assumptions: Binance retail taker fees plus estimated spread and slippage.
  Live forecasts use the measured spread.
- The backtest covers one exchange and about 11 months, including a strong bear phase in the
  test period. Patterns can disappear when the market regime changes, and validation-to-test
  decay was large.
- The live universe applies spread and tick-size filters that were impossible to apply
  historically, so it is a stricter subset of the backtest universe.
- Order book, funding, open interest, liquidations and sentiment are recorded as context only,
  because no free point-in-time history exists to backtest them.
- The probability calibration comes from backtest forecasts. It is not yet validated on
  forward data.

---

# Standalone upgrade results (2026-11-25)

Rules were fixed before the results (`docs/PREREGISTRATION_V2.md`). Every number below comes
from `data/research/*.json` and the ledger. Experiments A and B were not changed.

## Multi-horizon experiments (untouched test period Oct 1 – Nov 22)

| Horizon | Model | Test accuracy (flat = wrong) | Non-flat accuracy | Gross / net bps per forecast | 3-class Brier: model / no-signal / climatology | Touch probabilities |
|---|---|---|---|---|---|---|
| 5 min | `lgbm_reg_shallow` (new) | 48.1% (n = 15,264; 18.7% flat) | 59.2% | +3.2 (t = 12.7) / −21.4 | 0.370 / 0.373 / 0.434 | beat climatology |
| 10 min | `lgbm_reg_huber` (from A) | 51.5% (n = 7,632) | 56.8% | +3.1 (t = 5.2) / −21.5 | 0.529 / 0.532 / 0.591 | beat climatology |
| 60 min | `rsi_rev_60` (from B) | 53.1% (n = 1,272) | 54.1% | +1.8 (n.s.) / −22.8 | 0.616 / 0.617 / 0.665 | beat climatology |
| 4 h | `rv_rev_10` (new) | 53.8% (n = 318, p = 0.10) | – | +2.9 (n.s.) / −22.0 | 0.606 / 0.606 / 0.667 | beat climatology |
| 24 h | `quiet_drift` (new) | 40.0% (n = 50) | – | −56 / −83 | 0.613 / 0.583 / 0.662 | ±10% targets worse than climatology |

**Interpretation:**
- **The short horizons carry a statistically strong but economically tiny signal.** It amounts
  to about 3 bps gross per forecast, against a round-trip cost of about 25 bps.
- **The 3-class and touch probabilities beat base rates, but almost entirely through volatility
  forecasting.** A no-signal version with the same volatility forecast is nearly as good, and at
  24 hours it is better. "How far will it move" is predictable; "which way" mostly is not.
- **The 24-hour model failed its untouched test.**

## Friend's three strategies (fixed definitions, after costs)

| Strategy | Test signals | 4h gross / net bps | Stop/target paper trades (test) | Verdict |
|---|---|---|---|---|
| S1 Relative-Strength Reset (ETH/BTC) | 32 | +23.0 / 0.0 (beat random entries, p = 0.005) | +12.8 bps/trade (t = 0.6) | Mixed: **negative in training** (−52 bps/trade, 64 signals); too few signals; unproven |
| S2 Breakout Retest | 3,666 | +12.0 / −12.0 | −19.5 bps/trade (t = −8.2) | **Fails** after costs on validation and test |
| S3 Liquidity Recovery (proxy) | 399 | −15.4 / −39.6 | −48.9 bps/trade (t = −3.4) | **Fails** in the proxy form. The real order-book version is new live code with no track record |

None of the three strategies outperformed the existing models.

## Kalshi controlled test

The comparison is paired: the same base model, with and without Kalshi's 15-minute market price.
It was run on 5,181 untouched BTC and ETH windows from Oct 28 to Nov 25.
- **Brier:** 0.24899 without, 0.24898 with. The difference is −0.00001, with a 95% confidence
  interval of −0.00005 to +0.00002.
- **Accuracy:** 52.48% without and 52.50% with. A McNemar test gives p = 1.0.
- **Kalshi alone** is slightly worse, at 52.31% accuracy and a Brier of 0.24912.
- **Decision:** Kalshi is **not adopted**, as the pre-registered rule requires. The likely reason
  is that its mid-window price mostly restates the move since the window opened, which Binance
  data already contains.
- **Live:** both arms keep running in parallel to collect forward evidence.

## Paper trading

The **gated** portfolio may trade only strategies that passed an out-of-sample test after costs.
None did, so **it holds cash by design**. Shadow portfolios trade every signal of S1, S2 and S3
for measurement. The rules are:
- position ≤ 10% of equity, capped at $2,000;
- at most 5 positions;
- stops and targets, assuming the stop is hit first if both fall in the same minute;
- a 2% daily loss limit;
- a kill switch;
- 0.10% fees per side, measured or assumed spread, slippage, and a 2-second fill delay.

## Continuous operation: what was verified, and what was not

- **Crash recovery, verified live.** A worker killed with SIGKILL was restarted by the supervisor
  10 seconds later, and the exit, the restart and a checkpoint are in the ledger. A hung worker
  is killed and restarted, and outage start and end events are recorded; both have automated
  tests.
- **Restart without duplicates or back-filling, verified live.** After an unplanned stop,
  expired forecasts were verified late from historical trades, and the missed slot became a
  gap record rather than a fabricated forecast.
- **Not verifiable here: 24/7 operation.** In this cloud coding session, background processes
  stop when the session goes idle. The original daemon died minutes after a turn ended, at
  05:30 UTC. Continuous operation therefore requires deploying to your PC or a server
  (`docs/INSTALL.md`), where Task Scheduler, systemd or Docker restart it.
- **Windows installer:** written, but not executed on Windows; the build machine is Linux.
