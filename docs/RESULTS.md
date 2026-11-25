# Results (backtests completed 2026-11-24; forward tests started 2026-11-25 05:13 UTC)

## Summary

**Neither horizon shows a tradable edge.** Both pre-registered models show a statistically
significant but very small directional signal on the untouched test period. Neither model is
profitable after realistic retail trading costs. Even in a low-cost scenario (0.02% fees per
side), trading every forecast still loses money. Under the rule that trades only when the
expected move exceeds costs, a spot trader would almost never trade. The forward experiments
have only just started, and their samples are far too small to say anything yet.

## Data actually used

- 707 Binance USDT spot pairs considered, **including delisted ones**. 544 had data in the
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
