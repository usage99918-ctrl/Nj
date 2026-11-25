# Pre-registration — standalone upgrade (2026-11-25)

All rules below were fixed in code before any of the corresponding results were computed. The
machine-readable records are in the hash-chained ledger (type `preregistration`). The original
pre-registration for Experiments A and B (`docs/PREREGISTRATION.md`) is unchanged, and A and B
keep running exactly as frozen.

## New experiments run separately from A and B

| Experiment | Horizon | Cadence | Model |
|---|---|---|---|
| G2_5min | 5 min | 5 min | new selection (rule below) |
| G2_10min | 10 min | 10 min | A's v1 selection (`lgbm_reg_huber`) + probability layers |
| G2_60min | 60 min | 60 min | B's v1 selection (`rsi_rev_60`) + probability layers |
| G2_240min | 4 h | 4 h (00, 04, … UTC) | new selection |
| G2_1440min | 24 h | daily at 00:00 UTC | new selection |
| KX_{BTC,ETH}_{without,with} | to the Kalshi window close (~13 min) | every 15-min Kalshi window | controlled Kalshi test |

**Selection for new horizons.** This is the same rule as A/B:
- Use the highest validation **balanced accuracy** among 101 strategies.
- Eligible strategies need validation coverage ≥ 90% and ≥ min(300, 0.9 × validation slots)
  forecasts. The 24-hour horizon has only 92 validation days, hence the scaled minimum.
- Ties are broken by mean IC.
- The library adds `rev_1440` (24-hour reversal) to the A/B library.

## Probability definitions

- **SIDEWAYS** means |log return over the horizon| ≤ τ_h, where τ_h = max(0.25%,
  0.25% × √(h/60)). That gives 0.25% at 5/10/60 minutes, 0.50% at 4 hours and 1.22% at
  24 hours. **UP** means a return > τ_h; **DOWN** means a return < −τ_h.
- **Binary direction** (as in A/B): UP means the end price is higher than the start price; an
  unchanged price counts as wrong.
- **Touch +x% / −x%**: the price reaches p0·(1 ± x) at any moment inside the window. Targets
  per horizon:

  | Horizon | Targets |
  |---|---|
  | 5 min | 0.25%, 0.5% |
  | 10 min | 0.25%, 0.5%, 1% |
  | 60 min | 0.5%, 1%, 2% |
  | 4 h | 1%, 2%, 5% |
  | 24 h | 2%, 5%, 10% |

- **Calibration**, walk-forward on the strategy's own past out-of-sample forecasts:
  - 3-class probabilities come from μ, σ and the empirical distribution of standardized
    residuals;
  - touch probabilities come from a logistic model on ln(x/σ) and μ/σ.
- **Hypotheses on the untouched test period:**
  - H1: accuracy > 50%.
  - H2: gross return > 0.
  - H3: long-only net > 0.
  - H4: the 3-class Brier score is below climatology.
  - H5: every touch target's Brier score is below climatology.

  Directional accuracy, calibration, price error and interval coverage are reported separately.
  A higher win rate alone is never treated as evidence.

## Friend's strategies (no tuning)

Definitions are in `cryptolab/strategies/friend.py` (S1, S2, S3).
- **Event study** at 60 minutes, 4 hours and 24 hours.
- **Stop/target paper trades** after base-case costs. If both are hit in the same minute, the
  stop is assumed first.
- **Benchmarks:**
  - 200 random-entry samples on the same coins, in the same split, with the same direction mix;
  - buy-and-hold;
  - the existing models at matching horizons (`docs/RESULTS.md`).
- **S3:** the historical version uses a Corwin–Schultz spread *estimate* and is labelled a
  proxy. The order-book version can only be evaluated forward, and is labelled unvalidated
  until enough live signals exist.

## Kalshi controlled test

See the docstring of `cryptolab/backtest/kalshi_test.py`.
- **Base model:** trained only before the first Kalshi window.
- **Arms:** WITHOUT uses (base score, move since window open); WITH adds logit(Kalshi YES mid
  at open + 2 minutes).
- **Training:** both arms on the first 60% of Kalshi windows.
- **Test:** the last 40%, paired, using only windows with a usable quote in both arms.
- **Primary metric:** the Brier difference. **Kalshi is adopted only if** the 95% bootstrap
  confidence interval of (WITH − WITHOUT) Brier lies entirely below 0. Secondary: accuracy
  (McNemar), log loss, gross returns.
- Live, both arms keep running independently. When no usable quote exists, WITH records a gap
  rather than falling back.

## Paper trading

- The **gated** portfolio trades only strategies listed in `[paper].approved_strategies`. That
  list stays empty until a strategy passes an out-of-sample test net of costs.
- **Shadow** portfolios trade every signal of one strategy, purely for measurement.
- Risk rules and the execution model are in `cryptolab/live/paper.py`. The parameters live in
  `config/default.toml` `[paper]`, and changing them requires a new version.
