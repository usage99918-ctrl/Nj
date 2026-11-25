# Pre-registration

These rules were fixed in code (and committed) **before** any validation or test metric was
computed. The machine-readable record of each decision is in the hash-chained ledger
(`data/ledger/*.jsonl`, record types `preregistration` and `model_version`), written in this
order: validation selection → test simulation → test result → live model.

## Research design

| Item | Rule |
|---|---|
| Data | Binance spot 1-minute klines (official bulk files + public API), USDT quote |
| Universe | Each month: top 30 USDT pairs by trailing 30-day quote volume, computed only from data before the month. Candidates include **delisted** pairs (exchangeInfo status BREAK). Excluded: stablecoins/pegs (static list + trailing daily volatility < 0.3%), wrapped/staked duplicates, leveraged tokens, tokenized equities |
| Decision times | Every 10 min (Experiment A) / every 60 min (Experiment B), UTC-aligned |
| Target | log(last trade price at t+h / last trade price at t), from 1-minute bar closes |
| Splits | Train 2026-01-01 → 06-30 (walk-forward, first OOS month March) · Validation 07-01 → 09-30 · **Test 10-01 → 11-22 (untouched until the selection was frozen)** |
| Walk-forward | Monthly refit on all decisions with t < month start − 1-day embargo − horizon |
| Candidate selection per decision | coin with the largest \|z\| (ties → more liquid) — identical in backtest and live |
| Calibration | b (expected-return scale, ≥ 0), P(direction correct) = logistic on \|μ/σ\|, interval quantiles of standardized residuals — all fit on the strategy's own **past** out-of-sample forecasts |
| Costs (base case) | Binance taker 0.10%/side + half-spread 0.5 bp (top-10 liquidity) or 1.5 bp + 1 bp slippage per side; low-cost sensitivity 0.02%/side |
| Flat outcomes | identical start/end prices → **incorrect** for both UP and DOWN |

## Model selection rule (per horizon)

1. Candidates: all 100 strategies (constant/random baselines excluded; simple momentum and
   simple mean reversion are eligible because they are also family members).
2. Must have validation coverage ≥ 90% of decision times and ≥ 300 validation forecasts.
3. Choose the highest **validation balanced accuracy** (mean of hit rates on up-moves and on
   down-moves — a constant UP/DOWN rule scores exactly 50%, so market drift cannot win).
   Tie-break: validation mean cross-sectional rank IC.
4. Freeze the choice in the ledger before simulating the test months.

## Test hypotheses (evaluated once, on the selected model only)

- H1: test directional accuracy > 50% (one-sided exact binomial, α = 0.05)
- H2: test mean gross directional return > 0 (Newey-West t, one-sided, α = 0.05)
- H3: test mean long-only return after base-case costs > 0 (t > 1.645)

A predictive advantage is claimed only if H1 or H2 passes; a *tradable* advantage only if H3
also passes. All other strategies' test results are reported for transparency but are not
used to change the selection (multiple-testing corrections are reported alongside).

## Live (forward) experiments

- Live model = the selected strategy refit on all history through 2026-11-22, calibrated on
  all of its out-of-sample backtest forecasts. Version `<experiment>-v1-<strategy>`; its file
  hash is in the ledger and is checked before every forecast.
- One forecast per slot (10 min / 60 min). A slot is attempted only within 5 / 20 minutes of
  its start; otherwise it is recorded as a gap. No back-filling, no re-scoring.
- Model changes create a new version (v2, …) evaluated only on forecasts made after it was
  frozen. Versions are never changed after individual losses.
- Forward forecasts and backtests are reported separately and never pooled.
