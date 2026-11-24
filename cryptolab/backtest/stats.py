"""Statistical tools: confidence intervals, HAC t-stats, multiple-testing corrections,
bootstrap reality check, deflated Sharpe ratio, calibration helpers."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - hw, c + hw)


def binom_p_greater(k: int, n: int, p0: float = 0.5) -> float:
    if n == 0:
        return float("nan")
    return float(stats.binomtest(k, n, p0, alternative="greater").pvalue)


def newey_west_t(x: np.ndarray, lags: int | None = None) -> tuple[float, float, float]:
    """Mean, HAC standard error and t-stat (Newey-West, Bartlett kernel)."""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float("nan"),) * 3
    if lags is None:
        lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    m = x.mean()
    e = x - m
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    se = math.sqrt(max(s, 1e-30) / n)
    return m, se, m / se


def p_from_t(t: float) -> float:
    """One-sided p-value (H1: mean > 0) from a normal approximation."""
    if not np.isfinite(t):
        return float("nan")
    return float(1 - stats.norm.cdf(t))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(((p if np.isfinite(p) else 1.0, k) for k, p in pvals.items()))
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def benjamini_hochberg(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(((p if np.isfinite(p) else 1.0, k) for k, p in pvals.items()))
    m = len(items)
    out, prev = {}, 1.0
    for i in range(m - 1, -1, -1):
        p, k = items[i]
        prev = min(prev, p * m / (i + 1))
        out[k] = prev
    return out


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap (geometric block lengths), vectorized."""
    p = 1.0 / mean_block
    jumps = rng.random(n) < p
    jumps[0] = True
    starts = rng.integers(0, n, size=n)
    pos = np.arange(n)
    seg_first = np.maximum.accumulate(np.where(jumps, pos, 0))
    return (starts[seg_first] + (pos - seg_first)) % n


def reality_check(D: np.ndarray, n_boot: int = 1000, mean_block: float = 10.0, seed: int = 0) -> dict:
    """White's Reality Check (with Hansen's studentization-free re-centering).

    D: T x K matrix of performance differentials vs the benchmark (e.g. correct - 0.5).
    H0: no strategy beats the benchmark. Returns the p-value for the best strategy.
    """
    D = np.asarray(D, dtype="float64")
    T, K = D.shape
    rng = np.random.default_rng(seed)
    means = D.mean(axis=0)
    stat = math.sqrt(T) * means.max()
    centered = D - means
    count = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(T, mean_block, rng)
        bstat = math.sqrt(T) * centered[idx].mean(axis=0).max()
        if bstat >= stat:
            count += 1
    return {"stat": stat, "p_value": (count + 1) / (n_boot + 1), "best_index": int(means.argmax()), "T": T, "K": K}


def deflated_sharpe(sr: float, n: int, sr_trials: np.ndarray, skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Bailey & Lopez de Prado deflated Sharpe ratio (per-period SR)."""
    sr_trials = np.asarray([s for s in sr_trials if np.isfinite(s)])
    N = max(len(sr_trials), 1)
    var = sr_trials.var(ddof=1) if len(sr_trials) > 1 else 0.0
    emc = 0.5772156649
    if N > 1 and var > 0:
        sr0 = math.sqrt(var) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * math.e)))
    else:
        sr0 = 0.0
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    z = (sr - sr0) * math.sqrt(max(n - 1, 1)) / den
    return {"sr": sr, "sr0_expected_max_under_null": sr0, "dsr_prob": float(stats.norm.cdf(z)), "n_trials": N}


def fit_logistic_1d(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> tuple[float, float]:
    """Logistic regression y ~ sigmoid(a + b x) with mild L2 on b (Newton's method)."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return (0.0, 0.0)
    a, b = 0.0, 0.0
    for _ in range(50):
        z = np.clip(a + b * x, -30, 30)
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p)
        ga = np.sum(p - y)
        gb = np.sum((p - y) * x) + l2 * b
        haa = np.sum(w) + 1e-9
        hab = np.sum(w * x)
        hbb = np.sum(w * x * x) + l2 + 1e-9
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-8:
            break
    return float(a), float(b)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def reliability_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "freq": float(y[m].mean())})
    return out


def max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    return float((peak[1:] - cum).max())


def streaks(bools: np.ndarray) -> tuple[int, int]:
    best_w = best_l = cur_w = cur_l = 0
    for b in bools:
        if b:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return best_w, best_l
