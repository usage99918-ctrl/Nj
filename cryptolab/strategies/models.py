"""Fitted (walk-forward) strategies: time-series, regression, classification, trees,
ensembles and regime switching. Hyper-parameters are fixed a priori (not tuned on
validation or test data); each configuration counts as a separate tested strategy."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from cryptolab.features import model_feature_columns
from cryptolab.strategies.library import Strategy, _clipz, _sd

log = logging.getLogger(__name__)
MAX_TRAIN_ROWS = 400_000


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _target(df: pd.DataFrame) -> np.ndarray:
    return np.clip(df["y_std"].values.astype("float64"), -10, 10)


class _Standardizer:
    def fit(self, X: np.ndarray):
        self.med = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.scale = np.where((q3 - q1) > 1e-12, (q3 - q1) / 1.349, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.med) / self.scale
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return np.clip(Z, -8, 8)


class FeatureModel:
    """Generic sklearn/lightgbm wrapper over a feature subset."""

    def __init__(self, make, cols=None, classify=False, seed=0, standardize=True, max_rows=MAX_TRAIN_ROWS):
        self.make, self.cols_spec, self.classify, self.seed = make, cols, classify, seed
        self.standardize, self.max_rows = standardize, max_rows

    def _cols(self, df):
        if self.cols_spec is None:
            return model_feature_columns(df)
        return [c for c in self.cols_spec if c in df.columns]

    def fit(self, train: pd.DataFrame, h: int) -> None:
        tr = _subsample(train, self.max_rows, self.seed)
        self.cols = self._cols(tr)
        X = tr[self.cols].values.astype("float64")
        self.std = _Standardizer().fit(X)
        X = self.std.transform(X) if self.standardize else np.where(np.isfinite(X), X, np.nan)
        y = _target(tr)
        self.m = self.make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.classify:
                self.m.fit(X, (y > 0).astype(int))
            else:
                self.m.fit(X, y)

    def predict(self, df: pd.DataFrame, h: int) -> np.ndarray:
        X = df[self.cols].values.astype("float64")
        X = self.std.transform(X) if self.standardize else np.where(np.isfinite(X), X, np.nan)
        if self.classify:
            p = np.clip(self.m.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
            return np.log(p / (1 - p))
        return self.m.predict(X)


class ARPooled:
    """Pooled AR(3) on standardized lagged horizon returns (no intercept: no drift bet)."""

    def __init__(self, h):
        if h == 10:
            self.spec = [("r_10", None, 10), ("r_30", "r_10", 20), ("r_60", "r_30", 30)]
        else:
            self.spec = [("r_60", None, 60), ("r_240", "r_60", 180), ("r_1440", "r_240", 1200)]

    def _X(self, df):
        cols = []
        for a, b, L in self.spec:
            v = df[a].astype("float64") - (df[b].astype("float64") if b else 0.0)
            cols.append((v / _sd(df, L)).values)
        X = np.column_stack(cols)
        return np.clip(np.where(np.isfinite(X), X, 0.0), -10, 10)

    def fit(self, train, h):
        X, y = self._X(train), _target(train)
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(self, df, h):
        return self._X(df) @ self.beta


class HourSeasonality:
    """Mean standardized return by UTC hour, shrunk toward zero (n / (n + 20000))."""

    def fit(self, train, h):
        g = pd.DataFrame({"hour": train["hour"].values, "y": _target(train)}).groupby("hour")["y"].agg(["mean", "count"])
        self.m = (g["mean"] * g["count"] / (g["count"] + 20_000)).to_dict()

    def predict(self, df, h):
        return df["hour"].map(self.m).fillna(0.0).values * 20


def _rule_matrix(rules: list[Strategy], df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([r.predict(df, 0) for r in rules])


class EnsembleEqual:
    """Equal-weight average of cross-sectionally standardized representative rule scores."""
    REPS = ["mom_60", "rev_10", "brk_follow_240", "ofi_follow_10", "rv_mom_60", "resid_rev_60",
            "btc_lead_10", "rsi_rev_14"]

    def __init__(self, rules):
        self.rules = [r for r in rules if r.name in self.REPS]

    def fit(self, train, h):
        pass

    def predict(self, df, h):
        M = _rule_matrix(self.rules, df)
        t = df["t"].values
        out = np.zeros(len(df))
        Mdf = pd.DataFrame(M)
        Mdf["t"] = t
        g = Mdf.groupby("t")
        mean = g.transform("mean").values
        std = g.transform("std").values
        Z = np.where(std > 0, (M - mean) / std, 0.0)
        out = np.nanmean(Z, axis=1)
        return out


class EnsembleTopIC:
    """Average of the k rules with the highest positive correlation with y_std in training."""

    def __init__(self, rules, k=5):
        self.all_rules, self.k = [r for r in rules if not r.baseline or True], k

    def fit(self, train, h):
        tr = _subsample(train, 300_000, 1)
        M = _rule_matrix(self.all_rules, tr)
        y = _target(tr)
        sd = M.std(axis=0)
        cors = np.array([np.corrcoef(M[:, i], y)[0, 1] if sd[i] > 0 else -1 for i in range(M.shape[1])])
        cors = np.where(np.isfinite(cors), cors, -1)
        idx = [i for i in np.argsort(-cors)[: self.k] if cors[i] > 0]
        self.sel = [self.all_rules[i] for i in idx]
        self.scales = [sd[i] for i in idx]
        self.chosen = [(self.all_rules[i].name, float(cors[i])) for i in idx]

    def predict(self, df, h):
        if not self.sel:
            return np.zeros(len(df))
        M = _rule_matrix(self.sel, df) / np.array(self.scales)
        return M.mean(axis=1)


class EnsembleStack:
    """Ridge stacking of all rule scores (learned weights, walk-forward)."""

    def __init__(self, rules, alpha=100.0):
        self.rules, self.alpha = rules, alpha

    def fit(self, train, h):
        from sklearn.linear_model import Ridge
        tr = _subsample(train, 300_000, 2)
        M = _rule_matrix(self.rules, tr)
        self.sd = np.where(M.std(axis=0) > 0, M.std(axis=0), 1.0)
        self.m = Ridge(alpha=self.alpha).fit(M / self.sd, _target(tr))

    def predict(self, df, h):
        return self.m.predict(_rule_matrix(self.rules, df) / self.sd)


class EnsembleModels:
    """Average of standardized predictions of already-fitted component model strategies."""

    def __init__(self, components: list[Strategy]):
        self.components = components

    def fit(self, train, h):
        tr = _subsample(train, 100_000, 3)
        self.scales = []
        for c in self.components:
            s = np.std(c.predict(tr, h))
            self.scales.append(s if s > 0 else 1.0)

    def predict(self, df, h):
        return np.mean([c.predict(df, h) / s for c, s in zip(self.components, self.scales)], axis=0)


class RegimeSwitch:
    """Four regimes (BTC 24h trend up/down x BTC vol ratio high/low); in each regime use the
    candidate rule with the best positive training correlation (abstain if none)."""
    CANDIDATES = ["mom_60", "rev_10", "brk_follow_240", "rsi_rev_14", "resid_rev_60", "btc_lead_10",
                  "ofi_follow_10", "ofi_fade_10"]

    def __init__(self, rules):
        self.rules = [r for r in rules if r.name in self.CANDIDATES]

    @staticmethod
    def _regime(df):
        return (df["btc_trend"].values > 0).astype(int) * 2 + (df["btc_vol_ratio"].values > 1).astype(int)

    def fit(self, train, h):
        tr = _subsample(train, 300_000, 4)
        M = _rule_matrix(self.rules, tr)
        y = _target(tr)
        reg = self._regime(tr)
        self.choice = {}
        for r in range(4):
            m = reg == r
            best, bc = None, 0.0
            for i in range(M.shape[1]):
                if m.sum() < 1000 or M[m, i].std() == 0:
                    continue
                c = np.corrcoef(M[m, i], y[m])[0, 1]
                if np.isfinite(c) and c > bc:
                    best, bc = i, c
            self.choice[r] = best

    def predict(self, df, h):
        M = _rule_matrix(self.rules, df)
        reg = self._regime(df)
        out = np.zeros(len(df))
        for r, i in self.choice.items():
            if i is not None:
                m = reg == r
                out[m] = M[m, i]
        return out


def model_strategies(h: int, seed: int, rules: list[Strategy]) -> list[Strategy]:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor

    mom_cols = ["r_5", "r_10", "r_30", "r_60", "r_240", "r_1440", "ma_x_10_60", "ma_x_30_240", "ma_x_60_1440",
                "macd_hist", "ewm_drift_60", "ewm_drift_240", "brk_60", "brk_240", "brk_1440"]
    rev_cols = ["r_1", "r_5", "r_10", "dev_ma_30", "dev_ma_60", "dev_ma_240", "rsi_14", "rsi_60", "bb_pctb_60",
                "bb_pctb_240", "resid_r_10", "resid_r_60", "xs_rel_r_10", "xs_rel_r_60"]
    lgb_common = dict(learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                      random_state=seed, n_jobs=4, verbose=-1)
    S: list[Strategy] = []

    def add(name, family, hyp, model, **params):
        S.append(Strategy(name, family, hyp, model=model, kind="model", params=params))

    add("ar_pooled", "time_series", "Pooled AR(3) on standardized lagged horizon returns.", ARPooled(h))
    add("hour_seasonality", "time_series", "Hour-of-day return seasonality (shrunk).", HourSeasonality())
    for a in (1.0, 10.0, 100.0):
        add(f"ridge_all_a{int(a)}", "regression", f"Ridge regression (alpha={a}) on all features.",
            FeatureModel(lambda a=a: Ridge(alpha=a), seed=seed), alpha=a)
    add("ridge_momentum_set", "regression", "Ridge on trend/momentum features only.",
        FeatureModel(lambda: Ridge(alpha=10.0), cols=mom_cols, seed=seed))
    add("ridge_reversal_set", "regression", "Ridge on short-term reversal features only.",
        FeatureModel(lambda: Ridge(alpha=10.0), cols=rev_cols, seed=seed))
    add("huber_all", "regression", "Robust (Huber-loss) linear regression on all features.",
        FeatureModel(lambda: SGDRegressor(loss="huber", epsilon=1.0, alpha=1e-4, max_iter=20, tol=None,
                                          random_state=seed), seed=seed, max_rows=300_000))
    for C in (0.01, 0.1, 1.0):
        add(f"logit_C{C}", "classification", f"L2 logistic regression (C={C}) for P(up).",
            FeatureModel(lambda C=C: LogisticRegression(C=C, max_iter=300), classify=True, seed=seed,
                         max_rows=300_000), C=C)
    add("lgbm_reg_shallow", "tree_ml", "LightGBM regression, 15 leaves, strong regularization.",
        FeatureModel(lambda: LGBMRegressor(n_estimators=200, num_leaves=15, min_child_samples=2000, **lgb_common),
                     seed=seed, standardize=False))
    add("lgbm_reg_medium", "tree_ml", "LightGBM regression, 63 leaves.",
        FeatureModel(lambda: LGBMRegressor(n_estimators=300, num_leaves=63, min_child_samples=1000, **lgb_common),
                     seed=seed, standardize=False))
    add("lgbm_reg_huber", "tree_ml", "LightGBM with Huber loss (robust to fat tails).",
        FeatureModel(lambda: LGBMRegressor(objective="huber", alpha=1.5, n_estimators=200, num_leaves=15,
                                           min_child_samples=2000, **lgb_common), seed=seed, standardize=False))
    add("lgbm_clf_shallow", "tree_ml", "LightGBM classifier for P(up), 15 leaves.",
        FeatureModel(lambda: LGBMClassifier(n_estimators=200, num_leaves=15, min_child_samples=2000, **lgb_common),
                     classify=True, seed=seed, standardize=False))
    add("lgbm_clf_medium", "tree_ml", "LightGBM classifier for P(up), 63 leaves.",
        FeatureModel(lambda: LGBMClassifier(n_estimators=300, num_leaves=63, min_child_samples=1000, **lgb_common),
                     classify=True, seed=seed, standardize=False))
    add("extra_trees", "tree_ml", "Extremely randomized trees (depth 8).",
        FeatureModel(lambda: ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=500,
                                                 max_features=0.5, n_jobs=4, random_state=seed),
                     seed=seed, max_rows=200_000))
    comp = [s for s in S if s.name in ("ridge_all_a10", "lgbm_reg_shallow", "logit_C0.1")]
    add("ens_family_equal", "ensemble", "Equal-weight average of 8 family-representative rule signals.",
        EnsembleEqual(rules))
    add("ens_top5_ic", "ensemble", "Average of the 5 rules with the best positive training correlation.",
        EnsembleTopIC(rules))
    add("ens_stack_ridge", "ensemble", "Ridge-stacked combination of all rule signals.", EnsembleStack(rules))
    add("ens_ml_avg", "ensemble", "Average of ridge, LightGBM and logistic model forecasts.", EnsembleModels(comp))
    add("regime_switch", "regime", "Regime-dependent choice of rule (BTC trend x volatility).", RegimeSwitch(rules))
    return S
