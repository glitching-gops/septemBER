"""Gradient-boosted pair model (XGBoost, GPU) with out-of-fold isotonic calibration.

The decision stage (expected-F0.5 prefix search) consumes probabilities, so a raw GBDT
score is not enough: the P1 audit measured ECE 0.00006 for the calibrated linear model
and the prefix search relies on that. We keep the same calibration contract here --
isotonic, fitted out-of-fold with splits grouped by S1 so no entity's pairs leak across
the calibration boundary.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

DEFAULTS = dict(objective="binary:logistic", device="cuda", tree_method="hist",
                n_estimators=600, max_depth=8, learning_rate=0.08, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=10, max_bin=256, eval_metric="aucpr")


def make_xgb(cfg_xgb: dict | None = None, seed: int = 42, **over):
    """An XGBClassifier on CUDA. Raises if xgboost has no CUDA support (never CPU silently)."""
    import xgboost as xgb

    p = {**DEFAULTS, **(cfg_xgb or {}), **over}
    p.pop("early_stopping_rounds", None)  # only meaningful with an eval set
    if p.get("device") != "cuda":
        raise RuntimeError(f"GPU-first policy: xgb device must be 'cuda', got {p.get('device')!r}")
    return xgb.XGBClassifier(random_state=seed, n_jobs=1, verbosity=0, **p)


def fit_calibrated(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42,
                   n_inner: int = 1, cfg_xgb: dict | None = None):
    """XGBoost + isotonic calibration, inner splits grouped by S1.

    ``n_inner=1`` (default) uses ONE grouped 75/25 split: the booster is fitted on 75% of
    the training folds and the isotonic map on the held-out 25%. That is still strictly
    out-of-fold with respect to the outer test fold, and costs one GPU fit per outer fold
    instead of ``n_inner``. Raise ``n_inner`` for the usual averaged-ensemble behaviour.
    """
    if n_inner <= 1:
        splits = [list(GroupKFold(n_splits=4).split(X, y, groups))[0]]
    else:
        splits = list(GroupKFold(n_splits=n_inner).split(X, y, groups))
    clf = CalibratedClassifierCV(make_xgb(cfg_xgb, seed), method="isotonic", cv=splits)
    clf.fit(X, y)
    return clf


def fit_plain(X: np.ndarray, y: np.ndarray, seed: int = 42, cfg_xgb: dict | None = None,
              eval_set=None, early_stopping_rounds: int | None = None):
    """Uncalibrated fit, used for feature importance and for the NIL model."""
    import xgboost as xgb

    p = {**DEFAULTS, **(cfg_xgb or {})}
    p.pop("early_stopping_rounds", None)
    kw = {}
    if eval_set is not None and early_stopping_rounds:
        kw["eval_set"] = eval_set
        p["early_stopping_rounds"] = early_stopping_rounds
    m = xgb.XGBClassifier(random_state=seed, n_jobs=1, verbosity=0, **p)
    m.fit(X, y, verbose=False, **kw)
    return m


def predict(clf, X: np.ndarray, batch: int = 4_000_000) -> np.ndarray:
    """Calibrated positive-class probability, in batches so VRAM stays bounded."""
    out = np.empty(len(X), np.float32)
    for a in range(0, len(X), batch):
        b = min(a + batch, len(X))
        out[a:b] = clf.predict_proba(X[a:b])[:, 1].astype(np.float32)
    return out


def importances(clf, names: list[str]) -> dict[str, float]:
    """Mean gain importance across the calibrated inner models, normalised to sum 1."""
    tot = np.zeros(len(names))
    n = 0
    for cc in getattr(clf, "calibrated_classifiers_", []):
        est = cc.estimator
        est = est[-1] if hasattr(est, "__getitem__") and not hasattr(est, "get_booster") else est
        score = est.get_booster().get_score(importance_type="gain")
        for i, nm in enumerate(names):
            tot[i] += score.get(f"f{i}", 0.0)
        n += 1
    if n:
        tot /= n
    s = tot.sum() or 1.0
    return dict(zip(names, (tot / s).round(5).tolist()))
