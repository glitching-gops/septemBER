"""Fold bookkeeping for the verifier: strict holdout (D8), 2-fold verifier split (D7), features.

Invariants (tested in tests/test_verifier_folds.py):

* the holdout is a set of whole P0 components, so no true pair straddles it;
* ``verifier_fold = xgb_fold mod 2``, so the verifier model that scores a pair never
  trained on that pair's S1 (the OOF property), and a stage-1 fold k is always scored by
  one verifier model only;
* holdout S1 are in no verifier training set and no XGBoost training set; like test S1,
  they are scored by the mean of both verifier fold models.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

V_FEATURES = ["v_logit", "v_p", "v_inband", "v_rank", "v_margin"]
V_FEATURES_NLI = V_FEATURES + ["v_pcontra"]


def holdout_mask(component_ids, frac: float, salt: str = "p3-holdout") -> np.ndarray:
    """True for components whose stable hash falls in the first ``frac`` of the space."""
    cut = int(round(frac * 10_000))
    return np.fromiter((zlib.crc32(f"{salt}:{c}".encode()) % 10_000 < cut for c in component_ids),
                       bool, len(component_ids))


def verifier_fold(xgb_fold) -> np.ndarray:
    return np.asarray(xgb_fold, dtype=np.int64) % 2


def in_band(p, band) -> np.ndarray:
    p = np.asarray(p)
    return (p >= band[0]) & (p <= band[1])


def verifier_features(pairs: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Per-pair verifier features aligned to ``pairs`` (s1_id, m_id); NaN outside the band.

    ``scored`` holds the band pairs with ``logit`` and ``p`` (and ``p_contra`` for NLI).
    Rank and margin are computed among the S1's band pairs, by logit.
    """
    s = scored[["s1_id", "m_id", "logit", "p"] + (["p_contra"] if "p_contra" in scored else [])].copy()
    s["v_rank"] = s.groupby("s1_id")["logit"].rank(ascending=False, method="first")
    s["v_margin"] = s["logit"] - s.groupby("s1_id")["logit"].transform("max")
    s = s.rename(columns={"logit": "v_logit", "p": "v_p", "p_contra": "v_pcontra"})
    out = pairs[["s1_id", "m_id"]].merge(s, on=["s1_id", "m_id"], how="left")
    out["v_inband"] = out["v_logit"].notna().astype(np.float32)
    cols = V_FEATURES_NLI if "v_pcontra" in out else V_FEATURES
    return out[cols].astype(np.float32).reset_index(drop=True)
