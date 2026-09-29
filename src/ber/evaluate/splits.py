"""Leakage-safe validation splits.

* Connected components of S1 entities that share any matched S2/S3 id, so a
  given S2/S3 positive label never appears in two folds.
* Stratified group K-fold over those components (stratum = country x singleton).
* Leave-one-country-out splits derived from whatever country values the data holds.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold


def build_components(s1_ids, gt: Mapping[str, frozenset]) -> np.ndarray:
    """Component id per entry of ``s1_ids`` (bipartite S1 <-> matched-id graph).

    Component ids are relabelled 0..k-1 in order of first appearance in ``s1_ids``.
    """
    s1_ids = list(s1_ids)
    n1 = len(s1_ids)
    tgt_index: dict[str, int] = {}
    rows, cols = [], []
    for i, s1 in enumerate(s1_ids):
        for m in gt.get(s1, ()):
            j = tgt_index.setdefault(m, len(tgt_index))
            rows.append(i)
            cols.append(n1 + j)
    n = n1 + len(tgt_index)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    labels = labels[:n1]
    _, first_idx, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first_idx))
    return order[inverse]


def component_size_table(component_id: np.ndarray) -> pd.Series:
    sizes = np.bincount(component_id)
    return pd.Series(sizes).value_counts().sort_index().rename_axis("component_size").rename("n_components")


def make_cv_folds(
    component_id: np.ndarray,
    strata: np.ndarray,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Fold index per row using StratifiedGroupKFold over components.

    A component's stratum is taken from its first row, so every row passed to
    the splitter carries its component's stratum (StratifiedGroupKFold needs
    one label per group to behave predictably).
    """
    component_id = np.asarray(component_id)
    strata = pd.Series(np.asarray(strata))
    comp_stratum = strata.groupby(component_id).transform("first").to_numpy()
    y = pd.factorize(comp_stratum)[0]
    folds = np.full(len(component_id), -1, dtype=np.int16)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=component_id)):
        folds[te] = k
    assert (folds >= 0).all()
    return folds


def make_loco(countries: pd.Series) -> pd.DataFrame:
    """One held-out split per distinct country value observed in ``countries``.

    Returns a frame aligned with ``countries`` holding one column per split,
    named ``loco:<value>``, with ``"valid"`` for the held-out rows and
    ``"train"`` otherwise.
    """
    values = pd.Series(countries).reset_index(drop=True)
    return pd.DataFrame(
        {f"loco:{c}": np.where(values == c, "valid", "train") for c in sorted(values.unique())}
    )


def check_no_positive_leak(folds: np.ndarray, s1_ids, gt: Mapping[str, frozenset]) -> int:
    """Number of S2/S3 ids whose positives appear in more than one fold (should be 0)."""
    seen: dict[str, int] = {}
    leaks = set()
    for f, s1 in zip(folds, s1_ids):
        for m in gt.get(s1, ()):
            prev = seen.setdefault(m, int(f))
            if prev != f:
                leaks.add(m)
    return len(leaks)
