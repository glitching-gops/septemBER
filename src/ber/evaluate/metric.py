"""Per-entity F0.5 and its macro average over Source-1 entities.

Edge cases (from the challenge spec):
    truth empty, pred empty         -> 1.0
    truth empty, pred non-empty     -> 0.0
    truth non-empty, pred empty     -> 0.0
    otherwise F0.5 = 1.25 * P * R / (0.25 * P + R)
"""
from __future__ import annotations

import warnings
from typing import Iterable, Mapping

BETA2 = 0.25  # beta = 0.5


def f05_entity(pred: Iterable[str], truth: Iterable[str]) -> float:
    pred = set(pred)
    truth = set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(
    pred: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str],
) -> dict:
    """Macro-average F0.5 over ``s1_ids``.

    An id absent from ``pred`` is scored as an empty prediction (with a warning).
    An id absent from ``truth`` is treated as a singleton.
    """
    s1_ids = list(s1_ids)
    if not s1_ids:
        raise ValueError("s1_ids is empty")

    missing = 0
    total = sing_sum = nons_sum = 0.0
    n_sing = n_nons = 0
    n_sing_fp = n_missed_all = n_perfect = n_partial = 0

    for s1 in s1_ids:
        if s1 in pred:
            p = set(pred[s1])
        else:
            p = set()
            missing += 1
        t = set(truth.get(s1, ()))
        f = f05_entity(p, t)
        total += f
        if not t:
            n_sing += 1
            sing_sum += f
            if p:
                n_sing_fp += 1
        else:
            n_nons += 1
            nons_sum += f
            if f == 1.0:
                n_perfect += 1
            elif not (p & t):
                n_missed_all += 1
            else:
                n_partial += 1

    if missing:
        warnings.warn(
            f"{missing} S1 id(s) missing from predictions; scored as empty",
            UserWarning,
            stacklevel=2,
        )

    return {
        "macro": total / len(s1_ids),
        "n": len(s1_ids),
        "n_singleton": n_sing,
        "n_nonsingleton": n_nons,
        "singleton_mean": sing_sum / n_sing if n_sing else float("nan"),
        "nonsingleton_mean": nons_sum / n_nons if n_nons else float("nan"),
        "n_singleton_fp": n_sing_fp,
        "n_missed_all": n_missed_all,
        "n_perfect": n_perfect,
        "n_partial": n_partial,
        "n_missing_pred": missing,
    }


def macro_f05_fast(sel_s1, sel_m, truth_s1, truth_m, s1_ids) -> float:
    """Vectorised macro F0.5 for grids: predictions and truth given as parallel id arrays.

    Equal to ``macro_f05`` (tested); ids absent from both are empty/empty -> 1.0.
    """
    import numpy as np
    import pandas as pd

    s1_ids = pd.Index(list(s1_ids))
    pred = pd.DataFrame({"s": sel_s1, "m": sel_m}).drop_duplicates()
    truth = pd.DataFrame({"s": truth_s1, "m": truth_m}).drop_duplicates()
    npred = pred.groupby("s").size().reindex(s1_ids, fill_value=0).to_numpy(float)
    ntrue = truth.groupby("s").size().reindex(s1_ids, fill_value=0).to_numpy(float)
    tp = pred.merge(truth, on=["s", "m"]).groupby("s").size().reindex(s1_ids, fill_value=0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / npred, 0.0)
        r = np.where(ntrue > 0, tp / ntrue, 0.0)
        f = np.where(tp > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    f = np.where((ntrue == 0) & (npred == 0), 1.0, f)
    return float(f.mean())
