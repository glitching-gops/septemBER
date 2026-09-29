"""Turn scored candidate pairs into per-S1 match sets.

    exclusive(pairs)          each S2/S3 record keeps only its highest-p S1 (ties -> smaller s1_id)
    decide_threshold(pairs)   every candidate with p >= tau, optionally capped per source
    decide_expected_f(pairs)  per S1, the prefix (possibly empty) maximising expected F0.5
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ber import compute
from ber.decide.expected_f import best_prefix


def exclusive(pairs: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    """Drop every (s1, m) row whose m_id is claimed with a higher p by another S1."""
    order = pairs.sort_values(["m_id", p_col, "s1_id"], ascending=[True, False, True])
    return order.drop_duplicates("m_id", keep="first").sort_index()


def decide_threshold(pairs: pd.DataFrame, tau: float, per_source_cap: int = 0, p_col: str = "p") -> pd.DataFrame:
    sel = pairs[pairs[p_col] >= tau]
    if per_source_cap:
        sel = (sel.sort_values(["s1_id", "source", p_col], ascending=[True, True, False])
               .groupby(["s1_id", "source"], group_keys=False).head(per_source_cap))
    return sel


def _ef_chunk(groups: list[tuple[str, np.ndarray, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    out = []
    for s1, m_ids, p in groups:
        order = np.argsort(-p, kind="stable")
        k, _ = best_prefix(p[order])
        out.append((s1, m_ids[order[:k]]))
    return out


def decide_expected_f(pairs: pd.DataFrame, p_col: str = "p", processes: int | None = None) -> pd.DataFrame:
    """Per S1, select the expected-F0.5-optimal prefix of candidates sorted by p."""
    if pairs.empty:
        return pairs.iloc[:0]
    g = pairs.sort_values("s1_id")
    s1 = g["s1_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1, len(s1)]
    m = g["m_id"].to_numpy()
    p = g[p_col].to_numpy(np.float64)
    groups = [(s1[a], m[a:b], p[a:b]) for a, b in zip(starts[:-1], starts[1:])]
    procs = processes or compute.n_jobs()
    if procs == 1 or len(groups) < 5000:
        res = _ef_chunk(groups)
    else:
        size = int(np.ceil(len(groups) / (procs * 4)))
        chunks = [groups[i:i + size] for i in range(0, len(groups), size)]
        with compute.pool(procs) as pool:
            res = [x for part in pool.map(_ef_chunk, chunks) for x in part]
    keep = {(s, mm) for s, ms in res for mm in ms}
    mask = [(a, b) in keep for a, b in zip(pairs["s1_id"], pairs["m_id"])]
    return pairs[np.asarray(mask, dtype=bool)]


def to_sets(selected: pd.DataFrame, s1_ids) -> dict[str, set]:
    out: dict[str, set] = {s: set() for s in s1_ids}
    for s, m in zip(selected["s1_id"], selected["m_id"]):
        out[s].add(m)
    return out
