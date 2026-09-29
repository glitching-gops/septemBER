"""Relational, country-agnostic pair features for (S1, S2/S3) candidate pairs.

All similarity features compare the two records; nothing depends on the country
value itself. Missing data is encoded explicitly as "missing" (a 0/1 indicator
with the similarity set to 0), never as a mismatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from ber import compute

NORM_COLS = ["entity_id", "name_core", "name_alt", "legal_class", "addr_core", "addr_missing",
             "house_no", "postcode_like", "nums", "landmark"]
# plus "name_key" (transliteration-mapped / segmented name, from *_keys.parquet)

FEATURES = [
    "name_tsr", "name_tsort", "name_partial", "tfidf_name_cos", "name_exact", "key_tsr", "name_cos", "addr_cos",
    "legal_agree", "legal_conflict",
    "addr_tsr", "addr_missing",
    "hn_exact", "hn_fuzzy", "hn_missing",
    "pc_agree", "pc_conflict",
    "nums_jacc", "landmark_jacc",
    "best_rank_log", "b4_hit", "b6n_hit", "b6a_hit", "b7_hit", "n_blockers", "is_s3",
]

# PVT_LTD and LTD are the same family: one side often drops "private".
_COMPATIBLE = {frozenset({"PVT_LTD", "LTD"}), frozenset({"LLC", "PLLC"}), frozenset({"SAS", "SASU"}),
               frozenset({"INC", "CORP"})}


def _best_alt(left_alts: pd.Series, right_alts: pd.Series, scorer, workers: int = 1) -> np.ndarray:
    """Max ``scorer`` over all (left alt, right alt) combinations, per row."""
    li, ls, ri, rs = [], [], [], []
    for i, (la, ra) in enumerate(zip(left_alts, right_alts)):
        la = list(la) if la is not None and len(la) else [""]
        ra = list(ra) if ra is not None and len(ra) else [""]
        for x in la:
            for y in ra:
                li.append(i)
                ls.append(x)
                rs.append(y)
    scores = cpdist(ls, rs, scorer=scorer, workers=workers, dtype=np.float32)
    out = np.zeros(len(left_alts), dtype=np.float32)
    np.maximum.at(out, np.asarray(li), scores)
    return out


def _legal(a, b) -> tuple[int, int]:
    a = set(a) if a is not None else set()
    b = set(b) if b is not None else set()
    if not a or not b:
        return 0, 0
    if a & b:
        return 1, 0
    if any(frozenset({x, y}) in _COMPATIBLE for x in a for y in b):
        return 1, 0
    return 0, 1


def _house(a: str, b: str) -> tuple[int, int, int]:
    """(exact, fuzzy, missing). Fuzzy: edit distance <= 1, or one side's range covers the other."""
    if not a or not b:
        return 0, 0, 1
    if a == b:
        return 1, 0, 0

    def rng(x):
        parts = [p for p in x.replace("/", "-").split("-") if p.isdigit()]
        if len(parts) >= 2 and len(parts[0]) == len(parts[1]):
            lo, hi = sorted((int(parts[0]), int(parts[1])))
            return lo, hi
        if parts:
            v = int(parts[0])
            return v, v
        return None

    ra, rb = rng(a), rng(b)
    if ra and rb and ra[0] <= rb[1] and rb[0] <= ra[1]:
        return 0, 1, 0
    if Levenshtein.distance(a, b) <= 1:
        return 0, 1, 0
    return 0, 0, 0


def _jacc(a, b) -> float:
    a = set(a) if a is not None else set()
    b = set(b) if b is not None else set()
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(cand: pd.DataFrame, s1n: pd.DataFrame, mn: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """Feature matrix aligned with ``cand`` rows.

    ``cand`` needs s1_id, m_id, source and the blocker columns; ``s1n``/``mn`` are
    normalised S1 and S2+S3 tables (columns NORM_COLS).
    """
    # Accept tables that are already indexed by entity_id, so a sharded caller can index
    # the full normalised tables once and reuse them for every shard.
    li = s1n if s1n.index.name == "entity_id" else s1n.set_index("entity_id")
    ri = mn if mn.index.name == "entity_id" else mn.set_index("entity_id")
    L = li.loc[cand["s1_id"].to_numpy()].reset_index(drop=True)
    R = ri.loc[cand["m_id"].to_numpy()].reset_index(drop=True)
    w = workers or compute.n_jobs()
    f = pd.DataFrame(index=np.arange(len(cand)))

    f["name_tsr"] = _best_alt(L["name_alt"], R["name_alt"], fuzz.token_set_ratio, w)
    f["name_tsort"] = _best_alt(L["name_alt"], R["name_alt"], fuzz.token_sort_ratio, w)
    f["name_partial"] = cpdist(L["name_core"].tolist(), R["name_core"].tolist(), scorer=fuzz.partial_ratio,
                               workers=w, dtype=np.float32)
    f["tfidf_name_cos"] = cand["b1_score"].fillna(0).to_numpy(np.float32) if "b1_score" in cand else 0.0
    f["name_exact"] = (L["name_core"].to_numpy() == R["name_core"].to_numpy()) & (L["name_core"].to_numpy() != "")
    if "name_key" in L and "name_key" in R:
        f["key_tsr"] = cpdist(L["name_key"].tolist(), R["name_key"].tolist(), scorer=fuzz.token_set_ratio,
                              workers=w, dtype=np.float32)
    else:
        f["key_tsr"] = f["name_tsr"]
    # exact E5 cosines for every candidate pair (precomputed by the caller from the embeddings)
    f["name_cos"] = cand["name_cos"].fillna(0).to_numpy(np.float32) if "name_cos" in cand else 0.0

    lg = [_legal(a, b) for a, b in zip(L["legal_class"], R["legal_class"])]
    f["legal_agree"] = np.array([x[0] for x in lg], np.int8)
    f["legal_conflict"] = np.array([x[1] for x in lg], np.int8)

    miss = L["addr_missing"].to_numpy(bool) | R["addr_missing"].to_numpy(bool)
    at = cpdist(L["addr_core"].tolist(), R["addr_core"].tolist(), scorer=fuzz.token_set_ratio, workers=w,
                dtype=np.float32)
    f["addr_tsr"] = np.where(miss, 0.0, at).astype(np.float32)
    ac = cand["addr_cos"].fillna(0).to_numpy(np.float32) if "addr_cos" in cand else np.zeros(len(cand), np.float32)
    f["addr_cos"] = np.where(miss, 0.0, ac).astype(np.float32)
    f["addr_missing"] = miss.astype(np.int8)

    hs = [_house(a, b) for a, b in zip(L["house_no"], R["house_no"])]
    f["hn_exact"] = np.array([x[0] for x in hs], np.int8)
    f["hn_fuzzy"] = np.array([x[1] for x in hs], np.int8)
    f["hn_missing"] = np.array([x[2] for x in hs], np.int8)

    pa, pc = [], []
    for a, b in zip(L["postcode_like"], R["postcode_like"]):
        a = set(a) if a is not None else set()
        b = set(b) if b is not None else set()
        if a and b:
            pa.append(int(bool(a & b)))
            pc.append(int(not (a & b)))
        else:
            pa.append(0)
            pc.append(0)
    f["pc_agree"] = np.array(pa, np.int8)
    f["pc_conflict"] = np.array(pc, np.int8)

    f["nums_jacc"] = np.array([_jacc(a, b) for a, b in zip(L["nums"], R["nums"])], np.float32)
    f["landmark_jacc"] = np.array(
        [_jacc(a.split() if a else (), b.split() if b else ()) for a, b in zip(L["landmark"], R["landmark"])], np.float32)

    rcols = [c_ for c_ in ("b1_rank", "b2_rank", "b3_rank", "b6n_rank", "b6a_rank") if c_ in cand]
    ranks = np.array(cand[rcols].fillna(0).to_numpy(np.float32), copy=True)
    ranks[ranks == 0] = 99.0  # not retrieved by the TF-IDF blockers
    best = ranks.min(axis=1)
    f["best_rank_log"] = np.log1p(best).astype(np.float32)
    b4 = (cand["b4_hn"].fillna(False).to_numpy(bool) | cand["b4_name"].fillna(False).to_numpy(bool)) if "b4_hn" in cand \
        else np.zeros(len(cand), bool)
    f["b4_hit"] = b4.astype(np.int8)

    def hit(col):
        return (cand[col].fillna(0).to_numpy() > 0).astype(np.int8) if col in cand else np.zeros(len(cand), np.int8)

    f["b6n_hit"], f["b6a_hit"], f["b7_hit"] = hit("b6n_rank"), hit("b6a_rank"), hit("b7_rank")
    rank_cols = [c_ for c_ in ("b1_rank", "b2_rank", "b3_rank", "b6n_rank", "b6a_rank", "b7_rank") if c_ in cand]
    f["n_blockers"] = ((cand[rank_cols].fillna(0).to_numpy() > 0).sum(axis=1)
                       + f["b4_hit"].to_numpy()).astype(np.int8)
    f["is_s3"] = (cand["source"].to_numpy() == "S3").astype(np.int8)
    f["name_exact"] = f["name_exact"].astype(np.int8)
    return f[FEATURES]


_SHARED: dict = {}


def _feat_chunk(bounds: tuple[int, int]) -> pd.DataFrame:
    a, b = bounds
    c = _SHARED["cand"].iloc[a:b].reset_index(drop=True)
    return pair_features(c, _SHARED["s1n"], _SHARED["mn"], workers=1)


def pair_features_parallel(cand: pd.DataFrame, s1n: pd.DataFrame, mn: pd.DataFrame,
                           chunk: int | None = None) -> pd.DataFrame:
    """pair_features over row chunks in forked workers (inputs shared copy-on-write).

    The chunk size is derived from the worker count so that a small input (one feature
    shard) still spreads over every worker: a fixed 250k chunk left a 100k-row shard on a
    single core. Chunking is row-wise and every feature is row-local, so the chunk size
    never changes the values.
    """
    if chunk is None:
        w = compute.n_jobs()
        chunk = max(20_000, min(250_000, -(-len(cand) // max(1, w))))
    if len(cand) <= chunk:
        return pair_features(cand, s1n, mn)
    _SHARED.update(cand=cand, s1n=s1n, mn=mn)
    try:
        bounds = [(i, min(i + chunk, len(cand))) for i in range(0, len(cand), chunk)]
        with compute.pool() as p:
            parts = p.map(_feat_chunk, bounds, chunksize=1)
    finally:
        _SHARED.clear()
    return pd.concat(parts, ignore_index=True)


def pair_cosines(cand: pd.DataFrame, s1_rows: np.ndarray, m_rows: np.ndarray, E1: np.ndarray, E2: np.ndarray,
                 chunk: int = 1_000_000) -> np.ndarray:
    """Row-wise cosine of already-normalised fp16 embeddings: E1[s1_rows[i]] . E2[m_rows[i]]."""
    out = np.empty(len(cand), np.float32)
    for a in range(0, len(cand), chunk):
        b = min(a + chunk, len(cand))
        x = np.asarray(E1[s1_rows[a:b]], dtype=np.float32)
        y = np.asarray(E2[m_rows[a:b]], dtype=np.float32)
        out[a:b] = np.einsum("ij,ij->i", x, y)
    return out
