"""Staged, resumable blocking (P1 change of plan).

Stages, each checkpointed per (target source x country) partition:
    lexical  (CPU)  B1 name-key char TF-IDF, B2 address TF-IDF, B3 name+address words, B4 exact keys
    dense    (GPU)  B6 forward E5 top-k on name and on address;
                    B7 reverse: each target queries ALL S1 of its country (name and address);
                    keep (S1, target) when the S1 is in the target's top-k'
    union           outer-merge, reciprocal-rank fusion, cand_rank per (S1, source), optional K_max
The retired lexical reverse blocker (B5) is not part of this pipeline.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ber.blocking.dense import dense_topk
from ber.blocking.keys import key_pairs
from ber.blocking.pipeline import _BLOCKERS, BlockingConfig, _spec
from ber.blocking.tfidf import TfidfIndex, ranks_within_query

RANK_COLS = ["b1_rank", "b2_rank", "b3_rank", "b6n_rank", "b6a_rank", "b7_rank"]
SCORE_COLS = ["b1_score", "b2_score", "b3_score", "b6n_score", "b6a_score", "b7_score"]


@dataclass
class DenseConfig:
    k6: int = 20   # forward top-k per field (name, address) per source
    k7: int = 5    # reverse top-k' per target record


def partition_slug(source: str, country: str) -> str:
    """Filesystem-safe partition id derived from the data value (never a hard-coded list)."""
    return f"{source}__{re.sub(r'[^A-Za-z0-9]+', '_', str(country)).strip('_') or 'blank'}"


def lexical_partition(q: pd.DataFrame, t: pd.DataFrame, cfg: BlockingConfig,
                      blockers=("b1", "b2", "b3", "b4")) -> tuple[pd.DataFrame, dict]:
    frames, timings = [], {}
    for b, (spec_name, texts) in _BLOCKERS.items():
        if b not in blockers:
            continue
        t0 = time.time()
        idx = TfidfIndex(_spec(spec_name, cfg)).fit(texts(t))
        k = {"b1": cfg.k1, "b2": cfg.k2, "b3": cfg.k3}[b]
        qi, ti, sc = idx.forward(idx.transform(texts(q)), k, cfg.threshold)
        frames.append(pd.DataFrame({"qi": qi, "ti": ti, f"{b}_score": sc, f"{b}_rank": ranks_within_query(qi)}))
        timings[b] = round(time.time() - t0, 1)
    if "b4" in blockers:
        t0 = time.time()
        kp = key_pairs(q["addr_core"], q["house_no"], q["name_core"], t["addr_core"], t["house_no"], t["name_core"], cap=cfg.key_cap)
        frames.append(kp.rename(columns={"q": "qi", "t": "ti"}))
        timings["b4"] = round(time.time() - t0, 1)
    if not frames:
        return pd.DataFrame(columns=["qi", "ti"]), timings
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["qi", "ti"], how="outer")
    return out, timings


def dense_partition(q: pd.DataFrame, t: pd.DataFrame, s1_all: pd.DataFrame, emb: dict, dcfg: DenseConfig) -> tuple[pd.DataFrame, dict]:
    """``emb``: {"S1": {"name": M, "addr": M}, "T": {"name": M, "addr": M}} indexed by emb_row."""
    timings, frames = {}, []
    qn = np.asarray(emb["S1"]["name"][q["emb_row"].to_numpy()])
    tn = np.asarray(emb["T"]["name"][t["emb_row"].to_numpy()])
    t0 = time.time()
    qi, ti, sc = dense_topk(qn, tn, dcfg.k6)
    frames.append(pd.DataFrame({"qi": qi, "ti": ti, "b6n_score": sc, "b6n_rank": ranks_within_query(qi)}))
    timings["b6n"] = round(time.time() - t0, 1)

    # address: only records that have an address on both sides (missing is never a mismatch)
    t0 = time.time()
    qa_rows = np.flatnonzero(~q["addr_missing"].to_numpy(bool))
    ta_rows = np.flatnonzero(~t["addr_missing"].to_numpy(bool))
    if len(qa_rows) and len(ta_rows) and "addr" in emb["S1"]:
        qa = np.asarray(emb["S1"]["addr"][q["emb_row"].to_numpy()[qa_rows]])
        ta = np.asarray(emb["T"]["addr"][t["emb_row"].to_numpy()[ta_rows]])
        a, b, sc = dense_topk(qa, ta, dcfg.k6)
        qi, ti = qa_rows[a], ta_rows[b]
        frames.append(pd.DataFrame({"qi": qi, "ti": ti, "b6a_score": sc, "b6a_rank": ranks_within_query(qi)}))
    timings["b6a"] = round(time.time() - t0, 1)

    # B7 reverse against ALL S1 of the country; keep hits on the query set only
    t0 = time.time()
    pos = pd.Series(np.arange(len(q)), index=q["entity_id"].to_numpy())
    q_in_all = s1_all["entity_id"].map(pos).fillna(-1).astype(np.int64).to_numpy()
    rev = []
    for field in ("name", "addr"):
        if field not in emb["S1"]:
            continue
        if field == "addr":
            t_rows = np.flatnonzero(~t["addr_missing"].to_numpy(bool))
            a_rows = np.flatnonzero(~s1_all["addr_missing"].to_numpy(bool))
        else:
            t_rows, a_rows = np.arange(len(t)), np.arange(len(s1_all))
        if not len(t_rows) or not len(a_rows):
            continue
        TT = np.asarray(emb["T"][field][t["emb_row"].to_numpy()[t_rows]])
        AA = np.asarray(emb["S1"][field][s1_all["emb_row"].to_numpy()[a_rows]])
        ti_, ai_, sc = dense_topk(TT, AA, dcfg.k7)
        r = ranks_within_query(ti_)
        qi = q_in_all[a_rows[ai_]]
        keep = qi >= 0
        rev.append(pd.DataFrame({"qi": qi[keep], "ti": t_rows[ti_[keep]], "b7_rank": r[keep], "b7_score": sc[keep]}))
    if rev:
        frames.append(pd.concat(rev).groupby(["qi", "ti"], as_index=False).agg(b7_rank=("b7_rank", "min"), b7_score=("b7_score", "max")))
    timings["b7"] = round(time.time() - t0, 1)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["qi", "ti"], how="outer")
    return out, timings


def fuse(c: pd.DataFrame, k_max: int = 0) -> pd.DataFrame:
    """Fill blocker columns, compute the RRF score and cand_rank within (s1_id, source)."""
    c = c.copy()
    for col in RANK_COLS:
        c[col] = c[col].fillna(0).astype(np.int16) if col in c else np.int16(0)
    for col in SCORE_COLS:
        if col not in c:
            c[col] = np.nan
        c[col] = c[col].astype(np.float32)
    for col in ("b4_hn", "b4_name"):
        c[col] = c[col].fillna(False).astype(bool) if col in c else False
    c["b4_size"] = c["b4_size"].fillna(0).astype(np.int32) if "b4_size" in c else 0
    rrf = np.zeros(len(c))
    for col in RANK_COLS:
        r = c[col].to_numpy().astype(np.float64)
        rrf += np.where(r > 0, 1.0 / (10.0 + r), 0.0)
    rrf += np.where(c["b4_hn"].to_numpy() | c["b4_name"].to_numpy(), 1.0 / 20.0, 0.0)
    maxcos = c[SCORE_COLS].max(axis=1).fillna(0).to_numpy()
    c["fused"] = (rrf + 1e-4 * maxcos).astype(np.float32)
    c = c.sort_values(["s1_id", "source", "fused", "m_id"], ascending=[True, True, False, True])
    c["cand_rank"] = c.groupby(["s1_id", "source"]).cumcount().astype(np.int32) + 1
    if k_max:
        c = c[c["cand_rank"] <= k_max]
    return c.reset_index(drop=True)
