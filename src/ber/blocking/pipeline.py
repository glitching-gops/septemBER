"""Multi-key blocking: run B1-B5 inside every (exact country value x target source)
partition and union the results with per-blocker flags, scores and ranks.

Candidate table columns (one row per (s1_id, m_id)):
    s1_id, m_id, source, country
    b1_score, b1_rank     name char TF-IDF cosine / rank (0 = not retrieved)
    b2_score, b2_rank     address TF-IDF
    b3_score, b3_rank     name+address word TF-IDF
    b4_hn, b4_name, b4_size   exact-key hits and smallest block size
    b5_rank               rank of the S1 in the target's reverse top-k' (0 = none)
    best_norm, fused      fusion scores used to rank and cap candidates
    cand_rank             rank of the pair among its S1's candidates from this source
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ber.blocking.keys import key_pairs
from ber.blocking.tfidf import SPECS, TfidfIndex, TfidfSpec, ranks_within_query


@dataclass
class BlockingConfig:
    k1: int = 30          # B1 name top-k per source
    k2: int = 30          # B2 address top-k
    k3: int = 20          # B3 combined top-k
    k5: int = 5           # B5 reverse top-k' (name side only)
    k5_missing: int = 10  # B5 reverse top-k' for targets whose address is missing
    k6: int = 30          # B6 dense (multilingual-E5) name top-k; used when embeddings are given
    b5_sides: tuple = ("b1",)  # which TF-IDF spaces feed the reverse blocker
    key_cap: int = 500    # B4 max block size
    k_max: int = 0        # final cap per S1 per source (0 = keep all, rank only)
    threshold: float = 0.05
    use_b5: bool = True
    max_df: dict = field(default_factory=dict)  # per-blocker override of SPECS[...].max_df


def _texts_name(df: pd.DataFrame) -> list[str]:
    col = "b1_text" if "b1_text" in df else "name_core"  # name key + skeleton when available
    return df[col].fillna("").tolist()


def _texts_addr(df: pd.DataFrame) -> list[str]:
    return df["addr_core"].fillna("").tolist()


def _texts_both(df: pd.DataFrame) -> list[str]:
    return (df["name_core"].fillna("") + " " + df["addr_core"].fillna("")).str.strip().tolist()


_BLOCKERS = {"b1": ("b1_name", _texts_name), "b2": ("b2_addr", _texts_addr), "b3": ("b3_both", _texts_both)}


def _spec(name: str, cfg: BlockingConfig) -> TfidfSpec:
    s = SPECS[name]
    md = cfg.max_df.get(name)
    return s if md is None else TfidfSpec(parts=s.parts, max_df=md, min_df=s.min_df)


def block_partition(q: pd.DataFrame, t: pd.DataFrame, cfg: BlockingConfig, log=print,
                    s1_all: pd.DataFrame | None = None, emb: tuple | None = None) -> pd.DataFrame:
    """Candidates for queries ``q`` against targets ``t`` (one country x one source).

    ``s1_all`` is the full S1 population of this country (``q`` is a subset of it).
    The reverse blocker B5 ranks each target's top-k' among ALL of ``s1_all`` and
    keeps only hits on ``q``, so a query subset (the dev slice) sees exactly the
    reverse candidates it would get in a full-scale run.
    Returns positional indices ``qi``/``ti`` plus blocker columns.
    """
    if s1_all is None:
        s1_all = q
    pos = pd.Series(np.arange(len(q)), index=q["entity_id"].to_numpy())
    q_in_all = s1_all["entity_id"].map(pos).fillna(-1).astype(np.int64).to_numpy()  # all-row -> q row or -1
    all_rows_of_q = np.flatnonzero(q_in_all >= 0)[np.argsort(q_in_all[q_in_all >= 0])]
    frames = []
    timings = {}
    Qs = {}
    for b, (spec_name, texts) in _BLOCKERS.items():
        t0 = time.time()
        idx = TfidfIndex(_spec(spec_name, cfg)).fit(texts(t))
        Q_all = idx.transform(texts(s1_all)) if b in cfg.b5_sides and cfg.use_b5 else None
        Q = Q_all[all_rows_of_q] if Q_all is not None else idx.transform(texts(q))
        Qs[b] = (idx, Q_all)
        k = {"b1": cfg.k1, "b2": cfg.k2, "b3": cfg.k3}[b]
        qi, ti, sc = idx.forward(Q, k, cfg.threshold)
        frames.append(pd.DataFrame({"qi": qi, "ti": ti, f"{b}_score": sc, f"{b}_rank": ranks_within_query(qi)}))
        timings[b] = time.time() - t0

    t0 = time.time()
    kp = key_pairs(q["addr_core"], q["house_no"], q["name_core"], t["addr_core"], t["house_no"], t["name_core"],
                   cap=cfg.key_cap)
    frames.append(kp.rename(columns={"q": "qi", "t": "ti"}))
    timings["b4"] = time.time() - t0

    if cfg.use_b5:
        t0 = time.time()
        rev = []
        k5 = max(cfg.k5, cfg.k5_missing)
        missing = t["addr_missing"].to_numpy(bool) if "addr_missing" in t else np.zeros(len(t), bool)
        for b in cfg.b5_sides:
            idx, Q_all = Qs[b]
            ti, ai, sc = idx.reverse(Q_all, k5, cfg.threshold)
            r = ranks_within_query(ti)
            qi = q_in_all[ai]
            # deeper reverse list only for targets without an address (name is their only signal)
            keep = (qi >= 0) & ((r <= cfg.k5) | missing[ti])
            rev.append(pd.DataFrame({"qi": qi[keep], "ti": ti[keep], "b5_rank": r[keep], "b5_score": sc[keep]}))
        r = pd.concat(rev).groupby(["qi", "ti"], as_index=False).agg(b5_rank=("b5_rank", "min"), b5_score=("b5_score", "max"))
        frames.append(r)
        timings["b5"] = time.time() - t0
    del Qs

    if emb is not None:
        from ber.blocking.dense import dense_topk

        t0 = time.time()
        q_emb, t_emb = emb
        qi, ti, sc = dense_topk(q_emb, t_emb, cfg.k6)
        frames.append(pd.DataFrame({"qi": qi, "ti": ti, "b6_score": sc, "b6_rank": ranks_within_query(qi)}))
        timings["b6"] = time.time() - t0

    # union on (qi, ti)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["qi", "ti"], how="outer")
    if "b6_rank" not in out:
        out["b6_rank"], out["b6_score"] = 0, np.nan
    for b in ("b1", "b2", "b3", "b6"):
        out[f"{b}_rank"] = out[f"{b}_rank"].fillna(0).astype(np.int16)
    for c in ("b4_hn", "b4_name"):
        out[c] = out[c].fillna(False).astype(bool) if c in out else False
    out["b4_size"] = out["b4_size"].fillna(0).astype(np.int32) if "b4_size" in out else 0
    out["b5_rank"] = out["b5_rank"].fillna(0).astype(np.int16) if "b5_rank" in out else np.int16(0)

    # fusion: rank-normalised per-blocker scores in (0, 1]
    norms = []
    for b, k in (("b1", cfg.k1), ("b2", cfg.k2), ("b3", cfg.k3)):
        r = out[f"{b}_rank"].to_numpy()
        norms.append(np.where(r > 0, 1 - (r - 1) / k, 0.0))
    size = out["b4_size"].to_numpy()
    norms.append(np.where(size > 0, 1 / (1 + np.log(np.maximum(size, 1))), 0.0))
    r5 = out["b5_rank"].to_numpy()
    norms.append(np.where(r5 > 0, 1 - (r5 - 1) / cfg.k5, 0.0))
    r6 = out["b6_rank"].to_numpy()
    norms.append(np.where(r6 > 0, 1 - (r6 - 1) / cfg.k6, 0.0))
    N = np.vstack(norms)
    out["best_norm"] = N.max(axis=0).astype(np.float32)
    # ranking key: reciprocal-rank fusion (the best ordering on the dev slice, see p1_blocking.md)
    rrf = np.zeros(len(out))
    for b in ("b1", "b2", "b3", "b5", "b6"):
        r = out[f"{b}_rank"].to_numpy().astype(np.float64)
        rrf += np.where(r > 0, 1.0 / (10.0 + r), 0.0)
    rrf += np.where(out["b4_hn"].to_numpy(bool) | out["b4_name"].to_numpy(bool), 1.0 / 20.0, 0.0)
    maxcos = out[["b1_score", "b2_score", "b3_score", "b6_score"]].max(axis=1).fillna(0).to_numpy()
    out["fused"] = (rrf + 1e-4 * maxcos).astype(np.float32)
    out["_mc"] = maxcos
    out = out.sort_values(["qi", "fused", "_mc", "ti"], ascending=[True, False, False, True])
    out["cand_rank"] = out.groupby("qi").cumcount().astype(np.int32) + 1
    out = out.drop(columns="_mc")
    if cfg.k_max:
        out = out[out["cand_rank"] <= cfg.k_max]
    log("    timings " + " ".join(f"{k}={v:.1f}s" for k, v in timings.items()) + f"  pairs={len(out):,}")
    return out.reset_index(drop=True)


def run_blocking(queries: pd.DataFrame, targets: dict[str, pd.DataFrame], cfg: BlockingConfig, log=print,
                 s1_all: pd.DataFrame | None = None, emb: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Block every query against each target source, partitioned by exact country value.

    ``queries``: normalised S1 rows. ``targets``: {"S2": df, "S3": df} normalised rows.
    ``s1_all``: the full S1 population for the reverse blocker (defaults to ``queries``).
    ``emb``: optional {"S1": ..., "S2": ..., "S3": ...} embedding matrices indexed by the
    frames' ``emb_row`` column, enabling the dense blocker B6.
    Country values are taken from the data (never enumerated in code).
    """
    if s1_all is None:
        s1_all = queries
    parts = []
    for source, tdf in targets.items():
        for country, qg in queries.groupby("country", sort=True):
            tg = tdf[tdf["country"] == country]
            log(f"  [{source} | {country!r}] queries={len(qg):,} targets={len(tg):,}")
            if len(tg) == 0 or len(qg) == 0:
                continue
            qg = qg.reset_index(drop=True)
            tg = tg.reset_index(drop=True)
            ag = s1_all[s1_all["country"] == country].reset_index(drop=True)
            e = None
            if emb is not None:
                e = (emb["S1"][qg["emb_row"].to_numpy()], emb[source][tg["emb_row"].to_numpy()])
            c = block_partition(qg, tg, cfg, log=log, s1_all=ag, emb=e)
            c.insert(0, "s1_id", qg["entity_id"].to_numpy()[c["qi"].to_numpy()])
            c.insert(1, "m_id", tg["entity_id"].to_numpy()[c["ti"].to_numpy()])
            c.insert(2, "source", source)
            c.insert(3, "country", country)
            parts.append(c.drop(columns=["qi", "ti"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
