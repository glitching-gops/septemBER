"""Blocking quality metrics: pair completeness, marginal recall, ceiling, hard slices."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ber.evaluate.metric import macro_f05

def _rank(c, col):
    return (c[col].fillna(0) > 0) if col in c else pd.Series(False, index=c.index)


BLOCKER_FLAGS = {
    "B1 name (keys)": lambda c: _rank(c, "b1_rank"),
    "B2 addr": lambda c: _rank(c, "b2_rank"),
    "B3 name+addr": lambda c: _rank(c, "b3_rank"),
    "B4 keys": lambda c: (c["b4_hn"].fillna(False).astype(bool) | c["b4_name"].fillna(False).astype(bool)) if "b4_hn" in c else pd.Series(False, index=c.index),
    "B5 lexical reverse (retired)": lambda c: _rank(c, "b5_rank"),
    "B6 dense name": lambda c: _rank(c, "b6n_rank"),
    "B6 dense addr": lambda c: _rank(c, "b6a_rank"),
    "B7 dense reverse": lambda c: _rank(c, "b7_rank"),
}


def truth_pairs(gt: Mapping[str, frozenset], s1_ids) -> pd.DataFrame:
    rows = [(s, m) for s in s1_ids for m in gt.get(s, ())]
    df = pd.DataFrame(rows, columns=["s1_id", "m_id"])
    df["source"] = df["m_id"].str[:2]
    return df


def mark_found(truth: pd.DataFrame, cand: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    c = cand if not k else cand[cand["cand_rank"] <= k]
    cols = ["s1_id", "m_id", "cand_rank"] + [c_ for c_ in ("b1_rank", "b2_rank", "b3_rank", "b4_hn", "b4_name",
                                                            "b5_rank", "b6_rank", "b6n_rank", "b6a_rank", "b7_rank") if c_ in c]
    m = truth.merge(c[cols], on=["s1_id", "m_id"], how="left", indicator=True)
    m["found"] = m["_merge"] == "both"
    return m.drop(columns="_merge")


def ceiling(truth: pd.DataFrame, found: pd.DataFrame, s1_ids, gt) -> float:
    """Macro F0.5 of an oracle predicting exactly the true matches present in the candidates."""
    pred: dict[str, set] = {s: set() for s in s1_ids}
    for s, m in zip(found.loc[found["found"], "s1_id"], found.loc[found["found"], "m_id"]):
        pred[s].add(m)
    return macro_f05(pred, gt, list(s1_ids))["macro"]


def cands_per_s1(cand: pd.DataFrame, s1_ids, k: int | None = None) -> pd.Series:
    c = cand if not k else cand[cand["cand_rank"] <= k]
    n = c.groupby("s1_id").size()
    return n.reindex(pd.Index(list(s1_ids)), fill_value=0)


def marginal(found: pd.DataFrame) -> pd.DataFrame:
    f = found[found["found"]].copy()
    flags = pd.DataFrame({k: fn(f).to_numpy() for k, fn in BLOCKER_FLAGS.items()})
    flags = flags.loc[:, flags.any(axis=0)]
    n_true = len(found)
    rows = []
    for k in flags.columns:
        only = flags[k] & (flags.drop(columns=k).sum(axis=1) == 0)
        rows.append([k, flags[k].sum() / n_true, only.sum(), only.sum() / n_true])
    return pd.DataFrame(rows, columns=["blocker", "standalone_recall", "unique_pairs", "unique_recall"])


def reduction_ratio(n_cand: int, s1_country: pd.Series, pool_sizes: Mapping[str, int]) -> float:
    total = float(sum(pool_sizes.get(c, 0) for c in s1_country))
    return 1.0 - n_cand / total if total else float("nan")


def hard_slices(found: pd.DataFrame, s1n: pd.DataFrame, mn: pd.DataFrame) -> pd.DataFrame:
    """Recall on hard subsets of true pairs. s1n/mn: normalised S1 / S2+S3 tables."""
    s1a = s1n.set_index("entity_id")
    ma = mn.set_index("entity_id")
    f = found.copy()
    f["m_script"] = f["m_id"].map(ma["name_script"])
    f["dom"] = f["m_id"].map(ma["is_domain"]).fillna(False).astype(bool) | f["s1_id"].map(s1a["is_domain"]).fillna(False).astype(bool)
    s1_tok = f["s1_id"].map(s1a["name_core"]).fillna("").str.split().map(set)
    m_tok = f["m_id"].map(ma["name_core"]).fillna("").str.split().map(set)
    f["no_tok"] = [not (a & b) for a, b in zip(s1_tok, m_tok)]
    f["m_addr_missing"] = f["m_id"].map(ma["addr_missing"]).fillna(False).astype(bool)
    l1 = f["s1_id"].map(s1a["legal_class"]).map(lambda x: len(x) > 0 if x is not None else False)
    l2 = f["m_id"].map(ma["legal_class"]).map(lambda x: len(x) > 0 if x is not None else False)
    f["one_sided_legal"] = l1 != l2
    per_blocker = {k.split()[0] + ("n" if "dense name" in k else "a" if "dense addr" in k else ""): fn(f)
                   for k, fn in BLOCKER_FLAGS.items()}
    per_blocker = {k: v for k, v in per_blocker.items() if v.any()}

    def row(name, mask):
        g = f[mask]
        r = [name, len(g), g["found"].mean() if len(g) else np.nan]
        r += [per_blocker[b].to_numpy()[mask].mean() if len(g) else np.nan for b in per_blocker]
        return r

    rows = [row("all true pairs", np.ones(len(f), bool))]
    for sc in sorted(f["m_script"].dropna().unique()):
        if sc != "latin":
            rows.append(row(f"matched name script = {sc}", (f["m_script"] == sc).to_numpy()))
    rows.append(row("matched name non-Latin (all scripts)", (f["m_script"] != "latin").to_numpy()))
    for name, mask in (("domain-like name (either side)", f["dom"]), ("no shared name token", f["no_tok"]),
                       ("matched address empty", f["m_addr_missing"]), ("one-sided legal suffix", f["one_sided_legal"])):
        rows.append(row(name, mask.to_numpy()))
    return pd.DataFrame(rows, columns=["slice", "n_true_pairs", "pair_completeness"] + [f"{b} recall" for b in per_blocker])


def order_keys(c: pd.DataFrame, k: tuple[int, int, int, int] = (30, 30, 20, 5)) -> dict[str, np.ndarray]:
    """Alternative per-pair ranking scores (higher = better) for choosing the K_max ordering."""
    k1, k2, k3, k5 = k
    r = {b: c[f"{b}_rank"].fillna(0).to_numpy(np.float32) for b in ("b1", "b2", "b3", "b5", "b6", "b6n", "b6a", "b7") if f"{b}_rank" in c}
    rrf = sum(np.where(r[b] > 0, 1.0 / (10.0 + r[b]), 0.0) for b in r)
    b4 = (c["b4_hn"].fillna(False).to_numpy(bool) | c["b4_name"].fillna(False).to_numpy(bool)) if "b4_hn" in c else np.zeros(len(c), bool)
    rrf = rrf + np.where(b4, 1.0 / 20.0, 0.0)
    cos = c[[x for x in ("b1_score", "b2_score", "b3_score", "b6_score", "b6n_score", "b6a_score", "b7_score") if x in c]].max(axis=1).fillna(0).to_numpy()
    return {
        **({"best_norm": c["best_norm"].to_numpy() + 1e-3 * rrf} if "best_norm" in c else {}),
        **({"pipeline RRF (fused)": c["fused"].to_numpy()} if "fused" in c else {}),
        "max cosine": cos,
        "reciprocal-rank fusion": rrf + 1e-4 * cos,
    }


def rerank(c: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """cand_rank (1 = best) within (s1_id, source) under ``score``."""
    tmp = pd.DataFrame({"s": c["s1_id"].to_numpy(), "src": c["source"].to_numpy(), "v": -score, "m": c["m_id"].to_numpy()})
    tmp = tmp.sort_values(["s", "src", "v", "m"])
    ranks = tmp.groupby(["s", "src"]).cumcount().to_numpy() + 1
    out = np.empty(len(c), dtype=np.int32)
    out[tmp.index.to_numpy()] = ranks
    return out
