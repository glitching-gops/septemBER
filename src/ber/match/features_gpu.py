"""Token-level pair features computed in torch on GPU.

Motivation (P1 audit): the model is precision-heavy and under-predicts -- 76% of missed
true pairs were candidates the decision rejected. The rapidfuzz features measure *string*
similarity, which cannot tell a shared rare token ("Anakaputhur") from a shared filler one
("Services"). These features weight token agreement by inverse document frequency and
expose the most distinctive token each side holds that the other lacks, which is the
classic over/under-merge signal.

All of it is GPU work: token ids are gathered into padded int32 matrices and compared with
a batched [B, T, T] equality tensor, so a shard of ~10^5 pairs is a few kernels rather
than a Python loop.

D1 (transductive statistics): document frequencies are fitted per split on that split's
own unlabelled inputs, the same procedure for train and test.

Columns are prefixed ``g_`` so the feature list picks them up automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_TOKENS = {"name": 12, "addr": 20}

GPU_FEATURES = [
    "g_name_idf_jacc",      # IDF-weighted Jaccard of name tokens
    "g_name_idf_cont",      # IDF-weighted containment (shared / smaller side)
    "g_name_unmatched_l",   # max IDF of an S1 name token absent from the candidate
    "g_name_unmatched_r",   # max IDF of a candidate name token absent from the S1
    "g_name_shared_n",      # count of shared name tokens
    "g_name_rare_shared",   # max IDF among the shared name tokens
    "g_addr_idf_jacc",
    "g_addr_idf_cont",
    "g_addr_unmatched_l",
    "g_addr_unmatched_r",
    "g_addr_shared_n",
    "g_addr_rare_shared",
    "g_b6n_rank", "g_b6a_rank", "g_b7_rank",  # raw ranks; P1 kept only hit/no-hit
    "g_miss_x_nameidf",     # address-missing interaction with name evidence
]


class TokenSpace:
    """Padded token ids for S1 and for the stacked S2+S3 targets, over ONE shared vocabulary.

    A shared vocabulary is essential: the pair features compare token *ids*, so an id must
    mean the same string on both sides. ``ids`` is 0-padded and id 0 is the pad slot, which
    never matches and has IDF 0.
    """

    def __init__(self, s1_text: pd.Series, tgt_text: pd.Series, max_tokens: int):
        self.max_tokens = max_tokens
        n1, n2 = len(s1_text), len(tgt_text)
        allt = pd.concat([s1_text.fillna(""), tgt_text.fillna("")], ignore_index=True)
        s = allt.str.split()
        ex = s.explode()
        ex = ex[ex.notna() & (ex.astype(str) != "")]
        codes, uniques = pd.factorize(ex.astype(str), sort=False)
        row = ex.index.to_numpy()

        # Deduplicate (record, token): these features are set-based, so a token repeated
        # inside one record must count once. Leaving duplicates in lets the IDF-weighted
        # numerator exceed the union and pushes the Jaccard above 1.
        pairs = pd.DataFrame({"r": row, "c": codes}).drop_duplicates()
        row = pairs["r"].to_numpy()
        codes = pairs["c"].to_numpy()

        # document frequency over every record of the split (S1 + targets)
        df_count = np.bincount(codes, minlength=len(uniques)).astype(np.float64)
        n_docs = n1 + n2
        idf = np.log((n_docs + 1.0) / (df_count + 1.0)) + 1.0
        self.idf = np.concatenate([[0.0], idf]).astype(np.float32)  # index 0 = pad

        # keep the first max_tokens distinct ids of each record
        order = np.arange(len(row))
        first = np.r_[True, row[1:] != row[:-1]]
        start = np.maximum.accumulate(np.where(first, order, 0))
        within = order - start
        keep = within < max_tokens
        ids = np.zeros((n_docs, max_tokens), dtype=np.int32)
        ids[row[keep], within[keep]] = codes[keep] + 1
        self.s1 = ids[:n1]
        self.tgt = ids[n1:]


def _block(torch, tl, tr, idf, device) -> dict:
    """IDF overlap statistics for one batch, all on GPU."""
    il, ir = idf[tl.long()], idf[tr.long()]
    lv, rv = tl > 0, tr > 0
    eq = (tl.unsqueeze(2) == tr.unsqueeze(1)) & lv.unsqueeze(2) & rv.unsqueeze(1)  # [B,T,T]
    ml, mr = eq.any(dim=2), eq.any(dim=1)
    zero = torch.zeros((), device=device)
    shared = (il * ml).sum(1)                      # IDF mass shared, counted once (left side)
    tot_l, tot_r = (il * lv).sum(1), (ir * rv).sum(1)
    union = torch.clamp(tot_l + tot_r - shared, min=1e-6)
    smaller = torch.clamp(torch.minimum(tot_l, tot_r), min=1e-6)
    return {
        "idf_jacc": shared / union,
        "idf_cont": shared / smaller,
        "unmatched_l": torch.where(lv & ~ml, il, zero).amax(dim=1),
        "unmatched_r": torch.where(rv & ~mr, ir, zero).amax(dim=1),
        "shared_n": ml.sum(1).to(torch.float32),
        "rare_shared": torch.where(ml, il, zero).amax(dim=1),
    }


def pair_token_features(cand: pd.DataFrame, l_rows: np.ndarray, r_rows: np.ndarray,
                        spaces: dict[str, TokenSpace], device, batch: int = 200_000) -> pd.DataFrame:
    """The ``g_*`` features for ``cand``; ``l_rows``/``r_rows`` index ``space.s1``/``space.tgt``."""
    import torch

    n = len(cand)
    out = {c: np.zeros(n, np.float32) for c in GPU_FEATURES}
    for field, prefix in (("name", "g_name"), ("addr", "g_addr")):
        sp = spaces[field]
        idf = torch.as_tensor(sp.idf, device=device)
        for a in range(0, n, batch):
            b = min(a + batch, n)
            tl = torch.as_tensor(sp.s1[l_rows[a:b]], device=device)
            tr = torch.as_tensor(sp.tgt[r_rows[a:b]], device=device)
            r = _block(torch, tl, tr, idf, device)
            for k, v in r.items():
                out[f"{prefix}_{k}"][a:b] = v.cpu().numpy()

    for col, name in (("b6n_rank", "g_b6n_rank"), ("b6a_rank", "g_b6a_rank"), ("b7_rank", "g_b7_rank")):
        v = cand[col].fillna(0).to_numpy(np.float32) if col in cand else np.zeros(n, np.float32)
        out[name] = np.where(v > 0, v, 999.0).astype(np.float32)

    # when the address is missing on either side the name carries the whole decision
    miss = cand["addr_missing_pair"].to_numpy(np.float32) if "addr_missing_pair" in cand \
        else np.zeros(n, np.float32)
    out["g_miss_x_nameidf"] = (miss * out["g_name_idf_jacc"]).astype(np.float32)
    return pd.DataFrame(out)[GPU_FEATURES]
