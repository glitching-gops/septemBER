"""Stage-2 context features on GPU: how a candidate sits among its S1's other candidates.

GROUP features (from stage-1 probabilities, over ALL candidates of each S1 / of each record):

    grp_rank        rank of p within the S1 (1 = best)
    grp_rank_src    rank of p within the S1 x source
    grp_gap_top1    p(top-1 of the S1) - p
    grp_softmax     softmax of logit(p) within the S1
    grp_n_above     number of the S1's candidates with p >= tau
    grp_m_best_oth  the record's best p for any OTHER S1 (reverse-exclusivity signal; 0 if none)
    grp_m_rank      rank of this S1 among the S1s competing for the record (1 = best)
    grp_m_n         number of S1s competing for the record

CANDIDATE-TO-CANDIDATE (c2c) features, from the cached e5 name / address embeddings:

    c2c_{name,addr}_{max,mean}_{same,cross}   cosine of b to the S1's top-3 OTHER candidates by
                                              stage-1 p, split by same / other source than b
    c2c_n_close                               number of the S1's other candidates with name
                                              cosine > 0.9 to b
    c2c_hn_agree / c2c_pc_agree               house number / postcode equal to the top-1 other
                                              candidate's (1 / 0; NaN when either is missing)

Everything is torch segment arithmetic (sort, scatter_reduce, index_add, bmm) on the chosen
device; no pandas groupby on the pair table. Inputs are integer codes, so this module knows
nothing about ids, sources or countries.
"""
from __future__ import annotations

GROUP_FEATURES = ["grp_rank", "grp_rank_src", "grp_gap_top1", "grp_softmax", "grp_n_above", "grp_m_best_oth", "grp_m_rank",
                  "grp_m_n"]
C2C_FEATURES = [f"c2c_{f}_{s}_{w}" for f in ("name", "addr") for s in ("max", "mean") for w in ("same", "cross")] + \
    ["c2c_n_close", "c2c_hn_agree", "c2c_pc_agree"]


def group_features(s1, m, src, p, tau: float, n_ent: int) -> dict:
    """CUDA tensors in (``s1``/``m`` entity indices, ``src`` source code, ``p`` stage-1 p); CUDA out.

    Ranks break ties by the smaller record index (S1 side) / smaller S1 index (record side).
    """
    import torch

    from ber.gpu import segment as sgm
    P = p.to(torch.float32)
    out = {}
    out["grp_rank"] = sgm.rank_in_group(s1, P, m).float()
    out["grp_rank_src"] = sgm.rank_in_group(s1 * 8 + src.to(torch.int64), P, m).float()
    smax = sgm.seg_max(P, s1, n_ent, -1.0)
    out["grp_gap_top1"] = smax[s1] - P
    z = torch.logit(P.clamp(1e-6, 1 - 1e-6))
    zmax = sgm.seg_max(z, s1, n_ent, -1e9)
    ez = torch.exp(z - zmax[s1])
    out["grp_softmax"] = ez / sgm.seg_sum(ez, s1, n_ent)[s1]
    out["grp_n_above"] = sgm.seg_count(s1, n_ent, P >= tau)[s1].float()
    m_rank = sgm.rank_in_group(m, P, s1)
    out["grp_m_rank"] = m_rank.float()
    out["grp_m_n"] = sgm.seg_count(m, n_ent)[m].float()
    best = sgm.seg_max(P, m, n_ent, -1.0)
    second = sgm.seg_max(torch.where(m_rank == 1, torch.full_like(P, -1.0), P), m, n_ent, -1.0)
    out["grp_m_best_oth"] = torch.where(m_rank == 1, second[m], best[m]).clamp(min=0.0)
    return out


def c2c_features(s1, m, p, src, emb, addr_ok, hn, pc, top: int = 3, close: float = 0.9,
                 row_budget: int = 400_000, pair_budget: int = 40_000_000, max_cands: int = 4096) -> dict:
    """Per pair: similarity of its candidate record ``m`` to the S1's other candidates.

    All inputs are CUDA tensors: ``s1``/``m`` int64 entity indices, ``p`` stage-1 probability,
    ``src`` source code; ``addr_ok``/``hn``/``pc`` are PER-ENTITY tensors (address present,
    house-number / postcode int64 codes, 0 = missing); ``emb(field, ent)`` returns the
    L2-normalised e5 embeddings of entity indices ``ent``. Rows are ordered by (s1, p desc,
    row); S1 are bucketed by candidate count (powers of two) and processed in padded chunks
    with one bmm each. An S1 with more than ``max_cands`` candidates keeps its top
    ``max_cands`` by p; its other rows get NaN.
    """
    import torch

    from ber.gpu import segment as sgm
    dev = p.device
    n = p.numel()
    rowid = torch.arange(n, device=dev)
    perm = sgm.order([(s1, False), (p.to(torch.float64), True), (rowid, False)])
    ss = s1[perm]
    pos = sgm.pos_in_segment(ss)
    gid = torch.cumsum(sgm.seg_starts(ss).to(torch.int64), 0) - 1
    G = int(gid[-1]) + 1
    size = torch.clamp(sgm.seg_count(gid, G), max=max_cands)
    res = {k: torch.full((n,), float("nan"), dtype=torch.float32, device=dev) for k in C2C_FEATURES}
    bucket = torch.ceil(torch.log2(size.to(torch.float64))).to(torch.int64).clamp(min=1)
    keep_row = pos < max_cands
    for bk in torch.unique(bucket).tolist():
        Lb = min(2 ** int(bk), max_cands)
        groups = torch.nonzero(bucket == bk).squeeze(1)
        step = max(1, min(row_budget // Lb, pair_budget // (Lb * Lb)))
        for a in range(0, groups.numel(), step):
            gsel = groups[a:a + step]
            Gc = gsel.numel()
            lut = torch.full((G,), -1, dtype=torch.int64, device=dev)
            lut[gsel] = torch.arange(Gc, device=dev)
            loc = lut[gid]
            r = (loc >= 0) & keep_row & (pos < Lb)
            idx = torch.full((Gc, Lb), -1, dtype=torch.int64, device=dev)
            idx[loc[r], pos[r]] = perm[r]
            valid = idx >= 0
            safe = torch.where(valid, idx, torch.zeros_like(idx)).reshape(-1)
            ent = m[safe]
            En = emb("name", ent).view(Gc, Lb, -1).float()
            Ea = emb("addr", ent).view(Gc, Lb, -1).float()
            ok = addr_ok[ent].view(Gc, Lb) & valid
            Sr = src[safe].view(Gc, Lb)
            Cn = torch.bmm(En, En.transpose(1, 2))
            Ca = torch.bmm(Ea, Ea.transpose(1, 2))
            eye = torch.eye(Lb, dtype=torch.bool, device=dev).unsqueeze(0)
            other = valid.unsqueeze(1) & valid.unsqueeze(2) & ~eye
            topm = other & (torch.cumsum(other.int(), dim=2) <= top)
            same = Sr.unsqueeze(2) == Sr.unsqueeze(1)
            tgt = idx[valid]
            for nm, C, extra in (("name", Cn, None), ("addr", Ca, ok.unsqueeze(1) & ok.unsqueeze(2))):
                for w, sm in (("same", same), ("cross", ~same)):
                    msk = topm & sm if extra is None else topm & sm & extra
                    cnt = msk.sum(2)
                    mx = torch.where(msk, C, torch.full_like(C, -2.0)).amax(2)
                    mean = torch.where(msk, C, torch.zeros_like(C)).sum(2) / cnt.clamp(min=1)
                    nan = torch.full_like(mx, float("nan"))
                    res[f"c2c_{nm}_max_{w}"][tgt] = torch.where(cnt > 0, mx, nan)[valid]
                    res[f"c2c_{nm}_mean_{w}"][tgt] = torch.where(cnt > 0, mean, nan)[valid]
            res["c2c_n_close"][tgt] = ((Cn > close) & other).sum(2).float()[valid]
            # agreement with the best OTHER candidate: column 0, or column 1 for the top-1 itself
            col = torch.where(torch.arange(Lb, device=dev) == 0, 1, 0).clamp(max=Lb - 1).unsqueeze(0).expand(Gc, Lb)
            top_ok = torch.gather(valid, 1, col) & (Lb > 1)
            for key, arr in (("c2c_hn_agree", hn), ("c2c_pc_agree", pc)):
                A = arr[ent].view(Gc, Lb)
                B = torch.gather(A, 1, col)
                both = (A != 0) & (B != 0) & top_ok & valid
                val = torch.where(both, (A == B).float(), torch.full(A.shape, float("nan"), device=dev))
                res[key][tgt] = val[valid]
    return res
