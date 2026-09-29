"""D2 hard-negative mining and D4 scoring-set construction on the GPU (items 7, 8).

Semantics of P4's CPU versions (``build_training_pairs_r2``, ``run_p4_verifier.scoring_set``):
negatives ranked current false positives first, then by the combined key (ties: smaller
record index); ``n_rand`` random others per S1 from a seeded ``torch`` generator (so the
random picks differ from the numpy-seeded CPU version; everything else is identical).
"""
from __future__ import annotations

import torch

from ber.gpu import segment as sg


def training_rows(c_s1, c_m, c_y, c_key, c_fp, c_fn, s1_mask, gt_s1, gt_m, hot_ent, *, n_hard=8, n_rand=2,
                  oversample=2, fn_dup=2, seed=42) -> dict:
    """Rows (s1, m, y) for the S1 with ``s1_mask[s1]``; ``hot_ent``: bool per entity (non-Latin /
    empty address) -> the row is repeated ``oversample`` times if either side is hot."""
    dev = c_s1.device
    g = torch.Generator(device=dev).manual_seed(seed)
    inset = s1_mask[c_s1]
    neg = inset & (c_y == 0)
    ni = torch.nonzero(neg).squeeze(1)
    perm = sg.order([(c_s1[ni], False), (c_fp[ni].to(torch.int64), True), (c_key[ni].to(torch.float64), True),
                     (c_m[ni], False)])
    ni = ni[perm]
    r = sg.pos_in_segment(c_s1[ni])
    hard = ni[r < n_hard]
    rest = ni[r >= n_hard]
    u = torch.rand(rest.numel(), generator=g, device=dev, dtype=torch.float64)
    rr = sg.rank_in_group(c_s1[rest], -u)  # ascending u
    rand = rest[rr <= n_rand]
    gi = torch.nonzero(s1_mask[gt_s1]).squeeze(1)
    fn = torch.nonzero(inset & (c_y == 1) & c_fn.bool()).squeeze(1)
    s1 = torch.cat([gt_s1[gi]] + [c_s1[fn]] * max(0, fn_dup - 1) + [c_s1[hard], c_s1[rand]])
    m = torch.cat([gt_m[gi]] + [c_m[fn]] * max(0, fn_dup - 1) + [c_m[hard], c_m[rand]])
    y = torch.cat([torch.ones(gi.numel() + fn.numel() * max(0, fn_dup - 1), dtype=torch.int8, device=dev),
                   torch.zeros(hard.numel() + rand.numel(), dtype=torch.int8, device=dev)])
    hot = hot_ent[s1] | hot_ent[m]
    hi = torch.nonzero(hot).squeeze(1)
    idx = torch.cat([torch.arange(s1.numel(), device=dev)] + [hi] * (oversample - 1))
    idx = idx[torch.randperm(idx.numel(), generator=g, device=dev)]
    return {"s1": s1[idx], "m": m[idx], "y": y[idx], "n_hard": hard.numel(), "n_rand": rand.numel(),
            "n_pos": gi.numel(), "n_fn_dup": fn.numel()}


def sub3_empty(s1, m, p, groups, taus: dict, n_ent: int) -> torch.Tensor:
    """Bool per entity: S1 with no pair selected by exclusivity (within each group) + threshold."""
    from ber.gpu.decide import exclusive, threshold
    has = torch.zeros(n_ent, dtype=torch.bool, device=s1.device)
    present = torch.zeros(n_ent, dtype=torch.bool, device=s1.device)
    present[s1] = True
    for gcode, tau in taus.items():
        rows = groups == int(gcode)
        sel = threshold(exclusive(s1, m, p, rows), p, tau)
        has[s1[sel]] = True
    return present & ~has


def scoring_set(s1, m, p_e1b, p_s3, empty_ent, band, rescue_k: int, rescue_max_p: float, n_ent: int) -> dict:
    """D4: band (E1b p in ``band``) | rescue (top-``rescue_k`` by E1b p of S1 that sub3 predicts empty or
    whose sub3 max p < ``rescue_max_p``). Returns row masks ``inband``, ``rescued`` and their union."""
    mx = sg.seg_max(p_s3.to(torch.float64), s1, n_ent)
    resc_ent = empty_ent | (mx < rescue_max_p)
    top = sg.topk_mask(s1, p_e1b.to(torch.float64), rescue_k, tie=m)
    rescued = resc_ent[s1] & top
    pe = p_e1b.to(torch.float64)
    inband = (pe >= band[0]) & (pe <= band[1])
    return {"inband": inband, "rescued": rescued, "any": inband | rescued}
