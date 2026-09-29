"""Threshold and segment-threshold search as [n_tau x n_S1] computations (item 5).

For kept pairs, ``b = searchsorted(taus, p, right=True)`` is the number of grid values <= p,
so the pair is selected for tau index t iff t < b. Per-S1 histograms over b (``index_put_``
with accumulate) and a reverse cumulative sum give selected / true-positive counts for every
tau at once; F0.5 then follows per S1 and the macro is a column mean. Ties between taus go to
the smallest tau (first maximum), as in P3's ascending strict-improvement loop.
"""
from __future__ import annotations

import torch

from ber.gpu.metric import f_from_counts

DEFAULT_GRID = (0.10, 0.95, 426)  # step 0.002


def grid(lo: float = DEFAULT_GRID[0], hi: float = DEFAULT_GRID[1], n: int = DEFAULT_GRID[2], device="cuda"):
    return torch.linspace(lo, hi, n, dtype=torch.float64, device=device).round(decimals=6)


def _local(s1, ids, n_ent):
    lut = torch.full((n_ent,), -1, dtype=torch.int64, device=s1.device)
    lut[ids] = torch.arange(ids.numel(), device=s1.device)
    return lut[s1]


def counts_by_tau(s1, y, p, sel_rows, taus, ids, n_ent):
    """(npred, tp) of shape [n_ids, n_tau]: counts of rows with p >= tau_t per S1 in ``ids``."""
    loc = _local(s1, ids, n_ent)
    r = sel_rows & (loc >= 0)
    b = torch.searchsorted(taus, p[r].to(torch.float64), right=True)
    nt = taus.numel()
    H = torch.zeros((ids.numel(), nt + 1), dtype=torch.int32, device=p.device)
    Ht = torch.zeros_like(H)
    H.index_put_((loc[r], b), torch.ones_like(b, dtype=torch.int32), accumulate=True)
    Ht.index_put_((loc[r], b), y[r].to(torch.int32), accumulate=True)
    rc = torch.flip(torch.cumsum(torch.flip(H, [1]), 1), [1])
    rt = torch.flip(torch.cumsum(torch.flip(Ht, [1]), 1), [1])
    return rc[:, 1:], rt[:, 1:]


def counts_fixed(s1, y, sel, ids, n_ent):
    loc = _local(s1, ids, n_ent)
    r = sel & (loc >= 0)
    npred = torch.zeros(ids.numel(), dtype=torch.int32, device=s1.device).index_add_(
        0, loc[r], torch.ones(int(r.sum()), dtype=torch.int32, device=s1.device))
    tp = torch.zeros_like(npred).index_add_(0, loc[r], y[r].to(torch.int32))
    return npred, tp


def macro_grid(npred, tp, ntrue_ids, chunk: int = 50_000) -> torch.Tensor:
    """Macro F0.5 per tau column, S1 rows chunked to bound VRAM."""
    tot = torch.zeros(npred.shape[1], dtype=torch.float64, device=npred.device)
    for a in range(0, npred.shape[0], chunk):
        tot += f_from_counts(tp[a:a + chunk], npred[a:a + chunk], ntrue_ids[a:a + chunk, None]).sum(0)
    return tot / npred.shape[0]


def best_tau(s1, y, p, keep, ids, ntrue, taus) -> tuple[float, float]:
    """argmax-tau (first maximum) of the macro F0.5 over ``ids`` for the kept pairs."""
    n_ent = ntrue.numel()
    npred, tp = counts_by_tau(s1, y, p, keep, taus, ids, n_ent)
    m = macro_grid(npred, tp, ntrue[ids])
    i = int(torch.argmax(m))
    return float(taus[i]), float(m[i])


def segment_taus(s1, y, p, keep, seg, ids, ntrue, taus, tau0: float, sweeps: int = 2) -> dict:
    """Coordinate search of one tau per segment code in ``seg`` (start: ``tau0`` everywhere)."""
    n_ent = ntrue.numel()
    codes = [int(c) for c in torch.unique(seg[keep]).tolist()]
    cur = {c: tau0 for c in codes}
    nti = ntrue[ids]
    for _ in range(sweeps):
        for c in codes:
            tv = torch.zeros_like(p, dtype=torch.float64)
            for c2, t2 in cur.items():
                tv = torch.where(seg == c2, torch.full_like(tv, t2), tv)
            other = keep & (seg != c) & (p.to(torch.float64) >= tv)
            npo, tpo = counts_fixed(s1, y, other, ids, n_ent)
            npg, tpg = counts_by_tau(s1, y, p, keep & (seg == c), taus, ids, n_ent)
            m = macro_grid(npg + npo[:, None], tpg + tpo[:, None], nti)
            cur[c] = float(taus[int(torch.argmax(m))])
    return cur


def apply_segment_taus(p, keep, seg, taus: dict) -> torch.Tensor:
    tv = torch.full_like(p, float("inf"), dtype=torch.float64)
    for c, t in taus.items():
        tv = torch.where(seg == int(c), torch.full_like(tv, t), tv)
    return keep & (p.to(torch.float64) >= tv)
