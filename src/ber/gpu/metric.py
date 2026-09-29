"""Per-S1 F0.5 and macro averages on CUDA tensors (item 4).

|T| comes from the GROUND-TRUTH match counts per S1 (``n_true``), not from candidate labels,
so true matches missing from the candidates still count as misses. empty/empty = 1; one
side empty = 0.
"""
from __future__ import annotations

import torch

from ber.gpu import segment as sg

BETA2 = 0.25


def n_true(gt_s1: torch.Tensor, n_ent: int) -> torch.Tensor:
    return sg.seg_count(gt_s1, n_ent)


def labels(s1, m, gt_s1, gt_m, n_ent: int) -> torch.Tensor:
    return sg.isin_pairs(s1, m, gt_s1, gt_m, n_ent)


def f_from_counts(tp: torch.Tensor, npred: torch.Tensor, ntrue: torch.Tensor) -> torch.Tensor:
    tp, npred, ntrue = tp.to(torch.float64), npred.to(torch.float64), ntrue.to(torch.float64)
    P = torch.where(npred > 0, tp / npred.clamp(min=1), torch.zeros_like(tp))
    R = torch.where(ntrue > 0, tp / ntrue.clamp(min=1), torch.zeros_like(tp))
    f = torch.where(tp > 0, (1 + BETA2) * P * R / (BETA2 * P + R).clamp(min=1e-300), torch.zeros_like(tp))
    return torch.where((ntrue == 0) & (npred == 0), torch.ones_like(f), f)


def per_s1_f(sel: torch.Tensor, s1: torch.Tensor, y: torch.Tensor, ntrue: torch.Tensor) -> torch.Tensor:
    """F0.5 for EVERY entity index (length of ``ntrue``); read off the S1 you evaluate."""
    n = ntrue.numel()
    tp = sg.seg_count(s1[sel], n, y[sel])
    npred = sg.seg_count(s1[sel], n)
    return f_from_counts(tp, npred, ntrue)


def macro(f: torch.Tensor, ids: torch.Tensor) -> float:
    return float(f[ids].mean())


def breakdown(f: torch.Tensor, ids: torch.Tensor, country: torch.Tensor, ntrue: torch.Tensor,
              labels_: list[str]) -> dict:
    """Macro overall, per country (index_add over country codes), singleton / non-singleton."""
    fi, ci, ti = f[ids], country[ids].to(torch.int64), ntrue[ids]
    nc = len(labels_)
    s = torch.zeros(nc, dtype=torch.float64, device=f.device).index_add_(0, ci, fi)
    c = torch.zeros(nc, dtype=torch.float64, device=f.device).index_add_(0, ci, torch.ones_like(fi))
    out = {"macro": float(fi.mean())}
    for k, lab in enumerate(labels_):
        if float(c[k]) > 0:
            out[f"macro[{lab}]"] = float(s[k] / c[k])
    single = ti == 0
    out["singleton_mean"] = float(fi[single].mean()) if bool(single.any()) else float("nan")
    out["nonsingleton_mean"] = float(fi[~single].mean()) if bool((~single).any()) else float("nan")
    return out


def micro(sel, s1, y, ntrue, ids_mask: torch.Tensor) -> tuple[float, float]:
    """Micro precision / recall over the S1 in ``ids_mask`` (bool over entity indices)."""
    r = sel & ids_mask[s1]
    tp = float((y & r).sum())
    return tp / max(1.0, float(r.sum())), tp / max(1.0, float(ntrue[ids_mask].sum()))
