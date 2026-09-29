"""Grouping primitives over CUDA tensors (item 1).

Keys are int64 group ids; ``order`` sorts lexicographically by several keys with successive
stable sorts (least significant key first), so equal keys keep input order and every tie is
resolved by an explicit later key.
"""
from __future__ import annotations

import torch


def order(keys: list[tuple[torch.Tensor, bool]]) -> torch.Tensor:
    """Permutation sorting by ``keys`` = [(tensor, descending), ...], most significant FIRST."""
    n = keys[0][0].shape[0]
    perm = torch.arange(n, device=keys[0][0].device)
    for k, desc in reversed(keys):
        _, o = torch.sort(k[perm], stable=True, descending=desc)
        perm = perm[o]
    return perm


def seg_starts(sorted_key: torch.Tensor) -> torch.Tensor:
    """Bool mask: row starts a new segment of equal consecutive keys."""
    st = torch.ones_like(sorted_key, dtype=torch.bool)
    if sorted_key.numel() > 1:
        st[1:] = sorted_key[1:] != sorted_key[:-1]
    return st


def pos_in_segment(sorted_key: torch.Tensor) -> torch.Tensor:
    """0-based position of each (sorted) row inside its segment."""
    st = seg_starts(sorted_key)
    idx = torch.arange(sorted_key.numel(), device=sorted_key.device)
    first = torch.cummax(torch.where(st, idx, torch.zeros_like(idx)), 0).values
    return idx - first


def rank_in_group(group: torch.Tensor, score: torch.Tensor, tie: torch.Tensor | None = None) -> torch.Tensor:
    """1-based rank by ``score`` descending within ``group``; ties by ``tie`` ascending, then row order."""
    keys = [(group, False), (score, True)] + ([(tie, False)] if tie is not None else [])
    perm = order(keys)
    r = torch.empty(group.numel(), dtype=torch.int64, device=group.device)
    r[perm] = pos_in_segment(group[perm]) + 1
    return r


def topk_mask(group: torch.Tensor, score: torch.Tensor, k: int, tie: torch.Tensor | None = None) -> torch.Tensor:
    return rank_in_group(group, score, tie) <= k


def seg_sum(values: torch.Tensor, group: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros((n,) + values.shape[1:], dtype=values.dtype, device=values.device)
    return out.index_add_(0, group, values)


def seg_count(group: torch.Tensor, n: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    v = torch.ones(group.numel(), dtype=torch.int64, device=group.device)
    if mask is not None:
        v = v * mask.to(torch.int64)
    return seg_sum(v, group, n)


def seg_max(values: torch.Tensor, group: torch.Tensor, n: int, fill: float = float("-inf")) -> torch.Tensor:
    out = torch.full((n,), fill, dtype=values.dtype, device=values.device)
    return out.scatter_reduce(0, group, values, "amax", include_self=True)


def seg_min(values: torch.Tensor, group: torch.Tensor, n: int, fill) -> torch.Tensor:
    out = torch.full((n,), fill, dtype=values.dtype, device=values.device)
    return out.scatter_reduce(0, group, values, "amin", include_self=True)


def group_argmax(group: torch.Tensor, score: torch.Tensor, tie: torch.Tensor, n: int) -> torch.Tensor:
    """Row index of the max ``score`` per group (ties: smaller ``tie``); -1 for empty groups."""
    best = seg_max(score, group, n)
    cand = score == best[group]
    big = torch.iinfo(torch.int64).max
    tmin = seg_min(torch.where(cand, tie, torch.full_like(tie, big)), group, n, big)
    win = cand & (tie == tmin[group])
    rows = torch.arange(group.numel(), device=group.device)
    out = torch.full((n,), big, dtype=torch.int64, device=group.device)
    out = out.scatter_reduce(0, group[win], rows[win], "amin", include_self=True)
    return torch.where(out == big, torch.full_like(out, -1), out)


def isin_pairs(a1: torch.Tensor, a2: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor, width: int) -> torch.Tensor:
    """Membership of pairs (a1, a2) in the set of pairs (b1, b2); ids < ``width``."""
    return torch.isin(a1 * width + a2, b1 * width + b2)


def lookup_pairs(a1, a2, b1, b2, width: int) -> torch.Tensor:
    """Row index into (b1, b2) of each (a1, a2) pair, or -1 (b pairs must be unique)."""
    kb = b1 * width + b2
    sk, o = torch.sort(kb, stable=True)
    ka = a1 * width + a2
    pos = torch.searchsorted(sk, ka).clamp(max=max(0, sk.numel() - 1))
    hit = sk.numel() > 0
    found = (sk[pos] == ka) if hit else torch.zeros_like(ka, dtype=torch.bool)
    return torch.where(found, o[pos], torch.full_like(pos, -1))
