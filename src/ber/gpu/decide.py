"""Exclusivity and set decisions on CUDA tensors (items 2, 3).

Semantics are P3's ``ber.decide.sets``: each S2/S3 record is kept only for its highest-p S1
(ties: smaller s1 index = smaller s1_id string); ``threshold`` keeps p >= tau; the
expected-F0.5 rule picks, per S1, the prefix of its p-sorted candidates (possibly empty) with
the highest exact expected F0.5 under independent Bernoulli matches.
"""
from __future__ import annotations

import torch

from ber.gpu import segment as sg

BETA2 = 0.25


def exclusive(s1: torch.Tensor, m: torch.Tensor, p: torch.Tensor, rows: torch.Tensor | None = None,
              n_m: int | None = None) -> torch.Tensor:
    """Bool mask of kept pairs among ``rows`` (all rows if None)."""
    n_m = n_m or int(m.max()) + 1
    pp = p.to(torch.float64)
    if rows is not None:
        pp = torch.where(rows, pp, torch.full_like(pp, float("-inf")))
    best = sg.seg_max(pp, m, n_m)
    cand = (pp == best[m]) & torch.isfinite(pp)
    big = torch.iinfo(torch.int64).max
    smin = sg.seg_min(torch.where(cand, s1, torch.full_like(s1, big)), m, n_m, big)
    keep = cand & (s1 == smin[m])
    return keep if rows is None else keep & rows


def threshold(keep: torch.Tensor, p: torch.Tensor, tau) -> torch.Tensor:
    """``tau``: scalar or per-row tensor."""
    return keep & (p.to(torch.float64) >= (tau if torch.is_tensor(tau) else float(tau)))


def expected_f(s1: torch.Tensor, p: torch.Tensor, keep: torch.Tensor, max_n: int = 40, min_p: float = 1e-4,
               chunk: int = 65536) -> torch.Tensor:
    """Bool mask: the expected-F0.5-optimal prefix of each S1's kept candidates."""
    dev = p.device
    rows = torch.nonzero(keep).squeeze(1)
    out = torch.zeros_like(keep)
    if rows.numel() == 0:
        return out
    ss, pp = s1[rows], p[rows].to(torch.float64)
    perm = sg.order([(ss, False), (pp, True), (rows, False)])
    ss, pp, rows = ss[perm], pp[perm], rows[perm]
    pos = sg.pos_in_segment(ss)
    st = sg.seg_starts(ss)
    gid = torch.cumsum(st.to(torch.int64), 0) - 1
    G = int(gid[-1]) + 1
    L = int(pos.max()) + 1
    P = torch.zeros((G, L), dtype=torch.float64, device=dev)
    P[gid, pos] = pp
    n_sel = torch.clamp(sg.seg_count(gid, G, pp >= min_p), max=max_n)
    kbest = torch.empty(G, dtype=torch.int64, device=dev)
    a = torch.arange(L + 1, device=dev, dtype=torch.float64)
    for g0 in range(0, G, chunk):
        Pc = P[g0:g0 + chunk]
        Gc = Pc.shape[0]
        # suffix distributions of items k..L-1 (padding p = 0 is a no-op Bernoulli)
        suf = [None] * (L + 1)
        cur = torch.zeros((Gc, L + 1), dtype=torch.float64, device=dev)
        cur[:, 0] = 1
        suf[L] = cur
        for k in range(L - 1, -1, -1):
            q = Pc[:, k:k + 1]
            nxt = cur * (1 - q)
            nxt[:, 1:] += cur[:, :-1] * q
            suf[k] = cur = nxt
        ef = torch.full((Gc, L + 1), float("-inf"), dtype=torch.float64, device=dev)
        ef[:, 0] = suf[0][:, 0]
        pin = torch.zeros((Gc, L + 1), dtype=torch.float64, device=dev)
        pin[:, 0] = 1
        for k in range(1, L + 1):
            q = Pc[:, k - 1:k]
            nxt = pin * (1 - q)
            nxt[:, 1:] += pin[:, :-1] * q
            pin = nxt
            Fk = (1 + BETA2) * a[:, None] / (BETA2 * (a[:, None] + a[None, :]) + k)
            ef[:, k] = torch.einsum("ga,ab,gb->g", pin, Fk, suf[k])
        kk = torch.arange(L + 1, device=dev)[None, :]
        ef = torch.where(kk <= n_sel[g0:g0 + chunk, None], ef, torch.full_like(ef, float("-inf")))
        kbest[g0:g0 + chunk] = torch.argmax(ef, 1)
    out[rows[pos < kbest[gid]]] = True
    return out
