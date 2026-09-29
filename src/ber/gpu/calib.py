"""Calibration on CUDA tensors (item 6): Platt (LBFGS, float64), binned isotonic, two-sample KS.

Platt minimises sum(BCE) + 0.5 / C * a^2 (sklearn LogisticRegression's objective with the
intercept unpenalised), so C = 1e4 reproduces P3's calibrator.

Binned isotonic: scores are sorted and cut into <= ``n_bins`` equal-count bins; the weighted
isotonic (non-decreasing) regression of the bin label means is computed EXACTLY with the
max-min formula  f_i = max_{j<=i} min_{k>=i} mean(j..k)  as one [bins x bins] tensor op, and
predictions interpolate linearly between bin score means (``searchsorted``), clipped at the ends.
"""
from __future__ import annotations

import torch


def platt_fit(z: torch.Tensor, y: torch.Tensor, C: float = 1e4, iters: int = 30) -> tuple[float, float]:
    """Newton's method on the 2-parameter penalised logistic loss (float64, on the device).

    A fixed number of iterations with no convergence test, so the host never waits on the GPU
    inside the loop (the LBFGS version spent ~2.4 s of host time per fit driving tiny kernels).
    """
    z, y = z.to(torch.float64), y.to(torch.float64)
    ab = torch.zeros(2, dtype=torch.float64, device=z.device)
    reg = torch.tensor([1.0 / C, 0.0], dtype=torch.float64, device=z.device)
    for _ in range(iters):
        p = torch.sigmoid(ab[0] * z + ab[1])
        r = p - y
        w = p * (1 - p)
        g = torch.stack([(r * z).sum(), r.sum()]) + reg * ab
        h = torch.stack([torch.stack([(w * z * z).sum() + reg[0], (w * z).sum()]),
                         torch.stack([(w * z).sum(), w.sum()])])
        ab = ab - torch.linalg.solve(h, g)
    out = ab.tolist()
    return float(out[0]), float(out[1])


def platt_apply(z: torch.Tensor, ab: tuple[float, float]) -> torch.Tensor:
    return torch.sigmoid(ab[0] * z.to(torch.float64) + ab[1])


class BinnedIsotonic:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, n_bins: int = 2000):
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        xs, o = torch.sort(x, stable=True)
        ys = y[o]
        n = xs.numel()
        nb = min(n_bins, n)
        edges = torch.div(torch.arange(nb + 1, device=x.device) * n, nb, rounding_mode="floor")
        bid = torch.bucketize(torch.arange(n, device=x.device), edges[1:], right=True).clamp(max=nb - 1)
        w = torch.zeros(nb, dtype=torch.float64, device=x.device).index_add_(0, bid, torch.ones_like(xs))
        sx = torch.zeros_like(w).index_add_(0, bid, xs)
        sy = torch.zeros_like(w).index_add_(0, bid, ys)
        keep = w > 0
        w, sx, sy = w[keep], sx[keep], sy[keep]
        self.x = sx / w
        cw = torch.cat([torch.zeros(1, dtype=torch.float64, device=x.device), torch.cumsum(w, 0)])
        cy = torch.cat([torch.zeros(1, dtype=torch.float64, device=x.device), torch.cumsum(sy, 0)])
        m = w.numel()
        j = torch.arange(m, device=x.device)[:, None]
        k = torch.arange(m, device=x.device)[None, :]
        M = (cy[k + 1] - cy[j]) / (cw[k + 1] - cw[j]).clamp(min=1e-300)
        M = torch.where(k >= j, M, torch.full_like(M, float("inf")))
        A = torch.flip(torch.cummin(torch.flip(M, [1]), 1).values, [1])  # A[j, i] = min_{k>=i} M[j, k]
        A = torch.where(j <= k, A, torch.full_like(A, float("-inf")))    # only j <= i
        self.f = A.max(0).values                                          # f_i = max_{j<=i} A[j, i]

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        s = s.to(torch.float64).clamp(float(self.x[0]), float(self.x[-1]))
        i = torch.searchsorted(self.x, s, right=True).clamp(1, self.x.numel() - 1)
        x0, x1 = self.x[i - 1], self.x[i]
        f0, f1 = self.f[i - 1], self.f[i]
        t = torch.where(x1 > x0, (s - x0) / (x1 - x0).clamp(min=1e-300), torch.zeros_like(s))
        return (f0 + t * (f1 - f0)).clamp(0.0, 1.0)


class Isotonic(BinnedIsotonic):
    """EXACT isotonic regression on the GPU (sklearn ``IsotonicRegression(out_of_bounds="clip")``).

    Ties in x are pooled first (weighted mean, as sklearn does); then pool-adjacent-violators runs
    in parallel: every maximal run of adjacent blocks with non-increasing means is merged in one
    pass (each adjacent violating pair must share a block in the solution, so merging the whole run
    is valid); passes repeat until no violation is left. Knots = the unique x with their fitted
    block means; prediction interpolates linearly and clips at the ends (``BinnedIsotonic.__call__``),
    which is what sklearn's ``interp`` over its block thresholds computes.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, max_passes: int = 1_000_000):  # noqa: D107
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        xs, o = torch.sort(x, stable=True)
        ux, inv, cnt = torch.unique_consecutive(xs, return_inverse=True, return_counts=True)
        w = cnt.to(torch.float64)
        sy = torch.zeros_like(w).index_add_(0, inv, y[o])
        blk = torch.arange(ux.numel(), device=x.device)
        bw, bs = w, sy
        for _ in range(max_passes):
            mean = bs / bw
            viol = mean[:-1] >= mean[1:]
            if not bool(viol.any()):
                break
            start = torch.ones(mean.numel(), dtype=torch.bool, device=x.device)
            start[1:] = ~viol
            gid = torch.cumsum(start.to(torch.int64), 0) - 1
            G = int(gid[-1]) + 1
            bw = torch.zeros(G, dtype=torch.float64, device=x.device).index_add_(0, gid, bw)
            bs = torch.zeros(G, dtype=torch.float64, device=x.device).index_add_(0, gid, bs)
            blk = gid[blk]
        self.x = ux
        self.f = (bs / bw)[blk]
        if self.x.numel() == 1:  # single knot: constant prediction
            self.x = torch.cat([self.x, self.x + 1e-12])
            self.f = torch.cat([self.f, self.f])


def ks(a: torch.Tensor, b: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, _ = torch.sort(a.to(torch.float64))
    b, _ = torch.sort(b.to(torch.float64))
    allv = torch.cat([a, b])
    ca = torch.searchsorted(a, allv, right=True).to(torch.float64) / a.numel()
    cb = torch.searchsorted(b, allv, right=True).to(torch.float64) / b.numel()
    return float((ca - cb).abs().max())


def psi(ref: torch.Tensor, new: torch.Tensor, bins: int = 10) -> float:
    """PSI on reference deciles (P3's definition)."""
    ref, new = ref.to(torch.float64), new.to(torch.float64)
    q = torch.quantile(ref, torch.linspace(0, 1, bins + 1, dtype=torch.float64, device=ref.device))
    edges = torch.unique(q)
    inner = edges[1:-1]
    a = torch.bincount(torch.bucketize(ref, inner, right=True), minlength=inner.numel() + 1).to(torch.float64)
    b = torch.bincount(torch.bucketize(new, inner, right=True), minlength=inner.numel() + 1).to(torch.float64)
    a = a / ref.numel() + 1e-4
    b = b / new.numel() + 1e-4
    return float(((b - a) * torch.log(b / a)).sum())
