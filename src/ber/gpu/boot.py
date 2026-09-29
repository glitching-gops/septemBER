"""Bootstrap confidence intervals over S1 on the GPU (item 9)."""
from __future__ import annotations

import torch


def ci(d: torch.Tensor, n_boot: int = 1000, seed: int = 0, chunk: int = 100, q=(0.025, 0.975)) -> tuple[float, float]:
    """Percentile CI of the mean of per-S1 values ``d`` (resampling S1 with replacement)."""
    d = d.to(torch.float64)
    n = d.numel()
    g = torch.Generator(device=d.device).manual_seed(seed)
    means = []
    for a in range(0, n_boot, chunk):
        c = min(chunk, n_boot - a)
        idx = torch.randint(0, n, (c, n), generator=g, device=d.device)
        means.append(d[idx].mean(1))
    m = torch.cat(means)
    qs = torch.quantile(m, torch.tensor(q, dtype=torch.float64, device=d.device))
    return float(qs[0]), float(qs[1])
