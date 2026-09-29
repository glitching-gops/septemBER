"""Masked dense top-k retrieval on the GPU (E7 sub-index): chunked matmul + topk per partition."""
from __future__ import annotations

import torch


def topk(q: torch.Tensor, t: torch.Tensor, k: int, chunk: int = 16384, max_elems: int = 1 << 30) -> tuple:
    """For each query row: the top-``k`` target rows by cosine (inputs L2-normalised).

    Returns flattened (query row, target row, score, rank 1..k)."""
    kk = min(k, t.shape[0])
    chunk = max(16, min(chunk, max_elems // max(1, t.shape[0])))  # score block <= max_elems (fp16)
    qi, ti, sc = [], [], []
    tt = t.to(torch.float16).T.contiguous()
    for a in range(0, q.shape[0], chunk):
        s = q[a:a + chunk].to(torch.float16) @ tt
        v, i = torch.topk(s, kk, dim=1, sorted=True)
        v = v.float()
        qi.append(torch.arange(a, a + s.shape[0], device=q.device).repeat_interleave(kk))
        ti.append(i.reshape(-1))
        sc.append(v.reshape(-1))
    return torch.cat(qi), torch.cat(ti), torch.cat(sc), torch.arange(1, kk + 1, device=q.device).repeat(q.shape[0])
