"""B6: dense name retrieval with multilingual-E5 (intfloat/multilingual-e5-small, MIT).

Names are embedded in their ORIGINAL script (NFKC), so a Latin S1 name and its
Devanagari/Tamil/... rendering land close together without transliteration. The
model is used only as an encoder; there is no fine-tuning and no external lookup.
Embeddings are L2-normalised float16, and top-k cosine is a chunked matmul + topk
on the GPU set in ``compute.gpu_device``. GPU use requires ``compute.use_gpu: true``.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

from ber import compute

from ber.embed import MODEL_REVISION, MODEL_SUBDIR  # noqa: F401  (re-exported)


def _device():
    from ber.embed import gpu_setup

    return gpu_setup()


def dense_topk_numpy(Q: np.ndarray, T: np.ndarray, k: int, chunk: int = 4096, min_score: float = 0.0):
    """CPU fallback / reference: exact brute-force top-k with NumPy."""
    k = min(k, T.shape[0])
    qi, ti, sc = [], [], []
    Tf = np.asarray(T, dtype=np.float32)
    for s in range(0, Q.shape[0], chunk):
        S = np.asarray(Q[s:s + chunk], dtype=np.float32) @ Tf.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < S.shape[1] else np.tile(np.arange(S.shape[1]), (len(S), 1))
        v = np.take_along_axis(S, idx, axis=1)
        rows = np.repeat(np.arange(s, s + len(S)), idx.shape[1])
        keep = v.ravel() >= min_score
        qi.append(rows[keep]); ti.append(idx.ravel()[keep]); sc.append(v.ravel()[keep])
    q, t, s_ = np.concatenate(qi), np.concatenate(ti).astype(np.int64), np.concatenate(sc).astype(np.float32)
    order = np.lexsort((t, -s_, q))
    return q[order].astype(np.int64), t[order], s_[order]


def dense_topk(Q: np.ndarray, T: np.ndarray, k: int, chunk: int = 1024, min_score: float = 0.0):
    """(query_idx, target_idx, cosine) exact top-k per query row on the GPU (fp16 mm + topk).

    Falls back to NumPy only when compute.cpu_fallback is true; otherwise GPU errors propagate.
    """
    try:
        import torch

        dev = _device()
    except Exception:
        # GPU-first policy: only an explicit ``compute.cpu_fallback: true`` (unit tests on toy
        # data) may take the NumPy path; otherwise the GPU error propagates and the run stops.
        from ber import compute

        if not compute.cpu_fallback():
            raise
        return dense_topk_numpy(Q, T, k, min_score=min_score)
    Tg = torch.from_numpy(T).to(dev)
    k = min(k, T.shape[0])
    qi, ti, sc = [], [], []
    with torch.inference_mode():
        for s in range(0, Q.shape[0], chunk):
            q = torch.from_numpy(Q[s:s + chunk]).to(dev)
            v, i = torch.topk(q @ Tg.T, k, dim=1)
            v, i = v.float().cpu().numpy(), i.cpu().numpy()
            rows = np.repeat(np.arange(s, s + len(v)), k)
            keep = v.ravel() >= min_score
            qi.append(rows[keep])
            ti.append(i.ravel()[keep])
            sc.append(v.ravel()[keep])
    del Tg
    torch.cuda.empty_cache()
    q, t, s_ = np.concatenate(qi), np.concatenate(ti).astype(np.int64), np.concatenate(sc).astype(np.float32)
    order = np.lexsort((t, -s_, q))
    return q[order].astype(np.int64), t[order], s_[order]
