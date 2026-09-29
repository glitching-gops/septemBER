"""Loaders: parquet / npy bytes -> pinned host buffer -> CUDA tensor (no host computation).

Pair keys come from the cached int encoding (``ber.io.ids``); feature columns are moved one
at a time and stacked on the device.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.io import ids as idsmod


_STAGE = {"buf": None}


def _stage(nbytes: int):
    """One reusable page-locked staging buffer (grown on demand). Pinning a fresh buffer per array
    cost ~20 s wall / ~1,000 s CPU for the dev load (page-locking is expensive; reuse is free)."""
    import torch
    b = _STAGE["buf"]
    if b is None or b.numel() < nbytes:
        _STAGE["buf"] = b = torch.empty(max(nbytes, 1 << 26), dtype=torch.uint8).pin_memory()
    return b


def to_dev(a, device, dtype=None):
    """Host array -> CUDA tensor through a pinned staging buffer. The host copy is not used afterwards."""
    import torch
    a = np.asarray(a)
    src = torch.from_numpy(a if a.flags.writeable and a.flags.c_contiguous else np.asarray(a).copy())
    buf = _stage(src.numel() * src.element_size())
    pinned = buf[:src.numel() * src.element_size()].view(src.dtype).view(src.shape)
    pinned.copy_(src)
    out = pinned.to(device, non_blocking=True)
    torch.cuda.current_stream(device).synchronize()  # the staging buffer is reused by the next call
    return out.to(dtype) if dtype is not None and out.dtype != dtype else out


def feature_shards(cfg, scope: str) -> list[Path]:
    v = cfg["versions"]
    root = Path(cfg["abs"]["features_dir"]) / f"{v['norm_version']}__{v['blocker_set']}__K{v['K']}__{v['feature_version']}" / scope
    parts = sorted(root.glob("shard=*/part.parquet"))
    if not parts or any(not (p.parent / "_SUCCESS").exists() for p in parts):
        raise FileNotFoundError(f"missing / incomplete feature shards under {root}")
    return parts


def load_pairs(cfg, scope: str, split: str, columns: list[str], device, timer=None) -> dict:
    """{"s1", "m" (int64), "src" (int8: 2 / 3), "X" (float32 [n, len(columns)])} on ``device``.

    The whole read + host->device transfer is the permitted I/O phase (``timer.io``)."""
    import torch
    t0, c0 = time.time(), time.process_time()
    parts = feature_shards(cfg, scope)
    s1s, ms, cols = [], [], [[] for _ in columns]
    for i, p in enumerate(parts):
        k = idsmod.encode_pairs_cached(cfg, split, p, f"{scope}_shard{i:03d}")
        tab = pq.read_table(p, columns=columns)
        s1s.append(to_dev(k["s1"], device))
        ms.append(to_dev(k["m"], device))
        for j, c in enumerate(columns):
            cols[j].append(to_dev(tab.column(c).to_numpy(zero_copy_only=False), device, torch.float32))
    if timer:
        timer.io(time.time() - t0, time.process_time() - c0)
    out = {"s1": torch.cat(s1s), "m": torch.cat(ms)}
    out["X"] = torch.stack([torch.cat(c) for c in cols], 1) if columns else None
    ent = entity_src(cfg, split, device)
    out["src"] = ent[out["m"]]
    return out


def entity_src(cfg, split: str, device):
    import torch
    t = idsmod.entity_table(cfg, split)
    return to_dev(t["src"].to_numpy(), device, torch.int8)


def entity_country(cfg, split: str, device):
    import torch
    t = idsmod.entity_table(cfg, split)
    return to_dev(t["country"].to_numpy(), device, torch.int16)


def dev_meta(cfg, device) -> dict:
    """Dev slice as tensors: s1 idx, xgb fold, holdout flag (P3's component-hash holdout).
    The encoding is computed once and cached next to the id table."""
    import pandas as pd
    import torch
    from ber.verify import folds as vf
    a = cfg["abs"]
    src = a["runs_dir"] / "dev_slice.parquet"
    cache = idsmod.ids_dir(cfg) / f"dev_meta_{cfg['verifier']['holdout_frac']}.npz"
    if not cache.exists() or cache.stat().st_mtime < Path(src).resolve().stat().st_mtime:
        dev = pd.read_parquet(src, columns=["s1_id", "fold"])
        comp = pq.read_table(a["runs_dir"] / "splits.parquet", columns=["s1_id", "component_id"]).to_pandas()
        cid = pd.Index(comp["s1_id"]).get_indexer(dev["s1_id"])  # id lookup (encoding step)
        hold = vf.holdout_mask(comp["component_id"].to_numpy()[cid], cfg["verifier"]["holdout_frac"])
        np.savez(cache, s1=idsmod.encode(cfg, "train", dev["s1_id"]), fold=dev["fold"].to_numpy(np.int64), hold=hold)
    z = np.load(cache)
    return {"s1": to_dev(z["s1"], device), "fold": to_dev(z["fold"], device), "holdout": to_dev(z["hold"], device, torch.bool)}


def gt_dev(cfg, device) -> dict:
    s, m = idsmod.gt_pairs(cfg)
    return {"s1": to_dev(s, device), "m": to_dev(m, device)}


def pair_file(cfg, split: str, path: Path, tag: str, columns: list[str], device) -> dict:
    """Any (s1_id, m_id, ...) parquet -> int keys + float32 columns on device."""
    import torch
    k = idsmod.encode_pairs_cached(cfg, split, path, tag)
    tab = pq.read_table(path, columns=columns)
    out = {"s1": to_dev(k["s1"], device), "m": to_dev(k["m"], device)}
    for c in columns:
        out[c] = to_dev(tab.column(c).to_numpy(zero_copy_only=False), device, torch.float32)
    return out
