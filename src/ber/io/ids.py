"""One-time string id -> int64 encoding, cached to ``<cache_dir>/ids/`` (permitted CPU step).

Per split, every S1/S2/S3 entity id gets the index of its position in the LEXICOGRAPHICALLY
SORTED list of all ids of that split. So "smaller s1_id" (P3's exclusivity tie-break) is
"smaller int" on the GPU. Pair tables are encoded once per feature shard / pair file and the
int columns are cached next to the id table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ber.io_utils import read_ground_truth, read_source

SRC_CODE = {"source1": 1, "source2": 2, "source3": 3}


def ids_dir(cfg) -> Path:
    d = Path(cfg["abs"]["cache_dir"]) / "ids"
    d.mkdir(parents=True, exist_ok=True)
    return d


_TAB: dict = {}


def entity_table(cfg, split: str) -> pd.DataFrame:
    """entity_id, idx (sorted order), src (1/2/3), country code (int, codes in sorted label order)."""
    p = ids_dir(cfg) / f"{split}_entities.parquet"
    if str(p) in _TAB:
        return _TAB[str(p)]
    if p.exists():
        _TAB[str(p)] = pd.read_parquet(p)
        return _TAB[str(p)]
    parts = []
    for s, c in SRC_CODE.items():
        raw, _ = read_source(cfg["abs"][f"{split}.{s}"], validate=False)
        parts.append(pd.DataFrame({"entity_id": raw["entity_id"].astype(str), "src": np.int8(c),
                                   "country_label": raw["country"].astype(str)}))
    t = pd.concat(parts, ignore_index=True).sort_values("entity_id", kind="stable").reset_index(drop=True)
    assert t["entity_id"].is_unique, "duplicate entity ids across sources"
    t["idx"] = np.arange(len(t), dtype=np.int64)
    labels = sorted(t["country_label"].unique())
    t["country"] = t["country_label"].map({c: i for i, c in enumerate(labels)}).astype(np.int16)
    t.to_parquet(p, index=False)
    return t


_ENC: dict = {}


def encoder(cfg, split: str) -> pd.Index:
    key = (str(cfg["abs"]["cache_dir"]), split)
    if key not in _ENC:
        _ENC[key] = pd.Index(entity_table(cfg, split)["entity_id"].to_numpy())
    return _ENC[key]


def encode(cfg, split: str, ids) -> np.ndarray:
    idx = encoder(cfg, split).get_indexer(np.asarray(ids, dtype=object))
    if (idx < 0).any():
        raise KeyError(f"{int((idx < 0).sum())} ids not in the {split} entity table")
    return idx.astype(np.int64)


def country_labels(cfg, split: str) -> list[str]:
    t = entity_table(cfg, split)
    return sorted(t["country_label"].unique())


def gt_pairs(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Train ground truth as (s1 idx, match idx) int64 arrays (cached)."""
    p = ids_dir(cfg) / "train_gt.npz"
    if p.exists():
        z = np.load(p)
        return z["s1"], z["m"]
    a = cfg["abs"]
    s1raw, _ = read_source(a["train.source1"], validate=False)
    gt, _ = read_ground_truth(a["train.ground_truth"], s1_ids=s1raw["entity_id"])
    s = [k for k, v in gt.items() for _ in v]
    m = [x for v in gt.values() for x in v]
    s1i, mi = encode(cfg, "train", s), encode(cfg, "train", m)
    np.savez(p, s1=s1i, m=mi)
    return s1i, mi


def encode_pairs_cached(cfg, split: str, src_path: Path, tag: str, s1_col="s1_id", m_col="m_id") -> dict:
    """int64 s1 / m (and source code) for a pair parquet, cached as ``<ids>/<tag>.npz``."""
    p = ids_dir(cfg) / f"{tag}.npz"
    if p.exists() and p.stat().st_mtime >= Path(src_path).stat().st_mtime:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    d = pd.read_parquet(src_path, columns=[s1_col, m_col])
    out = {"s1": encode(cfg, split, d[s1_col]), "m": encode(cfg, split, d[m_col])}
    np.savez(p, **out)
    return out
