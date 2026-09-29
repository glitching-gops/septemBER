"""Parallel normalisation of source tables, cached as Parquet per split/source."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ber import compute
from ber.normalize.addresses import normalize_address
from ber.normalize.names import normalize_name

NAME_COLS = ["name_script", "name_ascii", "name_norm", "name_core", "name_alt", "legal_class", "is_domain"]
ADDR_COLS = ["addr_norm", "addr_core", "landmark", "nums", "house_no", "postcode_like", "region", "addr_missing"]
NORM_COLS = ["entity_id", "country"] + NAME_COLS + ADDR_COLS


def _normalize_chunk(args: tuple[list[str], list[str]]) -> dict[str, list]:
    names, addrs = args
    out: dict[str, list] = {c: [] for c in NAME_COLS + ADDR_COLS}
    for n, a in zip(names, addrs):
        rn = normalize_name(n)
        ra = normalize_address(a)
        for c in NAME_COLS:
            out[c].append(rn[c])
        for c in ADDR_COLS:
            out[c].append(ra[c])
    return out


def normalize_frame(df: pd.DataFrame, chunk: int = 20_000, processes: int | None = None) -> pd.DataFrame:
    """Normalise a source frame (entity_id, business_name, business_address, country)."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    procs = processes or compute.n_jobs()
    if procs == 1 or len(jobs) == 1:
        parts = [_normalize_chunk(j) for j in jobs]
    else:
        with compute.pool(procs) as p:
            parts = p.map(_normalize_chunk, jobs, chunksize=1)
    cols = {c: [v for part in parts for v in part[c]] for c in NAME_COLS + ADDR_COLS}
    out = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(), "country": df["country"].to_numpy(), **cols})
    return out[NORM_COLS]


def norm_cache_path(cfg: dict, split: str, source: str) -> Path:
    return cfg["abs"]["cache_dir"] / "norm" / f"{split}_{source}.parquet"


def load_norm(cfg: dict, split: str, source: str, columns: list[str] | None = None) -> pd.DataFrame:
    p = norm_cache_path(cfg, split, source)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/run_normalize.py")
    return pd.read_parquet(p, columns=columns)


def normalize_and_cache(cfg: dict, split: str, source: str, df: pd.DataFrame) -> tuple[Path, float]:
    t = time.time()
    out = normalize_frame(df)
    p = norm_cache_path(cfg, split, source)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    return p, time.time() - t
