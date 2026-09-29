"""Sharded, resumable pair-feature stage.

    python scripts/run_features.py --scope dev            # -> features/<key>/shard=NN/
    python scripts/run_features.py --scope test --shards 64
    python scripts/run_features.py --scope dev --verify    # re-derive 1 shard, compare to cache

Candidates are split into ``shards`` buckets by a STABLE hash of ``s1_id`` (crc32, not
Python's randomised ``hash``), so a pair always lands in the same shard and every S1's
candidates stay together -- group-wise decisions and grouped CV folds are never split.

Each shard directory holds ``part.parquet``, ``_SUCCESS`` and ``manifest.json``
(config hash, git sha, row count, feature list). A rerun skips shards that already have
``_SUCCESS``, so a long job survives interruption.

The cache key is (norm_version, blocker_set, K, feature_version); bump the relevant
``versions:`` entry in the config and the stage recomputes into a new directory instead
of silently reusing stale features.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ber import compute
from ber.evaluate.ledger import append_run, config_hash, git_hash
from ber.io_utils import load_config
from ber.match.build import FeatureContext
from ber.match.features import FEATURES

KEY_COLS = ["s1_id", "m_id", "source"]


def shard_of(s1_ids: np.ndarray, shards: int) -> np.ndarray:
    """Stable shard index per s1_id (crc32 of the utf-8 bytes)."""
    return np.array([zlib.crc32(s.encode()) % shards for s in s1_ids], dtype=np.int16)


def feature_key(cfg: dict) -> str:
    v = cfg["versions"]
    return f"{v['norm_version']}__{v['blocker_set']}__K{v['K']}__{v['feature_version']}"


def features_dir(cfg: dict, scope: str) -> Path:
    return Path(cfg["abs"]["features_dir"]) / feature_key(cfg) / scope


def load_candidates(cfg: dict, scope: str) -> pd.DataFrame:
    """Candidate table for a scope, cut to K_max per S1 per source."""
    path = Path(cfg["abs"]["cache_dir"]) / "cand" / f"{scope}.parquet"
    cand = pd.read_parquet(path)
    k = int(cfg["versions"]["K"])
    if "cand_rank" in cand:
        cand = cand[cand["cand_rank"] <= k]
    return cand.reset_index(drop=True)


def _write_shard(out: Path, feats: pd.DataFrame, keys: pd.DataFrame, cfg: dict, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([keys.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    tmp = out / "part.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(out / "part.parquet")  # atomic: a crash never leaves a half-written shard
    (out / "manifest.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "_SUCCESS").write_text("")


def run(cfg: dict, scope: str, shards: int, split: str, only: list[int] | None = None) -> dict:
    root = features_dir(cfg, scope)
    root.mkdir(parents=True, exist_ok=True)
    cand = load_candidates(cfg, scope)
    sh = shard_of(cand["s1_id"].to_numpy(), shards)
    chash = config_hash({k: v for k, v in cfg.items() if k != "abs"})
    sha = git_hash(Path(cfg["abs"]["root"]))
    todo = [i for i in range(shards) if not (root / f"shard={i:02d}" / "_SUCCESS").exists()]
    if only is not None:
        todo = [i for i in todo if i in only]
    print(f"[features] {scope}: {len(cand):,} pairs -> {shards} shards at {root}")
    print(f"[features] {len(todo)} shard(s) to build, {shards - len(todo)} already done")

    t0, rows = time.time(), 0
    peak = 0.0
    ctx = None
    for n, i in enumerate(todo, 1):
        part = cand[sh == i].reset_index(drop=True)
        if part.empty:
            _write_shard(root / f"shard={i:02d}", pd.DataFrame(columns=FEATURES),
                         part[KEY_COLS], cfg, {"rows": 0, "config_hash": chash, "git_sha": sha})
            continue
        if ctx is None:  # normalised tables + embedding maps, loaded once for all shards
            gpu = cfg["versions"]["feature_version"] != "f1"  # f1 == the P1 feature set
            with compute.RamMonitor(f"context[{split}]") as cmon:
                ctx = FeatureContext(cfg, split, gpu_features=gpu)
            print(f"  context loaded in {cmon.elapsed:.0f}s (resident {cmon.peak_gb:.1f} GB)", flush=True)
        with compute.RamMonitor(f"features[{scope} shard {i:02d}]") as mon:
            F = ctx.features(part)
        peak = max(peak, mon.peak_gb)
        _write_shard(root / f"shard={i:02d}", F, part[KEY_COLS], cfg,
                     {"rows": int(len(part)), "config_hash": chash, "git_sha": sha,
                      "features": list(F.columns), "seconds": round(mon.elapsed, 1),
                      "peak_pss_gb": round(mon.peak_gb, 1)})
        rows += len(part)
        done = time.time() - t0
        eta = done / n * (len(todo) - n)
        print(f"  shard {i:02d} [{n}/{len(todo)}] {len(part):,} pairs in {mon.elapsed:.0f}s "
              f"(peak {mon.peak_gb:.1f} GB) — ETA {eta / 60:.1f} min", flush=True)
    return {"root": root, "pairs": int(len(cand)), "built_rows": rows, "shards": shards,
            "seconds": round(time.time() - t0, 1), "peak_gb": round(peak, 1), "config_hash": chash}


def load_features(cfg: dict, scope: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Concatenate every finished shard, in shard order."""
    root = features_dir(cfg, scope)
    parts = sorted(root.glob("shard=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(f"no feature shards under {root}")
    missing = [p.parent.name for p in parts if not (p.parent / "_SUCCESS").exists()]
    if missing:
        raise FileNotFoundError(f"incomplete shards: {missing}")
    return pd.concat([pd.read_parquet(p, columns=columns) for p in parts], ignore_index=True)


def verify(cfg: dict, scope: str, shards: int, split: str, shard: int) -> None:
    """Rebuild one shard in memory and compare it to the cached copy."""
    cand = load_candidates(cfg, scope)
    part = cand[shard_of(cand["s1_id"].to_numpy(), shards) == shard].reset_index(drop=True)
    cached = pd.read_parquet(features_dir(cfg, scope) / f"shard={shard:02d}" / "part.parquet")
    from ber.match.build import build_features  # the unsharded reference implementation
    fresh = build_features(cfg, part, split)
    assert (cached["m_id"].to_numpy() == part["m_id"].to_numpy()).all(), "row order differs"
    bad = []
    for c in FEATURES:
        if not np.allclose(cached[c].to_numpy(np.float64), fresh[c].to_numpy(np.float64),
                           rtol=0, atol=0, equal_nan=True):
            bad.append(c)
    if bad:
        raise AssertionError(f"shard {shard}: {len(bad)} feature(s) differ: {bad[:5]}")
    print(f"verify: shard {shard:02d} ({len(part):,} pairs) reproduces bit-for-bit across "
          f"all {len(FEATURES)} features")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2.yaml")
    ap.add_argument("--scope", required=True, help="dev / test (names cache/cand/<scope>.parquet)")
    ap.add_argument("--split", default=None, help="train / test (defaults: dev->train, test->test)")
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--only", type=int, nargs="*", help="build only these shard ids")
    ap.add_argument("--verify", type=int, nargs="?", const=0, default=None,
                    help="rebuild this shard and compare to the cache")
    ap.add_argument("--feature-version", default=None, help="override versions.feature_version")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.feature_version:
        cfg["versions"]["feature_version"] = a.feature_version
    compute.limit_threads()
    shards = a.shards or int(cfg["compute"].get("shards", 64))
    split = a.split or ("test" if a.scope == "test" else "train")

    if a.verify is not None:
        verify(cfg, a.scope, shards, split, a.verify)
        return

    r = run(cfg, a.scope, shards, split, a.only)
    print(f"[features] {a.scope}: {r['pairs']:,} pairs, {r['seconds'] / 60:.1f} min, "
          f"peak {r['peak_gb']:.1f} GB -> {r['root']}")
    if not a.no_ledger:
        v = cfg["versions"]
        append_run(cfg["abs"]["runs_ledger"], f"p2_features_{a.scope}",
                   metrics={"stage": "e0_features", "config_hash": r["config_hash"],
                            "norm_version": v["norm_version"], "blocker_set": v["blocker_set"],
                            "K": v["K"], "feature_version": v["feature_version"],
                            f"n_pairs_{a.scope}": r["pairs"], "wall_s": r["seconds"],
                            "peak_ram_gb": r["peak_gb"], "device": "cpu+cuda"},
                   notes=f"sharded feature cache ({shards} shards) at {r['root']}")


if __name__ == "__main__":
    main()
