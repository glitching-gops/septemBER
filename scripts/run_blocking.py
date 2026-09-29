"""Staged, resumable blocking -> data/cache/cand/{scope}/<stage>_<partition>.parquet and {scope}.parquet

    python scripts/run_blocking.py --scope dev  --stage lexical   # CPU, B1-B4 per partition
    python scripts/run_blocking.py --scope dev  --stage dense     # GPU, B6/B7 per partition
    python scripts/run_blocking.py --scope dev  --stage union     # merge + RRF + cand_rank
    python scripts/run_blocking.py --scope test --stage all --blockers b2,b3

Every partition checkpoint is skipped when it already exists and its .done marker
matches (resumable). Partitions are (target source x exact country value).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber import compute
from ber.blocking.pipeline import BlockingConfig
from ber.blocking.staged import DenseConfig, dense_partition, fuse, lexical_partition, partition_slug
from ber.io_utils import load_config
from ber.normalize.run import load_norm

COLS = ["entity_id", "country", "name_core", "addr_core", "house_no", "addr_missing"]


def cand_path(cfg: dict, scope: str) -> Path:
    return cfg["abs"]["cache_dir"] / "cand" / f"{scope}.parquet"


def load_block_frame(cfg: dict, split: str, source: str) -> pd.DataFrame:
    df = load_norm(cfg, split, source, COLS)
    keys = pd.read_parquet(cfg["abs"]["cache_dir"] / "norm" / f"{split}_{source}_keys.parquet", columns=["entity_id", "b1_text"])
    assert (keys["entity_id"].to_numpy() == df["entity_id"].to_numpy()).all()
    df["b1_text"] = keys["b1_text"].to_numpy()
    df["emb_row"] = np.arange(len(df))
    return df


def _done(path: Path) -> Path:
    return path.with_suffix(".done.json")


def is_done(path: Path, key: dict) -> bool:
    """A stage checkpoint is reusable when its file and marker exist and agree on config and row count."""
    d = _done(path)
    if not (path.exists() and d.exists()):
        return False
    try:
        meta = json.loads(d.read_text())
        n = pd.read_parquet(path, columns=["s1_id"]).shape[0]
    except Exception:
        return False
    return meta.get("key") == key and meta.get("rows") == n


def mark_done(path: Path, key: dict, rows: int, extra: dict) -> None:
    _done(path).write_text(json.dumps({"key": key, "rows": rows, **extra}, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scope", choices=["dev", "train", "test"], required=True)
    ap.add_argument("--stage", choices=["lexical", "dense", "union", "all"], default="all")
    ap.add_argument("--blockers", default="b1,b2,b3,b4", help="lexical blockers to run (test-time pruning)")
    ap.add_argument("--k1", type=int, default=30)
    ap.add_argument("--k2", type=int, default=30)
    ap.add_argument("--k3", type=int, default=20)
    ap.add_argument("--k6", type=int, default=20)
    ap.add_argument("--k7", type=int, default=5)
    ap.add_argument("--key-cap", type=int, default=500)
    ap.add_argument("--k-max", type=int, default=0)
    ap.add_argument("--only-partition", default=None, help="run a single partition slug (debug/timing)")
    ap.add_argument("--union-stages", default="lexical,dense", help="stages merged by the union (e.g. 'dense')")
    ap.add_argument("--rank-cap", type=json.loads, default={},
                    help='keep only ranks <= cap, e.g. \'{"b6n_rank": 20, "b6a_rank": 20, "b7_rank": 5}\' '
                         '(top-k is nested, so this reproduces a smaller k exactly)')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    compute.limit_threads()
    split = "test" if args.scope == "test" else "train"
    pdir = cfg["abs"]["cache_dir"] / "cand" / args.scope
    pdir.mkdir(parents=True, exist_ok=True)

    s1_all = load_block_frame(cfg, split, "source1")
    s1 = s1_all
    if args.scope == "dev":
        dev = set(pd.read_parquet(cfg["abs"]["runs_dir"] / "dev_slice.parquet", columns=["s1_id"])["s1_id"])
        s1 = s1_all[s1_all["entity_id"].isin(dev)]
    targets = {"S2": load_block_frame(cfg, split, "source2"), "S3": load_block_frame(cfg, split, "source3")}
    bc = BlockingConfig(k1=args.k1, k2=args.k2, k3=args.k3, key_cap=args.key_cap, use_b5=False)
    dc = DenseConfig(k6=args.k6, k7=args.k7)
    lex_blockers = [b for b in args.blockers.split(",") if b]
    stages = ["lexical", "dense", "union"] if args.stage == "all" else [args.stage]
    parts = [(src, c) for src in targets for c in sorted(s1["country"].unique())]
    print(f"scope={args.scope} queries={len(s1):,} partitions={[partition_slug(*p) for p in parts]} stages={stages}", flush=True)

    emb = None
    for stage in [s for s in stages if s != "union"]:
        for src, country in parts:
            slug = partition_slug(src, country)
            if args.only_partition and slug != args.only_partition:
                continue
            q = s1[s1["country"] == country].reset_index(drop=True)
            t = targets[src][targets[src]["country"] == country].reset_index(drop=True)
            path = pdir / f"{stage}_{slug}.parquet"
            if stage == "lexical":
                key = {"blockers": lex_blockers, "k": [bc.k1, bc.k2, bc.k3], "key_cap": bc.key_cap, "n_q": len(q), "n_t": len(t)}
            else:
                key = {"k6": dc.k6, "k7": dc.k7, "n_q": len(q), "n_t": len(t)}
            if is_done(path, key):
                print(f"[{stage} {slug}] checkpoint valid, skipped", flush=True)
                continue
            if len(q) == 0 or len(t) == 0:
                continue
            t0 = time.time()
            with compute.RamMonitor(f"{stage}[{slug}]") as mon:
                if stage == "lexical":
                    out, timings = lexical_partition(q, t, bc, lex_blockers)
                else:
                    if emb is None:
                        from ber.embed.encoder import load_emb

                        emb = {k: {f: load_emb(cfg, split, s, f) for f in ("name", "addr")}
                               for k, s in (("S1", "source1"), ("S2", "source2"), ("S3", "source3"))}
                    a = s1_all[s1_all["country"] == country].reset_index(drop=True)
                    out, timings = dense_partition(q, t, a, {"S1": emb["S1"], "T": emb[src]}, dc)
                mon.check()
            out["s1_id"] = q["entity_id"].to_numpy()[out["qi"].to_numpy()]
            out["m_id"] = t["entity_id"].to_numpy()[out["ti"].to_numpy()]
            out = out.drop(columns=["qi", "ti"])
            tmp = path.with_suffix(".tmp.parquet")
            out.to_parquet(tmp, index=False)
            tmp.replace(path)
            mark_done(path, key, len(out), {"seconds": round(time.time() - t0, 1), "timings": timings,
                                            "peak_pss_gb": round(mon.peak_gb, 1)})
            print(f"[{stage} {slug}] q={len(q):,} t={len(t):,} pairs={len(out):,} {timings} "
                  f"{time.time() - t0:.0f}s peak {mon.peak_gb:.1f} GB", flush=True)

    if "union" in stages:
        t0 = time.time()
        frames, used = [], {}
        for src, country in parts:
            slug = partition_slug(src, country)
            got, used[slug] = [], []
            for stage in args.union_stages.split(","):
                p = pdir / f"{stage}_{slug}.parquet"
                if p.exists() and _done(p).exists():
                    g = pd.read_parquet(p)
                    for col, cap in args.rank_cap.items():
                        if col in g:
                            v = g[col].fillna(0)
                            g.loc[v > cap, [col, col.replace("_rank", "_score")]] = np.nan
                    rank_cols = [c for c in g.columns if c.endswith("_rank")]
                    flag_cols = [c for c in ("b4_hn", "b4_name") if c in g]
                    alive = (g[rank_cols].fillna(0) > 0).any(axis=1)
                    for c in flag_cols:
                        alive |= g[c].fillna(False).astype(bool)
                    got.append(g[alive])
                    used[slug].append(stage)
            if not got:
                continue
            m = got[0]
            for g in got[1:]:
                m = m.merge(g, on=["s1_id", "m_id"], how="outer")
            m["source"], m["country"] = src, country
            frames.append(m)
        cands = fuse(pd.concat(frames, ignore_index=True), args.k_max)
        out = Path(args.out) if args.out else cand_path(cfg, args.scope)
        cands.to_parquet(out, index=False)
        meta = {"scope": args.scope, "n_queries": len(s1), "n_pairs": len(cands), "stages_per_partition": used,
                "union_stages": args.union_stages, "rank_cap": args.rank_cap,
                "k_max": args.k_max, "lexical": {"k": [bc.k1, bc.k2, bc.k3], "blockers": lex_blockers},
                "dense": dc.__dict__, "union_seconds": round(time.time() - t0, 1),
                "partition_meta": {p.name: json.loads(p.read_text()) for p in sorted(pdir.glob("*.done.json"))}}
        Path(str(out).replace(".parquet", ".meta.json")).write_text(json.dumps(meta, indent=1, default=str))
        print(f"union: {len(cands):,} pairs for {cands['s1_id'].nunique():,} S1 -> {out}; stages used {used}", flush=True)


if __name__ == "__main__":
    main()
