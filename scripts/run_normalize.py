"""Normalise source files -> data/cache/norm/{split}_{source}.parquet.

    python scripts/run_normalize.py                      # all 6 source files
    python scripts/run_normalize.py --split train --source source1 --limit 1000000
"""
from __future__ import annotations

import argparse

from ber import compute
from ber.io_utils import load_config, read_source
from ber.normalize.run import normalize_and_cache, normalize_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["train", "test"], action="append")
    ap.add_argument("--source", choices=["source1", "source2", "source3"], action="append")
    ap.add_argument("--limit", type=int, default=0, help="benchmark on the first N rows; nothing is cached")
    args = ap.parse_args()
    cfg = load_config(args.config)
    compute.limit_threads()
    splits = args.split or ["train", "test"]
    sources = args.source or ["source1", "source2", "source3"]
    total_n, total_t = 0, 0.0
    with compute.RamMonitor("normalize") as mon:
        for split in splits:
            for src in sources:
                df, _ = read_source(cfg["abs"][f"{split}.{src}"])
                if args.limit:
                    import time

                    df = df.head(args.limit)
                    t = time.time()
                    out = normalize_frame(df)
                    dt = time.time() - t
                    print(f"{split}/{src}: {len(df):,} rows in {dt:.1f}s -> {len(df) / dt:,.0f} rec/s (n_jobs={compute.n_jobs()})")
                    print(out.sample(5, random_state=0).T.to_string()[:3000])
                    continue
                path, dt = normalize_and_cache(cfg, split, src, df)
                total_n += len(df)
                total_t += dt
                print(f"{split}/{src}: {len(df):,} rows in {dt:.1f}s ({len(df) / dt:,.0f} rec/s) -> {path}")
                mon.check()
    if total_n:
        print(f"TOTAL {total_n:,} rows in {total_t:.1f}s ({total_n / total_t:,.0f} rec/s), peak PSS {mon.peak_gb:.1f} GB")


if __name__ == "__main__":
    main()
