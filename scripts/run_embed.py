"""Encode names / addresses with multilingual-E5 -> data/cache/emb/{split}_{source}_{field}.npy (+ _ids.npy)

    python scripts/run_embed.py --field addr --benchmark 100000   # throughput + projection only
    python scripts/run_embed.py --field addr                      # resumable: valid caches are skipped
    python scripts/run_embed.py --field name

Inputs (original script kept, no transliteration):
    name -> raw business_name, NFKC (the name incl. aka/DBA alias text)
    addr -> addr_norm (normalised address)
Order: train S2/S3 -> train S1 -> test S1/S2/S3.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ber.embed import MODEL_SUBDIR
from ber.embed.encoder import MAX_LEN, E5Encoder, benchmark, emb_valid, save_emb
from ber.io_utils import load_config, read_source
from ber.normalize.run import load_norm

ORDER = [("train", "source2"), ("train", "source3"), ("train", "source1"),
         ("test", "source1"), ("test", "source2"), ("test", "source3")]


def texts_for(cfg, split, src, field) -> tuple[np.ndarray, list[str]]:
    norm = load_norm(cfg, split, src, ["entity_id", "addr_norm"])
    ids = norm["entity_id"].to_numpy()
    if field == "addr":
        return ids, norm["addr_norm"].fillna("").tolist()
    raw, _ = read_source(cfg["abs"][f"{split}.{src}"])
    assert (raw["entity_id"].to_numpy() == ids).all()
    return ids, raw["business_name"].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--field", choices=["name", "addr"], required=True)
    ap.add_argument("--split", action="append", choices=["train", "test"])
    ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()
    cfg = load_config(args.config)
    enc = E5Encoder(cfg["abs"]["cache_dir"] / MODEL_SUBDIR, batch=args.batch)
    order = [(sp, s) for sp, s in ORDER if not args.split or sp in args.split]

    if args.benchmark:
        ids, texts = texts_for(cfg, "train", "source2", args.field)
        rate = benchmark(enc, texts[:args.benchmark], MAX_LEN[args.field])
        total = sum(len(load_norm(cfg, sp, s, ["entity_id"])) for sp, s in order)
        print(f"{args.field}: {rate:,.0f} rec/s on 1 GPU; {total:,} records -> {total / rate / 60:.1f} min; "
              f"peak VRAM {enc.peak_vram_gb():.1f} GB")
        return

    for split, src in order:
        ids, texts = texts_for(cfg, split, src, args.field)
        if emb_valid(cfg, split, src, args.field, ids):
            print(f"{split}/{src}/{args.field}: cached and valid, skipped", flush=True)
            continue
        t = time.time()
        E = enc.encode(texts, MAX_LEN[args.field])
        save_emb(cfg, split, src, args.field, E, ids)
        print(f"{split}/{src}/{args.field}: {len(E):,} in {time.time() - t:.0f}s ({len(E) / (time.time() - t):,.0f}/s), "
              f"peak VRAM {enc.peak_vram_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
