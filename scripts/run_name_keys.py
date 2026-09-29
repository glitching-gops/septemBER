"""Learn the transliteration map(s) and write name keys for blocking.

    python scripts/run_name_keys.py

1. Learn TranslitMap from TRUE train pairs whose matched name is non-Latin:
     nondev   : pairs of train S1 entities NOT in the dev slice (used for train files,
                so dev-slice evaluation stays leak-free)
     alltrain : all train pairs (used for the test files)
2. Measure the gain on dev-slice non-Latin true pairs (share with >= 1 shared name
   token before vs after mapping).
3. Build a unigram Segmenter from S1 name tokens (train + test, label-free).
4. Write data/cache/norm/{split}_{source}_keys.parquet with name_key and b1_text.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from ber import compute
from ber.io_utils import load_config, read_ground_truth, read_source
from ber.normalize.namekeys import Segmenter, TranslitMap, block_text, name_key
from ber.normalize.run import load_norm

_TMAP = None
_SEG = None


def _keys_chunk(rows):
    return [(k := name_key(c, s, d, _TMAP, _SEG), block_text(k)) for c, s, d in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--min-precision", type=float, default=0.8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    a = cfg["abs"]
    out_dir = a["cache_dir"] / "translit"
    out_dir.mkdir(parents=True, exist_ok=True)

    s1raw, _ = read_source(a["train.source1"])
    gt, _ = read_ground_truth(a["train.ground_truth"], s1_ids=s1raw["entity_id"])
    dev = set(pd.read_parquet(a["runs_dir"] / "dev_slice.parquet", columns=["s1_id"])["s1_id"])
    s1n = load_norm(cfg, "train", "source1", ["entity_id", "name_norm"]).set_index("entity_id")["name_norm"]
    mn = pd.concat([load_norm(cfg, "train", s, ["entity_id", "name_norm", "name_script"]) for s in ("source2", "source3")])
    mn = mn[~mn["name_script"].isin(["latin", "none"])].set_index("entity_id")
    pairs = [(s, m) for s, ms in gt.items() for m in ms if m in mn.index]
    print(f"true pairs with a non-Latin matched name: {len(pairs):,}")
    exclude = set()  # legal atoms are handled by the legal lexicon already; keep them mappable anyway

    def learn(sel):
        return TranslitMap.learn(((s1n[s], mn.at[m, "name_norm"]) for s, m in sel),
                                 args.min_support, args.min_precision, exclude)

    nondev = learn([p for p in pairs if p[0] not in dev])
    alltrain = learn(pairs)
    nondev.save(out_dir / "nondev.json")
    alltrain.save(out_dir / "alltrain.json")
    print("nondev:", nondev.meta, "\nalltrain:", alltrain.meta)

    # gain on held-out dev pairs (token-overlap proxy)
    dev_pairs = [p for p in pairs if p[0] in dev]
    s1core = load_norm(cfg, "train", "source1", ["entity_id", "name_core"]).set_index("entity_id")["name_core"]
    mcore = pd.concat([load_norm(cfg, "train", s, ["entity_id", "name_core"]) for s in ("source2", "source3")]).set_index("entity_id")["name_core"]
    before = np.mean([bool(set(s1core[s].split()) & set(mcore[m].split())) for s, m in dev_pairs])
    after = np.mean([bool(set(s1core[s].split()) & set(nondev.apply(mcore[m]).split())) for s, m in dev_pairs])
    top = sorted(nondev.mapping.items(), key=lambda kv: -1)[:0]
    gain = {"dev_nonlatin_pairs": len(dev_pairs), "share_with_shared_name_token_before": round(float(before), 4),
            "share_with_shared_name_token_after": round(float(after), 4)}
    (out_dir / "gain.json").write_text(json.dumps(gain, indent=2))
    print("gain on dev non-Latin pairs:", gain)

    # segmenter vocabulary from S1 names (train + test), label-free
    counts: Counter = Counter()
    for split in ("train", "test"):
        for s in load_norm(cfg, split, "source1", ["name_core"])["name_core"]:
            counts.update(s.split())
    seg = Segmenter(counts)
    print(f"segmenter vocabulary: {len(seg.logp):,} words")

    global _TMAP, _SEG
    _SEG = seg
    for split, tmap in (("train", nondev), ("test", alltrain)):
        _TMAP = tmap
        for src in ("source1", "source2", "source3"):
            df = load_norm(cfg, split, src, ["entity_id", "name_core", "name_script", "is_domain"])
            rows = list(zip(df["name_core"], df["name_script"], df["is_domain"]))
            chunks = [rows[i:i + 100_000] for i in range(0, len(rows), 100_000)]
            with compute.pool() as p:
                res = [x for part in p.map(_keys_chunk, chunks) for x in part]
            out = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(),
                                "name_key": [r[0] for r in res], "b1_text": [r[1] for r in res]})
            path = a["cache_dir"] / "norm" / f"{split}_{src}_keys.parquet"
            out.to_parquet(path, index=False)
            changed = (out["name_key"].to_numpy() != df["name_core"].to_numpy()).mean()
            print(f"{split}/{src}: {len(out):,} keys, {changed:.2%} differ from name_core -> {path}")


if __name__ == "__main__":
    main()
