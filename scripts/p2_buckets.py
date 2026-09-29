"""Error-bucket table for a P2 model, using the P1 audit's bucket definitions.

    python scripts/p2_buckets.py --bundle e1b

Replays the shipped decision on the cached out-of-fold probabilities and reports the same
buckets as `reports/p1_audit.md` (§A3), so before/after is directly comparable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from ber import compute  # noqa: E402
from ber.decide.sets import decide_expected_f, decide_threshold, exclusive, to_sets  # noqa: E402
from ber.evaluate.metric import f05_entity  # noqa: E402
from ber.io_utils import load_config, read_ground_truth, read_source  # noqa: E402


def bucket(pred: set, true: set) -> str:
    if not true:
        return "singleton false positive" if pred else "correct"
    if not (pred & true):
        return "missed all"
    if pred == true:
        return "correct"
    fp, fn = pred - true, true - pred
    return "wrong entity (FP and FN)" if fp and fn else ("over-predicted" if fp else "under-predicted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2.yaml")
    ap.add_argument("--bundle", default="e1b")
    args = ap.parse_args()
    cfg = load_config(args.config)
    a = cfg["abs"]
    compute.limit_threads(8)  # be polite: a test-feature job may be running

    bundle = joblib.load(Path(a["models_dir"]) / f"{args.bundle}.joblib")
    oof = pd.read_parquet(Path(a["models_dir"]) / f"{args.bundle}_oof.parquet")
    winner, taus = bundle["winner"], bundle["fold_taus"]

    s1raw, _ = read_source(a["train.source1"])
    gt_all, _ = read_ground_truth(a["train.ground_truth"], s1_ids=s1raw["entity_id"])
    s1_country = dict(zip(s1raw["entity_id"], s1raw["country"]))
    dev = pd.read_parquet(a["runs_dir"] / "dev_slice.parquet")
    s1_ids = dev["s1_id"].tolist()
    gt = {s: gt_all[s] for s in s1_ids}

    sels = []
    for k in sorted(oof["fold"].unique()):
        ex = exclusive(oof[oof["fold"] == k])
        sels.append(decide_expected_f(ex) if winner.endswith("expected-F")
                    else decide_threshold(ex, taus[int(k)]))
    pred = to_sets(pd.concat(sels), s1_ids)

    ent = pd.DataFrame({"s1_id": s1_ids})
    ent["country"] = ent["s1_id"].map(s1_country)
    ent["n_true"] = [len(gt[s]) for s in s1_ids]
    ent["f"] = [f05_entity(pred[s], gt[s]) for s in s1_ids]
    ent["bucket"] = [bucket(pred[s], gt[s]) for s in s1_ids]
    ent["f_loss"] = 1 - ent["f"]

    eb = ent.groupby("bucket").agg(S1=("s1_id", "size"), loss=("f_loss", "sum")).reset_index()
    eb["share of S1"] = eb["S1"] / len(ent)
    eb["share of total F-loss"] = eb["loss"] / ent["f_loss"].sum()
    eb["macro pts lost"] = eb["loss"] / len(ent)
    eb = eb.drop(columns="loss").sort_values("S1", ascending=False)

    print(f"model {args.bundle}, decision {winner}, macro F0.5 {ent['f'].mean():.6f}")
    print(tabulate(eb, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    rows = []
    for c, g in ent.groupby("country"):
        rows.append({"country": c, "S1": len(g), "macro F0.5": g["f"].mean(),
                     "singleton mean": g.loc[g["n_true"] == 0, "f"].mean(),
                     "non-singleton mean": g.loc[g["n_true"] > 0, "f"].mean()})
    print("\n" + tabulate(pd.DataFrame(rows), headers="keys", tablefmt="github",
                          showindex=False, floatfmt=".4f"))
    out = Path(a["models_dir"]) / f"{args.bundle}_buckets.csv"
    eb.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
