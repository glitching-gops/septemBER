"""Build component-grouped stratified CV folds and leave-one-country-out splits.

Writes runs/splits.parquet (s1_id, component_id, fold, country_raw, is_singleton)
and runs/loco.parquet (s1_id, country_raw, one role column per country split).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from tabulate import tabulate

from ber.evaluate.splits import (
    build_components,
    check_no_positive_leak,
    component_size_table,
    make_cv_folds,
    make_loco,
)
from ber.io_utils import load_config, read_ground_truth, read_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    a = cfg["abs"]

    t0 = time.time()
    s1, _ = read_source(a["train.source1"])
    gt, _ = read_ground_truth(a["train.ground_truth"], s1_ids=s1["entity_id"])
    s1_ids = s1["entity_id"].tolist()
    print(f"loaded {len(s1_ids):,} train S1 entities in {time.time() - t0:.1f}s")

    comp = build_components(s1_ids, gt)
    sizes = component_size_table(comp)
    print("\nComponent sizes (S1 entities per component):")
    print(tabulate(sizes.reset_index(), headers="keys", tablefmt="github", showindex=False))
    if sizes.index.max() == 1:
        print("-> every component has size 1: no S2/S3 id is matched to more than one S1 entity.")

    singleton = np.array([len(gt[s]) == 0 for s in s1_ids])
    strata = s1["country"].astype(str).to_numpy() + "|" + np.where(singleton, "singleton", "matched")

    t0 = time.time()
    folds = make_cv_folds(comp, strata, cfg["n_folds"], cfg["seed"])
    print(f"\nStratifiedGroupKFold done in {time.time() - t0:.1f}s")
    leaks = check_no_positive_leak(folds, s1_ids, gt)
    assert leaks == 0, f"{leaks} S2/S3 ids have positives in more than one fold"

    splits = pd.DataFrame(
        {
            "s1_id": s1_ids,
            "component_id": comp.astype(np.int64),
            "fold": folds.astype(np.int16),
            "country_raw": s1["country"].to_numpy(),
            "is_singleton": singleton,
        }
    )
    a["splits"].parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(a["splits"], index=False)

    loco = make_loco(splits["country_raw"])
    loco.insert(0, "country_raw", splits["country_raw"].to_numpy())
    loco.insert(0, "s1_id", splits["s1_id"].to_numpy())
    loco.to_parquet(a["loco"], index=False)

    summary = splits.groupby("fold").agg(n=("s1_id", "size"), singleton_frac=("is_singleton", "mean"))
    mix = pd.crosstab(splits["fold"], splits["country_raw"], normalize="index")
    mix.columns = [f"share:{c}" for c in mix.columns]
    summary = summary.join(mix)
    print("\nCV fold summary:")
    summary = summary.reset_index()
    fmt = ["g", "g"] + [".4f"] * (summary.shape[1] - 2)
    print(tabulate(summary, headers="keys", tablefmt="github", showindex=False, floatfmt=fmt))

    rows = []
    for col in loco.columns[2:]:
        v = loco[col] == "valid"
        rows.append([col, int((~v).sum()), int(v.sum()), float(splits.loc[v.to_numpy(), "is_singleton"].mean())])
    print("\nLOCO splits:")
    print(tabulate(rows, headers=["split", "n_train", "n_valid", "valid_singleton_frac"], tablefmt="github", floatfmt=".4f"))
    print(f"\nwrote {a['splits']} and {a['loco']}; cross-fold positive leaks: {leaks}")


if __name__ == "__main__":
    main()
