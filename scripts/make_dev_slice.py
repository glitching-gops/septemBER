"""Build the 10% dev slice of train S1 entities for fast iteration.

Queries are the sampled S1 records; blocking still searches the FULL train S2/S3
pool of the same country, so distractor density (and hence precision) stays
realistic. Stratified by country x match-count bucket (0 = singleton, 1, 2, 3,
4+); each entity keeps its CV fold from runs/splits.parquet.

Caveat: exclusivity can only be enforced among the sampled S1 on the slice.

-> runs/dev_slice.parquet (s1_id, fold, country_raw, n_matches, bucket, stratum)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tabulate import tabulate

from ber.io_utils import load_config, read_ground_truth, read_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--frac", type=float, default=0.10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    a = cfg["abs"]

    s1, _ = read_source(a["train.source1"])
    gt, _ = read_ground_truth(a["train.ground_truth"], s1_ids=s1["entity_id"])
    splits = pd.read_parquet(a["splits"], columns=["s1_id", "fold"])
    df = pd.DataFrame({"s1_id": s1["entity_id"].to_numpy(), "country_raw": s1["country"].to_numpy()})
    df["n_matches"] = df["s1_id"].map(lambda k: len(gt[k])).astype(np.int16)
    df["bucket"] = df["n_matches"].clip(upper=4).astype(str).replace("4", "4+")
    df["stratum"] = df["country_raw"] + "|" + df["bucket"]
    df = df.merge(splits, on="s1_id", how="left", validate="one_to_one")
    assert df["fold"].notna().all(), "splits.parquet is stale; rerun make_splits.py"

    dev = (
        df.groupby("stratum", group_keys=False)
        .sample(frac=args.frac, random_state=cfg["seed"])
        .sort_values("s1_id")
        .reset_index(drop=True)
    )
    dev = dev[["s1_id", "fold", "country_raw", "n_matches", "bucket", "stratum"]]
    out = a["runs_dir"] / "dev_slice.parquet"
    dev.to_parquet(out, index=False)

    full = df["stratum"].value_counts(normalize=True).rename("full_share")
    sl = dev["stratum"].value_counts(normalize=True).rename("slice_share")
    comp = pd.concat([full, sl], axis=1).fillna(0)
    comp["diff_pp"] = (comp["slice_share"] - comp["full_share"]) * 100
    print(tabulate(comp.sort_index().reset_index(), headers="keys", tablefmt="github", floatfmt=".4f", showindex=False))
    print(f"\ndev slice: {len(dev):,} of {len(df):,} S1 ({len(dev) / len(df):.2%}); max |diff| {comp['diff_pp'].abs().max():.3f} pp")
    print("per fold:", dev["fold"].value_counts().sort_index().to_dict())
    assert comp["diff_pp"].abs().max() < 0.1
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
