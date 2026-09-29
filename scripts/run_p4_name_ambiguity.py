"""R2 plus label-free name ambiguity features; GPU training and counting."""
import json
import os
import pickle
import sys
from datetime import datetime, time
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import pandas as pd
import torch
import run_p4_stage as stage
from ber import compute
from ber.embed import gpu_setup
from ber.gpu import pipeline as pl
from ber.io import gpu_load as gl, ids as I
from ber.io_utils import load_config
from ber.normalize.run import load_norm
from ber.verify import set_deterministic

TAG = "s1_rr_name_ambiguity"


def freeze_check():
    if datetime.now(ZoneInfo("Asia/Kolkata")).time() >= time(22, 30):
        raise RuntimeError("22:30 IST experiment freeze reached.")


def name_frequencies(cfg, D, timer):
    # CPU phases only parse files and encode text keys/record IDs.
    print("A0 | encoding cached names | estimated 2–8 minutes | "
          "next: GPU counts", flush=True)
    fields = ("name_norm", "name_core")
    with timer.host("io"):
        first = load_norm(
            cfg, "train", "source1",
            columns=["entity_id", "country", *fields],
        )

    vocabs, counts = {}, {}
    for field in fields:
        with timer.host("tok"):
            keys = pd.MultiIndex.from_frame(first[["country", field]])
            codes, vocab = pd.factorize(keys, sort=False)
        code_gpu = gl.to_dev(codes, stage.DEV).long() + 1
        vocabs[field] = vocab
        counts[field] = torch.bincount(
            code_gpu, minlength=len(vocab) + 1
        ).float()
        counts[field][0] = 0
        del code_gpu, codes, keys

    frequencies = {
        field: torch.full(
            (D.n_ent,), float("nan"),
            dtype=torch.float32, device=stage.DEV,
        ) for field in fields
    }
    for source in ("source1", "source2", "source3"):
        if source == "source1":
            frame = first
        else:
            with timer.host("io"):
                frame = load_norm(
                    cfg, "train", source,
                    columns=["entity_id", "country", *fields],
                )
        with timer.host("io"):
            entity_indices = I.encode(cfg, "train", frame["entity_id"])
        entity_gpu = gl.to_dev(entity_indices, stage.DEV).long()
        for field in fields:
            with timer.host("tok"):
                keys = pd.MultiIndex.from_frame(frame[["country", field]])
                codes = vocabs[field].get_indexer(keys)
                empty = frame[field].fillna("").str.strip().eq("").to_numpy()
            code_gpu = gl.to_dev(codes, stage.DEV).long() + 1
            values = counts[field][code_gpu]
            values[gl.to_dev(empty, stage.DEV).bool()] = float("nan")
            frequencies[field][entity_gpu] = torch.log1p(values)
            del code_gpu, values, codes, keys, empty
        del frame, entity_gpu, entity_indices

    columns, names = [], []
    for field in fields:
        for side, index in (("a", D.s1), ("b", D.m)):
            columns.append(frequencies[field][index])
            names.append(f"ambiguity_{side}_{field}_logcount")
    return torch.stack(columns, 1), names


def main():
    freeze_check()
    cfg = load_config("configs/p4_noaug.yaml")
    compute.limit_threads()
    stage.DEV = gpu_setup()
    assert stage.DEV.type == "cuda", "No CPU fallback."
    set_deterministic(cfg["seed"])
    md = stage.p4_dir(cfg)
    if (md / f"{TAG}.pkl").exists():
        raise RuntimeError("Experiment exists; refusing to overwrite.")

    with open(md / "s1_rr.pkl", "rb") as f:
        baseline = pickle.load(f)
    stage.EXTRA.update(
        on=True, name="rr",
        no_rescue=baseline["verifier"].get("no_rescue", False),
    )
    original_load = stage.load_dev

    def augmented_load(cfg, timer):
        D, X, names = original_load(cfg, timer)
        A, added = name_frequencies(cfg, D, timer)
        missing = (X[:, names.index("addr_missing")] > 0.5).float()
        Z = torch.cat([A, A * missing[:, None]], 1)
        added += [n + "_missing" for n in added]
        assert Z.device.type == "cuda"
        print(f"A1 | features ready | {Z.shape[1]} added features | "
              "next: grouped GPU training", flush=True)
        return D, torch.cat([X, Z], 1), names + added

    stage.load_dev = augmented_load
    original_fit = pl.fit_calibrated

    def guarded_fit(*args, **kwargs):
        freeze_check()
        return original_fit(*args, **kwargs)

    pl.fit_calibrated = guarded_fit
    args = SimpleNamespace(
        tag=TAG, baseline="s1_rr", no_loco=False,
        verifier=baseline["verifier"]["key"],
    )
    stage.cmd_stage1(args, cfg)

    result_path = md / f"{TAG}_result.json"
    result = json.loads(result_path.read_text())
    reference = json.loads((md / "s1_rr_result.json").read_text())
    loco_delta = {
        k: 100 * (v - reference["loco"][k])
        for k, v in result["loco"].items()
    }
    checks = {
        "cv": result["delta_cv_vs_stage1_pp"] >= 0.10,
        "holdout": result["delta_holdout_vs_stage1_pp"] >= 0.10,
        "positive_ci": result["ci_vs_stage1_pp"][0] > 0,
        "loco": bool(loco_delta)
                and all(v >= -0.05 for v in loco_delta.values()),
        "holdout_gap": abs(result["holdout_minus_cv_pp"]) <= 0.5,
        "before_freeze": datetime.now(
            ZoneInfo("Asia/Kolkata")).time() < time(22, 30),
    }
    result.update(
        acceptance_checks=checks,
        gate_pass=all(checks.values()),
        loco_delta_vs_r2_pp=loco_delta,
        feature_scope="Full S1 name counts; no label-based features",
    )
    result_path.write_text(json.dumps(result, indent=2))
    print("A2 | " + ("PASS" if result["gate_pass"] else "FAIL")
          + f" | delta CV {result['delta_cv_vs_stage1_pp']:+.3f} pp, "
          + f"holdout {result['delta_holdout_vs_stage1_pp']:+.3f} pp"
          + " | next: review gate results", flush=True)
    print(json.dumps({
        "checks": checks,
        "ci_pp": result["ci_vs_stage1_pp"],
        "loco_delta_pp": loco_delta,
    }, indent=2), flush=True)
    print("REPORT:", result_path, flush=True)


if __name__ == "__main__":
    main()
