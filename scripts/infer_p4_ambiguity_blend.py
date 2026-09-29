"""Batched GPU inference for the accepted R2/name-ambiguity blend."""
import hashlib
import inspect
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import torch
import run_p4_name_ambiguity as features
from ber.gpu import pipeline as pl
from ber.gpu.xgb import predict_mean
from ber.io_utils import load_config

TAG = "s1_rr_ambiguity_blend50"
SUB = "subC_ambiguity_blend50_gpu"
cfg = load_config("configs/p4_noaug.yaml")
md = Path(cfg["abs"]["models_dir"]) / "p4"
report = json.loads((md / f"{TAG}_result.json").read_text())
assert report["gate_pass"] and report["status"] == "accepted"
out_root = Path(cfg["abs"]["cache_dir"]).parent / "submissions"
out = out_root / SUB
assert not out.exists(), f"Refusing to overwrite {out}"

# Reuse the exact training feature algorithm on unlabelled test records.
feature_source = inspect.getsource(features.name_frequencies)
assert feature_source.count('"train"') >= 3
feature_ns = dict(features.__dict__)
exec(feature_source.replace('"train"', '"test"'), feature_ns)
test_frequencies = feature_ns["name_frequencies"]


def score_batches(cfg, L, V, n_ent, base_feats, blend, timer):
    features.stage.DEV = L["X"].device
    assert features.stage.DEV.type == "cuda"
    assert blend["gate_pass"] and blend["alpha"] == 0.5
    left, right = blend["left"], blend["right"]
    assert left["verifier"] == right["verifier"]
    assert base_feats + pl.V_NAMES == left["features"]

    torch.cuda.empty_cache()
    D = SimpleNamespace(s1=L["s1"], m=L["m"], n_ent=n_ent)
    A, names = test_frequencies(cfg, D, timer)
    expected = base_feats + names + [n + "_missing" for n in names] + pl.V_NAMES
    assert expected == right["features"], "Name-feature order differs."
    missing_index = base_feats.index("addr_missing")
    left_models = list(left["folds"].values())
    right_models = list(right["folds"].values())

    def predict(start, end):
        base = L["X"][start:end]
        verifier = V[start:end]
        ambiguity = A[start:end]
        missing = (base[:, missing_index] > 0.5).float()
        XL = torch.cat([base, verifier], 1)
        XR = torch.cat(
            [base, ambiguity, ambiguity * missing[:, None], verifier], 1
        )
        return (
            0.5 * predict_mean(left_models, XL)
            + 0.5 * predict_mean(right_models, XR)
        )

    probe = min(4096, L["s1"].numel())
    midpoint = probe // 2
    whole = predict(0, probe)
    split = torch.cat([predict(0, midpoint), predict(midpoint, probe)])
    delta = float((whole - split).abs().max())
    assert delta <= 1e-6, f"Batch parity failed: {delta}"
    print(f"[gpu] blend batch parity max delta={delta}", flush=True)

    p = torch.empty(
        L["s1"].numel(), dtype=torch.float32, device=features.stage.DEV
    )
    started = time.monotonic()
    for start in range(0, p.numel(), 250000):
        end = min(start + 250000, p.numel())
        p[start:end] = predict(start, end)
        if start == 0 or end % 5000000 == 0 or end == p.numel():
            elapsed = time.monotonic() - started
            eta = elapsed * (p.numel() - end) / end / 60
            print(f"[gpu] {end:,}/{p.numel():,} pairs; ETA {eta:.1f} min",
                  flush=True)
    return p


# Preserve the canonical decision, output writer and both validators.
original = HERE / "run_p4_infer.py"
source = original.read_text()

def replace_once(old, new):
    global source
    assert source.count(old) == 1, f"Inference source differs: {old[:70]}"
    source = source.replace(old, new)

replace_once(
    '''    with timer.host("io"), open(md_ / f"{args.stage1}.pkl", "rb") as fh:
        b1 = pickle.load(fh)''',
    '''    with timer.host("io"), open(md_ / "s1_rr_ambiguity_blend50.pkl", "rb") as fh:
        blend_bundle = pickle.load(fh)
        assert blend_bundle["gate_pass"]
        b1 = blend_bundle["left"]'''
)
replace_once(
    '''        X1 = torch.cat([L["X"], V], 1)
        del L
        p = predict_mean(list(b1["folds"].values()), X1)
        del X1''',
    '''        p = score_batches(cfg, L, V, n_ent, base_feats, blend_bundle, timer)
        del L
        torch.cuda.empty_cache()'''
)
replace_once(
    '        bundle = b1',
    '''        bundle = {
            **b1,
            "tau_median": blend_bundle["tau_median"],
            "cv": {"macro_threshold": blend_bundle["evaluation"]["cv"]}
        }'''
)
anchor = '            L["X"] = torch.cat([L["X"], pk], 1)'
replace_once(anchor, anchor + "\n            del pk")
anchor = '            extra_sets.append(ep.with_name(f"test_{args.extra}_set.parquet"))'
replace_once(anchor, anchor + "\n            del E, ents")
replace_once(
    '"model": args.stage2 or args.stage1',
    '"model": "s1_rr_ambiguity_blend50"'
)

ns = {
    "__name__": "accepted_blend_inference",
    "__file__": str(original),
    "score_batches": score_batches,
}
exec(compile(source, str(original), "exec"), ns)
sys.argv = [
    str(original), "--config", "configs/p4_noaug.yaml",
    "--stage1", "s1_rr", "--extra", "rr", "--sub", SUB,
    "--out-root", str(out_root),
    "--link-dir", str(cfg["abs"]["root"] / "submissions"),
]
ns["main"]()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

metadata = {
    "model": TAG,
    "evaluation": report,
    "blend": "0.5 * calibrated R2 + 0.5 * calibrated name-ambiguity",
    "threshold": report["tau_median"],
    "feature_scope": "Name counts from the full unlabelled test S1 pool",
    "bundle_sha256": sha256(md / f"{TAG}.pkl"),
    "matching_sha256": sha256(out / "matching_results.tsv"),
    "runner_sha256": sha256(Path(__file__)),
}
(out / "BLEND.json").write_text(json.dumps(metadata, indent=2))
info_path = out / "INFER.json"
info = json.loads(info_path.read_text())
info["blend"] = metadata
info_path.write_text(json.dumps(info, indent=2))
(out / "SUMMARY.md").write_text(
    f"# {SUB}\n\n"
    f"Fixed 50/50 blend of R2 and the name-ambiguity matcher.\n\n"
    f"CV: {report['cv']:.6f}; holdout: {report['holdout']:.6f}; "
    f"global threshold: {report['tau_median']:.3f}.\n\n"
    "All acceptance gates passed. Numerical inference ran on GPU. "
    "Candidate pairs are the exact R2 scored set. "
    "See INFER.json for validation and BLEND.json for model details.\n"
)
print("READY:", out / "matching_results.tsv", flush=True)
