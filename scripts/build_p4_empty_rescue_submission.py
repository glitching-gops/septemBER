"""Frozen empty-entity rescue on test; GPU scoring; LOCO does not block preparation."""
import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import run_p4_stage as stage
import run_p4_infer as inference
import run_p4_name_ambiguity as ambiguity
from ber import compute
from ber.embed import gpu_setup
from ber.evaluate.ledger import append_run
from ber.gpu import StageTimer, decide as gd, pipeline as pl, segment as sg
from ber.gpu.retrieval import topk
from ber.gpu.xgb import predict_mean
from ber.io import entity_attrs as EA, gpu_load as gl, ids as I, submission
from ber.io_utils import load_config, read_source
from ber.verify import set_deterministic

TAG = "empty_rescue_test"
SUB = "subD_empty_rescue_gpu"
K = 100

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["all", "prepare", "score", "finish"], default="all")
parser.add_argument("--fold", type=int, choices=[0, 1])
args = parser.parse_args()

cfg = load_config("configs/p4_noaug.yaml")
root = Path(cfg["abs"]["cache_dir"]).parent
pilot = root / TAG
out = root / "submissions" / SUB
baseline = root / "submissions/subC_ambiguity_blend50_gpu/test_scores.parquet"
bundle_path = stage.p4_dir(cfg) / "s1_rr_ambiguity_blend50.pkl"
prefilter_path = stage.p4_dir(cfg) / "empty_retrieval_prefilter_pilot.pkl"
pilot.mkdir(parents=True, exist_ok=True)


def freeze_check():
    pass  # User revoked the 22:30 cutoff.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_inputs():
    paths = {
        "baseline_scores": baseline,
        "matcher_bundle": bundle_path,
        "prefilter": prefilter_path,
    }
    for path in paths.values():
        assert path.exists(), f"Missing required input: {path}"
    manifest = {
        "hashes": {name: digest(path) for name, path in paths.items()},
        "versions": cfg["versions"],
        "seed": cfg["seed"],
        "k": K,
        "keep_fraction": .05,
    }
    path = pilot / "INPUT.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Inputs changed; refusing stale checkpoints."
    else:
        path.write_text(json.dumps(manifest, indent=2))
    return manifest


def segment(filename, start, end):
    text = (HERE / filename).read_text()
    assert text.count(start) == 1 and text.count(end) == 1
    return textwrap.dedent(text[text.index(start):text.index(end)])


# Reuse the measured GPU retrieval and prefilter kernels.
body = segment(
    "retrieve_p4_empty_rescue_cv.py",
    "            def embeddings(field, entities):",
    "            new_true = gm.labels(",
)
body = body.replace("if time.monotonic()-started > 1200:", "if False:")
prefix = '''def deeper(cfg, D, queries, table, timer):
    with timer.host("io"):
        source = gl.to_dev(table.src.to_numpy(), stage.DEV).long()
        country = gl.to_dev(table.country.to_numpy(), stage.DEV).long()
        attributes = EA.attrs(cfg, "test")
        address_ok = gl.to_dev(attributes["addr_ok"], stage.DEV).bool()
        reader = EA.EmbReader(cfg, "test", stage.DEV)
'''
exec(compile(
    prefix + textwrap.indent(body, "    ") + "    return s1, m\n",
    "<test_gpu_retrieval>", "exec",
))

body = segment(
    "prefilter_p4_empty_rescue_cv.py",
    "            def embeddings(field, entities):",
    "            params = {",
)
prefix = '''def prefeatures(cfg, s1, m, table, timer):
    dev, n_ent = stage.DEV, len(table)
    with timer.host("io"):
        source = gl.to_dev(table.src.to_numpy(), dev).long()
        attributes = EA.attrs(cfg, "test")
        reader = EA.EmbReader(cfg, "test", dev)
'''
exec(compile(
    prefix + textwrap.indent(body, "    ") + "    return X, names\n",
    "<test_gpu_prefilter>", "exec",
))


def numeric(path, columns, timer):
    with timer.host("io"):
        tab = pq.read_table(path, columns=columns)
        return {
            c: gl.to_dev(tab[c].to_numpy(zero_copy_only=False), stage.DEV)
            for c in columns
        }


def save_numeric(path, values, timer):
    with timer.host("io"):
        pq.write_table(
            pa.table({k: v.cpu().numpy() for k, v in values.items()}), path
        )


def load_bundle(timer):
    with timer.host("io"):
        with bundle_path.open("rb") as handle:
            bundle = pickle.load(handle)
    assert bundle["gate_pass"] and bundle["alpha"] == .5
    assert bundle["left"]["verifier"] == bundle["right"]["verifier"]
    return bundle


def prepare(timer):
    if (pilot / "PREPARED.json").exists():
        print("PREPARE | cached | verified inputs unchanged | next: scoring", flush=True)
        return
    bundle = load_bundle(timer)
    B = numeric(baseline, ["s1", "m", "p", "selected"], timer)
    B["s1"], B["m"] = B["s1"].long(), B["m"].long()
    selected = B["selected"].bool()
    with timer.host("io"):
        table = I.entity_table(cfg, "test")
        sources = gl.to_dev(table.src.to_numpy(), stage.DEV).long()
    n_ent = len(table)
    replay = gd.threshold(
        gd.exclusive(B["s1"], B["m"], B["p"], None, n_ent),
        B["p"], bundle["tau_median"],
    )
    assert bool((replay == selected).all()), "Saved baseline decision parity failed."
    counts = sg.seg_count(B["s1"][selected], n_ent)
    queries = torch.nonzero((sources == 1) & (counts == 0)).flatten()
    D = SimpleNamespace(s1=B["s1"], m=B["m"], n_ent=n_ent)
    print(
        f"T0 | empty test entities | {queries.numel():,} "
        "| next: deeper retrieval", flush=True,
    )
    ns, nm = deeper(cfg, D, queries, table, timer)
    assert ns.numel() > 0
    raw_count = int(ns.numel())
    del B, replay, selected, counts, sources, D
    torch.cuda.empty_cache()

    X, names = prefeatures(cfg, ns, nm, table, timer)
    with timer.host("io"):
        with prefilter_path.open("rb") as handle:
            frozen = pickle.load(handle)
    assert frozen["features"] == names
    assert frozen["keep_fraction"] == .05
    scores = torch.zeros(ns.numel(), device=stage.DEV)
    models = frozen["folds"]
    for k in sorted(models):
        scores += models[k].raw(X) / len(models)
    chosen = torch.topk(scores, max(1, math.ceil(.05 * ns.numel()))).indices
    ns, nm = ns[chosen], nm[chosen]
    save_numeric(pilot / "retained_pairs.parquet", {"s1": ns, "m": nm}, timer)
    with timer.host("io"):
        ids = table.entity_id.to_numpy()
        pairs = pd.DataFrame({
            "s1_id": ids[ns.cpu().numpy()],
            "m_id": ids[nm.cpu().numpy()],
        })
        pairs.to_parquet(pilot / f"test_{TAG}_set.parquet", index=False)
        I.encode_pairs_cached(cfg, "test", pilot / f"test_{TAG}_set.parquet", TAG + "_test_keys")
        (pilot / "PREPARED.json").write_text(json.dumps({
            "empty_queries": int(queries.numel()),
            "raw_new_pairs": raw_count,
            "retained_pairs": int(ns.numel()),
            "policy": "Frozen five-model mean; global top five percent",
            "test_labels_used": False,
        }, indent=2))
    print(
        f"T1 | prepared | {ns.numel():,} new pairs "
        "| next: both verifier folds", flush=True,
    )


def score(fold):
    assert fold in (0, 1)
    assert (pilot / "PREPARED.json").exists()
    from ber.verify import rerankers
    from run_p4_verifier import cmd_score
    text = textwrap.dedent(inspect.getsource(rerankers.Verifier.score))
    old = "1.0 / (1.0 + np.exp(-logit))"
    assert text.count(old) == 1
    text = text.replace(
        old, "torch.sigmoid(torch.as_tensor(logit, device=self.device)).cpu().numpy()"
    )
    namespace = dict(vars(rerankers), torch=torch)
    exec(compile(text, "<test_cuda_probability>", "exec"), namespace)
    rerankers.Verifier.score = namespace["score"]
    os.environ["P4_EXTRA_ROOT"] = str(pilot)
    os.environ["P4_SCORE_ROOT"] = str(pilot / "verifier")
    cmd_score(SimpleNamespace(
        fold=fold, part="test", extra=TAG,
        reverse=False, middle=False, det_batch=128,
    ), cfg)


def affected_features(query_mask, names, timer):
    basic = [n for n in names if n not in stage.PROV]
    paths = gl.feature_shards(cfg, "test")
    inputs = [
        (path, basic, f"test_shard{i:03d}", False)
        for i, path in enumerate(paths)
    ]
    inputs.append((root / "rr/test_feats.parquet", names, "rr_test_feats", True))
    all_s, all_m, all_x = [], [], []
    print(
        "T2 | reading affected cached feature rows | estimated 2–5 minutes "
        "| next: matcher features", flush=True,
    )
    for path, columns, tag, is_extra in inputs:
        with timer.host("io"):
            keys = I.encode_pairs_cached(cfg, "test", path, tag)
            s = gl.to_dev(keys["s1"], stage.DEV).long()
            m = gl.to_dev(keys["m"], stage.DEV).long()
        rows = torch.nonzero(query_mask[s]).flatten()
        if not rows.numel():
            continue
        with timer.host("io"):
            tab = pq.read_table(path, columns=columns)
            tab = tab.take(pa.array(rows.cpu().numpy()))
            values = [
                gl.to_dev(tab[c].to_numpy(zero_copy_only=False), stage.DEV, torch.float32)
                for c in columns
            ]
        X = torch.stack(values, 1)
        if not is_extra:
            prov = torch.full(
                (rows.numel(), len(stage.PROV)), float("nan"), device=stage.DEV,
            )
            prov[:, :2] = 0
            X = torch.cat([X, prov], 1)
        all_s.append(s[rows])
        all_m.append(m[rows])
        all_x.append(X)
        del tab, values, s, m, rows
    return torch.cat(all_s), torch.cat(all_m), torch.cat(all_x)


def finish(timer):
    assert not (out / "matching_results.tsv").exists(), "Output already exists; refusing overwrite."
    bundle = load_bundle(timer)
    left, right = bundle["left"], bundle["right"]
    names = [n for n in left["features"] if n not in pl.V_NAMES]
    B = numeric(baseline, ["s1", "m", "p", "selected"], timer)
    s1, m, p = B["s1"].long(), B["m"].long(), B["p"].float()
    baseline_selected = B["selected"].bool()
    R = numeric(pilot / "retained_pairs.parquet", ["s1", "m"], timer)
    ns, nm = R["s1"].long(), R["m"].long()
    with timer.host("io"):
        table = I.entity_table(cfg, "test")
    n_ent = len(table)
    assert not bool(sg.isin_pairs(ns, nm, s1, m, n_ent).any())
    queries = torch.unique(ns)
    qmask = torch.zeros(n_ent, dtype=torch.bool, device=stage.DEV)
    qmask[queries] = True
    old_rows = qmask[s1]
    indices = torch.nonzero(old_rows).flatten()
    os1, om, old_x = affected_features(qmask, names, timer)
    assert torch.equal(os1, s1[old_rows]) and torch.equal(om, m[old_rows])

    # Read the original verifier scores without changing their files.
    for key in ("P4_SCORE_ROOT", "P4_TEST_SET"):
        os.environ.pop(key, None)
    inference.DEV = stage.DEV
    old_v = inference.test_verifier(
        cfg, os1, om, left["verifier"], n_ent, timer,
        extra_sets=[root / "rr/test_rr_set.parquet"],
    )

    with timer.host("io"):
        ids = table.entity_id.to_numpy()
        cand = pd.DataFrame({
            "s1_id": ids[ns.cpu().numpy()],
            "m_id": ids[nm.cpu().numpy()],
        })
        cand["source"] = table.src.to_numpy()[nm.cpu().numpy()]
        cand["source"] = cand["source"].map({2: "S2", 3: "S3"})
        cand["country"] = table.country_label.to_numpy()[ns.cpu().numpy()]
    for col in ("b6n_rank", "b6a_rank", "b7_rank"):
        cand[col] = float("nan")
    fp = pilot / "matcher_features.parquet"
    if fp.exists():
        with timer.host("io"):
            features = pd.read_parquet(fp)
        assert features[["s1_id", "m_id"]].equals(cand[["s1_id", "m_id"]])
    else:
        print(
            "T3 | new pair features | estimated 5–10 minutes "
            "| next: GPU matcher", flush=True,
        )
        from eval_p4_empty_rescue_holdout import gpu_token_counts
        gpu_token_counts()
        from ber.match.build import FeatureContext
        c2 = {**cfg, "versions": {**cfg["versions"], "feature_version": "f2"}}
        with timer.host("tok"):
            context = FeatureContext(c2, "test", gpu_features=True)
            features = context.features(cand)
        del context
        features.insert(0, "m_id", cand.m_id.to_numpy())
        features.insert(0, "s1_id", cand.s1_id.to_numpy())
        with timer.host("io"):
            features.to_parquet(fp, index=False)

    basic = [n for n in names if n not in stage.PROV]
    with timer.host("io"):
        new_x = gl.to_dev(features[basic].to_numpy(dtype="float32"), stage.DEV)
    prov = torch.full(
        (ns.numel(), len(stage.PROV)), float("nan"), device=stage.DEV,
    )
    prov[:, :2] = 0
    new_x = torch.cat([new_x, prov], 1)
    del features, cand, prov, table, R

    sx, mx = torch.cat([os1, ns]), torch.cat([om, nm])
    base_x = torch.cat([old_x, new_x])
    text = inspect.getsource(ambiguity.name_frequencies)
    namespace = dict(vars(ambiguity))
    exec(compile(text.replace('"train"', '"test"'), "<test_name_counts>", "exec"), namespace)
    A, added = namespace["name_frequencies"](
        cfg, SimpleNamespace(s1=sx, m=mx, n_ent=n_ent), timer,
    )
    assert names + pl.V_NAMES == left["features"]
    assert names + added + [n + "_missing" for n in added] + pl.V_NAMES == right["features"]
    missing = (base_x[:, names.index("addr_missing")] > .5).float()
    nold = os1.numel()

    def predict(base, ambiguity_x, miss, verifier):
        result = torch.empty(base.shape[0], device=stage.DEV)
        for start in range(0, base.shape[0], 250000):
            end = min(start + 250000, base.shape[0])
            xl = torch.cat([base[start:end], verifier[start:end]], 1)
            xr = torch.cat([
                base[start:end], ambiguity_x[start:end],
                ambiguity_x[start:end] * miss[start:end, None], verifier[start:end],
            ], 1)
            result[start:end] = (
                .5 * predict_mean(list(left["folds"].values()), xl)
                + .5 * predict_mean(list(right["folds"].values()), xr)
            )
        return result

    replay = predict(old_x, A[:nold], missing[:nold], old_v)
    parity = float((replay - p[old_rows]).abs().max())
    assert parity <= 1e-6, f"Existing test prediction parity failed: {parity}"
    print(f"T4 | baseline prediction parity | max delta={parity} | next: rescue decision", flush=True)

    scores = []
    for j in (0, 1):
        paths = sorted(
            (pilot / "verifier" / f"fold{j}" / "det/test_extra").glob(f"part-{TAG}-*.parquet")
        )
        assert paths, f"Missing verifier fold {j}"
        parts = []
        for path in paths:
            with timer.host("io"):
                parts.append(gl.pair_file(
                    cfg, "test", path, f"{TAG}_f{j}_{path.stem}", ["logit"], stage.DEV,
                ))
        q = {k: torch.cat([v[k] for v in parts]) for k in ("s1", "m", "logit")}
        ix = sg.lookup_pairs(ns, nm, q["s1"], q["m"], n_ent)
        assert q["s1"].numel() == ns.numel() and bool((ix >= 0).all())
        scores.append(q["logit"][ix])
    l0, l1 = scores
    a0, b0 = left["verifier"]["platt"][0]
    a1, b1 = left["verifier"]["platt"][1]
    lg = ((l0 + l1) / 2).double()
    pc = torch.sigmoid(((a0*l0+b0) + (a1*l1+b1)) / 2).double()
    hit = old_v[:, 2] > .5
    V = pl.verifier_features(
        sx, mx,
        torch.cat([os1[hit], ns]), torch.cat([om[hit], nm]),
        torch.cat([old_v[hit, 0].double(), lg]),
        torch.cat([old_v[hit, 1].double(), pc]),
        torch.cat([old_v[hit, 5].double(), torch.zeros_like(pc)]),
        n_ent,
    )
    updated = predict(base_x, A, missing, V)
    p[indices] = updated[:nold]
    p = torch.cat([p, updated[nold:]])
    s1, m = torch.cat([s1, ns]), torch.cat([m, nm])
    selected = gd.threshold(
        gd.exclusive(s1, m, p, None, n_ent), p, bundle["tau_median"],
    )

    order = sg.order([(s1, False), (m, False)])
    selected_order = selected[order]
    with timer.host("io"):
        first, _ = read_source(cfg["abs"]["test.source1"])
        print("T5 | writing both TSVs | estimated 5–15 minutes | next: validators", flush=True)
        submission.write(
            cfg, out, list(first.entity_id),
            s1[order].cpu().numpy(), m[order].cpu().numpy(),
            s1[order][selected_order].cpu().numpy(),
            m[order][selected_order].cpu().numpy(),
        )
        from validate_outputs import validate
        validation = validate(
            out / "matching_results.tsv", out / "candidate_pairs.tsv", cfg, "test",
        )
    save_numeric(
        out / "test_scores.parquet",
        {"s1": s1, "m": m, "p": p, "selected": selected}, timer,
    )
    info = {
        "status": "validated_experimental",
        "loco_status": "pending; not required for file preparation",
        "submission_approved": False,
        "model": "frozen_ambiguity_blend50_plus_empty_rescue",
        "tau": bundle["tau_median"],
        "new_scored_pairs": int(ns.numel()),
        "new_selected_pairs": int(selected[baseline_selected.numel():].sum()),
        "old_matches_removed": int(
            (baseline_selected & ~selected[:baseline_selected.numel()]).sum()
        ),
        "total_matches": int(selected.sum()),
        "baseline_prediction_parity": parity,
        "test_labels_used": False,
        "validation": validation,
        "inputs": json.loads((pilot / "INPUT.json").read_text()),
        "timings": timer.summary(),
    }
    with timer.host("io"):
        (out / "INFER.json").write_text(json.dumps(info, indent=2, default=str))
        (out / "SUMMARY.md").write_text(
            "# Experimental empty-entity rescue\n\n"
            "Frozen accepted matcher, verifier, prefilter and threshold.\n"
            "CV and holdout passed; LOCO remains unresolved/pending.\n"
            "Both submission validators passed. No test labels used.\n"
        )
        linkdir = Path(cfg["abs"]["root"]) / "submissions" / SUB
        linkdir.mkdir(parents=True, exist_ok=True)
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            link = linkdir / name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(out / name)
    print(validation.get("official", ""), flush=True)
    print("READY:", out / "matching_results.tsv", flush=True)


def run_child(phase, fold=None, env=None):
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", phase]
    if fold is not None:
        command += ["--fold", str(fold)]
    subprocess.run(command, env=env, check=True)


def main():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    check_inputs()
    if args.phase == "all":
        run_child("prepare")
        run_child("score", 0)
        run_child("score", 1)
        run_child("finish")
        return

    compute.limit_threads()
    try:
        stage.DEV = gpu_setup()
    except Exception as exc:
        print(f"[GPU ALERT] Cannot start GPU job: {exc}", flush=True)
        raise
    assert stage.DEV.type == "cuda", "No CPU fallback."
    inference.DEV = stage.DEV
    set_deterministic(cfg["seed"])
    timer = StageTimer(TAG + "_" + args.phase)
    status = "running"
    try:
        with timer.gpu():
            if args.phase == "prepare":
                prepare(timer)
            elif args.phase == "score":
                score(args.fold)
            else:
                finish(timer)
        status = "complete"
    except torch.cuda.OutOfMemoryError:
        print("[GPU ALERT] GPU memory exhausted; stopped without CPU fallback.", flush=True)
        raise
    finally:
        st = timer.summary()
        append_run(
            cfg["abs"]["runs_ledger"], TAG + "_" + args.phase,
            {
                "stage": "test_empty_rescue_" + args.phase,
                "model": "frozen_ambiguity_blend50",
                "device": "cuda",
                "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                **{k: st[k] for k in (
                    "wall_s", "cpu_time_s", "gpu_time_s",
                    "peak_vram_gb", "cpu_violation",
                )},
            },
            notes=status + "; experimental; LOCO pending", extra=st,
        )


if __name__ == "__main__":
    main()
