"""C2/C3: fine-tune a verifier on one verifier fold, then score out-of-fold, holdout and test.

    python scripts/run_verifier_ft.py test-band                       # E1b p on every test pair -> band list
    python scripts/run_verifier_ft.py pilot --model qwen3-rr-0.6b      # 200k-pair gate on the screen sample
    python scripts/run_verifier_ft.py fold --model qwen3-rr-0.6b --fold 0

Fold j trains on non-holdout dev S1 with ``verifier_fold == j`` and scores (a) the band
pairs of the OTHER verifier fold (out-of-fold), (b) the holdout band pairs and (c) the test
band pairs. Holdout and test features are later the mean of the two fold models, OOF
features come from the single model that never saw the S1 - see ber.verify.folds.

Test scoring is sharded every 250k pairs and resumes from finished shards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ber import compute  # noqa: E402
from ber.evaluate.ledger import append_run, config_hash  # noqa: E402
from ber.io_utils import load_config, read_ground_truth, read_source  # noqa: E402
from ber.verify import folds as vf  # noqa: E402
from ber.verify.rerankers import MODELS, Verifier  # noqa: E402
from ber.verify.serialize import load_records, pair_texts  # noqa: E402
from ber.verify.train import build_training_pairs, train  # noqa: E402
from run_features import features_dir  # noqa: E402

SHARD = 250_000
RECIPE = {  # D5
    "qwen_yesno": dict(lora=True, lr=1e-4, eff_batch=256),
    "cross": dict(lora=False, lr=2e-5, eff_batch=512),
    "nli": dict(lora=True, lr=1e-4, eff_batch=256),
}


def vroot(cfg) -> Path:
    return Path(cfg["abs"]["cache_dir"]).parent / "verifier"


def dev_frame(cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    a, vc = cfg["abs"], cfg["verifier"]
    dev = pd.read_parquet(a["runs_dir"] / "dev_slice.parquet")
    comp = pd.read_parquet(a["runs_dir"] / "splits.parquet", columns=["s1_id", "component_id"])
    dev = dev.merge(comp, on="s1_id", how="left")
    dev["holdout"] = vf.holdout_mask(dev["component_id"].to_numpy(), vc["holdout_frac"])
    dev["vfold"] = vf.verifier_fold(dev["fold"])
    oof = pd.read_parquet(Path(a["models_dir"]) / "e1b_oof.parquet")
    oof["band"] = vf.in_band(oof["p"], vc["band"])
    s1raw, _ = read_source(a["train.source1"], validate=False)
    gt, _ = read_ground_truth(a["train.ground_truth"], s1_ids=s1raw["entity_id"])
    return dev, oof, gt


def strata_sets(cfg) -> tuple[set, set]:
    norm = Path(cfg["abs"]["cache_dir"]) / "norm"
    d = pd.concat([pd.read_parquet(norm / f"train_{s}.parquet", columns=["entity_id", "name_script", "addr_missing"])
                   for s in ("source1", "source2", "source3")])
    return (set(d.loc[d["name_script"] != "latin", "entity_id"]),
            set(d.loc[d["addr_missing"].astype(bool), "entity_id"]))


def score_to(v, pairs, recs, style, path: Path, batch_size: int | None = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    t_txt, c_txt = time.time(), time.process_time()
    L, R = pair_texts(pairs, recs, style)
    v.last_text_s = time.time() - t_txt  # record text for the tokenizer (tokenisation phase)
    v.last_text_cpu = time.process_time() - c_txt
    t = time.time()
    s = v.score(L, R, log_every=100_000, batch_size=batch_size)
    out = pairs[["s1_id", "m_id"]].reset_index(drop=True).assign(logit=s["logit"], p=s["p"])
    if "p_contra" in s:
        out["p_contra"] = s["p_contra"]
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"  scored {len(out):,} -> {path.name} ({len(out) / (time.time() - t):,.0f} pairs/s)", flush=True)
    return out


def cmd_test_band(args, cfg) -> None:
    from ber.embed import gpu_setup
    gpu_setup()
    from run_inference import ensemble_predict
    out = vroot(cfg) / "test_band.parquet"
    if out.exists():
        print(f"{out} exists")
        return
    bundle = joblib.load(Path(cfg["abs"]["models_dir"]) / "e1b.joblib")
    feats = bundle["features"]
    band = cfg["verifier"]["band"]
    parts, n = [], 0
    t = time.time()
    for p in sorted(features_dir(cfg, "test").glob("shard=*/part.parquet")):
        d = pd.read_parquet(p, columns=["s1_id", "m_id", "source"] + feats)
        d["p"] = ensemble_predict(bundle, d[feats].to_numpy(np.float32))
        n += len(d)
        parts.append(d.loc[vf.in_band(d["p"], band), ["s1_id", "m_id", "source", "p"]])
    tb = pd.concat(parts, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tb.to_parquet(out, index=False)
    print(f"test band: {len(tb):,} of {n:,} pairs ({len(tb) / n:.2%}) in {time.time() - t:.0f}s -> {out}")


def load_model(key, cfg):
    from ber.embed import gpu_setup
    dev = gpu_setup()
    return Verifier(key, dev, max_len=cfg["verifier"]["max_len"]), os.environ.get("CUDA_VISIBLE_DEVICES", "")


def cmd_pilot(args, cfg) -> None:
    """Train on <= pilot_pairs from verifier fold 0 (screen sample excluded); dCV on the sample."""
    import torch
    key, spec = args.model, MODELS[args.model]
    dev, oof, gt = dev_frame(cfg)
    scr = pd.read_parquet(vroot(cfg) / "screen" / f"{key}_dev.parquet")  # zero-shot scores, sample band
    sample_s1 = set(scr["s1_id"])
    nonlatin, empty = strata_sets(cfg)
    pool = dev[(~dev["holdout"]) & (dev["vfold"] == 0) & (~dev["s1_id"].isin(sample_s1))]["s1_id"]
    rows = build_training_pairs(oof, gt, pool, nonlatin, empty, seed=cfg["seed"])
    keep = set(pd.Series(rows["s1_id"].unique()).sample(frac=1.0, random_state=1)
               .iloc[:max(1, int(args.pilot_pairs / (len(rows) / rows["s1_id"].nunique())))])
    rows = rows[rows["s1_id"].isin(keep)].head(args.pilot_pairs).reset_index(drop=True)
    print(f"[pilot] {key}: {len(rows):,} training pairs from {len(keep):,} S1 ({rows['y'].mean():.1%} pos)",
          flush=True)
    recs = load_records(cfg, "train", set(rows["s1_id"]) | set(rows["m_id"]) | set(scr["s1_id"]) | set(scr["m_id"]))
    v, gpu_id = load_model(key, cfg)
    L, R = pair_texts(rows, recs, spec["style"])
    out = vroot(cfg) / key / "pilot"
    torch.cuda.reset_peak_memory_stats()
    st = train(v, L, R, rows["y"].to_numpy(), out, max_minutes=args.max_minutes, seed=cfg["seed"],
               **RECIPE[spec["kind"]])
    print(f"[pilot] trained: {st}", flush=True)
    sc = score_to(v, scr[["s1_id", "m_id"]], recs, spec["style"], out / "sample_scores.parquet")
    from sklearn.metrics import roc_auc_score
    print(f"[pilot] AUC on sample band: zero-shot {roc_auc_score(scr['y'], scr['logit']):.4f} -> "
          f"pilot {roc_auc_score(scr['y'], sc['logit']):.4f}", flush=True)
    (out / "train_stats.json").write_text(json.dumps({**st, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                                                      "gpu": gpu_id}, indent=1))


def cmd_fold(args, cfg) -> None:
    import torch
    key, spec, j = args.model, MODELS[args.model], args.fold
    out = vroot(cfg) / key / f"fold{j}"
    out.mkdir(parents=True, exist_ok=True)
    dev, oof, gt = dev_frame(cfg)
    nonlatin, empty = strata_sets(cfg)
    tr_s1 = dev.loc[(~dev["holdout"]) & (dev["vfold"] == j), "s1_id"]
    rows = build_training_pairs(oof, gt, tr_s1, nonlatin, empty, n_hard=args.n_hard, n_rand=args.n_rand,
                                seed=cfg["seed"] + j)
    if args.max_pairs and len(rows) > args.max_pairs:
        rows = rows.head(args.max_pairs).reset_index(drop=True)  # rows are pre-shuffled
    hold = set(dev.loc[dev["holdout"], "s1_id"])
    oof_s1 = set(dev.loc[(~dev["holdout"]) & (dev["vfold"] != j), "s1_id"])
    # invariants (D7/D8): no scored pair's S1 was trained on; holdout never trained on
    assert not (set(rows["s1_id"]) & (hold | oof_s1)), "training S1 leaked into a scored set"
    band = oof[oof["band"]]
    s_oof = band[band["s1_id"].isin(oof_s1)][["s1_id", "m_id"]].reset_index(drop=True)
    s_hold = band[band["s1_id"].isin(hold)][["s1_id", "m_id"]].reset_index(drop=True)
    print(f"[fold {j}] {key}: train {len(rows):,} pairs / {rows['s1_id'].nunique():,} S1 "
          f"({rows['y'].mean():.1%} pos); score oof {len(s_oof):,}, holdout {len(s_hold):,}", flush=True)
    recs = load_records(cfg, "train", set(rows["s1_id"]) | set(rows["m_id"]) | set(s_oof["s1_id"]) |
                        set(s_oof["m_id"]) | set(s_hold["s1_id"]) | set(s_hold["m_id"]))
    v, gpu_id = load_model(key, cfg)
    torch.cuda.reset_peak_memory_stats()
    T0 = time.time()
    stats_p = out / "train_stats.json"
    if (out / "final").exists() and stats_p.exists():
        # resume: reload the finished fold model instead of retraining
        st = json.loads(stats_p.read_text())
        if spec["kind"] in ("qwen_yesno", "nli"):
            from peft import PeftModel
            v.model = PeftModel.from_pretrained(v.model, out / "final").merge_and_unload().to(torch.bfloat16).eval()
        else:
            from ber.verify.rerankers import _shim_extended_mask
            v.model.load_state_dict(type(v.model).from_pretrained(out / "final", trust_remote_code=True).state_dict())
            _shim_extended_mask(v.model)
        print(f"[fold {j}] reloaded trained model from {out / 'final'}", flush=True)
    else:
        L, R = pair_texts(rows, recs, spec["style"])
        st = train(v, L, R, rows["y"].to_numpy(), out, max_minutes=args.max_minutes, seed=cfg["seed"] + j,
                   token_budget=args.token_budget, **RECIPE[spec["kind"]])
        st["peak_vram_gb_train"] = torch.cuda.max_memory_allocated() / 1e9
        stats_p.write_text(json.dumps(st, indent=1))
        print(f"[fold {j}] trained: {st}", flush=True)
    score_to(v, s_oof, recs, spec["style"], out / "scores_oof.parquet")
    score_to(v, s_hold, recs, spec["style"], out / "scores_holdout.parquet")
    del recs
    tb_path = vroot(cfg) / "test_band.parquet"
    while not tb_path.exists():  # written by the `test-band` step, possibly still running
        print("  waiting for test_band.parquet ...", flush=True)
        time.sleep(60)
    tb = pd.read_parquet(tb_path, columns=["s1_id", "m_id"])
    print(f"[fold {j}] test band pairs to score: {len(tb):,}", flush=True)
    trecs = load_records(cfg, "test", set(tb["s1_id"]) | set(tb["m_id"]))
    tdir = out / "test"
    tdir.mkdir(exist_ok=True)
    t = time.time()
    for s in range(0, len(tb), SHARD):
        score_to(v, tb.iloc[s:s + SHARD].reset_index(drop=True), trecs, spec["style"],
                 tdir / f"part-{s // SHARD:04d}.parquet")
    print(f"[fold {j}] test scored in {(time.time() - t) / 60:.1f} min", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    append_run(cfg["abs"]["runs_ledger"], f"p3_verifier_{key}_fold{j}",
               {"stage": "p3_c2_finetune", "model": spec["hf"], "verifier_model": spec["hf"],
                "verifier_band": "-".join(map(str, cfg["verifier"]["band"])), "device": "cuda",
                "gpu_ids": gpu_id, "peak_vram_gb": round(peak, 2), "wall_s": round(time.time() - T0, 1),
                "n_pairs_test": len(tb),
                "config_hash": config_hash({k: x for k, x in cfg.items() if k != "abs"})},
               notes=f"verifier fold {j}: trained on {st['pairs']:,} pairs; OOF/holdout/test scored",
               extra=st)


def reload_fold_model(key: str, cfg, j: int):
    """The trained fold-j verifier (merged LoRA or full fine-tune), bf16, eval mode."""
    import torch
    from transformers import AutoModelForSequenceClassification

    spec = MODELS[key]
    final = vroot(cfg) / key / f"fold{j}" / "final"
    v, gpu_id = load_model(key, cfg)
    if spec["kind"] in ("qwen_yesno", "nli"):
        from peft import PeftModel
        v.model = PeftModel.from_pretrained(v.model, final).merge_and_unload()
    else:
        v.model = AutoModelForSequenceClassification.from_pretrained(
            final, dtype=torch.bfloat16, trust_remote_code=bool(spec.get("remote_code")))
        if spec.get("remote_code"):
            from ber.verify.rerankers import _shim_extended_mask
            _shim_extended_mask(v.model)
    v.model.to(v.device).to(torch.bfloat16).eval()
    return v, gpu_id


def cmd_score_extra(args, cfg) -> None:
    """Score test band pairs that the fold runs did not cover (e.g. pairs E4 moved into the band)."""
    key, spec = args.model, MODELS[args.model]
    miss = pd.read_parquet(vroot(cfg) / key / "test_missing.parquet")
    print(f"[extra] {len(miss):,} test band pairs to score with both fold models", flush=True)
    split_cfg = load_config(args.records_config) if args.records_config else cfg
    trecs = load_records(split_cfg, "test", set(miss["s1_id"]) | set(miss["m_id"]))
    for j in (0, 1):
        v, _ = reload_fold_model(key, cfg, j)
        d = vroot(cfg) / key / f"fold{j}" / "test_extra"
        d.mkdir(exist_ok=True)
        n = len(list(d.glob("part-*.parquet")))
        score_to(v, miss, trecs, spec["style"], d / f"part-{n:04d}.parquet")
        del v
        import torch
        torch.cuda.empty_cache()


def cmd_rescore(args, cfg) -> None:
    """Deterministic re-score (fixed batch size, stable length sort, seeds, deterministic flags)
    of fold j's OOF, holdout and test band pairs with its trained model -> fold{j}/det/."""
    from ber.verify import score_dir, set_deterministic
    set_deterministic(cfg["seed"])  # before any CUDA call
    key, spec, j = args.model, MODELS[args.model], args.fold
    cfg = {**cfg, "verifier": {**cfg["verifier"], "scores": "det"}}
    out = score_dir(cfg, key, j)
    (out / "test").mkdir(parents=True, exist_ok=True)
    dev, oof, gt = dev_frame(cfg)
    hold = set(dev.loc[dev["holdout"], "s1_id"])
    oof_s1 = set(dev.loc[(~dev["holdout"]) & (dev["vfold"] != j), "s1_id"])
    band = oof[oof["band"]]
    s_oof = band[band["s1_id"].isin(oof_s1)][["s1_id", "m_id"]].reset_index(drop=True)
    s_hold = band[band["s1_id"].isin(hold)][["s1_id", "m_id"]].reset_index(drop=True)
    v, gpu_id = reload_fold_model(key, cfg, j)
    bs = args.det_batch
    recs = load_records(cfg, "train", set(s_oof["s1_id"]) | set(s_oof["m_id"]) | set(s_hold["s1_id"]) | set(s_hold["m_id"]))
    t0 = time.time()
    if not (args.reverse or args.test_only):
        score_to(v, s_oof, recs, spec["style"], out / "scores_oof.parquet", batch_size=bs)
        score_to(v, s_hold, recs, spec["style"], out / "scores_holdout.parquet", batch_size=bs)
    del recs
    tb = pd.read_parquet(vroot(cfg) / "test_band.parquet", columns=["s1_id", "m_id"])
    trecs = load_records(cfg, "test", set(tb["s1_id"]) | set(tb["m_id"]))
    starts = list(range(0, len(tb), SHARD))
    if args.reverse:  # a second worker (e.g. on the other GPU) takes shards from the end
        starts = starts[::-1]
    for s in starts:
        score_to(v, tb.iloc[s:s + SHARD].reset_index(drop=True), trecs, spec["style"],
                 out / "test" / f"part-{s // SHARD:04d}.parquet", batch_size=bs)
    if args.reverse or args.test_only:
        print(f"[rescore fold {j}] {'reverse' if args.reverse else 'test-only'} worker done in {(time.time() - t0) / 60:.1f} min", flush=True)
        return
    (out / "DETERMINISTIC.json").write_text(json.dumps({"batch_size": bs, "seed": cfg["seed"], "gpu": gpu_id,
                                                       "minutes": (time.time() - t0) / 60}))
    print(f"[rescore fold {j}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test-band", "pilot", "fold", "score-extra", "rescore"])
    ap.add_argument("--config", default="configs/p3.yaml")
    ap.add_argument("--model", default="qwen3-rr-0.6b")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--pilot-pairs", type=int, default=200_000)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--n-hard", type=int, default=8)
    ap.add_argument("--n-rand", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--records-config", default=None,
                    help="config whose norm cache supplies the record text (only name_script/name_key are read)")
    ap.add_argument("--reverse", action="store_true", help="rescore: take test shards from the end")
    ap.add_argument("--test-only", action="store_true", help="rescore: test shards only, from the start")
    ap.add_argument("--det-batch", type=int, default=256, help="rescore: fixed batch size")
    ap.add_argument("--token-budget", type=int, default=16_000, help="padded tokens per micro-batch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    compute.limit_threads()
    {"test-band": cmd_test_band, "pilot": cmd_pilot, "fold": cmd_fold,
     "score-extra": cmd_score_extra, "rescore": cmd_rescore}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
