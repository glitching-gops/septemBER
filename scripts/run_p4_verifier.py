"""Final phase (P4) verifier round 2: refreshed negatives, French augmentation, continued training.

    python scripts/run_p4_verifier.py negatives               # D2 training rows per verifier fold
    python scripts/run_p4_verifier.py train --fold 0          # D1 + D3: continue fold 0 for one epoch
    python scripts/run_p4_verifier.py sets                    # D4 scoring sets (band + rescue), dev and test
    python scripts/run_p4_verifier.py score --fold 0 [--part oof|holdout|test] [--reverse]

Fold bookkeeping is P3's (ber.verify.folds): fold j trains on non-holdout dev S1 with
``verifier_fold == j`` and scores the OTHER fold's S1 (OOF), the holdout and test. Every
output is resumable (existing files / finished shards are skipped; training resumes from its
latest full-state checkpoint) and deterministic (seeds, fixed batch order and batch size).
Round-2 artefacts live under ``<p4>/verifier/<verifier.key>``; the sub3 fold models under
``<p4>/verifier/<verifier.init_key>`` (a symlink to P3) are only read.
"""
from __future__ import annotations

import argparse
import hashlib
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
from ber.io_utils import load_config  # noqa: E402
from ber.verify import folds as vf  # noqa: E402
from ber.verify.rerankers import MODELS  # noqa: E402
from ber.verify.serialize import ditto, load_records, pair_texts, plain  # noqa: E402
from ber.verify.train import build_training_pairs_r2, train  # noqa: E402
from run_verifier_ft import dev_frame, load_model, score_to, strata_sets  # noqa: E402

SHARD = 250_000


def vroot(cfg) -> Path:
    return Path(cfg["abs"]["cache_dir"]).parent / "verifier"


def r2dir(cfg) -> Path:
    return vroot(cfg) / cfg["verifier"]["key"]


def sub3_calibrated_oof(cfg) -> pd.DataFrame:
    """sub3 verifier OOF scores (each S1 from the fold model that never saw it), Platt-calibrated."""
    b = joblib.load(Path(cfg["abs"]["models_dir"]) / "e8_bge_det.joblib")
    platt = b["verifier"]["platt"]
    parts = []
    for j in (0, 1):
        s = pd.read_parquet(vroot(cfg) / cfg["verifier"]["init_key"] / f"fold{j}" / "det" / "scores_oof.parquet")
        a, c = platt[j]
        parts.append(s[["s1_id", "m_id"]].assign(v_p=1 / (1 + np.exp(-(a * s["logit"] + c)))))
    return pd.concat(parts, ignore_index=True)


def cmd_negatives(args, cfg) -> None:
    vc = cfg["verifier"]
    out = r2dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    dev, oof, gt = dev_frame(cfg)
    nonlatin, empty = strata_sets(cfg)
    s3 = pd.read_parquet(Path(cfg["abs"]["models_dir"]) / "e8_bge_det_oof.parquet")
    taus = joblib.load(Path(cfg["abs"]["models_dir"]) / "e8_bge_det.joblib")["fold_taus"]
    s3["tau"] = s3["fold"].map(taus)
    cand = oof[["s1_id", "m_id", "p", "y"]].rename(columns={"p": "p_e1b"})
    cand = cand.merge(s3[["s1_id", "m_id", "p", "tau"]].rename(columns={"p": "p_s3"}), on=["s1_id", "m_id"], how="left")
    cand = cand.merge(sub3_calibrated_oof(cfg), on=["s1_id", "m_id"], how="left")
    cand["key"] = np.where(cand["v_p"].notna(), (cand["p_e1b"] + cand["v_p"]) / 2, cand["p_e1b"])
    cand["fp"] = ((cand["y"] == 0) & (cand["p_s3"] >= cand["tau"])).astype(np.int8)
    cand["fn"] = ((cand["y"] == 1) & (cand["p_s3"] < cand["tau"])).astype(np.int8)
    for j in (0, 1):
        tr = dev.loc[(~dev["holdout"]) & (dev["vfold"] == j), "s1_id"]
        rows = build_training_pairs_r2(cand, gt, tr, nonlatin, empty, n_hard=vc["n_hard"], n_rand=vc["n_rand"],
                                       fn_dup=vc["fn_dup"], seed=cfg["seed"] + 100 + j)
        hold = set(dev.loc[dev["holdout"], "s1_id"])
        other = set(dev.loc[(~dev["holdout"]) & (dev["vfold"] != j), "s1_id"])
        assert not (set(rows["s1_id"]) & (hold | other)), "training S1 leaked into a scored set"
        c = cand[cand["s1_id"].isin(set(tr))]
        rows.to_parquet(out / f"train_rows_fold{j}.parquet", index=False)
        print(f"[neg] fold {j}: {len(rows):,} rows / {rows['s1_id'].nunique():,} S1, {rows['y'].mean():.1%} pos; "
              f"FP {int(c['fp'].sum()):,}, FN {int(c['fn'].sum()):,} (sub3 OOF)", flush=True)


def load_init_weights(v, cfg, j: int) -> None:
    """Replace the base weights with the sub3 fold-j fine-tuned weights, in fp32 (D1)."""
    import torch
    from transformers import AutoModelForSequenceClassification
    final = vroot(cfg) / cfg["verifier"]["init_key"] / f"fold{j}" / "final"
    sd = AutoModelForSequenceClassification.from_pretrained(final, dtype=torch.float32).state_dict()
    v.model = v.model.float()
    missing, unexpected = v.model.load_state_dict(sd, strict=False)
    assert not unexpected and all("position_ids" in m for m in missing), (missing, unexpected)
    print(f"[train] fold {j}: initialised from {final}", flush=True)


def cmd_train(args, cfg) -> None:
    import torch
    from ber.verify import set_deterministic
    from ber.verify.augment import augment_pairs
    set_deterministic(cfg["seed"])
    vc, j = cfg["verifier"], args.fold
    key = vc["init_key"]
    spec = MODELS[key]
    out = r2dir(cfg) / f"fold{j}"
    out.mkdir(parents=True, exist_ok=True)
    stats_p = out / "train_stats.json"
    if (out / "final").exists() and stats_p.exists():
        print(f"[train] fold {j} already trained -> {out / 'final'}")
        return
    rows = pd.read_parquet(r2dir(cfg) / f"train_rows_fold{j}.parquet")
    if args.max_pairs and len(rows) > args.max_pairs:
        rows = rows.head(args.max_pairs).reset_index(drop=True)
    recs = load_records(cfg, "train", set(rows["s1_id"]) | set(rows["m_id"]))
    fn = {"plain": plain, "ditto": ditto}[spec["style"]]
    t = time.time()
    frac = 0.0 if args.no_aug else vc["aug_frac"]
    L, R, side = augment_pairs(rows, recs, fn, frac, seed=cfg["seed"] + 200 + j)
    del recs
    print(f"[train] fold {j}: {len(rows):,} rows, {int((side >= 0).sum()):,} augmented "
          f"({(side >= 0).mean():.1%}) [{time.time() - t:.0f}s]", flush=True)
    pd.DataFrame({"aug_side": side}).to_parquet(out / "aug_side.parquet", index=False)
    v, gpu_id = load_model(key, cfg)
    load_init_weights(v, cfg, j)
    torch.cuda.reset_peak_memory_stats()
    T0 = time.time()
    st = train(v, L, R, rows["y"].to_numpy(), out, lora=False, lr=vc["lr"], eff_batch=vc["eff_batch"],
               token_budget=args.token_budget, ckpt_every=vc["ckpt_every"], seed=cfg["seed"] + 100 + j,
               max_minutes=args.max_minutes, resumable=True)
    st["peak_vram_gb_train"] = torch.cuda.max_memory_allocated() / 1e9
    st["augmented"] = int((side >= 0).sum())
    st["init"] = str(vroot(cfg) / key / f"fold{j}" / "final")
    stats_p.write_text(json.dumps(st, indent=1))
    import shutil
    shutil.rmtree(out / "ckpt-latest", ignore_errors=True)
    print(f"[train] fold {j} done: {st}", flush=True)
    append_run(cfg["abs"]["runs_ledger"], f"p4_verifier_r2_fold{j}",
               {"stage": "p4_verifier_train", "model": spec["hf"], "verifier_model": spec["hf"],
                "verifier_band": "-".join(map(str, vc["band"])), "device": "cuda", "gpu_ids": gpu_id,
                "peak_vram_gb": round(st["peak_vram_gb_train"], 2), "wall_s": round(time.time() - T0, 1),
                "config_hash": config_hash({k: x for k, x in cfg.items() if k != "abs"})},
               notes=f"round-2 fold {j}: continued from sub3 weights, {st['pairs']:,}/{st['pairs_total']:,} pairs, "
                     f"{st['steps']} steps, {st['augmented']:,} French-style augmented", extra=st)


def rescue_pairs(pairs: pd.DataFrame, rescue_s1: set, k: int) -> pd.DataFrame:
    """Top-``k`` candidates by E1b p of each rescued S1 (D4)."""
    r = pairs[pairs["s1_id"].isin(rescue_s1)].sort_values(["s1_id", "p_e1b", "m_id"], ascending=[True, False, True])
    return r[r.groupby("s1_id").cumcount() < k]


def scoring_set(pairs: pd.DataFrame, empty_s1: set, vc: dict) -> pd.DataFrame:
    """Band (E1b p in ``vc['band']``) + rescue, one row per pair, with ``inband`` / ``rescued`` flags.

    ``pairs`` has s1_id, m_id, p_e1b, p_s3. Rescue S1 = sub3 predicts empty, or sub3 max p <
    ``rescue_max_p``.
    """
    mx = pairs.groupby("s1_id")["p_s3"].max()
    rescue_s1 = set(empty_s1) | set(mx.index[mx < vc["rescue_max_p"]])
    band = pairs.loc[vf.in_band(pairs["p_e1b"], vc["band"]), ["s1_id", "m_id"]]
    resc = rescue_pairs(pairs, rescue_s1, vc["rescue_k"])[["s1_id", "m_id"]]
    out = band.assign(inband=np.int8(1)).merge(resc.assign(rescued=np.int8(1)), on=["s1_id", "m_id"], how="outer")
    out = out.fillna({"inband": 0, "rescued": 0}).astype({"inband": np.int8, "rescued": np.int8})
    return out.sort_values(["s1_id", "m_id"]).reset_index(drop=True)


def sub3_empty(pairs: pd.DataFrame, tau_of, groups) -> set:
    """S1 with an empty sub3 set: exclusivity within each group, then threshold."""
    from ber.decide.sets import decide_threshold, exclusive
    empty = set()
    for g, d in pairs.groupby(groups):
        sel = decide_threshold(exclusive(d[["s1_id", "m_id", "p"]]), tau_of(g))
        empty |= set(d["s1_id"]) - set(sel["s1_id"])
    return empty


def cmd_sets(args, cfg) -> None:
    """D4 scoring sets. dev: E1b OOF p + sub3 OOF p (holdout: sub3 bundle, mean of folds);
    test: E1b and sub3 p on every test pair (also kept for the group / c2c features)."""
    from ber.embed import gpu_setup
    gpu_setup()
    from run_features import load_features
    from run_inference import attach_verifier, ensemble_predict
    from run_e1 import load_matrix
    from run_verifier_integrate import verifier_scores
    vc, a = cfg["verifier"], cfg["abs"]
    out = r2dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a["models_dir"])
    s3b = joblib.load(mdir / "e8_bge_det.joblib")
    cfg3 = {**cfg, "verifier": {**vc, "band": s3b["verifier"]["band"]}}
    T0 = time.time()
    if not (out / "dev_pairs_p.parquet").exists():
        dev, oof, gt = dev_frame(cfg)
        hold = set(dev.loc[dev["holdout"], "s1_id"])
        s3 = pd.read_parquet(mdir / "e8_bge_det_oof.parquet")
        # holdout: sub3 bundle (mean of its fold models) on holdout features + verifier features
        F, _ = load_matrix(cfg)
        H = F[F["s1_id"].isin(hold)].reset_index(drop=True)
        y = oof[["s1_id", "m_id", "y"]]
        sc_h, _ = verifier_scores(vc["init_key"], cfg3, y, "holdout")
        V = vf.verifier_features(H[["s1_id", "m_id"]], sc_h)
        H = pd.concat([H, V], axis=1)
        ph = ensemble_predict(s3b, H[s3b["features"]].to_numpy(np.float32))
        s3h = H[["s1_id", "m_id", "source"]].assign(p=ph, fold=-1)
        del F, H
        s3all = pd.concat([s3[["s1_id", "m_id", "p", "fold"]], s3h[["s1_id", "m_id", "p", "fold"]]], ignore_index=True)
        dp = oof[["s1_id", "m_id", "source", "p", "y", "fold"]].rename(columns={"p": "p_e1b", "fold": "xfold"})
        dp = dp.merge(s3all.rename(columns={"p": "p_s3", "fold": "s3fold"}), on=["s1_id", "m_id"], how="left")
        assert dp["p_s3"].notna().all(), "sub3 p missing for some dev pairs"
        dp["holdout"] = dp["s1_id"].isin(hold).astype(np.int8)
        dp.to_parquet(out / "dev_pairs_p.parquet", index=False)
        print(f"[sets] dev pairs with E1b + sub3 p: {len(dp):,} [{time.time() - T0:.0f}s]", flush=True)
    dp = pd.read_parquet(out / "dev_pairs_p.parquet")
    taus = s3b["fold_taus"]
    empty = sub3_empty(dp.rename(columns={"p_s3": "p"}), lambda g: taus[g] if g >= 0 else s3b["tau_median"],
                       dp["s3fold"])
    ds = scoring_set(dp, empty, vc)
    ds.to_parquet(out / "dev_set.parquet", index=False)
    print(f"[sets] dev: {len(ds):,} pairs (band {int(ds['inband'].sum()):,}, rescued {int(ds['rescued'].sum()):,}, "
          f"rescue-only {int(((ds['rescued'] == 1) & (ds['inband'] == 0)).sum()):,}); sub3-empty S1 {len(empty):,}",
          flush=True)

    if not (out / "test_pairs_p.parquet").exists():
        e1b = joblib.load(mdir / "e1b.joblib")
        feats = list(dict.fromkeys(e1b["features"] + [f for f in s3b["features"] if not f.startswith("v_")]))
        F = load_features(cfg, "test", columns=["s1_id", "m_id", "source"] + feats)
        p_e1b = ensemble_predict(e1b, F[e1b["features"]].to_numpy(np.float32))
        print(f"[sets] test E1b p on {len(F):,} pairs [{time.time() - T0:.0f}s]", flush=True)
        F = attach_verifier(F, s3b, cfg3)
        p_s3 = ensemble_predict(s3b, F[s3b["features"]].to_numpy(np.float32))
        tp = F[["s1_id", "m_id", "source"]].assign(p_e1b=p_e1b, p_s3=p_s3)
        del F
        tp.to_parquet(out / "test_pairs_p.parquet", index=False)
        print(f"[sets] test sub3 p [{time.time() - T0:.0f}s]", flush=True)
    tp = pd.read_parquet(out / "test_pairs_p.parquet")
    empty_t = sub3_empty(tp.rename(columns={"p_s3": "p"}), lambda g: s3b["tau_median"], np.zeros(len(tp), int))
    ts = scoring_set(tp, empty_t, vc)
    ts.to_parquet(out / "test_set.parquet", index=False)
    n_s1 = tp["s1_id"].nunique()
    print(f"[sets] test: {len(ts):,} pairs (band {int(ts['inband'].sum()):,}, rescued {int(ts['rescued'].sum()):,}); "
          f"sub3-empty S1 {len(empty_t):,} / {n_s1:,} [{time.time() - T0:.0f}s]", flush=True)


def cmd_score(args, cfg) -> None:
    """Deterministic scoring with the round-2 fold model: OOF (other fold's S1), holdout, test shards.

    Row selection (which dev S1 a fold model may score) is done on the GPU with int ids; the host
    only builds the pair texts and tokenises them (timed as the tokenisation phase)."""
    from ber.verify import set_deterministic
    set_deterministic(cfg["seed"])
    import torch
    from transformers import AutoModelForSequenceClassification
    from ber.gpu import StageTimer
    from ber.io import gpu_load as gl
    from ber.io import ids as I
    vc, j = cfg["verifier"], args.fold
    spec = MODELS[vc["init_key"]]
    fdir = r2dir(cfg) / f"fold{j}"
    out = (Path(os.environ["P4_SCORE_ROOT"]) / f"fold{j}" / "det") if os.environ.get("P4_SCORE_ROOT") else fdir / "det"
    (out / "test").mkdir(parents=True, exist_ok=True)
    v, gpu_id = load_model(vc["init_key"], cfg)
    v.model = AutoModelForSequenceClassification.from_pretrained(fdir / "final", dtype=torch.bfloat16)
    v.model.to(v.device).eval()
    timer = StageTimer(f"score_fold{j}")
    bs = args.det_batch

    def run(pairs, recs, path):
        if path.exists():
            return
        t = time.time()
        v.last_text_s = v.last_tok_s = v.last_text_cpu = v.last_tok_cpu = 0.0
        with timer.gpu():
            score_to(v, pairs, recs, spec["style"], path, batch_size=bs)
        timer.tok(v.last_text_s + v.last_tok_s, v.last_text_cpu + v.last_tok_cpu)
        print(f"  {path.name}: {len(pairs):,} pairs in {(time.time() - t) / 60:.1f} min "
              f"(text+tokenise {v.last_text_s + v.last_tok_s:.0f} s)", flush=True)

    if args.part in ("all", "dev", "oof", "holdout") and not args.reverse:
        dev_ = torch.device("cuda")
        e7 = args.extra is not None
        X = args.extra
        ds_path = (r2dir(cfg).parent.parent / X / f"dev_{X}_set.parquet") if e7 else r2dir(cfg) / "dev_set.parquet"
        ds = pd.read_parquet(ds_path, columns=["s1_id", "m_id"])
        k = I.encode_pairs_cached(cfg, "train", ds_path, f"{X}_dev_set_keys" if e7 else "dev_set")
        meta = gl.dev_meta(cfg, dev_)
        n_ent = len(I.entity_table(cfg, "train"))
        vfold = torch.full((n_ent,), -1, dtype=torch.int64, device=dev_)
        vfold[meta["s1"]] = meta["fold"] % 2
        hold = torch.zeros(n_ent, dtype=torch.bool, device=dev_)
        hold[meta["s1"]] = meta["holdout"]
        s1t = gl.to_dev(k["s1"], dev_)
        is_hold = hold[s1t]
        is_oof = (~is_hold) & (vfold[s1t] == 1 - j) & (vfold[s1t] >= 0)
        s_oof = ds.iloc[torch.nonzero(is_oof).squeeze(1).cpu().numpy()].reset_index(drop=True)
        s_hold = ds.iloc[torch.nonzero(is_hold).squeeze(1).cpu().numpy()].reset_index(drop=True)
        t, tc = time.time(), time.process_time()
        recs = load_records(cfg, "train", set(s_oof["s1_id"]) | set(s_oof["m_id"]) | set(s_hold["s1_id"]) | set(s_hold["m_id"]))
        timer.io(time.time() - t, time.process_time() - tc)
        sfx = f"_{X}" if e7 else ""
        if args.part in ("all", "dev", "oof"):
            run(s_oof, recs, out / f"scores_oof{sfx}.parquet")
        if args.part in ("all", "dev", "holdout"):
            run(s_hold, recs, out / f"scores_holdout{sfx}.parquet")
        del recs
    if args.part in ("all", "test"):
        e7 = args.extra is not None
        X = args.extra
        ts_path = (Path(os.environ.get("P4_EXTRA_ROOT", str(r2dir(cfg).parent.parent / X))) / f"test_{X}_set.parquet") if e7 else r2dir(cfg) / "test_set.parquet"
        if os.environ.get("P4_TEST_SET") and not e7:
            ts_path = Path(os.environ["P4_TEST_SET"])  # reproduction: the GPU-rebuilt scoring set
        ts = pd.read_parquet(ts_path, columns=["s1_id", "m_id"])
        tdir = out / ("test_extra" if e7 else "test")
        tdir.mkdir(parents=True, exist_ok=True)
        t, tc = time.time(), time.process_time()
        trecs = load_records(cfg, "test", set(ts["s1_id"]) | set(ts["m_id"]))
        timer.io(time.time() - t, time.process_time() - tc)
        starts = list(range(0, len(ts), SHARD))
        order_ = starts[::-1] if args.reverse else starts
        if args.middle:  # a third worker: from the middle shard upwards, then the rest
            order_ = starts[len(starts) // 2:] + starts[:len(starts) // 2]
        for s in order_:
            run(ts.iloc[s:s + SHARD].reset_index(drop=True), trecs,
                tdir / f"part-{X + '-' if e7 else ''}{s // SHARD:04d}.parquet")
        n_done = len(list(tdir.glob(f"part-{X + '-' if e7 else ''}*.parquet")))
        print(f"[score fold {j}] test {n_done}/{len(starts)} shards", flush=True)
        if n_done == len(starts) and not e7:
            (out / "DETERMINISTIC.json").write_text(json.dumps({"batch_size": bs, "seed": cfg["seed"], "gpu": gpu_id}))
    st = timer.summary()
    print(f"[score fold {j}] {st}", flush=True)
    append_run(cfg["abs"]["runs_ledger"], f"p4_verifier_r2_score_fold{j}_{args.part}{'_rev' if args.reverse else ''}",
               {"stage": "p4_verifier_score", "model": spec["hf"], "verifier_model": spec["hf"],
                "verifier_band": "-".join(map(str, vc["band"])), "device": "cuda", "gpu_ids": gpu_id,
                "peak_vram_gb": st["peak_vram_gb"], "wall_s": st["wall_s"], "gpu_time_s": st["gpu_time_s"],
                "cpu_time_s": st["cpu_time_s"], "cpu_violation": st["cpu_violation"],
                "config_hash": config_hash({k_: x for k_, x in cfg.items() if k_ != "abs"})},
               notes=f"round-2 fold {j} scoring ({args.part}); host phases: record I/O {st['io_s']} s, "
                     f"text+tokenisation {st['tok_s']} s (blocking CUDA sync)", extra=st)

def cmd_sets_gpu(args, cfg) -> None:
    """D4 test scoring set rebuilt on the GPU (reproduction path): E1b p and sub3 p on every test pair with
    exact GPU ports of the P2/P3 bundles (``ber.gpu.xgb.from_sklearn``), sub3-empty S1 via exclusivity +
    threshold, band + rescue via ``ber.gpu.mining``. Writes ``<out>/test_set.parquet``; compares with the
    shipped set if present."""
    import torch
    from ber.embed import gpu_setup
    dev = gpu_setup()
    from ber.gpu import StageTimer, mining
    from ber.gpu import pipeline as pl
    from ber.gpu import segment as sg
    from ber.gpu.xgb import from_sklearn, predict_mean
    from ber.io import gpu_load as gl
    from ber.io import ids as I
    from ber.verify import set_deterministic
    set_deterministic(cfg["seed"])
    import pyarrow as pa
    import pyarrow.parquet as pq
    timer = StageTimer("sets_gpu")
    vc, md = cfg["verifier"], Path(cfg["abs"]["models_dir"])
    e1b = joblib.load(md / "e1b.joblib")
    s3b = joblib.load(md / "e8_bge_det.joblib")
    base = list(dict.fromkeys(e1b["features"] + [f for f in s3b["features"] if not f.startswith("v_")]))
    n_ent = len(I.entity_table(cfg, "test"))
    L = gl.load_pairs(cfg, "test", "test", base, dev, timer)
    with timer.gpu():
        X = L["X"]
        idx = {f: i for i, f in enumerate(base)}
        p_e1b = predict_mean([from_sklearn(c) for c in e1b["folds"].values()], X[:, [idx[f] for f in e1b["features"]]])
        # sub3 verifier features on its own band [0.02, 0.95], mean of both fold models' Platt logits
        band3 = s3b["verifier"]["band"]
        pe = p_e1b.to(torch.float64)
        inb = (pe >= band3[0]) & (pe <= band3[1])
        root = vroot(cfg) / s3b["verifier"]["key"]
        fr = []
        for j in (0, 1):
            parts = sorted((root / f"fold{j}" / "det" / "test").glob("part-*.parquet"))
            with timer.host("io"):
                ps = [gl.pair_file(cfg, "test", q, f"sub3_f{j}_{q.stem}", ["logit"], dev) for q in parts]
            fr.append({c: torch.cat([x[c] for x in ps]) for c in ("s1", "m", "logit")})
        j1 = sg.lookup_pairs(fr[0]["s1"], fr[0]["m"], fr[1]["s1"], fr[1]["m"], n_ent)
        (a0, b0), (a1, b1) = s3b["verifier"]["platt"][0], s3b["verifier"]["platt"][1]
        lg = (fr[0]["logit"] + fr[1]["logit"][j1]) / 2
        pc = torch.sigmoid(((a0 * fr[0]["logit"] + b0) + (a1 * fr[1]["logit"][j1] + b1)) / 2)
        V = pl.verifier_features(L["s1"], L["m"], fr[0]["s1"], fr[0]["m"], lg.to(torch.float64), pc,
                                 torch.zeros_like(pc), n_ent)[:, :5]
        assert bool((V[:, 2].bool() == inb).all()), "sub3 band scores do not cover the recomputed E1b band"
        X3 = torch.cat([X[:, [idx[f] for f in s3b["features"] if not f.startswith("v_")]], V], 1)
        p_s3 = predict_mean([from_sklearn(c) for c in s3b["folds"].values()], X3)
        empty = mining.sub3_empty(L["s1"], L["m"], p_s3, torch.zeros_like(L["s1"]), {0: s3b["tau_median"]}, n_ent)
        ss = mining.scoring_set(L["s1"], L["m"], p_e1b, p_s3, empty, vc["band"], vc["rescue_k"], vc["rescue_max_p"], n_ent)
        keep = ss["any"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    o = sg.order([(L["s1"][keep], False), (L["m"][keep], False)])
    _h = timer.host("io")
    _h.__enter__()
    ids = I.entity_table(cfg, "test")["entity_id"].to_numpy(object)
    s1h, mh = L["s1"][keep][o].cpu().numpy(), L["m"][keep][o].cpu().numpy()
    pq.write_table(pa.table({"s1_id": ids[s1h], "m_id": ids[mh], "inband": ss["inband"][keep][o].to(torch.int8).cpu().numpy(),
                             "rescued": ss["rescued"][keep][o].to(torch.int8).cpu().numpy()}), out / "test_set.parquet")
    ref = r2dir(cfg) / "test_set.parquet"
    if ref.exists():
        a = pd.read_parquet(ref)
        b = pd.read_parquet(out / "test_set.parquet")
        same = set(zip(a["s1_id"], a["m_id"])) == set(zip(b["s1_id"], b["m_id"]))
        print(f"[sets-gpu] {len(b):,} pairs (shipped {len(a):,}); identical pair set: {same}; "
              f"sub3-empty S1 {int(empty.sum()):,}", flush=True)
    _h.__exit__(None, None, None)
    print(f"[sets-gpu] {timer.summary()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["negatives", "train", "sets", "score", "sets-gpu"])
    ap.add_argument("--out-dir", default=None, help="sets-gpu: output directory")
    ap.add_argument("--config", default="configs/p4.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=300.0, help="D1 cap: 5 h")
    ap.add_argument("--token-budget", type=int, default=16_000)
    ap.add_argument("--no-aug", action="store_true", help="D3 control arm")
    ap.add_argument("--part", default="all", choices=["all", "dev", "oof", "holdout", "test"])
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--middle", action="store_true", help="score: start from the middle shard (third worker)")
    ap.add_argument("--det-batch", type=int, default=256)
    ap.add_argument("--extra", default=None, choices=[None, "e7", "rr"], help="score an extra pair set (E7 survivors / R2 union)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    compute.limit_threads()
    {"negatives": cmd_negatives, "train": cmd_train, "sets": cmd_sets, "score": cmd_score,
     "sets-gpu": cmd_sets_gpu}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
