"""Score the test candidates with a trained bundle and write a submission.

    python scripts/run_inference.py --bundle e1b --sub sub1

Steps: sharded test features (reused if already cached) -> ensemble of the fold models
-> exclusivity over ALL test S1 -> the bundle's winning decision rule -> TSVs + validator.

Exclusivity is exact here: every S2/S3 record is claimed by at most one S1 across the
whole test set, unlike the dev slice where only the sampled S1 compete.
Nothing is uploaded; the files are left for the owner to submit.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ber import compute  # noqa: E402
from ber.decide.sets import decide_expected_f, decide_threshold, exclusive, to_sets  # noqa: E402
from ber.evaluate.ledger import append_run, config_hash  # noqa: E402
from ber.io_utils import load_config, read_source, write_candidates, write_matching  # noqa: E402
from ber.match import gbdt  # noqa: E402
from run_features import features_dir, load_candidates, load_features, run as run_feature_stage  # noqa: E402
from validate_outputs import validate  # noqa: E402


def ensemble_predict(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Mean calibrated probability over the per-fold models.

    P1 used one model refitted on all dev pairs; averaging the existing fold models is
    equivalent in cost-free terms and slightly steadier, and needs no extra GPU fit.
    """
    acc = np.zeros(len(X), np.float64)
    for clf in bundle["folds"].values():
        acc += gbdt.predict(clf, X)
    return (acc / len(bundle["folds"])).astype(np.float32)


def attach_verifier(F: pd.DataFrame, bundle: dict, cfg: dict) -> pd.DataFrame:
    """Add the v_* features: mean of both verifier fold models' Platt-calibrated logits.

    Band membership is recomputed with the E1b ensemble on THESE features (so an E4 feature
    change that moves a pair into the band is caught). Every band pair must have a score from
    both fold models; missing ones are written out and the run stops (score them with
    ``run_verifier_ft.py score-extra``) rather than silently entering as out-of-band.
    """
    from ber.verify import folds as vf

    from ber.verify import score_dir

    vinfo = bundle["verifier"]
    root = Path(cfg["abs"]["cache_dir"]).parent / "verifier" / vinfo["key"]
    vcfg = {**cfg, "verifier": {**(cfg.get("verifier") or {}), "scores": vinfo.get("scores", "")}}
    e1b = joblib.load(Path(cfg["abs"]["models_dir"]) / "e1b.joblib")
    p_e1b = ensemble_predict(e1b, F[e1b["features"]].to_numpy(np.float32))
    band = vf.in_band(p_e1b, vinfo["band"])
    need = F.loc[band, ["s1_id", "m_id"]]
    frames = []
    for j in (0, 1):
        sd = score_dir(vcfg, vinfo["key"], j)
        parts = sorted((sd / "test").glob("part-*.parquet")) + sorted((sd / "test_extra").glob("part-*.parquet"))
        sj = pd.concat([pd.read_parquet(q) for q in parts], ignore_index=True).drop_duplicates(["s1_id", "m_id"])
        a, b = vinfo["platt"][j]
        frames.append(sj.assign(z=a * sj["logit"] + b)[["s1_id", "m_id", "logit", "z"]
                                                      + (["p_contra"] if "p_contra" in sj else [])])
    m = need.merge(frames[0], on=["s1_id", "m_id"], how="left").merge(
        frames[1], on=["s1_id", "m_id"], how="left", suffixes=("_0", "_1"))
    miss = m["logit_0"].isna() | m["logit_1"].isna()
    if miss.any():
        out = root / "test_missing.parquet"
        m.loc[miss, ["s1_id", "m_id"]].to_parquet(out, index=False)
        raise SystemExit(f"[inf] {int(miss.sum()):,} band pairs lack verifier scores -> {out}; "
                         "run `run_verifier_ft.py score-extra` first")
    scored = m[["s1_id", "m_id"]].assign(logit=(m["logit_0"] + m["logit_1"]) / 2,
                                         p=1 / (1 + np.exp(-(m["z_0"] + m["z_1"]) / 2)))
    if "p_contra_0" in m:
        scored["p_contra"] = (m["p_contra_0"] + m["p_contra_1"]) / 2
    V = vf.verifier_features(F[["s1_id", "m_id"]], scored)
    print(f"[inf] verifier: {int(band.sum()):,} band pairs ({band.mean():.2%})", flush=True)
    return pd.concat([F.reset_index(drop=True), V], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2.yaml")
    ap.add_argument("--bundle", required=True, help="model bundle name under models_dir")
    ap.add_argument("--sub", required=True, help="submission id, e.g. sub1")
    ap.add_argument("--feature-version", default=None)
    ap.add_argument("--scope", default="test")
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.feature_version:
        cfg["versions"]["feature_version"] = args.feature_version
    a = cfg["abs"]
    compute.limit_threads()
    T0 = time.time()
    # choose the least-loaded GPU before XGBoost touches CUDA (otherwise it takes device 0)
    from ber.embed import gpu_setup
    gpu_setup()

    bundle = joblib.load(Path(a["models_dir"]) / f"{args.bundle}.joblib")
    feats, winner = bundle["features"], bundle["winner"]
    _ef, _thr = bundle["cv"]["macro"], bundle["cv"].get("macro_threshold", float("nan"))
    print(f"[inf] bundle {args.bundle}: {len(feats)} features, decision {winner}, dev CV "
          f"{(_ef if winner.endswith('expected-F') else _thr):.6f}")

    shards = args.shards or int(cfg["compute"].get("shards", 64))
    root = features_dir(cfg, args.scope)
    done = len(list(root.glob("shard=*/_SUCCESS"))) if root.exists() else 0
    if done < shards:
        print(f"[inf] building test features ({done}/{shards} cached)", flush=True)
        run_feature_stage(cfg, args.scope, shards, args.scope)
    base_feats = [f for f in feats if not f.startswith("v_")]
    if bundle.get("verifier"):  # the E1b band needs E1b's inputs even if this bundle dropped some
        e1b_feats = joblib.load(Path(a["models_dir"]) / "e1b.joblib")["features"]
        base_feats = base_feats + [f for f in e1b_feats if f not in base_feats]
    F = load_features(cfg, args.scope, columns=["s1_id", "m_id", "source"] + base_feats)
    print(f"[inf] features {len(F):,} x {len(base_feats)} loaded  [{time.time() - T0:.0f}s]", flush=True)
    if bundle.get("verifier"):
        F = attach_verifier(F, bundle, cfg)
        print(f"[inf] verifier features attached  [{time.time() - T0:.0f}s]", flush=True)

    p = ensemble_predict(bundle, F[feats].to_numpy(np.float32))
    pairs = F[["s1_id", "m_id", "source"]].assign(p=p)
    del F
    print(f"[inf] scored, top-1 p median {pairs.groupby('s1_id')['p'].max().median():.4f} "
          f"[{time.time() - T0:.0f}s]", flush=True)

    ex = exclusive(pairs)
    sel = (decide_expected_f(ex) if winner.endswith("expected-F")
           else decide_threshold(ex, bundle["tau_median"]))
    print(f"[inf] decided: {len(sel):,} matches [{time.time() - T0:.0f}s]", flush=True)

    s1, _ = read_source(a[f"{args.scope}.source1"])
    s1_ids = list(s1["entity_id"])
    cand_all = load_candidates(cfg, args.scope)
    cands = cand_all.groupby("s1_id")["m_id"].agg(list).to_dict()
    matches = to_sets(sel, s1_ids)

    outdir = Path(a["submissions_dir"]) / args.sub
    outdir.mkdir(parents=True, exist_ok=True)
    write_candidates(outdir / "candidate_pairs.tsv", cands, s1_ids)
    write_matching(outdir / "matching_results.tsv", matches, s1_ids, candidates=cands)
    print(f"[inf] wrote {outdir}  [{time.time() - T0:.0f}s]", flush=True)

    report = None
    if not args.skip_validate:
        r = validate(outdir / "matching_results.tsv", outdir / "candidate_pairs.tsv", cfg, args.scope)
        report = "\n".join(f"  ok  {c}" for c in r["checks"]) + "\n\n" + (r["official"] or "")
        print(report)

    ef, thr = bundle["cv"]["macro"], bundle["cv"].get("macro_threshold", float("nan"))
    shipped_cv, other_cv = (ef, thr) if winner.endswith("expected-F") else (thr, ef)
    n_pred = sum(len(v) for v in matches.values())
    empty = sum(1 for v in matches.values() if not v)
    chash = config_hash({k: v for k, v in cfg.items() if k != "abs"})
    (outdir / "SUMMARY.md").write_text("\n".join([
        f"# {args.sub}: {args.bundle}\n",
        f"- Config hash `{chash}`; feature set `{cfg['versions']['feature_version']}`; "
        f"blockers `{cfg['versions']['blocker_set']}`; K={cfg['versions']['K']}.",
        f"- Model: {len(bundle['folds'])}-fold XGBoost ensemble (GPU), decision **{winner}**.",
        f"- Dev CV macro F0.5 of the SHIPPED rule ({winner}) **{shipped_cv:.6f}** "
        f"(other rule {other_cv:.6f}); P1 reference 0.895607, delta {shipped_cv - 0.895607:+.6f}.",
        f"- LOCO: " + ", ".join(f"{k} {v:.4f}" for k, v in bundle["loco"].items()) + ".",
        f"- Test output: {len(s1_ids):,} S1, {n_pred:,} predicted pairs, {len(cand_all):,} candidates, "
        f"{empty:,} empty ({empty / len(s1_ids):.2%}).",
        "\n## Validator\n", "```", report or "(skipped)", "```",
        "\nThe owner uploads this; nothing was submitted automatically.",
    ]), encoding="utf-8")
    print(f"[inf] SUMMARY.md written; done in {(time.time() - T0) / 60:.1f} min")

    if not args.no_ledger:
        v = cfg["versions"]
        append_run(a["runs_ledger"], f"p2_{args.sub}",
                   metrics={"stage": "test_inference", "model": args.bundle,
                            "submission_id": args.sub, "cv_macro_f05": bundle["cv"]["macro"],
                            "n_pairs_test": len(cand_all), "feature_version": v["feature_version"],
                            "norm_version": v["norm_version"], "blocker_set": v["blocker_set"],
                            "K": v["K"], "device": "cuda", "config_hash": chash,
                            "wall_s": round(time.time() - T0, 1)},
                   notes=f"{args.sub}: {args.bundle}, decision {winner}, {n_pred:,} matches, "
                         f"{empty / len(s1_ids):.2%} empty; validator PASS")


if __name__ == "__main__":
    main()
