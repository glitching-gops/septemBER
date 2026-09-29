"""C4 (D7): retrain the E1b XGBoost with verifier features; compare, paired, against E1b.

    python scripts/run_verifier_integrate.py --model qwen3-rr-0.6b [--drop g_b7_rank,n_blockers] --tag e8

Both arms (E1b features only / + verifier) are refitted on the SAME non-holdout 85% of the
dev slice with the same 5 component-grouped folds, so dCV is a paired comparison; the 95% CI
is a bootstrap over S1 of the per-S1 F0.5 difference. The holdout (15%, whole components,
never trained on by the verifier or by either XGBoost) is scored like test: the mean of
the 5 fold models, the median fold tau, and verifier features from the mean of both
verifier fold models. LOCO is run for both arms. The decision rule is re-selected on CV.

Verifier features: Platt calibration (logistic on the raw logit) is fitted per verifier fold
model on that model's OOF pairs only; holdout/test use the mean of the two calibrated
logits (and of the raw logits).

``--drop`` removes features from the NEW arm only (the D10 provenance ablation uses
``--model none --drop g_b7_rank,n_blockers``); ``--subindex`` makes the new arm E1b + the
E6/E7 sub-index features over the K15 + sub-index pair set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from ber import compute  # noqa: E402
from ber.decide.sets import decide_expected_f, decide_threshold, exclusive, to_sets  # noqa: E402
from ber.evaluate.ledger import append_run, config_hash  # noqa: E402
from ber.evaluate.metric import f05_entity  # noqa: E402
from ber.io_utils import load_config, read_ground_truth, read_source  # noqa: E402
from ber.match import gbdt  # noqa: E402
from ber.verify import folds as vf  # noqa: E402
from ber.verify import score_dir  # noqa: E402
from ber.verify.rerankers import MODELS  # noqa: E402
from p2_buckets import bucket  # noqa: E402
from run_e1 import best_tau, load_matrix  # noqa: E402


def md(df, floatfmt=".4f"):
    return tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=floatfmt)


def vroot(cfg) -> Path:
    return Path(cfg["abs"]["cache_dir"]).parent / "verifier"


def logit_of(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrators(key: str, cfg, oof_y: pd.DataFrame) -> dict:
    """Platt calibrator per verifier fold model, fitted on that model's OOF scores only."""
    cal = {}
    for j in (0, 1):
        s = pd.read_parquet(score_dir(cfg, key, j) / "scores_oof.parquet")
        s = s.merge(oof_y, on=["s1_id", "m_id"], how="inner")
        lr = LogisticRegression(C=1e4).fit(s[["logit"]].to_numpy(), s["y"].to_numpy())
        cal[j] = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    return cal


def combine(frames: list[pd.DataFrame], cal: list[tuple[float, float]]) -> pd.DataFrame:
    """Mean raw logit and mean calibrated logit over the given fold models' scores."""
    base = frames[0][["s1_id", "m_id"]].copy()
    raw, calz, con = [], [], []
    for f, (a, b) in zip(frames, cal):
        f = base.merge(f, on=["s1_id", "m_id"], how="left")
        raw.append(f["logit"].to_numpy())
        calz.append(a * f["logit"].to_numpy() + b)
        if "p_contra" in f:
            con.append(f["p_contra"].to_numpy())
    out = base.assign(logit=np.mean(raw, 0), p=1 / (1 + np.exp(-np.mean(calz, 0))))
    if con:
        out["p_contra"] = np.mean(con, 0)
    return out


def verifier_scores(key: str, cfg, oof_y: pd.DataFrame, which: str) -> tuple[pd.DataFrame, dict]:
    cal = calibrators(key, cfg, oof_y)
    if which == "oof":  # each S1 from the single model that never saw it
        parts = [combine([pd.read_parquet(score_dir(cfg, key, j) / "scores_oof.parquet")], [cal[j]])
                 for j in (0, 1)]
        return pd.concat(parts, ignore_index=True), cal
    if which == "holdout":
        fr = [pd.read_parquet(score_dir(cfg, key, j) / "scores_holdout.parquet") for j in (0, 1)]
    else:
        fr = [pd.concat([pd.read_parquet(p) for p in sorted((score_dir(cfg, key, j) / "test").glob("part-*.parquet"))],
                        ignore_index=True) for j in (0, 1)]
    return combine(fr, [cal[0], cal[1]]), cal


def per_s1_f(sel, ids, gt) -> pd.Series:
    pred = to_sets(sel, ids)
    return pd.Series({s: f05_entity(pred[s], gt[s]) for s in ids})


def run_arm(name, X, y, groups, fold, base, ids_by_fold, tr_ids_by_fold, gt, seed, cfg_xgb, hold_X, hold_base,
            hold_ids):
    oof = np.zeros(len(X), np.float32)
    models = {}
    for k in sorted(ids_by_fold):
        t = time.time()
        tr = fold != k
        models[k] = gbdt.fit_calibrated(X[tr], y[tr], groups[tr], seed=seed, cfg_xgb=cfg_xgb)
        oof[fold == k] = gbdt.predict(models[k], X[fold == k])
        print(f"  [{name}] fold {k} {time.time() - t:.0f}s", flush=True)
    taus, f_thr, f_ef = {}, {}, {}
    for k in sorted(ids_by_fold):
        te, tr = fold == k, fold != k
        taus[k], _ = best_tau(base[tr].assign(p=oof[tr]), gt, tr_ids_by_fold[k])
        ex = exclusive(base[te].assign(p=oof[te]))
        f_thr.update(per_s1_f(decide_threshold(ex, taus[k]), ids_by_fold[k], gt).to_dict())
        f_ef.update(per_s1_f(decide_expected_f(ex), ids_by_fold[k], gt).to_dict())
    f_thr, f_ef = pd.Series(f_thr), pd.Series(f_ef)
    winner = "threshold" if f_thr.mean() > f_ef.mean() else "expected-F"
    f_cv = f_thr if winner == "threshold" else f_ef
    tau_med = float(np.median(list(taus.values())))
    hp = np.mean([gbdt.predict(m, hold_X) for m in models.values()], 0)
    hex_ = exclusive(hold_base.assign(p=hp))
    hsel = decide_threshold(hex_, tau_med) if winner == "threshold" else decide_expected_f(hex_)
    f_hold = per_s1_f(hsel, hold_ids, gt)
    print(f"  [{name}] CV thr {f_thr.mean():.6f} EF {f_ef.mean():.6f} -> {winner}; holdout {f_hold.mean():.6f}",
          flush=True)
    return {"models": models, "oof": oof, "taus": taus, "tau_median": tau_med, "winner": winner,
            "f_cv": f_cv, "f_thr": f_thr, "f_ef": f_ef, "f_hold": f_hold}


def loco(X, y, groups, base, cser, s1_ids, s1_country, gt, seed, cfg_xgb, winner, tau):
    out = {}
    cs = sorted(set(cser))
    for a_ in cs:
        for b_ in cs:
            if a_ == b_:
                continue
            tr, te = cser == a_, cser == b_
            ids = [s for s in s1_ids if s1_country[s] == b_]
            clf = gbdt.fit_calibrated(X[tr], y[tr], groups[tr], seed=seed, cfg_xgb=cfg_xgb)
            ex = exclusive(base[te].assign(p=gbdt.predict(clf, X[te])))
            sel = decide_threshold(ex, tau) if winner == "threshold" else decide_expected_f(ex)
            out[f"{a_}->{b_}"] = float(per_s1_f(sel, ids, gt).mean())
    return out


def summary(f: pd.Series, gt, s1_country, sel_sets=None) -> dict:
    ids = list(f.index)
    n_true = np.array([len(gt[s]) for s in ids])
    r = {"macro": f.mean(), "singleton_mean": f[n_true == 0].mean(), "nonsingleton_mean": f[n_true > 0].mean()}
    for c in sorted({s1_country[s] for s in ids}):
        r[f"macro[{c}]"] = f[[s for s in ids if s1_country[s] == c]].mean()
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p3.yaml")
    ap.add_argument("--model", default="qwen3-rr-0.6b", help="verifier key, or 'none'")
    ap.add_argument("--drop", default="", help="comma-separated features removed from both arms")
    ap.add_argument("--tag", default="e8")
    ap.add_argument("--subindex", action="store_true",
                    help="new arm = E1b features + E6/E7 sub-index features on K15 + sub-index pairs")
    ap.add_argument("--no-loco", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    a, vc = cfg["abs"], cfg["verifier"]
    compute.limit_threads()
    from ber.embed import gpu_setup
    gpu_setup()  # choose the GPU before XGBoost initialises CUDA
    T0 = time.time()

    s1raw, _ = read_source(a["train.source1"], validate=False)
    gt_all, _ = read_ground_truth(a["train.ground_truth"], s1_ids=s1raw["entity_id"])
    s1_country = dict(zip(s1raw["entity_id"], s1raw["country"]))
    dev = pd.read_parquet(a["runs_dir"] / "dev_slice.parquet")
    comp = pd.read_parquet(a["runs_dir"] / "splits.parquet", columns=["s1_id", "component_id"])
    dev = dev.merge(comp, on="s1_id", how="left")
    dev["holdout"] = vf.holdout_mask(dev["component_id"].to_numpy(), vc["holdout_frac"])
    gt = {s: gt_all[s] for s in dev["s1_id"]}
    fold_of = dict(zip(dev["s1_id"], dev["fold"]))
    hold_set = set(dev.loc[dev["holdout"], "s1_id"])

    F, feats0 = load_matrix(cfg)
    drop = [d for d in args.drop.split(",") if d]
    feats_new0 = [f for f in feats0 if f not in drop]  # --drop applies to the NEW arm only (D10)
    assert not (args.subindex and args.model != "none"), "verifier + sub-index in one arm is not supported"
    y_all = np.fromiter((m in gt[s] for s, m in zip(F["s1_id"], F["m_id"])), np.int8, len(F))
    oof_y = F[["s1_id", "m_id"]].assign(y=y_all)
    V = None
    if args.model != "none":
        sc_oof, cal = verifier_scores(args.model, cfg, oof_y, "oof")
        sc_hold, _ = verifier_scores(args.model, cfg, oof_y, "holdout")
        V = vf.verifier_features(F[["s1_id", "m_id"]], pd.concat([sc_oof, sc_hold], ignore_index=True))
        print(f"[{args.tag}] verifier {args.model}: {int(V['v_inband'].sum()):,} band pairs with scores; "
              f"Platt per fold {cal}", flush=True)
    hm = F["s1_id"].isin(hold_set).to_numpy()
    tr_m = ~hm
    base_all = F[["s1_id", "m_id", "source"]]
    fold = F["s1_id"].map(fold_of).to_numpy()
    groups = pd.factorize(F["s1_id"])[0]
    ids_all = [s for s in dev["s1_id"] if s not in hold_set]
    ids_by_fold = {int(k): [s for s in ids_all if fold_of[s] == k] for k in sorted(set(fold))}
    tr_ids_by_fold = {k: [s for s in ids_all if fold_of[s] != k] for k in ids_by_fold}
    hold_ids = [s for s in dev["s1_id"] if s in hold_set]
    cser = F["country"].to_numpy()

    # the new arm's pair set: K15 (+ sub-index pairs with --subindex)
    Fn, extra_cols = F, []
    if args.subindex:
        root = Path(a["cache_dir"]).parent / "subindex"
        sr = pd.read_parquet(root / "dev_subranks.parquet")
        nf = pd.read_parquet(root / "dev_new_features.parquet")
        nf["country"] = nf["s1_id"].map(s1_country)
        Fn = pd.concat([F, nf[F.columns.intersection(nf.columns)]], ignore_index=True)
        Fn = Fn.merge(sr, on=["s1_id", "m_id"], how="left")
        extra_cols = ["e6_rank", "e6_score", "e7_rank", "e7_score"]
        print(f"[{args.tag}] sub-index: {len(nf):,} new pairs; new arm has {len(Fn):,} pairs", flush=True)
    yn = np.fromiter((m in gt[s] for s, m in zip(Fn["s1_id"], Fn["m_id"])), np.int8, len(Fn))
    hmn = Fn["s1_id"].isin(hold_set).to_numpy()
    fold_n = Fn["s1_id"].map(fold_of).to_numpy()
    groups_n = pd.factorize(Fn["s1_id"])[0]
    base_n = Fn[["s1_id", "m_id", "source"]]
    cser_n = Fn["country"].to_numpy()

    arms = {}
    X0 = F[feats0].to_numpy(np.float32)
    arms["base"] = run_arm("base", X0[tr_m], y_all[tr_m], groups[tr_m], fold[tr_m], base_all[tr_m].reset_index(drop=True),
                           ids_by_fold, tr_ids_by_fold, gt, cfg["seed"], cfg.get("xgb"), X0[hm],
                           base_all[hm].reset_index(drop=True), hold_ids)
    feats1 = feats_new0 + (list(V.columns) if V is not None else []) + extra_cols
    X1 = np.hstack([Fn[feats_new0].to_numpy(np.float32)]
                   + ([V.to_numpy(np.float32)] if V is not None else [])
                   + ([Fn[extra_cols].to_numpy(np.float32)] if extra_cols else []))
    has_new = V is not None or bool(extra_cols) or bool(drop)
    trn = ~hmn
    if has_new:
        arms["new"] = run_arm("new", X1[trn], yn[trn], groups_n[trn], fold_n[trn], base_n[trn].reset_index(drop=True),
                              ids_by_fold, tr_ids_by_fold, gt, cfg["seed"], cfg.get("xgb"), X1[hmn],
                              base_n[hmn].reset_index(drop=True), hold_ids)
    else:
        arms["new"] = arms["base"]

    # paired deltas vs the SHIPPED E1b protocol (threshold) on the same S1
    d_cv = (arms["new"]["f_cv"] - arms["base"]["f_cv"]).loc[ids_all].to_numpy()
    d_h = (arms["new"]["f_hold"] - arms["base"]["f_hold"]).loc[hold_ids].to_numpy()
    rng = np.random.default_rng(0)
    boot = np.array([d_cv[rng.integers(0, len(d_cv), len(d_cv))].mean() for _ in range(1000)])
    res = {k: {"cv": summary(v["f_cv"], gt, s1_country), "holdout": v["f_hold"].mean(), "winner": v["winner"],
               "tau_median": v["tau_median"]} for k, v in arms.items()}
    res["delta_cv_pp"] = 100 * d_cv.mean()
    res["ci_pp"] = [100 * np.quantile(boot, 0.025), 100 * np.quantile(boot, 0.975)]
    res["delta_holdout_pp"] = 100 * d_h.mean()
    print(json.dumps(res, indent=1, default=float), flush=True)

    if not args.no_loco:
        inp = {"base": (X0[tr_m], y_all[tr_m], groups[tr_m], base_all[tr_m].reset_index(drop=True), cser[tr_m]),
               "new": (X1[trn], yn[trn], groups_n[trn], base_n[trn].reset_index(drop=True), cser_n[trn])}
        for k in ("base", "new") if has_new else ("base",):
            Xk, yk, gk, bk, ck = inp[k]
            res[k]["loco"] = loco(Xk, yk, gk, bk, ck, ids_all, s1_country, gt, cfg["seed"], cfg.get("xgb"),
                                  arms[k]["winner"], arms[k]["tau_median"])
            print(f"  [{k}] LOCO {res[k]['loco']}", flush=True)
        if not has_new:
            res["new"]["loco"] = res["base"]["loco"]

    # buckets (P1 audit definitions) for both arms on the CV S1
    btab = []
    for k in ("base", "new"):
        arm = arms[k]
        sel = []
        tb, fk = ((base_all[tr_m].reset_index(drop=True), fold[tr_m]) if k == "base" or not has_new
                  else (base_n[trn].reset_index(drop=True), fold_n[trn]))
        for kk in ids_by_fold:
            te = fk == kk
            ex = exclusive(tb[te].assign(p=arm["oof"][te]))
            sel.append(decide_threshold(ex, arm["taus"][kk]) if arm["winner"] == "threshold" else decide_expected_f(ex))
        pred = to_sets(pd.concat(sel), ids_all)
        bb = pd.Series([bucket(pred[s], gt[s]) for s in ids_all]).value_counts()
        loss = pd.Series(1 - arm["f_cv"].loc[ids_all].to_numpy()).groupby([bucket(pred[s], gt[s]) for s in ids_all]).sum()
        for b in bb.index:
            btab.append({"arm": k, "bucket": b, "S1": int(bb[b]), "macro pts lost": loss.get(b, 0) / len(ids_all)})
        tp = sum(len(pred[s] & gt[s]) for s in ids_all)
        res[k]["micro_p"] = tp / max(1, sum(len(pred[s]) for s in ids_all))
        res[k]["micro_r"] = tp / max(1, sum(len(gt[s]) for s in ids_all))
    btab = pd.DataFrame(btab).pivot_table(index="bucket", columns="arm", values=["S1", "macro pts lost"]).reset_index()
    btab.columns = [" ".join(c).strip() for c in btab.columns]

    # bundle for inference
    new = arms["new"]
    bundle = {"folds": new["models"], "features": feats1, "fold_taus": new["taus"], "k_max": cfg["versions"]["K"],
              "winner": f"xgb/{new['winner']}", "tau_median": new["tau_median"],
              "cv": {"macro": new["f_ef"].mean(), "macro_threshold": new["f_thr"].mean()},
              "loco": res["new"].get("loco", {}), "holdout": new["f_hold"].mean(),
              "verifier": None if V is None else {"key": args.model, "band": vc["band"], "platt": cal,
                                                  "scores": vc.get("scores", "")},
              "trained_on": "non-holdout 85% of dev slice", "dropped": drop,
              "subindex": bool(extra_cols)}
    mdir = Path(a["models_dir"])
    joblib.dump(bundle, mdir / f"{args.tag}.joblib")
    (base_n[trn] if has_new else base_all[tr_m]).reset_index(drop=True).assign(
        p=new["oof"], y=(yn[trn] if has_new else y_all[tr_m]),
        fold=(fold_n[trn] if has_new else fold[tr_m])).to_parquet(mdir / f"{args.tag}_oof.parquet", index=False)
    imp = gbdt.importances(new["models"][0], feats1)
    top = pd.DataFrame(sorted(imp.items(), key=lambda kv: -kv[1]), columns=["feature", "gain"]).head(20)

    rows = []
    for k in ("base", "new"):
        r = res[k]
        rows.append({"arm": k, "decision": r["winner"], "CV macro": r["cv"]["macro"],
                     **{kk: vv for kk, vv in r["cv"].items() if kk.startswith("macro[")},
                     "singleton": r["cv"]["singleton_mean"], "non-singleton": r["cv"]["nonsingleton_mean"],
                     "micro P": r["micro_p"], "micro R": r["micro_r"], "holdout": r["holdout"],
                     **{f"LOCO {kk}": vv for kk, vv in r.get("loco", {}).items()}})
    rep = a["reports_dir"] / f"p3_{args.tag}.md"
    rep.write_text("\n".join([
        f"# P3 {args.tag}: E1b vs E1b " + ("+ verifier " + args.model if V is not None else
                                            "+ E6/E7 sub-index pairs" if extra_cols else f"minus {drop}") + "\n",
        f"- Both arms refitted on the non-holdout {len(ids_all):,} dev S1 (same 5 folds); holdout "
        f"{len(hold_ids):,} S1 scored like test. Dropped from the new arm: {drop or 'none'}.",
        f"- **dCV {res['delta_cv_pp']:+.3f} pp** (95% CI [{res['ci_pp'][0]:+.3f}, {res['ci_pp'][1]:+.3f}]); "
        f"**d holdout {res['delta_holdout_pp']:+.3f} pp**.\n",
        md(pd.DataFrame(rows), ".4f"), "\n## Error buckets (CV S1, P1 audit definitions)\n", md(btab, ".4f"),
        "\n## Feature importance (gain, fold 0, new arm)\n", md(top, ".5f"),
        f"\nWall time {(time.time() - T0) / 60:.1f} min."]), encoding="utf-8")
    (mdir / f"{args.tag}_result.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {rep}")
    if not args.no_ledger:
        r = res["new"]
        cs = sorted({s1_country[s] for s in ids_all})
        append_run(a["runs_ledger"], f"p3_{args.tag}",
                   {"stage": f"p3_{args.tag}", "model": f"xgboost_{r['winner']}",
                    "verifier_model": MODELS[args.model]["hf"] if V is not None else "",
                    "verifier_band": "-".join(map(str, vc["band"])) if V is not None else "",
                    "cv_macro_f05": r["cv"]["macro"], "cv_india": r["cv"].get(f"macro[{cs[0]}]"),
                    "cv_us": r["cv"].get(f"macro[{cs[-1]}]"), "singleton_mean": r["cv"]["singleton_mean"],
                    "nonsingleton_mean": r["cv"]["nonsingleton_mean"], "micro_p": r["micro_p"],
                    "micro_r": r["micro_r"], "loco_in_to_us": r.get("loco", {}).get(f"{cs[0]}->{cs[-1]}"),
                    "loco_us_to_in": r.get("loco", {}).get(f"{cs[-1]}->{cs[0]}"), "device": "cuda",
                    "n_pairs_dev": int(trn.sum()), "feature_version": cfg["versions"]["feature_version"],
                    "wall_s": round(time.time() - T0, 1),
                    "config_hash": config_hash({k: x for k, x in cfg.items() if k != "abs"})},
                   notes=f"{args.tag}: dCV {res['delta_cv_pp']:+.3f} pp CI {res['ci_pp']}, "
                         f"d holdout {res['delta_holdout_pp']:+.3f} pp (paired vs E1b refit on same 85%)",
                   extra={"base": res["base"], "holdout_new": r["holdout"]})


if __name__ == "__main__":
    main()
