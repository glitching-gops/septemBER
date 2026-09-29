"""E1: stage-1 XGBoost pair model on the sharded feature cache -> reports/p2_e1.md

    python scripts/run_e1.py                       # 5-fold grouped CV + LOCO
    python scripts/run_e1.py --baseline             # also score the P1 linear model for the delta

Same CV design as P1: the P0 component-grouped folds, exclusivity inside each held-out
fold, then the expected-F0.5 prefix decision. Only the scorer changes, so the CV delta is
attributable to the model and not to the protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from ber import compute  # noqa: E402
from ber.decide.sets import decide_expected_f, decide_threshold, exclusive, to_sets  # noqa: E402
from ber.evaluate.ledger import append_run, config_hash  # noqa: E402
from ber.evaluate.metric import f05_entity, macro_f05_fast  # noqa: E402
from ber.io_utils import load_config, read_ground_truth, read_source  # noqa: E402
from ber.match import gbdt  # noqa: E402
from ber.match.features import FEATURES  # noqa: E402
from run_features import load_candidates, load_features  # noqa: E402

DEAD = ["tfidf_name_cos", "b4_hit"]  # measured zero-coefficient under the dense-only pipeline
TAU_GRID = np.round(np.arange(0.20, 0.91, 0.025), 3)


def md(df, floatfmt=".4f"):
    return tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=floatfmt)


def active_features(columns) -> list[str]:
    return [f for f in FEATURES if f not in DEAD and f in columns] + \
           [c for c in columns if c not in FEATURES and c.startswith("g_")]


def load_matrix(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Sharded dev features realigned to candidate order, with keys."""
    cand = load_candidates(cfg, "dev")
    F = load_features(cfg, "dev")
    pos = pd.Series(np.arange(len(cand)), index=pd.MultiIndex.from_arrays(
        [cand["s1_id"].to_numpy(), cand["m_id"].to_numpy()]))
    idx = pos.loc[pd.MultiIndex.from_arrays(
        [F["s1_id"].to_numpy(), F["m_id"].to_numpy()])].to_numpy()
    F = F.iloc[np.argsort(idx, kind="stable")].reset_index(drop=True)
    assert (F["m_id"].to_numpy() == cand["m_id"].to_numpy()).all()
    F["country"] = cand["country"].to_numpy()
    return F, active_features(F.columns)


def evaluate(sel: pd.DataFrame, gt, s1_ids, s1_country) -> dict:
    pred = to_sets(sel, s1_ids)
    f = np.array([f05_entity(pred[s], gt[s]) for s in s1_ids])
    n_true = np.array([len(gt[s]) for s in s1_ids])
    tp = sum(len(pred[s] & gt[s]) for s in s1_ids)
    npred = sum(len(pred[s]) for s in s1_ids)
    r = {"macro": float(f.mean()),
         "singleton_mean": float(f[n_true == 0].mean()) if (n_true == 0).any() else float("nan"),
         "nonsingleton_mean": float(f[n_true > 0].mean()) if (n_true > 0).any() else float("nan"),
         "micro_p": tp / npred if npred else float("nan"),
         "micro_r": tp / max(1, int(n_true.sum())),
         "pred_per_s1": npred / len(s1_ids),
         "pred_empty": float(np.mean([len(pred[s]) == 0 for s in s1_ids]))}
    for c in sorted({s1_country[s] for s in s1_ids}):
        ids = [s for s in s1_ids if s1_country[s] == c]
        r[f"macro[{c}]"] = float(np.mean([f05_entity(pred[s], gt[s]) for s in ids]))
    return r


def best_tau(pairs: pd.DataFrame, gt, s1_ids) -> tuple[float, float]:
    ex = exclusive(pairs)
    ts = [(s, m) for s in s1_ids for m in gt[s]]
    t_s, t_m = [a for a, _ in ts], [b for _, b in ts]
    best, arg = -1.0, 0.5
    for t in TAU_GRID:
        s = ex[ex["p"] >= t]
        v = macro_f05_fast(s["s1_id"].to_numpy(), s["m_id"].to_numpy(), t_s, t_m, s1_ids)
        if v > best:
            best, arg = v, float(t)
    return arg, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2.yaml")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--feature-version", default=None, help="override versions.feature_version")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.feature_version:
        cfg["versions"]["feature_version"] = args.feature_version
    a = cfg["abs"]
    compute.limit_threads()
    T0 = time.time()

    s1raw, _ = read_source(a["train.source1"])
    gt_all, _ = read_ground_truth(a["train.ground_truth"], s1_ids=s1raw["entity_id"])
    s1_country = dict(zip(s1raw["entity_id"], s1raw["country"]))
    dev = pd.read_parquet(a["runs_dir"] / "dev_slice.parquet")
    s1_ids = dev["s1_id"].tolist()
    fold_of = dict(zip(dev["s1_id"], dev["fold"]))
    gt = {s: gt_all[s] for s in s1_ids}

    F, feats = load_matrix(cfg)
    print(f"[e1] {len(F):,} pairs x {len(feats)} features (dropped dead: {DEAD})", flush=True)
    y = np.fromiter((m in gt[s] for s, m in zip(F["s1_id"], F["m_id"])), np.int8, len(F))
    fold = F["s1_id"].map(fold_of).to_numpy()
    groups = pd.factorize(F["s1_id"])[0]
    X = F[feats].to_numpy(np.float32)
    base = F[["s1_id", "m_id", "source"]].copy()
    print(f"[e1] positives {y.sum():,} ({y.mean():.3%}); features ready {time.time() - T0:.0f}s",
          flush=True)

    oof = np.zeros(len(F), np.float32)
    models = {}
    for k in sorted(set(fold)):
        t0 = time.time()
        tr = fold != k
        clf = gbdt.fit_calibrated(X[tr], y[tr], groups[tr], seed=cfg["seed"],
                                  cfg_xgb=cfg.get("xgb"))
        models[int(k)] = clf
        oof[fold == k] = gbdt.predict(clf, X[fold == k])
        print(f"  fold {k}: fitted+scored in {time.time() - t0:.0f}s", flush=True)

    rows, taus = [], {}
    for k in sorted(set(fold)):
        te, tr = fold == k, fold != k
        te_ids = [s for s in s1_ids if fold_of[s] == k]
        tr_ids = [s for s in s1_ids if fold_of[s] != k]
        tau, _ = best_tau(base[tr].assign(p=oof[tr]), gt, tr_ids)
        taus[int(k)] = tau
        ex = exclusive(base[te].assign(p=oof[te]))
        rows.append({"fold": int(k), "tau": tau,
                     "threshold": evaluate(decide_threshold(ex, tau), gt, te_ids, s1_country)["macro"],
                     **{f"EF_{kk}": v for kk, v in
                        evaluate(decide_expected_f(ex), gt, te_ids, s1_country).items()}})
        print(f"  fold {k}: thr {rows[-1]['threshold']:.4f}  EF {rows[-1]['EF_macro']:.4f}", flush=True)

    per_fold = pd.DataFrame(rows)
    summary = {"macro": per_fold["EF_macro"].mean(), "std": per_fold["EF_macro"].std(ddof=0),
               "macro_threshold": per_fold["threshold"].mean()}
    for c in [c for c in per_fold.columns if c.startswith("EF_")]:
        summary[c[3:]] = per_fold[c].mean()
    print(f"\n[e1] CV macro F0.5 (expected-F) = {summary['macro']:.6f} "
          f"+/- {summary['std']:.6f}   (P1 = 0.895607, delta {summary['macro'] - 0.895607:+.6f})")

    # LOCO: fit on one training country, evaluate on the other
    loco = {}
    countries = sorted({s1_country[s] for s in s1_ids})
    cser = F["country"].to_numpy()
    for tr_c in countries:
        for te_c in countries:
            if tr_c == te_c:
                continue
            tr, te = cser == tr_c, cser == te_c
            te_ids = [s for s in s1_ids if s1_country[s] == te_c]
            clf = gbdt.fit_calibrated(X[tr], y[tr], groups[tr], seed=cfg["seed"], cfg_xgb=cfg.get("xgb"))
            p = gbdt.predict(clf, X[te])
            r = evaluate(decide_expected_f(exclusive(base[te].assign(p=p))), gt, te_ids, s1_country)
            loco[f"{tr_c}->{te_c}"] = r["macro"]
            print(f"  LOCO train {tr_c} -> eval {te_c}: {r['macro']:.4f}", flush=True)

    imp = gbdt.importances(models[0], feats)
    top = pd.DataFrame(sorted(imp.items(), key=lambda kv: -kv[1]), columns=["feature", "gain"])

    out = a["reports_dir"] / f"p2_{args.tag}.md"
    L = [f"# P2 {args.tag.upper()}: stage-1 XGBoost pair model\n",
         f"- Dev slice {len(s1_ids):,} S1, {len(F):,} candidate pairs at K={cfg['versions']['K']}, "
         f"{y.sum():,} positive ({y.mean():.2%}).",
         f"- Features: {len(feats)} (P1 set minus the measured-dead {DEAD}).",
         f"- Model: XGBoost {cfg['xgb']['n_estimators']} trees, depth {cfg['xgb']['max_depth']}, "
         f"device=cuda, isotonic OOF calibration.\n",
         "## CV summary (mean over 5 folds)\n",
         md(pd.DataFrame([{"scorer/decision": "xgb/threshold", "macro F0.5": summary["macro_threshold"]},
                          {"scorer/decision": "xgb/expected-F", "macro F0.5": summary["macro"]}])),
         "\n## Per fold\n", md(per_fold),
         "\n## LOCO\n", md(pd.DataFrame([{"split": k, "macro F0.5": v} for k, v in loco.items()])),
         "\n## Feature importance (gain, fold 0)\n", md(top.head(25), ".5f")]
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {out}")

    import joblib
    mdir = Path(a["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    # pick the decision rule that actually won on CV rather than assuming expected-F:
    # with a sharper scorer the plain threshold can beat the prefix search (E1a did).
    winner = ("xgb/expected-F" if summary["macro"] >= summary["macro_threshold"]
              else "xgb/threshold")
    bundle = {"folds": models, "features": feats, "fold_taus": taus, "k_max": cfg["versions"]["K"],
              "winner": winner, "tau_median": float(np.median(list(taus.values()))),
              "cv": summary, "loco": loco}
    print(f"[e1] winner: {winner} "
          f"(expected-F {summary['macro']:.6f} vs threshold {summary['macro_threshold']:.6f})")
    joblib.dump(bundle, mdir / f"{args.tag}.joblib")
    np.save(mdir / f"{args.tag}_oof.npy", oof)
    base.assign(p=oof, y=y, fold=fold).to_parquet(mdir / f"{args.tag}_oof.parquet", index=False)
    print(f"wrote {mdir / f'{args.tag}.joblib'}")

    if not args.no_ledger:
        v = cfg["versions"]
        append_run(a["runs_ledger"], f"p2_{args.tag}",
                   metrics={"stage": args.tag, "model": f"xgboost_{winner.split('/')[-1]}",
                            # log the SHIPPED rule's score, not always the expected-F one
                            "cv_macro_f05": (summary["macro"] if winner.endswith("expected-F")
                                             else summary["macro_threshold"]), "cv_macro_f05_mean": summary["macro"],
                            "cv_macro_f05_std": summary["std"],
                            "cv_india": summary.get(f"macro[{countries[0]}]"),
                            "cv_us": summary.get(f"macro[{countries[-1]}]"),
                            "singleton_mean": summary["singleton_mean"],
                            "nonsingleton_mean": summary["nonsingleton_mean"],
                            "micro_p": summary["micro_p"], "micro_r": summary["micro_r"],
                            "loco_us_to_in": loco.get(f"{countries[-1]}->{countries[0]}"),
                            "loco_in_to_us": loco.get(f"{countries[0]}->{countries[-1]}"),
                            "n_pairs_dev": len(F), "feature_version": v["feature_version"],
                            "norm_version": v["norm_version"], "blocker_set": v["blocker_set"],
                            "K": v["K"], "wall_s": round(time.time() - T0, 1), "device": "cuda",
                            "config_hash": config_hash({k: x for k, x in cfg.items() if k != "abs"})},
                   notes=f"E1 stage-1 XGBoost; CV {summary['macro']:.6f} vs P1 0.895607 "
                         f"(delta {summary['macro'] - 0.895607:+.6f})",
                   extra={"loco": loco, "importance_top10": dict(list(zip(top['feature'], top['gain']))[:10])})
    print(f"[e1] done in {(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
