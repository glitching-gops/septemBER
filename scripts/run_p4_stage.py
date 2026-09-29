"""P4 stage-1 / stage-2 / decision on the dev slice, GPU-only, paired against sub3 (the fallback).

    python scripts/run_p4_stage.py stage1 --verifier bge-rr-v2-m3-r2 --tag s1_r2
    python scripts/run_p4_stage.py stage2 --stage1 s1_r2 --tag s2_r2 [--no-c2c] [--drop-m-side]
    python scripts/run_p4_stage.py decide --arm s2_r2

All computation is torch on one GPU (``ber.gpu``); the host only reads / writes files and formats
the report. Protocol = P3's (5 component-grouped folds on the non-holdout 85%, per-fold tau from
the other folds' OOF, holdout = mean of fold models + median tau), with a 426-value tau grid.

stage1 (D5): E1b's 40 features + verifier features [v_logit, v_p, v_inband, v_rank, v_margin,
v_rescued]; Platt per verifier fold model on its own OOF pairs (LBFGS on GPU); holdout / test use
the mean of both fold models' calibrated logits.
stage2 (D5): [stage-1 p, logit, verifier features, group features, c2c features]; context from
stage-1 p (OOF for CV S1, fold-mean for holdout). LOCO is stacked (see ``loco_stacked``).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from ber import compute  # noqa: E402
from ber.evaluate.ledger import append_run, config_hash  # noqa: E402
from ber.gpu import StageTimer, calib, search  # noqa: E402
from ber.gpu import decide as gd  # noqa: E402
from ber.gpu import metric as gm  # noqa: E402
from ber.gpu import pipeline as pl  # noqa: E402
from ber.gpu import segment as sg  # noqa: E402
from ber.gpu.xgb import fit_calibrated  # noqa: E402
from ber.io import entity_attrs as EA  # noqa: E402
from ber.io import gpu_load as gl  # noqa: E402
from ber.io import ids as I  # noqa: E402
from ber.io_utils import load_config  # noqa: E402
from ber.match.context import C2C_FEATURES, GROUP_FEATURES, c2c_features, group_features  # noqa: E402

M_SIDE = ["grp_m_best_oth", "grp_m_rank", "grp_m_n"]
DEV = None


def p4_dir(cfg) -> Path:
    d = Path(cfg["abs"]["models_dir"]) / "p4"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vdir(cfg, key) -> Path:
    return Path(cfg["abs"]["cache_dir"]).parent / "verifier" / key


EXTRA = {"on": False, "name": None}
PROV = ["rr_src_flags", "rr_n_sources", "rr_lx4_score", "rr_lx4_rank", "rr_lx3a_score", "rr_lx3a_rank",
        "rr_lx1_score", "rr_lx1_rank", "rr_h2_parent_p", "rr_h2_parent_src", "rr_h2_flags"]


def e7dir(cfg) -> Path:
    return Path(cfg["abs"]["cache_dir"]).parent / (EXTRA["name"] or "e7")


def load_dev(cfg, timer) -> tuple[pl.Dev, torch.Tensor, list]:
    """Read + transfer the dev pairs and metadata (permitted I/O phase, timed as such)."""
    with timer.host("io"):
        return _load_dev(cfg)


def _load_dev(cfg) -> tuple[pl.Dev, torch.Tensor, list]:
    feats0 = joblib.load(Path(cfg["abs"]["models_dir"]) / "e1b.joblib")["features"]
    L = gl.load_pairs(cfg, "dev", "train", feats0, DEV)
    if EXTRA["on"]:  # extra pairs (E7 survivors / R2 union) appended after the K15 pairs
        rr = EXTRA["name"] == "rr"
        cols = feats0 + (PROV if rr else [])
        E = gl.pair_file(cfg, "train", e7dir(cfg) / "dev_feats.parquet", f"{EXTRA['name']}_dev_feats", cols, DEV)
        src_e = gl.entity_src(cfg, "train", DEV)
        XE = torch.stack([E[c] for c in cols], 1)
        XK = L["X"]
        if rr:  # K15 rows: no new source (flags 0, count 0), provenance scores NaN
            pk = torch.full((XK.shape[0], len(PROV)), float("nan"), device=DEV)
            pk[:, :2] = 0.0
            XK = torch.cat([XK, pk], 1)
            feats0 = feats0 + PROV
        L = {"s1": torch.cat([L["s1"], E["s1"]]), "m": torch.cat([L["m"], E["m"]]),
             "src": torch.cat([L["src"], src_e[E["m"]]]), "X": torch.cat([XK, XE])}
    n_ent = len(I.entity_table(cfg, "train"))
    gt = gl.gt_dev(cfg, DEV)
    meta = gl.dev_meta(cfg, DEV)
    fold_e = torch.full((n_ent,), -1, dtype=torch.int64, device=DEV)
    fold_e[meta["s1"]] = meta["fold"]
    hold_e = torch.zeros(n_ent, dtype=torch.bool, device=DEV)
    hold_e[meta["s1"]] = meta["holdout"]
    cv_ids = meta["s1"][~meta["holdout"]]
    D = pl.Dev(s1=L["s1"], m=L["m"], src=L["src"], y=gm.labels(L["s1"], L["m"], gt["s1"], gt["m"], n_ent),
               fold=fold_e[L["s1"]], hold=hold_e[L["s1"]], country=gl.entity_country(cfg, "train", DEV),
               ntrue=gm.n_true(gt["s1"], n_ent), cv_ids=cv_ids, hold_ids=meta["s1"][meta["holdout"]],
               ids_by_fold={int(k): cv_ids[fold_e[cv_ids] == k] for k in torch.unique(meta["fold"]).tolist()},
               n_ent=n_ent, labels=I.country_labels(cfg, "train"))
    assert bool((D.fold >= 0).all()), "dev pair whose S1 has no fold"
    return D, L["X"], list(feats0)


def verifier_block(cfg, D: pl.Dev, key: str, timer) -> tuple[torch.Tensor, dict]:
    """V [n, 6] for the dev pairs + info (Platt per fold, KS)."""
    vd = vdir(cfg, key)
    r2 = key == cfg["verifier"]["key"]
    sub = "det"
    oof_parts, cal, raw_oof = [], {}, {}
    for j in (0, 1):
        f = vd / f"fold{j}" / sub / "scores_oof.parquet"
        t, tc = time.time(), time.process_time()
        s = gl.pair_file(cfg, "train", f, f"{key}_f{j}_oof", ["logit"], DEV)
        fe = vd / f"fold{j}" / sub / f"scores_oof_{EXTRA['name']}.parquet"
        if EXTRA["on"] and r2:
            e = gl.pair_file(cfg, "train", fe, f"{key}_f{j}_oof_{EXTRA['name']}", ["logit"], DEV)
            s = {k: torch.cat([s[k], e[k]]) for k in s}
        timer.io(time.time() - t, time.process_time() - tc)
        y = gm.labels(s["s1"], s["m"], *gt_tensors(cfg), D.n_ent)
        cal[j] = calib.platt_fit(s["logit"], y)
        s["z"] = cal[j][0] * s["logit"].to(torch.float64) + cal[j][1]
        oof_parts.append(s)
        raw_oof[j] = torch.sigmoid(s["z"])
    ks_cal = calib.ks(raw_oof[0], raw_oof[1])
    ks_raw = calib.ks(oof_parts[0]["logit"], oof_parts[1]["logit"])
    with timer.host("io"):
        hp = [gl.pair_file(cfg, "train", vd / f"fold{j}" / sub / "scores_holdout.parquet", f"{key}_f{j}_hold", ["logit"], DEV)
              for j in (0, 1)]
        if EXTRA["on"] and r2:
            for j in (0, 1):
                e = gl.pair_file(cfg, "train", vd / f"fold{j}" / sub / f"scores_holdout_{EXTRA['name']}.parquet",
                                 f"{key}_f{j}_hold_{EXTRA['name']}", ["logit"], DEV)
                hp[j] = {k: torch.cat([hp[j][k], e[k]]) for k in hp[j]}
    j1 = sg.lookup_pairs(hp[0]["s1"], hp[0]["m"], hp[1]["s1"], hp[1]["m"], D.n_ent)
    assert bool((j1 >= 0).all()), "holdout scored sets differ between fold models"
    hz = ((cal[0][0] * hp[0]["logit"] + cal[0][1]) + (cal[1][0] * hp[1]["logit"][j1] + cal[1][1])) / 2
    hl = (hp[0]["logit"] + hp[1]["logit"][j1]) / 2
    sc_s1 = torch.cat([oof_parts[0]["s1"], oof_parts[1]["s1"], hp[0]["s1"]])
    sc_m = torch.cat([oof_parts[0]["m"], oof_parts[1]["m"], hp[0]["m"]])
    logit = torch.cat([oof_parts[0]["logit"], oof_parts[1]["logit"], hl]).to(torch.float64)
    pcal = torch.sigmoid(torch.cat([oof_parts[0]["z"], oof_parts[1]["z"], hz.to(torch.float64)]))
    resc = torch.zeros_like(pcal)
    if r2:
        with timer.host("io"):
            ds = gl.pair_file(cfg, "train", vdir(cfg, key) / "dev_set.parquet", "dev_set_resc", ["rescued", "inband"], DEV)
            if EXTRA["on"]:
                e = gl.pair_file(cfg, "train", e7dir(cfg) / f"dev_{EXTRA['name']}_set.parquet", f"{EXTRA['name']}_dev_set",
                                 ["rescued", "inband"], DEV)
                ds = {k: torch.cat([ds[k], e[k]]) for k in ds}
        k = sg.lookup_pairs(sc_s1, sc_m, ds["s1"], ds["m"], D.n_ent)
        assert bool((k >= 0).all()), "scored pair outside the dev scoring set"
        resc = ds["rescued"][k] * (1 - ds["inband"][k])  # rescue-only pairs
    if EXTRA.get("no_rescue") and r2:  # D4 rescue ablation: rescue-only pairs count as unscored
        keep = resc == 0
        sc_s1, sc_m, logit, pcal, resc = sc_s1[keep], sc_m[keep], logit[keep], pcal[keep], resc[keep]
    V = pl.verifier_features(D.s1, D.m, sc_s1, sc_m, logit, pcal, resc, D.n_ent)
    info = {"key": key, "platt": cal, "ks_oof_calibrated": ks_cal, "ks_oof_raw": ks_raw,
            "n_scored": int(V[:, 2].sum()), "n_rescue_only": int(V[:, 5].sum())}
    return V, info


_GT = {}


def gt_tensors(cfg):
    if "gt" not in _GT:
        g = gl.gt_dev(cfg, DEV)
        _GT["gt"] = (g["s1"], g["m"])
    return _GT["gt"]


def sub3_ref(cfg, D: pl.Dev, timer) -> dict:
    """sub3's per-entity F0.5: OOF p with its per-fold taus (CV S1); fold-mean p + tau median (holdout)."""
    b3 = joblib.load(Path(cfg["abs"]["models_dir"]) / "e8_bge_det.joblib")
    path = vdir(cfg, cfg["verifier"]["key"]) / "dev_pairs_p.parquet"
    t, tc = time.time(), time.process_time()
    s = gl.pair_file(cfg, "train", path, "dev_pairs_p", ["p_s3"], DEV)
    timer.io(time.time() - t, time.process_time() - tc)
    j = sg.lookup_pairs(D.s1, D.m, s["s1"], s["m"], D.n_ent)
    assert EXTRA["on"] or bool((j >= 0).all())
    p = torch.where(j >= 0, s["p_s3"][j.clamp(min=0)], torch.zeros_like(s["p_s3"][:1]).expand(j.numel()))  # E7 pairs: not sub3's
    f_cv = gm.per_s1_f(pl.select_cv(D, p, b3["fold_taus"]), D.s1, D.y, D.ntrue)
    f_h = gm.per_s1_f(pl.select_hold(D, p, b3["tau_median"]), D.s1, D.y, D.ntrue)
    res = json.loads((Path(cfg["abs"]["models_dir"]) / "e8_bge_det_result.json").read_text())
    return {"f_cv": f_cv, "f_hold": f_h, "p": p, "loco": res["new"]["loco"], "taus": b3["fold_taus"],
            "tau_median": b3["tau_median"]}


def save_p(path: Path, D: pl.Dev, p: torch.Tensor) -> None:
    pq.write_table(pa.table({"s1": D.s1.cpu().numpy(), "m": D.m.cpu().numpy(), "p": p.cpu().numpy(),
                             "hold": D.hold.cpu().numpy()}), path)


def load_p(path: Path, D: pl.Dev) -> torch.Tensor:
    t = pq.read_table(path)
    s1, m = gl.to_dev(t.column("s1").to_numpy(), DEV), gl.to_dev(t.column("m").to_numpy(), DEV)
    p = gl.to_dev(t.column("p").to_numpy(), DEV)
    j = sg.lookup_pairs(D.s1, D.m, s1, m, D.n_ent)
    assert bool((j >= 0).all())
    return p[j]


def save_f(path: Path, D: pl.Dev, f: torch.Tensor) -> None:
    ids = torch.cat([D.cv_ids, D.hold_ids])
    pq.write_table(pa.table({"s1": ids.cpu().numpy(), "f": f[ids].cpu().numpy(),
                             "hold": torch.cat([torch.zeros_like(D.cv_ids), torch.ones_like(D.hold_ids)]).cpu().numpy()}),
                   path)


def load_f(path: Path, D: pl.Dev) -> torch.Tensor:
    t = pq.read_table(path)
    f = torch.full((D.n_ent,), float("nan"), dtype=torch.float64, device=DEV)
    f[gl.to_dev(t.column("s1").to_numpy(), DEV)] = gl.to_dev(t.column("f").to_numpy(), DEV)
    return f


def finish(args, cfg, D, arm, X, feats, ref, timer, stage, verifier, prev_f=None, loco=None) -> dict:
    """Paired comparisons, buckets, bundle, report, ledger."""
    out = p4_dir(cfg)
    f_cv, f_h = arm["f_cv"], arm["f_hold"]
    cvm = torch.zeros(D.n_ent, dtype=torch.bool, device=DEV)
    cvm[D.cv_ids] = True
    sel_cv = pl.select_cv(D, arm["oof"], arm["taus"], arm["winner"])
    res = {"cv": gm.breakdown(f_cv, D.cv_ids, D.country, D.ntrue, D.labels), "holdout": gm.macro(f_h, D.hold_ids),
           "winner": arm["winner"], "cv_threshold": gm.macro(arm["f_thr"], D.cv_ids),
           "cv_expected_f": gm.macro(arm["f_ef"], D.cv_ids), "tau_median": arm["tau_median"],
           "taus": arm["taus"], "features": feats}
    res["delta_cv_vs_sub3_pp"], res["ci_vs_sub3_pp"] = pl.paired(f_cv, ref["f_cv"], D.cv_ids)
    res["delta_holdout_vs_sub3_pp"] = 100 * float((f_h[D.hold_ids] - ref["f_hold"][D.hold_ids]).mean())
    res["sub3"] = {"cv": gm.macro(ref["f_cv"], D.cv_ids), "holdout": gm.macro(ref["f_hold"], D.hold_ids),
                   "loco": ref["loco"]}
    if prev_f is not None:
        res["delta_cv_vs_stage1_pp"], res["ci_vs_stage1_pp"] = pl.paired(f_cv, prev_f, D.cv_ids)
        res["delta_holdout_vs_stage1_pp"] = 100 * float((f_h[D.hold_ids] - prev_f[D.hold_ids]).mean())
    res["micro_p"], res["micro_r"] = gm.micro(sel_cv, D.s1, D.y, D.ntrue, cvm)
    res["holdout_minus_cv_pp"] = 100 * (res["holdout"] - res["cv"]["macro"])
    res["loco"] = loco if loco is not None else ({} if args.no_loco else pl.loco_plain(
        D, X, D.country[D.s1], arm["tau_median"], cfg["seed"], cfg.get("xgb")))
    sel3 = pl.select_cv(D, ref["p"], ref["taus"])
    res["buckets"] = {"sub3": pl.buckets(D, sel3, ref["f_cv"], D.cv_ids), args.tag: pl.buckets(D, sel_cv, f_cv, D.cv_ids)}
    res["verifier"] = {k: v for k, v in verifier.items() if k != "platt"}
    res.update(timer.summary())
    print(json.dumps({k: v for k, v in res.items() if k not in ("features", "taus")}, indent=1, default=float), flush=True)
    bundle = {"folds": arm["models"], "features": feats, "fold_taus": arm["taus"], "tau_median": arm["tau_median"],
              "winner": arm["winner"], "cv": {"macro_threshold": res["cv_threshold"], "macro_expected_f": res["cv_expected_f"]},
              "holdout": res["holdout"], "loco": res["loco"], "verifier": verifier, "stage": stage,
              "trained_on": "non-holdout 85% of dev slice"}
    with timer.host("io"):
        with open(out / f"{args.tag}.pkl", "wb") as fh:
            pickle.dump(bundle, fh)
        save_p(out / f"{args.tag}_p.parquet", D, arm["p_all"])
        save_f(out / f"{args.tag}_f.parquet", D, f_cv.where(cvm, f_h))
        (out / f"{args.tag}_result.json").write_text(json.dumps(res, indent=1, default=float))
    imp = arm["models"][0].importances(feats)
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:25]
    rows = [{"arm": "sub3 (fallback)", "CV": res["sub3"]["cv"], "holdout": res["sub3"]["holdout"],
             **{f"LOCO {k}": v for k, v in ref["loco"].items()}},
            {"arm": args.tag, "CV": res["cv"]["macro"], **{k: v for k, v in res["cv"].items() if k.startswith("macro[")},
             "holdout": res["holdout"], **{f"LOCO {k}": v for k, v in res["loco"].items()},
             "micro P": res["micro_p"], "micro R": res["micro_r"]}]
    bt = [{"bucket": b, **{f"{a} S1": res["buckets"][a][b][0] for a in res["buckets"]},
           **{f"{a} pts lost": res["buckets"][a][b][1] for a in res["buckets"]}} for b in pl.BUCKETS]
    lines = [f"# P4 {args.tag} ({stage}, GPU-only)\n",
             f"- dCV vs sub3 **{res['delta_cv_vs_sub3_pp']:+.3f} pp** (95% CI [{res['ci_vs_sub3_pp'][0]:+.3f}, "
             f"{res['ci_vs_sub3_pp'][1]:+.3f}]); d holdout vs sub3 **{res['delta_holdout_vs_sub3_pp']:+.3f} pp**."]
    if prev_f is not None:
        lines.append(f"- dCV vs stage-1 **{res['delta_cv_vs_stage1_pp']:+.3f} pp** (CI [{res['ci_vs_stage1_pp'][0]:+.3f}, "
                     f"{res['ci_vs_stage1_pp'][1]:+.3f}]); d holdout vs stage-1 **{res['delta_holdout_vs_stage1_pp']:+.3f} pp**.")
    lines += [f"- decision on CV: {arm['winner']} (threshold {res['cv_threshold']:.6f}, expected-F "
              f"{res['cv_expected_f']:.6f}); tau median {arm['tau_median']:.3f}; holdout - CV {res['holdout_minus_cv_pp']:+.3f} pp.",
              f"- verifier {verifier.get('key')}: {res['verifier']}",
              f"- time: {timer.summary()}\n",
              tabulate(pd.DataFrame(rows), headers="keys", tablefmt="github", showindex=False, floatfmt=".5f"),
              "\n## Error buckets (CV S1)\n",
              tabulate(pd.DataFrame(bt), headers="keys", tablefmt="github", showindex=False, floatfmt=".5f"),
              "\n## Feature importance (gain, fold 0)\n",
              tabulate(pd.DataFrame(top, columns=["feature", "gain"]), headers="keys", tablefmt="github",
                       showindex=False, floatfmt=".5f")]
    (Path(cfg["abs"]["reports_dir"]) / f"p4_{args.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    st = timer.summary()
    lab = D.labels
    append_run(cfg["abs"]["runs_ledger"], f"p4_{args.tag}",
               {"stage": f"p4_{stage}", "model": f"xgboost_{arm['winner']}", "cv_macro_f05": res["cv"]["macro"],
                "cv_india": res["cv"].get(f"macro[{lab[0]}]"), "cv_us": res["cv"].get(f"macro[{lab[-1]}]"),
                "singleton_mean": res["cv"]["singleton_mean"], "nonsingleton_mean": res["cv"]["nonsingleton_mean"],
                "micro_p": res["micro_p"], "micro_r": res["micro_r"],
                "loco_in_to_us": res["loco"].get(f"{lab[0]}->{lab[-1]}"), "loco_us_to_in": res["loco"].get(f"{lab[-1]}->{lab[0]}"),
                "device": "cuda", "gpu_ids": compute_gpu(), "peak_vram_gb": st["peak_vram_gb"], "wall_s": st["wall_s"],
                "gpu_time_s": st["gpu_time_s"], "cpu_time_s": st["cpu_time_s"], "cpu_violation": st["cpu_violation"],
                "verifier_model": verifier.get("key", ""), "verifier_band": "-".join(map(str, cfg["verifier"]["band"])),
                "n_pairs_dev": int((~D.hold).sum()),
                "config_hash": config_hash({k: x for k, x in cfg.items() if k != "abs"})},
               notes=f"{args.tag}: dCV vs sub3 {res['delta_cv_vs_sub3_pp']:+.3f} pp CI {res['ci_vs_sub3_pp']}, "
                     f"d holdout {res['delta_holdout_vs_sub3_pp']:+.3f} pp (GPU-only pipeline)")
    if abs(res["holdout_minus_cv_pp"]) > 0.5:
        print(f"STOP: holdout - CV = {res['holdout_minus_cv_pp']:+.3f} pp > 0.5 pp (suspected leakage)", flush=True)
    return res


def compute_gpu() -> str:
    import os
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")


def cmd_stage1(args, cfg) -> None:
    timer = StageTimer(args.tag)
    with timer.gpu():
        D, X0, feats0 = load_dev(cfg, timer)
        V, vinfo = verifier_block(cfg, D, args.verifier, timer)
        X = torch.cat([X0, V], 1)
        feats = feats0 + pl.V_NAMES
        print(f"[{args.tag}] stage-1 {tuple(X.shape)}; verifier {vinfo}", flush=True)
        arm = pl.run_arm(D, X, cfg["seed"], cfg.get("xgb"), search.grid(device=DEV), log=print)
        ref = sub3_ref(cfg, D, timer)
        prev = load_f(p4_dir(cfg) / f"{args.baseline}_f.parquet", D) if args.baseline else None
        finish(args, cfg, D, arm, X, feats, ref, timer, "stage1", prev_f=prev, verifier=
               {**vinfo, "band": cfg["verifier"]["band"], "rescue_k": cfg["verifier"]["rescue_k"],
                "rescue_max_p": cfg["verifier"]["rescue_max_p"], "scores": "det",
                "no_rescue": bool(EXTRA.get("no_rescue")), "extra": EXTRA["name"]})


def context(cfg, D: pl.Dev, p1, tau1, c2c: bool, drop_m: bool, timer) -> tuple[torch.Tensor, list]:
    g = group_features(D.s1, D.m, D.src, p1, tau1, D.n_ent)
    cols, names = [g[k] for k in GROUP_FEATURES], list(GROUP_FEATURES)
    if c2c:
        t, tc = time.time(), time.process_time()
        a = EA.attrs(cfg, "train")
        emb = EA.EmbReader(cfg, "train", DEV)
        timer.io(time.time() - t, time.process_time() - tc)
        cc = c2c_features(D.s1, D.m, p1, D.src, emb, gl.to_dev(a["addr_ok"], DEV), gl.to_dev(a["hn"], DEV),
                          gl.to_dev(a["pc"], DEV))
        cols += [cc[k] for k in C2C_FEATURES]
        timer.io(emb.io_wall, emb.io_cpu)
        names += list(C2C_FEATURES)
    keep = [i for i, n in enumerate(names) if not (drop_m and n in M_SIDE)]
    return torch.stack([cols[i] for i in keep], 1), [names[i] for i in keep]


def stage2_X(p1, V, C) -> torch.Tensor:
    z = torch.logit(p1.clamp(1e-6, 1 - 1e-6))
    return torch.cat([p1[:, None], z[:, None], V, C], 1).to(torch.float32)


def cmd_stage2(args, cfg) -> None:
    timer = StageTimer(args.tag)
    out = p4_dir(cfg)
    with timer.gpu():
        D, X0, feats0 = load_dev(cfg, timer)
        with timer.host("io"):
            with open(out / f"{args.stage1}.pkl", "rb") as fh:
                b1 = pickle.load(fh)
            p1 = load_p(out / f"{args.stage1}_p.parquet", D)
        V, vinfo = verifier_block(cfg, D, b1["verifier"]["key"], timer)
        C, cnames = context(cfg, D, p1, b1["tau_median"], not args.no_c2c, args.drop_m_side, timer)
        X2 = stage2_X(p1, V, C)
        feats2 = ["s1_p", "s1_logit"] + pl.V_NAMES + cnames
        print(f"[{args.tag}] stage-2 {tuple(X2.shape)}", flush=True)
        arm = pl.run_arm(D, X2, cfg["seed"], cfg.get("xgb"), search.grid(device=DEV), log=print)
        ref = sub3_ref(cfg, D, timer)
        with timer.host("io"):
            f1 = load_f(out / f"{args.stage1}_f.parquet", D)
        loco = None if args.no_loco else loco_stacked(cfg, D, X0, V, args, timer)
        res = finish(args, cfg, D, arm, X2, feats2, ref, timer, "stage2", {**b1["verifier"]}, prev_f=f1, loco=loco)
        with open(out / f"{args.tag}.pkl", "rb") as fh:
            b = pickle.load(fh)
        b.update({"stage1": args.stage1, "context": {"c2c": not args.no_c2c, "drop_m_side": args.drop_m_side,
                                                      "tau1": b1["tau_median"], "names": cnames}})
        with open(out / f"{args.tag}.pkl", "wb") as fh:
            pickle.dump(b, fh)
        print(f"[{args.tag}] gate: dCV vs stage-1 {res['delta_cv_vs_stage1_pp']:+.3f} pp, d holdout "
              f"{res['delta_holdout_vs_stage1_pp']:+.3f} pp", flush=True)


def loco_stacked(cfg, D: pl.Dev, X0, V, args, timer) -> dict:
    """A -> B: stage-1 5-fold within A (OOF on A, fold mean on B); stage-2 fitted on A's OOF-based
    matrix, tau from a 5-fold stage-2 OOF within A; both decided on B with their own A-tau."""
    X1 = torch.cat([X0, V], 1)
    cser = D.country[D.s1]
    cv = ~D.hold
    out = {}
    grid = search.grid(device=DEV)
    codes = torch.unique(cser[cv]).tolist()
    for a in codes:
        for b in codes:
            if a == b:
                continue
            ra, rb = cv & (cser == a), cv & (cser == b)
            ids_a = D.cv_ids[D.country[D.cv_ids] == a]
            ids_b = D.cv_ids[D.country[D.cv_ids] == b]
            p1, _ = pl.oof_within(D, X1, ra, cfg["seed"], cfg.get("xgb"), rb)
            tau1, _ = search.best_tau(D.s1, D.y, p1, gd.exclusive(D.s1, D.m, p1, ra, D.n_ent), ids_a, D.ntrue, grid)
            f1 = gm.macro(gm.per_s1_f(gd.threshold(gd.exclusive(D.s1, D.m, p1, rb, D.n_ent), p1, tau1),
                                      D.s1, D.y, D.ntrue), ids_b)
            C, _ = context(cfg, D, torch.where(ra | rb, p1, torch.zeros_like(p1)), tau1, not args.no_c2c,
                           args.drop_m_side, timer)
            X2 = stage2_X(p1, V, C)
            p2, _ = pl.oof_within(D, X2, ra, cfg["seed"], cfg.get("xgb"))
            tau2, _ = search.best_tau(D.s1, D.y, p2, gd.exclusive(D.s1, D.m, p2, ra, D.n_ent), ids_a, D.ntrue, grid)
            mdl = fit_calibrated(X2[ra], D.y[ra], D.s1[ra], seed=cfg["seed"], cfg_xgb=cfg.get("xgb"))
            pb = torch.zeros_like(p1)
            pb[rb] = mdl.predict(X2[rb])
            f2 = gm.macro(gm.per_s1_f(gd.threshold(gd.exclusive(D.s1, D.m, pb, rb, D.n_ent), pb, tau2),
                                      D.s1, D.y, D.ntrue), ids_b)
            la, lb = D.labels[a], D.labels[b]
            out[f"{la}->{lb}"] = f2
            out[f"stage1 {la}->{lb}"] = f1
            print(f"  stacked LOCO {la}->{lb}: stage-1 {f1:.4f}, stage-2 {f2:.4f}", flush=True)
    return out


def cmd_decide(args, cfg) -> None:
    """D6: global threshold vs segment thresholds (source; rescue-only) vs expected-F, CV + holdout."""
    timer = StageTimer(f"decide_{args.arm}")
    out = p4_dir(cfg)
    with timer.gpu():
        D, _, _ = load_dev(cfg, timer)
        with timer.host("io"):
            with open(out / f"{args.arm}.pkl", "rb") as fh:
                b = pickle.load(fh)
            p = load_p(out / f"{args.arm}_p.parquet", D)
        V, _ = verifier_block(cfg, D, b["verifier"]["key"], timer)
        grid = search.grid(device=DEV)
        res = {}
        f_g = gm.per_s1_f(pl.select_cv(D, p, b["fold_taus"]), D.s1, D.y, D.ntrue)
        fh_g = gm.per_s1_f(pl.select_hold(D, p, b["tau_median"]), D.s1, D.y, D.ntrue)
        res["global"] = {"cv": gm.macro(f_g, D.cv_ids), "holdout": gm.macro(fh_g, D.hold_ids), "tau": b["tau_median"]}
        f_e = gm.per_s1_f(pl.select_cv(D, p, b["fold_taus"], "expected_f"), D.s1, D.y, D.ntrue)
        fh_e = gm.per_s1_f(pl.select_hold(D, p, b["tau_median"], "expected_f"), D.s1, D.y, D.ntrue)
        res["expected_f"] = {"cv": gm.macro(f_e, D.cv_ids), "holdout": gm.macro(fh_e, D.hold_ids)}
        segs = {"segment_source": D.src.to(torch.int64),
                "segment_rescue": (V[:, 5] > 0).to(torch.int64)}
        cv = ~D.hold
        for name, seg in segs.items():
            sel = torch.zeros_like(D.y)
            for k in sorted(D.ids_by_fold):
                tr, te = cv & (D.fold != k), cv & (D.fold == k)
                ids = torch.cat([D.ids_by_fold[j] for j in sorted(D.ids_by_fold) if j != k])
                tk = search.segment_taus(D.s1, D.y, p, gd.exclusive(D.s1, D.m, p, tr, D.n_ent), seg, ids, D.ntrue, grid,
                                         b["fold_taus"][k])
                sel |= search.apply_segment_taus(p, gd.exclusive(D.s1, D.m, p, te, D.n_ent), seg, tk)
            t_all = search.segment_taus(D.s1, D.y, p, gd.exclusive(D.s1, D.m, p, cv, D.n_ent), seg, D.cv_ids, D.ntrue,
                                        grid, b["tau_median"])
            selh = search.apply_segment_taus(p, gd.exclusive(D.s1, D.m, p, D.hold, D.n_ent), seg, t_all)
            res[name] = {"cv": gm.macro(gm.per_s1_f(sel, D.s1, D.y, D.ntrue), D.cv_ids),
                         "holdout": gm.macro(gm.per_s1_f(selh, D.s1, D.y, D.ntrue), D.hold_ids), "taus": t_all}
            print(f"  {name}: {res[name]}", flush=True)
        g = res["global"]
        ok = {k: v for k, v in res.items() if k.startswith("segment") and 100 * (v["cv"] - g["cv"]) >= 0.02
              and 100 * (v["holdout"] - g["holdout"]) >= 0.02}
        res["choice"] = max(ok, key=lambda k: ok[k]["cv"]) if ok else (
            "expected_f" if res["expected_f"]["cv"] > g["cv"] else "global")
    res["time"] = timer.summary()
    print(json.dumps(res, indent=1, default=float), flush=True)
    (out / f"{args.arm}_decide.json").write_text(json.dumps(res, indent=1, default=float))
    rows = [{"rule": k, "CV": v["cv"], "holdout": v["holdout"], "dCV pp": 100 * (v["cv"] - g["cv"]),
             "d holdout pp": 100 * (v["holdout"] - g["holdout"])} for k, v in res.items() if k not in ("choice", "time")]
    (Path(cfg["abs"]["reports_dir"]) / f"p4_{args.arm}_decide.md").write_text("\n".join([
        f"# P4 decision rules on {args.arm} (D6, GPU)\n",
        "Segment thresholds kept only if >= +0.02 pp on BOTH CV and holdout; expected-F only if it beats the "
        "global threshold on CV. Taus from a 426-value grid.\n",
        tabulate(pd.DataFrame(rows), headers="keys", tablefmt="github", showindex=False, floatfmt=".5f"),
        f"\n**Choice: {res['choice']}.** {res['time']}"]), encoding="utf-8")
    st = res["time"]
    append_run(cfg["abs"]["runs_ledger"], f"p4_decide_{args.arm}",
               {"stage": "p4_decide", "model": res["choice"], "cv_macro_f05": res[res["choice"]]["cv"],
                "device": "cuda", "gpu_ids": compute_gpu(), "peak_vram_gb": st["peak_vram_gb"], "wall_s": st["wall_s"],
                "gpu_time_s": st["gpu_time_s"], "cpu_time_s": st["cpu_time_s"], "cpu_violation": st["cpu_violation"]},
               notes=f"D6 on {args.arm}: " + "; ".join(f"{r['rule']} CV {r['CV']:.6f} hold {r['holdout']:.6f}" for r in rows))


def main() -> None:
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2", "decide"])
    ap.add_argument("--config", default="configs/p4.yaml")
    ap.add_argument("--verifier", default="bge-rr-v2-m3-r2")
    ap.add_argument("--stage1", default="s1_r2")
    ap.add_argument("--arm", default="s2_r2")
    ap.add_argument("--tag", default="s1_r2")
    ap.add_argument("--no-c2c", action="store_true")
    ap.add_argument("--drop-m-side", action="store_true")
    ap.add_argument("--no-loco", action="store_true")
    ap.add_argument("--extra", default=None, choices=[None, "e7", "rr"], help="append extra pairs (E7 survivors / R2 union)")
    ap.add_argument("--baseline", default=None, help="stage1: arm whose per-S1 F0.5 is the paired baseline (e.g. subA's)")
    ap.add_argument("--no-rescue", action="store_true", help="D4 ablation: drop the verifier scores of rescue-only pairs")
    args = ap.parse_args()
    EXTRA["on"] = args.extra is not None
    EXTRA["name"] = args.extra
    EXTRA["no_rescue"] = args.no_rescue
    cfg = load_config(args.config)
    compute.limit_threads()
    from ber.embed import gpu_setup
    DEV = gpu_setup()
    from ber.verify import set_deterministic
    set_deterministic(cfg["seed"])  # deterministic index_add / scatter kernels (float atomics), fixed seeds
    {"stage1": cmd_stage1, "stage2": cmd_stage2, "decide": cmd_decide}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
