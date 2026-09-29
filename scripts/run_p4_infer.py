"""P4 test inference, GPU-only: stage-1 (+ stage-2) bundle -> exclusivity -> decision -> TSVs + validators.

    python scripts/run_p4_infer.py --stage1 s1_r2 [--stage2 s2_r2] [--decide s2_r2] --sub subA \
        [--link-dir <repo>/submissions] [--extra e7]

Inputs checked here: test feature shards (P2 f2), the D4 test scoring set, verifier scores of EVERY
scoring-set pair from BOTH round-2 fold models (the run stops, listing what is missing, rather than
letting a pair silently enter as unscored), and the stage bundles in ``<models>/p4``. Verifier
features for test = mean of the two fold models' Platt-calibrated logits (P3 protocol). Stage-2
context is computed over ALL test S1. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ber import compute  # noqa: E402
from ber.evaluate.ledger import append_run, config_hash  # noqa: E402
from ber.gpu import StageTimer  # noqa: E402
from ber.gpu import decide as gd  # noqa: E402
from ber.gpu import pipeline as pl  # noqa: E402
from ber.gpu import search  # noqa: E402
from ber.gpu import segment as sg  # noqa: E402
from ber.gpu.xgb import predict_mean  # noqa: E402
from ber.io import entity_attrs as EA  # noqa: E402
from ber.io import gpu_load as gl  # noqa: E402
from ber.io import ids as I  # noqa: E402
from ber.io import submission  # noqa: E402
from ber.io_utils import load_config, read_source  # noqa: E402
from ber.match.context import C2C_FEATURES, GROUP_FEATURES, c2c_features, group_features  # noqa: E402

DEV = None


def vdir(cfg, key) -> Path:
    return Path(cfg["abs"]["cache_dir"]).parent / "verifier" / key


def test_verifier(cfg, s1, m, vinfo, n_ent, timer, extra_sets=()) -> torch.Tensor:
    key = vinfo["key"]
    t, tc = time.time(), time.process_time()
    import os
    ts_path = Path(os.environ["P4_TEST_SET"]) if os.environ.get("P4_TEST_SET") else \
        vdir(cfg, cfg["verifier"]["key"]) / "test_set.parquet"
    ts = gl.pair_file(cfg, "test", ts_path, "test_set" + ("_repro" if os.environ.get("P4_TEST_SET") else ""),
                      ["inband", "rescued"], DEV)
    for es in extra_sets:
        e = gl.pair_file(cfg, "test", es, Path(es).stem, ["inband", "rescued"], DEV)
        ts = {k: torch.cat([ts[k], e[k]]) for k in ts}
    fr = []
    root = Path(os.environ["P4_SCORE_ROOT"]) if os.environ.get("P4_SCORE_ROOT") else vdir(cfg, key)
    tag = "_repro" if os.environ.get("P4_SCORE_ROOT") else ""
    for j in (0, 1):
        parts = sorted((root / f"fold{j}" / "det" / "test").glob("part-*.parquet")) + \
            sorted((root / f"fold{j}" / "det" / "test_extra").glob("part-*.parquet"))
        ps = [gl.pair_file(cfg, "test", q, f"{key}_f{j}_{q.parent.name}_{q.stem}{tag}", ["logit"], DEV) for q in parts]
        fr.append({k: torch.cat([x[k] for x in ps]) for k in ("s1", "m", "logit")})
    timer.io(time.time() - t, time.process_time() - tc)
    j0 = sg.lookup_pairs(ts["s1"], ts["m"], fr[0]["s1"], fr[0]["m"], n_ent)
    j1 = sg.lookup_pairs(ts["s1"], ts["m"], fr[1]["s1"], fr[1]["m"], n_ent)
    miss = (j0 < 0) | (j1 < 0)
    if bool(miss.any()):
        out = vdir(cfg, cfg["verifier"]["key"]) / "test_missing.parquet"
        pq.write_table(pa.table({"s1": ts["s1"][miss].cpu().numpy(), "m": ts["m"][miss].cpu().numpy()}), out)
        raise SystemExit(f"[p4-inf] {int(miss.sum()):,} scoring-set pairs lack a verifier score -> {out}")
    (a0, b0), (a1, b1) = vinfo["platt"][0], vinfo["platt"][1]
    l0, l1 = fr[0]["logit"][j0].to(torch.float64), fr[1]["logit"][j1].to(torch.float64)
    pcal = torch.sigmoid(((a0 * l0 + b0) + (a1 * l1 + b1)) / 2)
    resc = ts["rescued"] * (1 - ts["inband"])
    ss1, sm, lg = ts["s1"], ts["m"], (l0 + l1) / 2
    if vinfo.get("no_rescue"):  # the bundle was trained with rescue-only pairs unscored: same here
        keep = resc == 0
        ss1, sm, lg, pcal, resc = ss1[keep], sm[keep], lg[keep], pcal[keep], resc[keep]
    return pl.verifier_features(s1, m, ss1, sm, lg, pcal, resc, n_ent)


def main() -> None:
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p4.yaml")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--decide", default=None, help="arm whose <arm>_decide.json picks the rule (default: global tau)")
    ap.add_argument("--sub", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--link-dir", default=None)
    ap.add_argument("--extra", default=None, choices=[None, "e7", "rr"], help="append extra pairs (<p4>/<extra>/test_feats.parquet)")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    a = cfg["abs"]
    if args.out_root is None:
        args.out_root = Path(a["cache_dir"]).parent / "submissions"
    compute.limit_threads()
    from ber.embed import gpu_setup
    DEV = gpu_setup()
    from ber.verify import set_deterministic
    set_deterministic(cfg["seed"])  # deterministic index_add / scatter kernels (float atomics), fixed seeds
    timer = StageTimer(f"infer_{args.sub}")
    md_ = Path(a["models_dir"]) / "p4"
    with timer.host("io"), open(md_ / f"{args.stage1}.pkl", "rb") as fh:
        b1 = pickle.load(fh)
    base_feats = [f for f in b1["features"] if f not in pl.V_NAMES]
    with timer.host("io"):
        n_ent = len(I.entity_table(cfg, "test"))
        I.encoder(cfg, "test")  # one-time id index (encoding step)
    prov = [f for f in base_feats if f.startswith("rr_")]
    shard_feats = [f for f in base_feats if f not in prov]
    with timer.gpu():
        L = gl.load_pairs(cfg, "test", "test", shard_feats, DEV, timer)
        if prov:  # K15 rows: no new source, provenance scores NaN (as in training)
            pk = torch.full((L["X"].shape[0], len(prov)), float("nan"), device=DEV)
            pk[:, [prov.index("rr_src_flags"), prov.index("rr_n_sources")]] = 0.0
            L["X"] = torch.cat([L["X"], pk], 1)
        extra_sets = []
        if args.extra:
            import os
            ep = Path(os.environ.get("P4_EXTRA_ROOT", str(Path(a["cache_dir"]).parent / args.extra))) / "test_feats.parquet"
            with timer.host("io"):
                E = gl.pair_file(cfg, "test", ep, f"{args.extra}_test_feats", base_feats, DEV)
            ents = gl.entity_src(cfg, "test", DEV)
            L = {"s1": torch.cat([L["s1"], E["s1"]]), "m": torch.cat([L["m"], E["m"]]),
                 "src": torch.cat([L["src"], ents[E["m"]]]),
                 "X": torch.cat([L["X"], torch.stack([E[c] for c in base_feats], 1)])}
            extra_sets.append(ep.with_name(f"test_{args.extra}_set.parquet"))
        s1, m, src = L["s1"], L["m"], L["src"]
        print(f"[p4-inf] {s1.numel():,} test pairs x {len(base_feats)} base features [{timer.summary()['wall_s']}s]", flush=True)
        V = test_verifier(cfg, s1, m, b1["verifier"], n_ent, timer, extra_sets)
        print(f"[p4-inf] verifier: {int(V[:, 2].sum()):,} scored, {int(V[:, 5].sum()):,} rescue-only", flush=True)
        X1 = torch.cat([L["X"], V], 1)
        del L
        p = predict_mean(list(b1["folds"].values()), X1)
        del X1
        bundle = b1
        if args.stage2:
            with timer.host("io"), open(md_ / f"{args.stage2}.pkl", "rb") as fh:
                b2 = pickle.load(fh)
            cx = b2["context"]
            g = group_features(s1, m, src, p, cx["tau1"], n_ent)
            cols = {k: g[k] for k in GROUP_FEATURES}
            if cx["c2c"]:
                t, tc = time.time(), time.process_time()
                at = EA.attrs(cfg, "test")
                emb = EA.EmbReader(cfg, "test", DEV)
                timer.io(time.time() - t, time.process_time() - tc)
                cc = c2c_features(s1, m, p, src, emb, gl.to_dev(at["addr_ok"], DEV), gl.to_dev(at["hn"], DEV),
                                  gl.to_dev(at["pc"], DEV))
                cols.update({k: cc[k] for k in C2C_FEATURES})
                timer.io(emb.io_wall, emb.io_cpu)
            C = torch.stack([cols[k] for k in cx["names"]], 1)
            z = torch.logit(p.clamp(1e-6, 1 - 1e-6))
            X2 = torch.cat([p[:, None], z[:, None], V, C], 1).to(torch.float32)
            assert b2["features"] == ["s1_p", "s1_logit"] + pl.V_NAMES + cx["names"], "stage-2 feature order"
            p = predict_mean(list(b2["folds"].values()), X2)
            del X2, C
            bundle = b2
            print(f"[p4-inf] stage-2 scored [{timer.summary()['wall_s']}s]", flush=True)
        rule = {"rule": "global", "tau": bundle["tau_median"]}
        if args.decide:
            dj = json.loads((md_ / f"{args.decide}_decide.json").read_text())
            ch = dj["choice"]
            rule = {"rule": ch, "tau": bundle["tau_median"], **({"taus": dj[ch]["taus"]} if ch.startswith("segment") else {})}
        keep = gd.exclusive(s1, m, p, None, n_ent)
        if rule["rule"] == "global":
            sel = gd.threshold(keep, p, rule["tau"])
        elif rule["rule"] == "expected_f":
            sel = gd.expected_f(s1, p, keep)
        else:
            seg = src.to(torch.int64) if rule["rule"] == "segment_source" else (V[:, 5] > 0).to(torch.int64)
            sel = search.apply_segment_taus(p, keep, seg, {int(k): v for k, v in rule["taus"].items()})
        print(f"[p4-inf] decided with {rule}: {int(sel.sum()):,} matches", flush=True)
        # per-country stats on GPU
        country = gl.entity_country(cfg, "test", DEV)
        with timer.host("io"):
            s1raw, _ = read_source(a["test.source1"])
            s1_file = gl.to_dev(I.encode(cfg, "test", s1raw["entity_id"]), DEV)
        npred = sg.seg_count(s1[sel], n_ent)[s1_file]
        cs = country[s1_file].to(torch.int64)
        labels = I.country_labels(cfg, "test")
        nc = len(labels)
        cnt = torch.bincount(cs, minlength=nc).double()
        per_c = [{"country": labels[i], "S1": int(cnt[i]),
                  "pairs_per_s1": float(torch.zeros(nc, dtype=torch.float64, device=DEV).index_add_(0, cs, npred.double())[i] / cnt[i]),
                  "empty": float(torch.zeros(nc, dtype=torch.float64, device=DEV).index_add_(0, cs, (npred == 0).double())[i] / cnt[i])}
                 for i in range(nc) if float(cnt[i]) > 0]
        o = sg.order([(s1, False), (m, False)])
        so = sel[o]
        with timer.host("io"):
            cand_s1, cand_m = s1[o].cpu().numpy(), m[o].cpu().numpy()
            sel_s1, sel_m = s1[o][so].cpu().numpy(), m[o][so].cpu().numpy()
    outdir = Path(args.out_root) / args.sub
    t, tc = time.time(), time.process_time()
    submission.write(cfg, outdir, list(s1raw["entity_id"]), cand_s1, cand_m, sel_s1, sel_m)
    pq.write_table(pa.table({"s1": s1.cpu().numpy(), "m": m.cpu().numpy(), "p": p.cpu().numpy(),
                             "selected": sel.cpu().numpy()}), outdir / "test_scores.parquet")
    from validate_outputs import validate
    r = validate(outdir / "matching_results.tsv", outdir / "candidate_pairs.tsv", cfg, "test")
    timer.io(time.time() - t, time.process_time() - tc)
    report = "\n".join(f"  ok  {c}" for c in r["checks"]) + "\n\n" + (r["official"] or "")
    print(report, flush=True)
    st = timer.summary()
    info = {"sub": args.sub, "stage1": args.stage1, "stage2": args.stage2, "rule": rule, "n_matches": int(sel.sum()),
            "n_candidates": int(s1.numel()), "empty": float((npred == 0).double().mean()), "per_country": per_c,
            "validator": report, "extra": args.extra, **st}
    (outdir / "INFER.json").write_text(json.dumps(info, indent=1, default=float))
    if args.link_dir:
        ld = Path(args.link_dir) / args.sub
        ld.mkdir(parents=True, exist_ok=True)
        for f in ("candidate_pairs.tsv", "matching_results.tsv"):
            (ld / f).unlink(missing_ok=True)
            (ld / f).symlink_to(outdir / f)
    print(f"[p4-inf] {args.sub}: {info['n_matches']:,} matches, empty {info['empty']:.2%}; {st}", flush=True)
    if not args.no_ledger:
        append_run(a["runs_ledger"], f"p4_{args.sub}",
                   {"stage": "test_inference", "model": args.stage2 or args.stage1, "submission_id": args.sub,
                    "cv_macro_f05": bundle["cv"]["macro_threshold"], "n_pairs_test": int(s1.numel()), "device": "cuda",
                    "gpu_ids": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""), "peak_vram_gb": st["peak_vram_gb"],
                    "gpu_time_s": st["gpu_time_s"], "cpu_time_s": st["cpu_time_s"], "cpu_violation": st["cpu_violation"],
                    "verifier_model": b1["verifier"]["key"], "verifier_band": "-".join(map(str, b1["verifier"]["band"])),
                    "config_hash": config_hash({k: v for k, v in cfg.items() if k != "abs"}), "wall_s": st["wall_s"]},
                   notes=f"{args.sub}: {info['n_matches']:,} matches, {info['empty']:.2%} empty, rule {rule['rule']}; "
                         "validators PASS")


if __name__ == "__main__":
    main()
