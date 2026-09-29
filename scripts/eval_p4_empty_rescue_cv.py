"""Evaluate novel retrieval candidates using the fixed accepted GPU matcher."""
import inspect
import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import pandas as pd
import torch
import run_p4_stage as stage
from run_p4_name_ambiguity import freeze_check, name_frequencies
from ber import compute
from ber.embed import gpu_setup
from ber.evaluate.ledger import append_run
from ber.gpu import StageTimer, calib
from ber.gpu import metric as gm, pipeline as pl, segment as sg
from ber.io import gpu_load as gl, ids as I
from ber.io_utils import load_config

TAG = "empty_rescue_matcher_cv"


def gpu_token_counts():
    """Retain CPU tokenisation; compute document frequencies and IDF on CUDA."""
    from ber.match import features_gpu as fg

    source = inspect.getsource(fg.TokenSpace)
    old = 'df_count = np.bincount(codes, minlength=len(uniques)).astype(np.float64)'
    new = (
        'df_count = torch.bincount('
        'torch.as_tensor(codes, dtype=torch.long, device=gpu_setup()), '
        'minlength=len(uniques)).double()'
    )
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = 'idf = np.log((n_docs + 1.0) / (df_count + 1.0)) + 1.0'
    new = (
        'idf = (torch.log((n_docs + 1.0) / (df_count + 1.0))'
        ' + 1.0).cpu().numpy()'
    )
    assert source.count(old) == 1
    source = source.replace(old, new)
    namespace = dict(vars(fg), torch=torch, gpu_setup=gpu_setup)
    exec(compile(source, "<pilot_gpu_token_counts>", "exec"), namespace)
    fg.TokenSpace = namespace["TokenSpace"]


def main():
    freeze_check()
    cfg = load_config("configs/p4_noaug.yaml")
    compute.limit_threads()
    stage.DEV = gpu_setup()
    assert stage.DEV.type == "cuda", "No CPU fallback."

    timer = StageTimer(TAG)
    root = Path(cfg["abs"]["cache_dir"]).parent
    pilot = root / "empty_rescue_cv"
    md = stage.p4_dir(cfg)
    report_path = root / "reports" / f"{TAG}.json"
    assert not report_path.exists(), "Evaluation exists; refusing to overwrite."
    report = {"status": "running", "submission_approved": False}

    try:
        with timer.gpu():
            with timer.host("io"):
                with open(md / "s1_rr_ambiguity_blend50.pkl", "rb") as f:
                    bundle = pickle.load(f)
                pairs = pd.read_parquet(
                    root / "reports/empty_retrieval_prefilter_cv_retained_pairs.parquet",
                    columns=["s1", "m"],
                )
                table = I.entity_table(cfg, "train")

            assert bundle["gate_pass"] and bundle["alpha"] == 0.5
            left, right = bundle["left"], bundle["right"]
            assert left["verifier"] == right["verifier"]
            taus = {int(k): float(v) for k, v in bundle["fold_taus"].items()}
            stage.EXTRA.update(
                on=True, name="rr",
                no_rescue=left["verifier"].get("no_rescue", False),
            )

            D, X, names = stage.load_dev(cfg, timer)
            with timer.host("io"):
                baseline_p = stage.load_p(
                    md / "s1_rr_ambiguity_blend50_p.parquet", D
                )

            baseline_sel = pl.select_cv(D, baseline_p, taus)
            baseline_f = gm.per_s1_f(
                baseline_sel, D.s1, D.y, D.ntrue
            )
            assert abs(
                gm.macro(baseline_f, D.cv_ids) - bundle["evaluation"]["cv"]
            ) < 1e-8

            ns = gl.to_dev(pairs.s1.to_numpy(), stage.DEV).long()
            nm = gl.to_dev(pairs.m.to_numpy(), stage.DEV).long()
            assert not bool(sg.isin_pairs(ns, nm, D.s1, D.m, D.n_ent).any())
            query_ids = torch.unique(ns)
            query_mask = torch.zeros(
                D.n_ent, dtype=torch.bool, device=stage.DEV
            )
            query_mask[query_ids] = True
            old_rows = query_mask[D.s1]
            assert not bool(D.hold[old_rows].any())
            old_indices = torch.nonzero(old_rows).flatten()
            old_x = X[old_rows]
            del X

            # Reuse the shipped Platt coefficients without refitting.
            original_platt = calib.platt_fit
            coefficients = iter(
                left["verifier"]["platt"][j] for j in (0, 1)
            )
            calib.platt_fit = lambda *a, **kw: next(coefficients)
            try:
                old_v, _ = stage.verifier_block(
                    cfg, D, left["verifier"]["key"], timer
                )
            finally:
                calib.platt_fit = original_platt
            old_v = old_v[old_rows]

            freeze_check()
            with timer.host("io"):
                ids = table.entity_id.to_numpy()
                cand = pd.DataFrame({
                    "s1_id": ids[ns.cpu().numpy()],
                    "m_id": ids[nm.cpu().numpy()],
                })
                cand["source"] = table.src.to_numpy()[
                    nm.cpu().numpy()
                ]
                cand["source"] = cand["source"].map({2: "S2", 3: "S3"})
                cand["country"] = table.country_label.to_numpy()[
                    ns.cpu().numpy()
                ]
            assert cand.source.notna().all()
            for col in ("b6n_rank", "b6a_rank", "b7_rank"):
                cand[col] = float("nan")

            feature_path = pilot / "matcher_features.parquet"
            if feature_path.exists():
                with timer.host("io"):
                    F = pd.read_parquet(feature_path)
                assert F[["s1_id", "m_id"]].equals(
                    cand[["s1_id", "m_id"]]
                )
            else:
                print(
                    "M0 | building features | estimated 5–10 minutes; "
                    "GPU similarities, permitted CPU text/string work | next: matcher",
                    flush=True,
                )
                gpu_token_counts()
                from ber.match.build import FeatureContext
                c2 = {
                    **cfg,
                    "versions": {
                        **cfg["versions"], "feature_version": "f2"
                    },
                }
                # CPU work here is cached-record parsing, tokenisation,
                # and the existing permitted string-feature implementation.
                with timer.host("tok"):
                    context = FeatureContext(c2, "train", gpu_features=True)
                    F = context.features(cand)
                del context
                F.insert(0, "m_id", cand.m_id.to_numpy())
                F.insert(0, "s1_id", cand.s1_id.to_numpy())
                with timer.host("io"):
                    F.to_parquet(feature_path, index=False)
                torch.cuda.empty_cache()

            freeze_check()
            base_names = [n for n in names if n not in stage.PROV]
            assert all(n in F for n in base_names)
            with timer.host("io"):
                new_x = gl.to_dev(
                    F[base_names].to_numpy(dtype="float32"), stage.DEV
                )
            provenance = torch.full(
                (len(F), len(stage.PROV)), float("nan"),
                device=stage.DEV,
            )
            provenance[:, :2] = 0.0
            new_x = torch.cat([new_x, provenance], 1)
            assert names == base_names + stage.PROV
            del F, cand, provenance, table, pairs

            sx = torch.cat([D.s1[old_rows], ns])
            mx = torch.cat([D.m[old_rows], nm])
            base_x = torch.cat([old_x, new_x])
            local = SimpleNamespace(s1=sx, m=mx, n_ent=D.n_ent)
            A, added = name_frequencies(cfg, local, timer)
            missing = (
                base_x[:, names.index("addr_missing")] > 0.5
            ).float()
            assert names + pl.V_NAMES == left["features"]
            assert (
                names + added + [n + "_missing" for n in added]
                + pl.V_NAMES
            ) == right["features"]

            fold_e = torch.full(
                (D.n_ent,), -1, dtype=torch.long, device=stage.DEV
            )
            fold_e[D.s1] = D.fold
            local_folds = fold_e[sx]
            assert bool((local_folds >= 0).all())

            def predict(base, ambiguity, miss, verifier, folds):
                p = torch.empty(
                    base.shape[0], device=stage.DEV, dtype=torch.float32
                )
                for k in sorted(D.ids_by_fold):
                    freeze_check()
                    rows = folds == k
                    if not bool(rows.any()):
                        continue
                    xl = torch.cat([base[rows], verifier[rows]], 1)
                    xr = torch.cat([
                        base[rows], ambiguity[rows],
                        ambiguity[rows] * miss[rows, None],
                        verifier[rows],
                    ], 1)
                    p[rows] = (
                        0.5 * left["folds"][k].predict(xl)
                        + 0.5 * right["folds"][k].predict(xr)
                    )
                return p

            nold = old_indices.numel()
            check_p = predict(
                old_x, A[:nold], missing[:nold], old_v,
                local_folds[:nold],
            )
            parity = float(
                (check_p - baseline_p[old_rows]).abs().max()
            )
            assert parity <= 1e-6, f"Baseline prediction parity failed: {parity}"
            print(
                f"M1 | baseline parity passed | max delta {parity} | "
                "next: expanded candidate scoring", flush=True,
            )

            score_parts = []
            for j in (0, 1):
                with timer.host("io"):
                    path = (
                        pilot / "verifier" / f"fold{j}" / "det"
                        / "scores_oof_empty_rescue_cv.parquet"
                    )
                    q = gl.pair_file(
                        cfg, "train", path,
                        f"empty_rescue_cv_f{j}", ["logit"], stage.DEV,
                    )
                assert bool((fold_e[q["s1"]] % 2 == 1-j).all())
                a, b = left["verifier"]["platt"][j]
                q["pcal"] = torch.sigmoid(
                    a * q["logit"].double() + b
                )
                score_parts.append(q)

            ss = torch.cat([q["s1"] for q in score_parts])
            sm = torch.cat([q["m"] for q in score_parts])
            lg = torch.cat([q["logit"] for q in score_parts]).double()
            pc = torch.cat([q["pcal"] for q in score_parts])
            lookup = sg.lookup_pairs(ns, nm, ss, sm, D.n_ent)
            assert bool((lookup >= 0).all()) and ss.numel() == ns.numel()
            assert torch.unique(
                torch.stack([ss, sm], 1), dim=0
            ).shape[0] == ss.numel()

            scored_old = old_v[:, 2] > 0.5
            V = pl.verifier_features(
                sx, mx,
                torch.cat([sx[:nold][scored_old], ss]),
                torch.cat([mx[:nold][scored_old], sm]),
                torch.cat([old_v[scored_old, 0].double(), lg]),
                torch.cat([old_v[scored_old, 1].double(), pc]),
                torch.cat([
                    old_v[scored_old, 5].double(),
                    torch.zeros_like(pc),
                ]),
                D.n_ent,
            )
            updated_p = predict(
                base_x, A, missing, V, local_folds
            )

            truth = stage.gt_tensors(cfg)
            new_y = gm.labels(ns, nm, *truth, D.n_ent)
            new_source = gl.entity_src(cfg, "train", stage.DEV)[nm]
            E = replace(
                D,
                s1=torch.cat([D.s1, ns]),
                m=torch.cat([D.m, nm]),
                src=torch.cat([D.src, new_source]),
                y=torch.cat([D.y, new_y]),
                fold=torch.cat([D.fold, fold_e[ns]]),
                hold=torch.cat([
                    D.hold, torch.zeros_like(ns, dtype=torch.bool)
                ]),
            )
            p = torch.cat([baseline_p.clone(), updated_p[nold:]])
            p[old_indices] = updated_p[:nold]
            selected = pl.select_cv(E, p, taus)
            f = gm.per_s1_f(selected, E.s1, E.y, E.ntrue)
            delta, ci = pl.paired(
                f, baseline_f, D.cv_ids, seed=cfg["seed"]
            )
            novel_sel = selected[D.s1.numel():]
            old_sel = selected[:D.s1.numel()]
            report.update(
                status="complete",
                baseline_cv=gm.macro(baseline_f, D.cv_ids),
                pilot_cv=gm.macro(f, D.cv_ids),
                delta_cv_pp=delta,
                ci_pp=ci,
                new_pairs=int(ns.numel()),
                new_true_pairs=int(new_y.sum()),
                selected_new_true_pairs=int((novel_sel & new_y).sum()),
                selected_new_false_pairs=int((novel_sel & ~new_y).sum()),
                lost_old_true_pairs=int(
                    (baseline_sel & ~old_sel & D.y).sum()
                ),
                removed_old_false_pairs=int(
                    (baseline_sel & ~old_sel & ~D.y).sum()
                ),
                baseline_parity_max_delta=parity,
                scope=(
                    "Full predicted-empty CV expansion. Fixed model and thresholds; "
                    "full-CV exclusivity recomputed. No holdout, LOCO, "
                    "population extrapolation, or submission approval."
                ),
            )
            report["cv_screen_pass"] = delta >= 0.10 and ci[0] > 0
            with timer.host("io"):
                stage.save_p(
                    pilot / "cv_expanded_predictions.parquet", E, p
                )
            print(json.dumps(report, indent=2), flush=True)
            print(
                f"M2 | pilot evaluated | delta {delta:+.4f} pp; "
                f"new TP {report['selected_new_true_pairs']}, "
                f"new FP {report['selected_new_false_pairs']} | "
                "next: review expansion feasibility", flush=True,
            )
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        st = timer.summary()
        report["timings"] = st
        with timer.host("io"):
            report_path.write_text(json.dumps(report, indent=2))
            append_run(
                cfg["abs"]["runs_ledger"], TAG,
                {
                    "stage": "empty_rescue_matcher_cv",
                    "model": "fixed_ambiguity_blend50",
                    "device": "cuda",
                    "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    **{k: st[k] for k in (
                        "wall_s", "cpu_time_s", "gpu_time_s",
                        "peak_vram_gb", "cpu_violation",
                    )},
                },
                notes=report["status"], extra=report,
            )
        print("REPORT:", report_path, flush=True)


if __name__ == "__main__":
    main()
