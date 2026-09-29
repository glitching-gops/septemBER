"""Fixed 50/50 R2/name-ambiguity blend; GPU evaluation and LOCO."""
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import torch
import run_p4_stage as stage
from run_p4_name_ambiguity import name_frequencies, freeze_check
from ber import compute
from ber.embed import gpu_setup
from ber.evaluate.ledger import append_run
from ber.gpu import StageTimer, calib, search
from ber.gpu import decide as gd
from ber.gpu import metric as gm
from ber.gpu import pipeline as pl
from ber.gpu.xgb import fit_calibrated
from ber.io_utils import load_config
from ber.verify import set_deterministic

TAG = "s1_rr_ambiguity_blend50"
LEFT = "s1_rr"
RIGHT = "s1_rr_name_ambiguity"


def main():
    freeze_check()
    cfg = load_config("configs/p4_noaug.yaml")
    compute.limit_threads()
    stage.DEV = gpu_setup()
    assert stage.DEV.type == "cuda", "No CPU fallback."
    set_deterministic(cfg["seed"])
    md = stage.p4_dir(cfg)
    rp = md / f"{TAG}_result.json"
    if rp.exists():
        raise RuntimeError("Experiment exists; refusing to overwrite.")
    timer = StageTimer(TAG)
    report = {"gate_pass": False, "alpha": 0.5, "status": "running"}

    try:
        with timer.gpu():
            with timer.host("io"):
                with open(md / f"{LEFT}.pkl", "rb") as f:
                    left = pickle.load(f)
                with open(md / f"{RIGHT}.pkl", "rb") as f:
                    right = pickle.load(f)
                reference = json.loads(
                    (md / f"{LEFT}_result.json").read_text()
                )
            stage.EXTRA.update(
                on=True, name="rr",
                no_rescue=left["verifier"].get("no_rescue", False),
            )
            D, X0, names0 = stage.load_dev(cfg, timer)
            with timer.host("io"):
                pleft = stage.load_p(md / f"{LEFT}_p.parquet", D)
                pright = stage.load_p(md / f"{RIGHT}_p.parquet", D)

            p = 0.5 * pleft + 0.5 * pright
            grid = search.grid(device=stage.DEV)
            taus = {}
            for k in sorted(D.ids_by_fold):
                freeze_check()
                rows = (~D.hold) & (D.fold != k)
                ids = torch.cat([
                    D.ids_by_fold[j] for j in sorted(D.ids_by_fold) if j != k
                ])
                taus[k], _ = search.best_tau(
                    D.s1, D.y, p,
                    gd.exclusive(D.s1, D.m, p, rows, D.n_ent),
                    ids, D.ntrue, grid,
                )
            tau = float(torch.quantile(torch.tensor(
                list(taus.values()), dtype=torch.float64, device=stage.DEV
            ), 0.5))

            base_taus = {
                int(k): float(v) for k, v in left["fold_taus"].items()
            }
            base_f = gm.per_s1_f(
                pl.select_cv(D, pleft, base_taus), D.s1, D.y, D.ntrue
            )
            assert abs(gm.macro(base_f, D.cv_ids)
                       - reference["cv"]["macro"]) < 1e-8
            new_f = gm.per_s1_f(
                pl.select_cv(D, p, taus), D.s1, D.y, D.ntrue
            )
            dcv, ci = pl.paired(new_f, base_f, D.cv_ids, seed=cfg["seed"])
            report.update(
                cv=gm.macro(new_f, D.cv_ids),
                delta_cv_pp=dcv, ci_pp=ci,
                fold_taus=taus, tau_median=tau,
            )
            print(f"B1 | CV evaluated | delta {dcv:+.3f} pp, CI {ci} | "
                  "next: holdout gate", flush=True)
            if dcv < 0.10 or ci[0] <= 0:
                report["status"] = "rejected_cv"
                return

            base_h = gm.per_s1_f(
                pl.select_hold(D, pleft, left["tau_median"]),
                D.s1, D.y, D.ntrue,
            )
            new_h = gm.per_s1_f(
                pl.select_hold(D, p, tau), D.s1, D.y, D.ntrue
            )
            holdout = gm.macro(new_h, D.hold_ids)
            dh = 100 * float(
                (new_h[D.hold_ids] - base_h[D.hold_ids]).mean()
            )
            report.update(holdout=holdout, delta_holdout_pp=dh)
            print(f"B2 | holdout evaluated | delta {dh:+.3f} pp | "
                  "next: LOCO", flush=True)
            if dh < 0.10 or abs(100 * (holdout - report["cv"])) > 0.5:
                report["status"] = "rejected_holdout"
                return

            freeze_check()
            # Reconstruct exactly the two trained feature matrices.
            original_platt = calib.platt_fit
            coefficients = iter(
                left["verifier"]["platt"][j] for j in (0, 1)
            )
            calib.platt_fit = lambda *a, **kw: next(coefficients)
            try:
                V, _ = stage.verifier_block(
                    cfg, D, left["verifier"]["key"], timer
                )
            finally:
                calib.platt_fit = original_platt

            A, added = name_frequencies(cfg, D, timer)
            missing = (X0[:, names0.index("addr_missing")] > 0.5).float()
            Z = torch.cat([A, A * missing[:, None]], 1)
            added += [n + "_missing" for n in added]
            assert names0 + pl.V_NAMES == left["features"]
            assert names0 + added + pl.V_NAMES == right["features"]
            XL = torch.cat([X0, V], 1)
            XR = torch.cat([X0, Z, V], 1)
            del X0, V, A, Z

            country = D.country[D.s1]
            cv = ~D.hold
            codes = torch.unique(country[cv]).tolist()
            loco = {}
            for a in codes:
                for b in codes:
                    if a == b:
                        continue
                    tr = cv & (country == a)
                    te = cv & (country == b)
                    pb = torch.zeros_like(p)
                    for X in (XL, XR):
                        freeze_check()
                        model = fit_calibrated(
                            X[tr], D.y[tr], D.s1[tr],
                            seed=cfg["seed"], cfg_xgb=cfg.get("xgb"),
                        )
                        pb[te] += 0.5 * model.predict(X[te])
                        del model
                    sel = gd.threshold(
                        gd.exclusive(D.s1, D.m, pb, te, D.n_ent), pb, tau
                    )
                    ids = D.cv_ids[D.country[D.cv_ids] == b]
                    key = f"{D.labels[a]}->{D.labels[b]}"
                    loco[key] = gm.macro(
                        gm.per_s1_f(sel, D.s1, D.y, D.ntrue), ids
                    )
                    change = 100 * (loco[key] - reference["loco"][key])
                    print(f"B3 | {key} | delta {change:+.3f} pp | "
                          "next: final gate", flush=True)

            freeze_check()
            deltas = {
                k: 100 * (v - reference["loco"][k])
                for k, v in loco.items()
            }
            passed = bool(deltas) and all(v >= -0.05 for v in deltas.values())
            report.update(
                loco=loco, loco_delta_pp=deltas,
                gate_pass=passed,
                status="accepted" if passed else "rejected_loco",
            )
            if passed:
                with timer.host("io"):
                    bundle = {
                        "left": left, "right": right, "alpha": 0.5,
                        "fold_taus": taus, "tau_median": tau,
                        "gate_pass": True, "evaluation": dict(report),
                    }
                    with open(md / f"{TAG}.pkl", "wb") as f:
                        pickle.dump(bundle, f)
                    stage.save_p(md / f"{TAG}_p.parquet", D, p)
            print(f"B4 | {'PASS' if passed else 'FAIL'} | "
                  f"CV {report['cv']:.6f}, holdout {holdout:.6f} | "
                  + ("next: prepare test inference" if passed else "keep R2"),
                  flush=True)
    except BaseException as exc:
        report.update(status="failed", gate_pass=False, error=repr(exc))
        raise
    finally:
        st = timer.summary()
        report["timings"] = st
        with timer.host("io"):
            rp.write_text(json.dumps(report, indent=2))
            append_run(
                cfg["abs"]["runs_ledger"], TAG,
                {
                    "stage": "ambiguity_blend", "model": "fixed_blend",
                    "device": "cuda",
                    "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "cv_macro_f05": report.get("cv", ""),
                    **{k: st[k] for k in (
                        "wall_s", "cpu_time_s", "gpu_time_s",
                        "peak_vram_gb", "cpu_violation"
                    )},
                },
                notes=report["status"], extra=report,
            )
        print("REPORT:", rp, flush=True)


if __name__ == "__main__":
    main()
