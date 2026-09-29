"""GPU matcher/prefilter LOCO rescue checks; no 22:30 cutoff."""
import argparse
import inspect
import math
import os
import pickle
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import pandas as pd
import torch
import torch.nn.functional as F
import run_p4_stage as stage
from ber.gpu import calib, decide as gd, metric as gm, pipeline as pl, segment as sg
from ber.gpu.retrieval import topk
from ber.gpu.xgb import fit_calibrated
from ber.io import entity_attrs as EA, gpu_load as gl
from ber.verify import set_deterministic
from run_p4_name_ambiguity import name_frequencies

parser = argparse.ArgumentParser()
parser.add_argument("--train-code", type=int, required=True)
parser.add_argument("--test-code", type=int, required=True)
ARGS = parser.parse_args()
assert ARGS.train_code != ARGS.test_code
TAG = f"empty_rescue_loco_{ARGS.train_code}_{ARGS.test_code}_v2"
K = 100


def freeze_check():
    # User explicitly revoked the old 22:30 experiment freeze.
    pass


def segment(filename, start, end):
    text = (HERE / filename).read_text()
    assert text.count(start) == 1 and text.count(end) == 1
    return textwrap.dedent(text[text.index(start):text.index(end)])


# Extract existing GPU kernels without executing the old script main loops.
retrieval = segment(
    "retrieve_p4_empty_rescue_cv.py",
    "            def embeddings(field, entities):",
    "            new_true = gm.labels(",
)
assert retrieval.count("if time.monotonic()-started > 1200:") == 1
retrieval = retrieval.replace("if time.monotonic()-started > 1200:", "if False:")
prefix = '''def retrieve(cfg, D, queries, table, timer):
    with timer.host("io"):
        source = gl.to_dev(table.src.to_numpy(), stage.DEV).long()
        country = gl.to_dev(table.country.to_numpy(), stage.DEV).long()
        attributes = EA.attrs(cfg, "train")
        address_ok = gl.to_dev(attributes["addr_ok"], stage.DEV).bool()
        reader = EA.EmbReader(cfg, "train", stage.DEV)
'''
exec(compile(
    prefix + textwrap.indent(retrieval, "    ") + "    return s1, m\n",
    "<loco_gpu_retrieval>", "exec",
))

prefilter = segment(
    "prefilter_p4_empty_rescue_cv.py",
    "            def embeddings(field, entities):",
    "            params = {",
)
prefix = '''def prefeatures(cfg, s1, m, table, timer):
    dev, n_ent = stage.DEV, len(table)
    with timer.host("io"):
        source = gl.to_dev(table.src.to_numpy(), dev).long()
        attributes = EA.attrs(cfg, "train")
        reader = EA.EmbReader(cfg, "train", dev)
'''
exec(compile(
    prefix + textwrap.indent(prefilter, "    ") + "    return X, names\n",
    "<loco_gpu_prefilter>", "exec",
))


def loco_select(D, p, tau):
    mask = (~D.hold) & (D.country[D.s1] == ARGS.test_code)
    keep = gd.exclusive(D.s1, D.m, p, mask, D.n_ent)
    return gd.threshold(keep, p, tau)


def train_matchers(cfg, D, X, names, left, right, timer):
    set_deterministic(cfg["seed"])
    country = D.country[D.s1]
    tr = (~D.hold) & (country == ARGS.train_code)
    te = (~D.hold) & (country == ARGS.test_code)
    assert bool(tr.any()) and bool(te.any())

    original = calib.platt_fit
    coefficients = iter(left["verifier"]["platt"][j] for j in (0, 1))
    calib.platt_fit = lambda *a, **kw: next(coefficients)
    try:
        V, _ = stage.verifier_block(cfg, D, left["verifier"]["key"], timer)
    finally:
        calib.platt_fit = original

    A, added = name_frequencies(cfg, D, timer)
    miss = (X[:, names.index("addr_missing")] > .5).float()
    assert names + pl.V_NAMES == left["features"]
    assert (
        names + added + [n + "_missing" for n in added] + pl.V_NAMES
        == right["features"]
    )

    with timer.host("io"):
        shipped_p = stage.load_p(
            stage.p4_dir(cfg) / "s1_rr_ambiguity_blend50_p.parquet", D
        )
    shipped_replay = torch.zeros_like(shipped_p)
    models = []
    pb = torch.zeros_like(D.y, dtype=torch.float32)
    for i in (0, 1):
        Z = (
            torch.cat([X, V], 1) if i == 0
            else torch.cat([X, A, A * miss[:, None], V], 1)
        )
        arm = (left, right)[i]
        for k in sorted(D.ids_by_fold):
            rows = (~D.hold) & (D.fold == k)
            shipped_replay[rows] += .5 * arm["folds"][k].predict(Z[rows])
        model = fit_calibrated(
            Z[tr], D.y[tr], D.s1[tr],
            seed=cfg["seed"], cfg_xgb=cfg.get("xgb"),
        )
        pb[te] += .5 * model.predict(Z[te])
        models.append(model)
        del Z
        print(
            f"L0 | matcher {i} fitted | source-country labels only "
            "| next: baseline parity", flush=True,
        )
    feature_parity = float(
        (shipped_replay[~D.hold] - shipped_p[~D.hold]).abs().max()
    )
    print(
        f"L0 | shipped feature parity | max delta={feature_parity:.9g} "
        "| next: frozen LOCO comparison", flush=True,
    )
    assert feature_parity <= 1e-6, (
        f"Shipped CV prediction parity failed: {feature_parity}"
    )
    del shipped_p, shipped_replay
    return models, pb, V


def prepare_pairs(cfg, D, pb, bundle, table, root, pilot, timer, report):
    selected = loco_select(D, pb, bundle["tau_median"])
    counts = sg.seg_count(D.s1[selected], D.n_ent)
    queries = D.cv_ids[counts[D.cv_ids] == 0]
    assert queries.numel() > 0
    print(
        f"L1 | LOCO predicted-empty queries | {queries.numel():,} "
        "| next: k100 retrieval", flush=True,
    )
    ns, nm = retrieve(cfg, D, queries, table, timer)
    assert ns.numel() > 0

    # Source-country training labels only. Target labels are evaluated later.
    with timer.host("io"):
        raw = pd.read_parquet(
            root / "reports/empty_retrieval_cv_k100_new_pairs.parquet",
            columns=["s1", "m"],
        )
        ts = gl.to_dev(raw.s1.to_numpy(), stage.DEV).long()
        tm = gl.to_dev(raw.m.to_numpy(), stage.DEV).long()
    rows = D.country[ts] == ARGS.train_code
    ts, tm = ts[rows], tm[rows]
    held = torch.zeros(D.n_ent, dtype=torch.bool, device=stage.DEV)
    held[D.s1] = D.hold
    assert ts.numel() > 0 and not bool(held[ts].any())

    truth = stage.gt_tensors(cfg)
    ty = gm.labels(ts, tm, *truth, D.n_ent)
    assert int(ty.sum()) > 0
    TX, fnames = prefeatures(cfg, ts, tm, table, timer)
    params = dict(
        device="cuda", tree_method="hist", n_estimators=250,
        max_depth=3, learning_rate=.05, min_child_weight=5,
        scale_pos_weight=100,
    )
    pf = fit_calibrated(
        TX, ty, ts, seed=cfg["seed"], cfg_xgb=params,
    )
    report.update(
        prefilter_training_pairs=int(ts.numel()),
        prefilter_training_positives=int(ty.sum()),
        empty_queries=int(queries.numel()),
        raw_new_pairs=int(ns.numel()),
    )
    del TX, ty, ts, tm, rows, held, raw

    NX, names = prefeatures(cfg, ns, nm, table, timer)
    assert names == fnames
    scores = pf.raw(NX)
    chosen = torch.topk(
        scores, max(1, math.ceil(.05 * ns.numel()))
    ).indices
    ns, nm = ns[chosen], nm[chosen]
    with timer.host("io"):
        frame = pd.DataFrame(dict(
            s1=ns.cpu().numpy(), m=nm.cpu().numpy(),
        ))
        frame.to_parquet(pilot / "retained_pairs.parquet", index=False)
        with open(pilot / "prefilter.pkl", "wb") as handle:
            pickle.dump(dict(
                model=pf, features=fnames, keep_fraction=.05,
                parameters=params, train_code=ARGS.train_code,
            ), handle)
    del NX, scores, chosen, pf
    print(
        f"L2 | source-country prefilter complete | {len(frame):,} retained "
        "| next: verifier scoring", flush=True,
    )
    return frame


def score_new(cfg, pairs, table, pilot, timer):
    from ber.verify import rerankers
    from run_p4_verifier import cmd_score

    text = textwrap.dedent(inspect.getsource(rerankers.Verifier.score))
    old = "1.0 / (1.0 + np.exp(-logit))"
    assert text.count(old) == 1
    text = text.replace(
        old,
        "torch.sigmoid(torch.as_tensor(logit, device=self.device)).cpu().numpy()",
    )
    namespace = dict(vars(rerankers), torch=torch)
    exec(compile(text, "<loco_cuda_verifier_probability>", "exec"), namespace)
    rerankers.Verifier.score = namespace["score"]

    ids = table.entity_id.to_numpy()
    with timer.host("io"):
        frame = pd.DataFrame(dict(
            s1_id=ids[pairs.s1.to_numpy()],
            m_id=ids[pairs.m.to_numpy()],
        ))
        frame.to_parquet(pilot / f"dev_{TAG}_set.parquet", index=False)
    os.environ["P4_SCORE_ROOT"] = str(pilot / "verifier")
    for j in (0, 1):
        cmd_score(SimpleNamespace(
            fold=j, part="oof", extra=TAG,
            reverse=False, middle=False, det_batch=128,
        ), cfg)
        torch.cuda.empty_cache()


# Preserve the established feature construction, OOF routing and parity checks.
source = (HERE / "eval_p4_empty_rescue_cv.py").read_text()


def change(old, new):
    global source
    assert source.count(old) == 1, f"Reviewed evaluator changed: {old[:80]!r}"
    source = source.replace(old, new)


change(
    "from run_p4_name_ambiguity import freeze_check, name_frequencies",
    "from run_p4_name_ambiguity import name_frequencies",
)
change('TAG = "empty_rescue_matcher_cv"', "")
change(
    '    pilot = root / "empty_rescue_cv"',
    '    pilot = root / TAG\n    pilot.mkdir(parents=True, exist_ok=True)',
)
change(
    '                pairs = pd.read_parquet(\n'
    '                    root / "reports/empty_retrieval_prefilter_cv_retained_pairs.parquet",\n'
    '                    columns=["s1", "m"],\n'
    '                )\n',
    "",
)

start = source.index(
    '            with timer.host("io"):\n                baseline_p = stage.load_p('
)
end = source.index("            ns = gl.to_dev(pairs.s1.to_numpy()", start)
source = source[:start] + '''            loco_models, baseline_p, full_v = train_matchers(
                cfg, D, X, names, left, right, timer
            )
            direction = f"{D.labels[ARGS.train_code]}->{D.labels[ARGS.test_code]}"
            D = replace(D, cv_ids=D.cv_ids[D.country[D.cv_ids] == ARGS.test_code])
            baseline_sel = loco_select(D, baseline_p, bundle["tau_median"])
            baseline_f = gm.per_s1_f(baseline_sel, D.s1, D.y, D.ntrue)
            baseline_score = gm.macro(baseline_f, D.cv_ids)
            expected = float(bundle["evaluation"]["loco"][direction])
            difference_pp = 100 * (baseline_score - expected)
            report.update(
                direction=direction,
                baseline_loco=baseline_score,
                saved_baseline_loco=expected,
                baseline_difference_pp=difference_pp,
                tau=float(bundle["tau_median"]),
                seed=cfg["seed"],
                xgb_parameters=cfg.get("xgb"),
                train_code=ARGS.train_code,
                test_code=ARGS.test_code,
                target_entities=int(D.cv_ids.numel()),
            )
            print(
                f"L0 | parity | {direction}: reproduced={baseline_score:.12f}, "
                f"saved={expected:.12f}, difference={difference_pp:+.6f} pp, "
                f"tau={bundle['tau_median']} | next: parity check",
                flush=True,
            )
            with timer.host("io"):
                with open(pilot / "diagnostic_matchers.pkl", "wb") as handle:
                    pickle.dump(loco_models, handle)
                stage.save_p(
                    pilot / "diagnostic_baseline_predictions.parquet",
                    D, baseline_p,
                )
            report["historical_score_reproduced"] = (
                abs(baseline_score - expected) < 1e-6
            )
            report["comparison_protocol"] = (
                "Shipped CV feature/prediction parity required. "
                "Rescue compared against the same frozen current LOCO models "
                "and separately against the historical LOCO score. "
                "Historical retraining drift remains unresolved."
            )
            report.update(
                direction=direction, baseline_loco=baseline_score,
                neural_scope="Existing frozen BGE/Platt protocol; neural training is not country-held-out",
            )
            with timer.host("io"):
                with open(pilot / "matchers.pkl", "wb") as handle:
                    pickle.dump(loco_models, handle)
            pairs = prepare_pairs(
                cfg, D, baseline_p, bundle, table, root, pilot, timer, report
            )
            score_new(cfg, pairs, table, pilot, timer)
''' + source[end:]

start = source.index("            # Reuse the shipped Platt coefficients without refitting.")
end = source.index("            freeze_check()", start)
source = (
    source[:start] + "            old_v = full_v[old_rows]\n"
    "            del full_v\n\n" + source[end:]
)

start = source.index("            def predict(base, ambiguity, miss, verifier, folds):")
end = source.index("            nold = old_indices.numel()", start)
source = source[:start] + '''            def predict(base, ambiguity, miss, verifier, folds):
                xl = torch.cat([base, verifier], 1)
                xr = torch.cat([
                    base, ambiguity, ambiguity * miss[:, None], verifier
                ], 1)
                return (
                    .5 * loco_models[0].predict(xl)
                    + .5 * loco_models[1].predict(xr)
                )

''' + source[end:]

change('"scores_oof_empty_rescue_cv.parquet"', 'f"scores_oof_{TAG}.parquet"')
change('f"empty_rescue_cv_f{j}"', 'f"{TAG}_f{j}"')
change(
    "            selected = pl.select_cv(E, p, taus)",
    '            selected = loco_select(E, p, bundle["tau_median"])',
)
change(
    '            report["cv_screen_pass"] = delta >= 0.10 and ci[0] > 0',
    '''            historical_delta = 100 * (
                gm.macro(f, D.cv_ids) - report["saved_baseline_loco"]
            )
            report["delta_vs_historical_loco_pp"] = historical_delta
            report["loco_screen_pass"] = (
                delta >= -0.05 and historical_delta >= -0.05
            )
            print(
                f"LOCO | comparisons | current={delta:+.4f} pp, "
                f"historical={historical_delta:+.4f} pp "
                "| next: acceptance gate", flush=True,
            )
            report["delta_loco_pp"] = delta
            report["rescued_loco"] = gm.macro(f, D.cv_ids)
            report["scope"] = (
                "Target-country evaluation only; matcher/prefilter trained on "
                "source-country labels. Existing frozen BGE features. "
                "No target fitting or threshold tuning."
            )
            verdict = "PASS" if report["loco_screen_pass"] else "FAIL"
            print(
                f"LOCO | {verdict} | {direction}: {delta:+.4f} pp "
                "| next: combined gate", flush=True,
            )''',
)
change('"stage": "empty_rescue_matcher_cv"', '"stage": "empty_rescue_loco"')
change('if __name__ == "__main__":\n    main()', "")
namespace = dict(globals())
namespace["__file__"] = str(HERE / "eval_p4_empty_rescue_cv.py")
exec(compile(source, "<loco_rescue_evaluator>", "exec"), namespace)
namespace["main"]()
