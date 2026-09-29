"""Grouped GPU prefilter pilot for saved predicted-empty retrieval pairs."""
import json
import math
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from run_p4_name_ambiguity import freeze_check
from ber import compute
from ber.embed import gpu_setup
from ber.evaluate.ledger import append_run
from ber.gpu import StageTimer
from ber.gpu import metric as gm, segment as sg
from ber.gpu.xgb import fit_calibrated
from ber.io import entity_attrs as EA, gpu_load as gl, ids as I
from ber.io_utils import load_config
from ber.verify import set_deterministic

TAG = "empty_retrieval_prefilter_pilot"
KEEP_FRACTION = 0.05


def main():
    freeze_check()
    cfg = load_config("configs/p4_noaug.yaml")
    compute.limit_threads()
    dev = gpu_setup()
    assert dev.type == "cuda", "No CPU fallback."
    set_deterministic(cfg["seed"])
    timer = StageTimer(TAG)
    root = Path(cfg["abs"]["cache_dir"]).parent / "reports"
    report_path = root / f"{TAG}.json"
    assert not report_path.exists(), "Pilot exists; refusing to overwrite."
    report = {"status": "running", "submission_approved": False}

    try:
        with timer.gpu():
            with timer.host("io"):
                pairs = pq.read_table(
                    root / "empty_retrieval_pilot_k100_new_pairs.parquet",
                    columns=["s1", "m"],
                )
                s1 = gl.to_dev(pairs["s1"].to_numpy(), dev).long()
                m = gl.to_dev(pairs["m"].to_numpy(), dev).long()
                table = I.entity_table(cfg, "train")
                n_ent = len(table)
                source = gl.to_dev(table["src"].to_numpy(), dev).long()
                metadata = gl.dev_meta(cfg, dev)
                truth = gl.gt_dev(cfg, dev)
                attributes = EA.attrs(cfg, "train")
                reader = EA.EmbReader(cfg, "train", dev)
            del pairs, table

            fold_e = torch.full((n_ent,), -1, dtype=torch.long, device=dev)
            hold_e = torch.zeros(n_ent, dtype=torch.bool, device=dev)
            fold_e[metadata["s1"]] = metadata["fold"]
            hold_e[metadata["s1"]] = metadata["holdout"]
            folds = fold_e[s1]
            assert bool((folds >= 0).all())
            assert not bool(hold_e[s1].any()), "Holdout pair in pilot."
            y = gm.labels(s1, m, truth["s1"], truth["m"], n_ent)
            assert s1.numel() > 0 and int(y.sum()) > 0
            print(f"F0 | input checked | {s1.numel():,} pairs, "
                  f"{int(y.sum())} positives; holdout excluded | "
                  "next: GPU features", flush=True)

            def embeddings(field, entities):
                before = reader.io_wall, reader.io_cpu
                try:
                    result = reader(field, entities)
                finally:
                    timer.io(
                        reader.io_wall-before[0],
                        reader.io_cpu-before[1],
                    )
                return F.normalize(result.float(), dim=1, eps=1e-6)

            cosine = torch.empty((s1.numel(), 2), device=dev)
            print("Embedding row I/O estimated 1–5 minutes; "
                  "all similarities computed on GPU.", flush=True)
            for start in range(0, s1.numel(), 65536):
                freeze_check()
                end = min(start+65536, s1.numel())
                for j, field in enumerate(("name", "addr")):
                    a = embeddings(field, s1[start:end])
                    b = embeddings(field, m[start:end])
                    cosine[start:end, j] = (a*b).sum(1)
                    del a, b

            hn = gl.to_dev(attributes["hn"], dev).long()
            pc = gl.to_dev(attributes["pc"], dev).long()
            ok = gl.to_dev(attributes["addr_ok"], dev).bool()
            both_hn = (hn[s1] != 0) & (hn[m] != 0)
            both_pc = (pc[s1] != 0) & (pc[m] != 0)
            both_address = ok[s1] & ok[m]

            count = sg.seg_count(s1, n_ent)[s1].float()
            nr = sg.rank_in_group(s1, cosine[:, 0].double(), m).float()
            ar = sg.rank_in_group(s1, cosine[:, 1].double(), m).float()
            nmax = sg.seg_max(cosine[:, 0], s1, n_ent)[s1]
            amax = sg.seg_max(cosine[:, 1], s1, n_ent)[s1]
            columns = [
                cosine[:, 0],
                torch.where(
                    both_address, cosine[:, 1],
                    torch.full_like(cosine[:, 1], float("nan")),
                ),
                torch.maximum(cosine[:, 0], torch.where(
                    both_address, cosine[:, 1], torch.zeros_like(count)
                )),
                nr/count.clamp(min=1), ar/count.clamp(min=1),
                cosine[:, 0]-nmax, cosine[:, 1]-amax,
                both_address.float(),
                both_hn.float(), (both_hn & (hn[s1] == hn[m])).float(),
                both_pc.float(), (both_pc & (pc[s1] == pc[m])).float(),
                (source[m] == 3).float(),
            ]
            names = [
                "name_cos", "addr_cos", "max_cos",
                "name_rank_fraction", "addr_rank_fraction",
                "name_margin", "addr_margin", "both_addresses",
                "both_house_numbers", "house_agree",
                "both_postcodes", "postcode_agree", "source3",
            ]
            X = torch.stack(columns, 1).float()
            del columns, cosine, hn, pc, ok, reader, attributes
            torch.cuda.empty_cache()

            params = {
                "device": "cuda", "tree_method": "hist",
                "n_estimators": 250, "max_depth": 3,
                "learning_rate": 0.05, "min_child_weight": 5,
                "scale_pos_weight": 100,
            }
            scores = torch.zeros(s1.numel(), device=dev)
            retained = torch.zeros_like(y)
            models = {}
            fold_reports = []
            for k in torch.unique(folds).tolist():
                freeze_check()
                tr, te = folds != k, folds == k
                assert int(y[tr].sum()) > 0, "No training positives."
                model = fit_calibrated(
                    X[tr], y[tr], s1[tr],
                    seed=cfg["seed"], cfg_xgb=params,
                )
                # Ranking uses raw GPU booster scores; no calibrated cutoff.
                scores[te] = model.raw(X[te])
                indices = torch.nonzero(te).flatten()
                budget = max(1, math.ceil(KEEP_FRACTION*indices.numel()))
                chosen = indices[torch.topk(scores[te], budget).indices]
                retained[chosen] = True
                models[int(k)] = model
                positives = int(y[te].sum())
                kept_positive = int((retained & te & y).sum())
                fold_reports.append({
                    "fold": int(k), "pairs": int(te.sum()),
                    "positives": positives,
                    "retained_positives": kept_positive,
                    "recall": kept_positive/positives if positives else None,
                })
                print(f"F1/fold{k} | scored out-of-fold | "
                      f"{kept_positive}/{positives} positives retained | "
                      "next: remaining folds", flush=True)

            positives = int(y.sum())
            kept_positive = int((retained & y).sum())
            recall = kept_positive/positives
            reduction = 1-float(retained.float().mean())
            recovered_entities = int(torch.unique(s1[y]).numel())
            retained_entities = int(torch.unique(s1[retained & y]).numel())
            passed = recall >= 0.80 and reduction >= 0.94999

            report.update(
                status="prefilter_pass" if passed else "prefilter_fail",
                gate_pass=passed,
                total_pairs=int(s1.numel()),
                retained_pairs=int(retained.sum()),
                positives=positives,
                retained_positives=kept_positive,
                positive_recall=recall,
                candidate_reduction=reduction,
                positive_entities=recovered_entities,
                retained_positive_entities=retained_entities,
                folds=fold_reports,
                features=names,
                scope=(
                    "Grouped CV pilot only. Five-percent budgets use scores, "
                    "never labels. No holdout or test evaluation."
                ),
            )
            with timer.host("io"):
                pq.write_table(pa.table({
                    "s1": s1[retained].cpu().numpy(),
                    "m": m[retained].cpu().numpy(),
                    "prefilter_score": scores[retained].cpu().numpy(),
                }), root / f"{TAG}_retained_pairs.parquet")
                if passed:
                    md = Path(cfg["abs"]["models_dir"])/"p4"
                    with open(md/f"{TAG}.pkl", "wb") as f:
                        pickle.dump({
                            "folds": models, "features": names,
                            "parameters": params, "keep_fraction": KEEP_FRACTION,
                            "submission_approved": False,
                        }, f)
            print(f"F2 | {'PASS' if passed else 'FAIL'} | "
                  f"positive recall {recall:.1%}, reduction {reduction:.1%} | "
                  "next: review before expanding retrieval", flush=True)
            print(json.dumps(report, indent=2), flush=True)

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
                    "stage": "retrieval_prefilter_pilot",
                    "model": "xgboost",
                    "device": "cuda",
                    "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    **{k: st[k] for k in (
                        "wall_s", "cpu_time_s", "gpu_time_s",
                        "peak_vram_gb", "cpu_violation"
                    )},
                },
                notes=report["status"], extra=report,
            )
        print("REPORT:", report_path, flush=True)


if __name__ == "__main__":
    main()
