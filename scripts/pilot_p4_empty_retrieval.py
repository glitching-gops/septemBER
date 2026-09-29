"""Bounded GPU retrieval pilot for predicted-empty CV entities."""
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import run_p4_stage as stage
from run_p4_name_ambiguity import freeze_check
from ber import compute
from ber.embed import gpu_setup
from ber.evaluate.ledger import append_run
from ber.gpu import StageTimer
from ber.gpu import metric as gm, pipeline as pl, segment as sg
from ber.gpu.retrieval import topk
from ber.io import entity_attrs as EA, gpu_load as gl, ids as I
from ber.io_utils import load_config
from ber.verify import set_deterministic

TAG = "empty_retrieval_pilot_k100"
MAX_QUERIES = 2000
K = 100


def main():
    freeze_check()
    cfg = load_config("configs/p4_noaug.yaml")
    compute.limit_threads()
    stage.DEV = gpu_setup()
    assert stage.DEV.type == "cuda", "No CPU fallback."
    set_deterministic(cfg["seed"])
    timer = StageTimer(TAG)
    md = stage.p4_dir(cfg)
    root = Path(cfg["abs"]["cache_dir"]).parent / "reports"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / f"{TAG}.json"
    assert not report_path.exists(), "Pilot exists; refusing to overwrite."
    report = {"status": "running", "k": K, "submission_approved": False}

    try:
        with timer.gpu():
            with timer.host("io"):
                with open(md / "s1_rr_ambiguity_blend50.pkl", "rb") as f:
                    baseline = pickle.load(f)
            assert baseline["gate_pass"]
            stage.EXTRA.update(
                on=True, name="rr",
                no_rescue=baseline["left"]["verifier"].get("no_rescue", False),
            )
            D, X, _ = stage.load_dev(cfg, timer)
            del X
            with timer.host("io"):
                p = stage.load_p(
                    md / "s1_rr_ambiguity_blend50_p.parquet", D
                )
            taus = {int(k): float(v) for k, v in baseline["fold_taus"].items()}
            selected = pl.select_cv(D, p, taus)
            npred = sg.seg_count(D.s1[selected], D.n_ent)
            available = D.cv_ids[npred[D.cv_ids] == 0]

            # Sample BEFORE consulting any labels for target selection.
            rng = torch.Generator(device=stage.DEV).manual_seed(cfg["seed"])
            queries = available[torch.randperm(
                available.numel(), device=stage.DEV, generator=rng
            )[:MAX_QUERIES]]

            with timer.host("io"):
                table = I.entity_table(cfg, "train")
                source = gl.to_dev(table["src"].to_numpy(), stage.DEV).long()
                country = gl.to_dev(table["country"].to_numpy(), stage.DEV).long()
                attributes = EA.attrs(cfg, "train")
                address_ok = gl.to_dev(
                    attributes["addr_ok"], stage.DEV
                ).bool()
                reader = EA.EmbReader(cfg, "train", stage.DEV)
                gt = gl.gt_dev(cfg, stage.DEV)

            report.update(
                available_empty_cv_entities=int(available.numel()),
                sampled_queries=int(queries.numel()),
                selection="Random sample of predicted-empty CV S1; label-free",
            )
            print(f"P0 | queries selected | {queries.numel():,}/"
                  f"{available.numel():,} empty CV entities | "
                  "next: deeper name/address search", flush=True)

            def embeddings(field, entities):
                before = reader.io_wall, reader.io_cpu
                try:
                    result = reader(field, entities)
                finally:
                    timer.io(
                        reader.io_wall-before[0],
                        reader.io_cpu-before[1],
                    )
                return F.normalize(result, dim=1, eps=1e-6)

            chunks = []
            started = time.monotonic()
            for code in torch.unique(country[queries]).tolist():
                for src in (2, 3):
                    for field in ("name", "addr"):
                        freeze_check()
                        qids = queries[country[queries] == code]
                        targets = torch.nonzero(
                            (country == code) & (source == src)
                        ).flatten()
                        if field == "addr":
                            qids = qids[address_ok[qids]]
                            targets = targets[address_ok[targets]]
                        if not qids.numel() or not targets.numel():
                            continue
                        if time.monotonic()-started > 1200:
                            raise RuntimeError("20-minute internal pilot budget reached.")

                        began = time.monotonic()
                        Q = embeddings(field, qids)
                        T = embeddings(field, targets)
                        qi, ti, score, rank = topk(
                            Q, T, K, chunk=128, max_elems=1 << 27
                        )
                        chunks.append(qids[qi]*D.n_ent + targets[ti])
                        print(f"P1 | country-code {code}, S{src}, {field} | "
                              f"{qids.numel():,} queries x {targets.numel():,} targets; "
                              f"{time.monotonic()-began:.1f}s | next: remaining searches",
                              flush=True)
                        del Q, T, qi, ti, score, rank, targets, qids
                        torch.cuda.empty_cache()

            assert chunks, "No searches produced candidates."
            keys = torch.unique(torch.cat(chunks))
            del chunks
            s1, m = keys // D.n_ent, keys % D.n_ent
            existing = sg.isin_pairs(s1, m, D.s1, D.m, D.n_ent)
            s1, m = s1[~existing], m[~existing]
            new_true = gm.labels(
                s1, m, gt["s1"], gt["m"], D.n_ent
            )
            recovered = sg.seg_count(s1[new_true], D.n_ent)
            retrieved = sg.seg_count(D.s1[D.y & ~D.hold], D.n_ent)
            oracle_before = gm.f_from_counts(retrieved, retrieved, D.ntrue)
            oracle_after = gm.f_from_counts(
                retrieved+recovered, retrieved+recovered, D.ntrue
            )
            actual_f = gm.per_s1_f(selected, D.s1, D.y, D.ntrue)

            # Optimistic gain from perfectly accepting only new true pairs.
            old_tp = sg.seg_count(D.s1[selected], D.n_ent, D.y[selected])
            ideal_f = gm.f_from_counts(
                old_tp+recovered, npred+recovered, D.ntrue
            )
            report.update(
                status="complete",
                new_candidate_pairs=int(s1.numel()),
                new_true_pairs=int(new_true.sum()),
                new_pairs_per_sampled_query=(
                    s1.numel()/max(1, queries.numel())
                ),
                sampled_entities_with_new_true_candidates=int(
                    (recovered[queries] > 0).sum()
                ),
                previously_zero_true_candidate_entities_recovered=int((
                    (retrieved[queries] == 0)
                    & (recovered[queries] > 0)
                ).sum()),
                measured_full_cv_candidate_oracle_gain_pp=100*float(
                    (oracle_after[D.cv_ids]-oracle_before[D.cv_ids]).mean()
                ),
                optimistic_full_cv_matcher_gain_pp=100*float(
                    (ideal_f[D.cv_ids]-actual_f[D.cv_ids]).mean()
                ),
                scope=(
                    "Sampled-query CV pilot only. No extrapolation, matcher fitting "
                    "or test inference. Ideal gain ignores competing false claims."
                ),
            )
            with timer.host("io"):
                pq.write_table(pa.table({
                    "s1": s1.cpu().numpy(), "m": m.cpu().numpy(),
                }), root / f"{TAG}_new_pairs.parquet")
            print("P2 | pilot complete | "
                  f"{int(new_true.sum()):,} new true pairs | next: review return per hour",
                  flush=True)
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
                    "stage": "empty_retrieval_pilot",
                    "model": "cached_dense_k100",
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
