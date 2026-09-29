"""Final push, R1: measure the ORACLE value of each retrieval source on the dev slice (GPU-only).

    python scripts/run_p4_rr.py measure --arm s1_r2na_norescue

Base = the shipped dev candidate table (dense K=15 per source). Sources measured (new pairs only):
  KX<K>   the same fused dense blockers cut deeper (cand_rank 16..K); test has the same blocker
          depth (rank caps 20/20/5), so this is exactly what re-fusing test with K produces
  H2<k>   second hop: for anchor pairs (S1, a) with the shipped stage-1 OOF p > 0.80, the top-k
          e5 name / address neighbours of record a among the S2/S3 records of the same country,
          (s)ame source as a and (o)ther source; ``H2<k>nc`` drops neighbours whose house number /
          postcode conflicts with the anchor's. Anchors are chosen from p only, never labels.
  E7      the empty-address name sub-index (k = 3), as built in P4
  LXP1    P1's lexical blockers b1-b4 (archived dev candidates): cross-check / upper bound for LX
Per source / union: new pairs (per S1 mean, p95), true pairs recovered (never retrieved vs all
missed by subA), completeness, oracle ceiling (each S1 predicts exactly its retrieved true
pairs) and its gain, projected test pairs and ceiling gain per million test pairs, overlaps,
and slices (non-Latin target name, empty target address, S1 under-predicted by subA).
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
from ber.evaluate.ledger import append_run  # noqa: E402
from ber.gpu import StageTimer  # noqa: E402
from ber.gpu import metric as gm  # noqa: E402
from ber.gpu import pipeline as pl  # noqa: E402
from ber.gpu import retrieval  # noqa: E402
from ber.gpu import segment as sg  # noqa: E402
from ber.io import entity_attrs as EA  # noqa: E402
from ber.io import gpu_load as gl  # noqa: E402
from ber.io import ids as I  # noqa: E402
from ber.io_utils import load_config  # noqa: E402

DEV = None
N_TEST_S1 = 1_732_544
N_TEST_PAIRS = 51_976_320


def rrdir(cfg) -> Path:
    import os
    d = Path(os.environ.get("P4_EXTRA_ROOT", str(Path(cfg["abs"]["cache_dir"]).parent / "rr")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def pairs_key(s1, m, n):
    return s1 * n + m


def uniq_pairs(s1, m, n):
    k = torch.unique(pairs_key(s1, m, n))
    return k // n, k % n


def cmd_measure(args, cfg, timer) -> None:
    sys.argv = sys.argv[:1]
    import run_p4_stage as S
    S.DEV = DEV
    n = len(I.entity_table(cfg, "train"))
    D, X0, feats0 = S.load_dev(cfg, timer)
    md = Path(cfg["abs"]["models_dir"]) / "p4"
    with timer.host("io"):
        with open(md / f"{args.arm}.pkl", "rb") as fh:
            b = pickle.load(fh)
        p = S.load_p(md / f"{args.arm}_p.parquet", D)
        gt = gl.gt_dev(cfg, DEV)
        meta = gl.dev_meta(cfg, DEV)
        ent = I.entity_table(cfg, "train")
        src_e = gl.to_dev(ent["src"].to_numpy(), DEV)
        cty_e = gl.to_dev(ent["country"].to_numpy(), DEV)
        at = EA.attrs(cfg, "train")
        ok_e, hn_e, pc_e = gl.to_dev(at["addr_ok"], DEV), gl.to_dev(at["hn"], DEV), gl.to_dev(at["pc"], DEV)
        norm_script = __import__("pandas").concat([__import__("pandas").read_parquet(
            Path(cfg["abs"]["cache_dir"]) / "norm" / f"train_{s}.parquet", columns=["entity_id", "name_script"])
            for s in ("source2", "source3")])
        nl_ids = I.encode(cfg, "train", norm_script.loc[norm_script["name_script"] != "latin", "entity_id"])
        nonlatin_e = torch.zeros(n, dtype=torch.bool, device=DEV)
        nonlatin_e[gl.to_dev(nl_ids, DEV)] = True
        candidate_path = Path(cfg["abs"]["p2_cache_dir"]) / "cand/dev.parquet"
        raw = pq.read_table(candidate_path, columns=["cand_rank"])
        kr = I.encode_pairs_cached(cfg, "train", candidate_path, "dev_cand_raw")
        raw_s1, raw_m = gl.to_dev(kr["s1"], DEV), gl.to_dev(kr["m"], DEV)
        raw_rank = gl.to_dev(raw.column("cand_rank").to_numpy(), DEV)
    out = {"arm": args.arm}
    with timer.gpu():
        dev_s1 = meta["s1"]
        in_dev = torch.zeros(n, dtype=torch.bool, device=DEV)
        in_dev[dev_s1] = True
        gsel = in_dev[gt["s1"]]
        t_s1, t_m = gt["s1"][gsel], gt["m"][gsel]
        tkey = pairs_key(t_s1, t_m, n)
        base_key = torch.unique(pairs_key(D.s1, D.m, n))
        in_base = torch.isin(tkey, base_key)
        # subA predictions on dev (CV S1 with fold taus, holdout with the median tau)
        sel = pl.select_cv(D, p, b["fold_taus"], b["winner"]) | pl.select_hold(D, p, b["tau_median"], b["winner"])
        pred_key = pairs_key(D.s1[sel], D.m[sel], n)
        predicted = torch.isin(tkey, pred_key)
        missed = ~predicted
        never = ~in_base
        # buckets -> under-predicted S1 in subA
        tp = sg.seg_count(D.s1[sel], n, D.y[sel])
        npred = sg.seg_count(D.s1[sel], n)
        under = (D.ntrue > 0) & (tp > 0) & (npred - tp == 0) & (D.ntrue - tp > 0)
        n_dev = dev_s1.numel()

        def ceiling(extra_key=None):
            ck = base_key if extra_key is None else torch.unique(torch.cat([base_key, extra_key]))
            hit = torch.isin(tkey, ck)
            ntp = sg.seg_count(t_s1[hit], n)
            f = gm.f_from_counts(ntp, ntp, D.ntrue)
            return float(f[dev_s1].mean()), float(hit.double().mean())

        c0, comp0 = ceiling()
        out["base"] = {"pairs": int(base_key.numel()), "completeness": comp0, "ceiling": c0,
                       "true_pairs": int(tkey.numel()), "missed_by_subA": int(missed.sum()),
                       "never_retrieved": int(never.sum()), "never_share_of_missed": float(never.sum() / missed.sum())}
        print(json.dumps(out["base"]), flush=True)
        sources = {}

        def register(name, s1, m, secs):
            s1, m = uniq_pairs(s1, m, n)
            k = pairs_key(s1, m, n)
            new = ~torch.isin(k, base_key)
            s1, m, k = s1[new], m[new], k[new]
            sources[name] = k
            if not name.startswith("U:") and not args.saved:
                pq.write_table(pa.table({"s1": s1.cpu().numpy(), "m": m.cpu().numpy()}), rrdir(cfg) / f"dev_srcm_{name}.parquet")
            hit = torch.isin(tkey, k)
            per = sg.seg_count(s1, n)[dev_s1].double()
            c1, comp1 = ceiling(k)
            ppm = per.mean().item() * N_TEST_S1 / 1e6
            r = {"new_pairs": int(k.numel()), "per_s1_mean": float(per.mean()), "per_s1_p95": float(torch.quantile(per, 0.95)),
                 "true_recovered": int(hit.sum()), "rec_of_never": float((hit & never).sum() / never.sum()),
                 "rec_of_missed": float((hit & missed).sum() / missed.sum()),
                 "precision": float(hit.sum() / max(1, k.numel())),
                 "completeness": comp1, "ceiling": c1, "ceiling_gain_pp": 100 * (c1 - c0),
                 "test_pairs_M_proj": ppm, "test_pairs_pct": 100 * ppm * 1e6 / N_TEST_PAIRS,
                 "gain_pp_per_M_test": 100 * (c1 - c0) / max(1e-9, ppm),
                 "slice_nonlatin": int((hit & nonlatin_e[t_m]).sum()), "slice_nonlatin_never": int((never & nonlatin_e[t_m]).sum()),
                 "slice_empty_addr": int((hit & ~ok_e[t_m]).sum()), "slice_empty_never": int((never & ~ok_e[t_m]).sum()),
                 "slice_under_subA": int((hit & under[t_s1]).sum()), "slice_under_missed": int((missed & under[t_s1]).sum()),
                 "seconds": round(secs, 1)}
            out[name] = r
            print(name, json.dumps({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in r.items()}), flush=True)

        # ---- KX: deeper cut of the same fused blockers
        for K in (20, 25, 30):
            t0 = time.time()
            m_ = (raw_rank > 15) & (raw_rank <= K) & in_dev[raw_s1]
            register(f"KX{K}", raw_s1[m_], raw_m[m_], time.time() - t0)
        if args.saved:  # reuse the saved dev sources (H2 variants, capped P1 lexical, GPU lexical)
            for nm in ("H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc", "H2a5snc", "H2n5onc", "LXP1r1", "LXP1r2", "LXP1r3",
                       "LX1", "LX2n", "LX2a", "LX3n", "LX3a", "LX4"):
                f = rrdir(cfg) / f"dev_src_{nm}.parquet"
                if not f.exists():
                    f = rrdir(cfg) / f"dev_srcm_{nm}.parquet"
                if f.exists():
                    with timer.host("io"):
                        t_ = pq.read_table(f, columns=["s1", "m"])
                        a_, b_ = gl.to_dev(t_.column("s1").to_numpy(), DEV), gl.to_dev(t_.column("m").to_numpy(), DEV)
                    register(nm, a_, b_, 0.0)
            ls = sources
            for un, parts in {"LX1+LX2n+LX2a": ["LX1", "LX2n", "LX2a"],
                              "LXG+H2sel": ["LX1", "LX2n", "LX2a", "H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc"],
                              "LXG+H2sel+E7": ["LX1", "LX2n", "LX2a", "H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc", "E7x"],
                              "LXP1r2+H2sel": ["LXP1r2", "H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc"],
                              "LX3n+LX3a": ["LX3n", "LX3a"],
                              "LX4+LX3a+LX1+H2sel": ["LX4", "LX3a", "LX1", "H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc"],
                              "LX4+H2sel": ["LX4", "H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc"],
                              "LX3+LX1+LX2a+H2sel": ["LX3n", "LX3a", "LX1", "LX2a", "H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc"]}.items():
                if all(pn in ls for pn in parts):
                    k = torch.cat([ls[pn] for pn in parts])
                    register("U:" + un, k // n, k % n, 0.0)
        # ---- E7
        with timer.host("io"):
            e7 = pq.read_table(Path(cfg["abs"]["cache_dir"]).parent / "e7" / "dev_raw.parquet")
            e7s1, e7m = gl.to_dev(e7.column("s1").to_numpy(), DEV), gl.to_dev(e7.column("m").to_numpy(), DEV)
        register("E7", e7s1, e7m, 71.0)
        if args.saved:
            sources["E7x"] = sources["E7"]
            for un, parts in {"LXG+H2sel+E7": ["LX1", "LX2n", "LX2a", "H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc", "E7"],
                              "LXG+H2sel+KX20": ["LX1", "LX2n", "LX2a", "H2a3snc", "H2a3onc", "H2n3snc", "H2n3onc", "KX20"]}.items():
                k = torch.cat([sources[pn] for pn in parts])
                register("U:" + un, k // n, k % n, 0.0)
        if not args.saved:
            # ---- LX (P1 lexical blockers, archived): cross-check / upper bound
            with timer.host("io"):
                A = Path("/scratch/user4/archive/p1/scratch/user4/ber/cache/cand/dev")
                ls1, lm, lr = [], [], []
                for f in sorted(A.glob("lexical_*.parquet")):
                    k = I.encode_pairs_cached(cfg, "train", f, f"p1_{f.stem}")
                    ls1.append(gl.to_dev(k["s1"], DEV))
                    lm.append(gl.to_dev(k["m"], DEV))
                    tb = pq.read_table(f, columns=["b1_rank", "b2_rank", "b3_rank", "b4_hn", "b4_name"])
                    rk = [gl.to_dev(tb.column(c).fill_null(0).to_numpy(), DEV, torch.float32) for c in ("b1_rank", "b2_rank", "b3_rank")]
                    best = torch.stack([torch.where(r > 0, r, torch.full_like(r, 1e9)) for r in rk], 1).amin(1)
                    b4 = gl.to_dev(tb.column("b4_hn").fill_null(False).to_numpy(zero_copy_only=False), DEV, torch.bool) & \
                        gl.to_dev(tb.column("b4_name").fill_null(False).to_numpy(zero_copy_only=False), DEV, torch.bool)
                    lr.append(torch.where(b4, torch.ones_like(best), best))
            ls1, lm, lr = torch.cat(ls1), torch.cat(lm), torch.cat(lr)
            register("LXP1", ls1, lm, 0.0)
            for r_ in (1, 2, 3, 5):
                register(f"LXP1r{r_}", ls1[lr <= r_], lm[lr <= r_], 0.0)
            # ---- H2: second hop from anchors (OOF / holdout fold-mean p > 0.80; labels never used)
            anc = (p > 0.80)
            a_s1, a_m = D.s1[anc], D.m[anc]
            emb = EA.EmbReader(cfg, "train", DEV)
            for field in ("name", "addr"):
                t0 = time.time()
                res = {k_: {"s": ([], []), "o": ([], []), "snc": ([], []), "onc": ([], [])} for k_ in (3, 5)}
                recs = torch.unique(a_m)
                for c in torch.unique(cty_e[recs]).tolist():
                    rc = recs[cty_e[recs] == c]
                    qe = emb(field, rc)
                    for s in (2, 3):
                        tgt = torch.nonzero((src_e == s) & (cty_e == c)).squeeze(1)
                        if field == "addr":
                            tgt = tgt[ok_e[tgt]]
                        te = emb(field, tgt)
                        qi, ti, sc, rk = retrieval.topk(qe, te, 6)
                        q_ent, t_ent = rc[qi], tgt[ti]
                        keep = q_ent != t_ent
                        q_ent, t_ent, rk = q_ent[keep], t_ent[keep], rk[keep]
                        # re-rank without self
                        rr = sg.rank_in_group(q_ent, -rk.to(torch.float64), t_ent)
                        same = src_e[q_ent] == s
                        conflict = ((hn_e[q_ent] != 0) & (hn_e[t_ent] != 0) & (hn_e[q_ent] != hn_e[t_ent])) | \
                                   ((pc_e[q_ent] != 0) & (pc_e[t_ent] != 0) & (pc_e[q_ent] != pc_e[t_ent]))
                        for k_ in (3, 5):
                            w = rr <= k_
                            for tag, msk in (("s", same), ("o", ~same)):
                                res[k_][tag][0].append(q_ent[w & msk]); res[k_][tag][1].append(t_ent[w & msk])
                                res[k_][tag + "nc"][0].append(q_ent[w & msk & ~conflict]); res[k_][tag + "nc"][1].append(t_ent[w & msk & ~conflict])
                secs = time.time() - t0
                # expand record -> neighbour into (S1, neighbour) via the anchor pairs (record -> its anchor S1s)
                order = torch.argsort(a_m)
                am_sorted, as1_sorted = a_m[order], a_s1[order]
                for k_ in (3, 5):
                    for tag in ("s", "o", "snc", "onc"):
                        q = torch.cat(res[k_][tag][0]); t = torch.cat(res[k_][tag][1])
                        lo = torch.searchsorted(am_sorted, q)
                        hi = torch.searchsorted(am_sorted, q, right=True)
                        cnt = hi - lo
                        rep = torch.repeat_interleave(torch.arange(q.numel(), device=DEV), cnt)
                        off = torch.arange(rep.numel(), device=DEV) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
                        s1n = as1_sorted[lo[rep] + off]
                        register(f"H2{field[0]}{k_}{tag}", s1n, t[rep], secs)
            timer.io(emb.io_wall, emb.io_cpu)
        # ---- unions + overlaps (by true pairs recovered)
        names = list(sources)
        ov = {}
        for i, a in enumerate(names):
            ha = torch.isin(tkey, sources[a])
            for bname in names[i + 1:]:
                if bname[:2] != a[:2]:
                    ov[f"{a}&{bname}"] = int((ha & torch.isin(tkey, sources[bname])).sum())
        out["overlap_true"] = ov
        unions = {"KX25+E7": ["KX25", "E7"], "KX25+H2n3o": ["KX25", "H2n3o"], "KX25+H2n3o+E7": ["KX25", "H2n3o", "E7"],
                  "KX25+H2n3o+H2a3o+E7": ["KX25", "H2n3o", "H2a3o", "E7"], "KX30+H2n5o+H2a5o+E7": ["KX30", "H2n5o", "H2a5o", "E7"],
                  "KX25+H2n3onc+H2a3onc+E7": ["KX25", "H2n3onc", "H2a3onc", "E7"], "KX25+LXP1": ["KX25", "LXP1"],
                  "KX25+H2n3o+E7+LXP1": ["KX25", "H2n3o", "E7", "LXP1"]}
        for un, parts in unions.items():
            if all(pn in sources for pn in parts):
                k = torch.cat([sources[pn] for pn in parts])
                register("U:" + un, k // n, k % n, 0.0)
        # greedy union by marginal ceiling gain per projected test pair, under test-pair budgets
        pool = [nm for nm in sources if not nm.startswith("U:") and nm not in ("LXP1",)]
        per_pair = N_TEST_S1 / n_dev
        out["greedy"] = {}
        for budget in (0.25, 0.40, 0.60):
            chosen, cur = [], torch.empty(0, dtype=torch.int64, device=DEV)
            c_cur = c0
            while True:
                best = None
                for nm in pool:
                    if nm in chosen:
                        continue
                    u = torch.unique(torch.cat([cur, sources[nm]]))
                    if u.numel() * per_pair > budget * N_TEST_PAIRS:
                        continue
                    c_new, _ = ceiling(u)
                    add = (u.numel() - cur.numel()) * per_pair / 1e6
                    eff = (c_new - c_cur) / max(add, 1e-9)
                    if c_new > c_cur and (best is None or eff > best[0]):
                        best = (eff, nm, u, c_new)
                if best is None:
                    break
                chosen.append(best[1]); cur, c_cur = best[2], best[3]
            out["greedy"][str(budget)] = {"sources": chosen, "ceiling_gain_pp": 100 * (c_cur - c0),
                                          "test_pairs_pct": 100 * cur.numel() * per_pair / N_TEST_PAIRS}
            print("greedy", budget, out["greedy"][str(budget)], flush=True)
    st = timer.summary()
    out["time"] = st
    (rrdir(cfg) / "r1_measure.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(st), flush=True)
    append_run(cfg["abs"]["runs_ledger"], "p4_rr_r1_measure",
               {"stage": "p4_rr_r1", "model": "none", "device": "cuda", "peak_vram_gb": st["peak_vram_gb"],
                "wall_s": st["wall_s"], "gpu_time_s": st["gpu_time_s"], "cpu_time_s": st["cpu_time_s"],
                "cpu_violation": st["cpu_violation"], "pair_completeness": out["base"]["completeness"],
                "oracle_ceiling": out["base"]["ceiling"]},
               notes="R1 oracle measurement of retrieval sources vs dev K15 (see rr/r1_measure.json)")


def cmd_lx(args, cfg, timer) -> None:
    """GPU lexical sources for a split: LX1 (MinHash LSH on names), LX2n (rare name tokens), LX2a (rare
    address token + equal house number); top-``k`` per (S1, target source) partition; new pairs vs K15."""
    from ber.gpu import lexical as lx
    from ber.io import text_codes as tc
    split = args.split
    scope = "dev" if split == "train" else "test"
    n = len(I.entity_table(cfg, split))
    with timer.host("io"):
        ent = I.entity_table(cfg, split)
        src_e = gl.to_dev(ent["src"].to_numpy(), DEV)
        cty_e = gl.to_dev(ent["country"].to_numpy(), DEV)
        nb = tc.name_bytes(cfg, split)
        wn = tc.word_ids(cfg, split, "name_core")
        wa = tc.word_ids(cfg, split, "addr_core")
        hn = gl.to_dev(EA.attrs(cfg, split)["hn"], DEV)
        keys = gl.load_pairs(cfg, scope, split, [], DEV)
        q_all = gl.dev_meta(cfg, DEV)["s1"] if split == "train" else torch.nonzero(src_e == 1).squeeze(1)
        wn_e, wn_t = gl.to_dev(wn["ent"], DEV), gl.to_dev(wn["tok"], DEV)
        wa_e, wa_t = gl.to_dev(wa["ent"], DEV), gl.to_dev(wa["tok"], DEV)
    base_key = torch.unique(keys["s1"] * n + keys["m"])
    res = {nm: ([], [], [], []) for nm in ("LX1", "LX2n", "LX2a")}
    secs = {}
    with timer.gpu():
        for c in torch.unique(cty_e[q_all]).tolist():
            q = q_all[cty_e[q_all] == c]
            with timer.host("io"):
                qsig_nb = gl.to_dev(nb[q.cpu().numpy()], DEV)
            t0 = time.time()
            qsig = lx.minhash(qsig_nb)
            for s in (2, 3):
                tgt = torch.nonzero((src_e == s) & (cty_e == c)).squeeze(1)
                with timer.host("io"):
                    tnb = gl.to_dev(nb[tgt.cpu().numpy()], DEV)
                t1 = time.time()
                tsig = lx.minhash(tnb)
                a = lx.lsh_candidates(q, qsig, tgt, tsig, args.k, n_ent=n)
                secs["LX1"] = secs.get("LX1", 0) + time.time() - t1
                t1 = time.time()
                b = lx.rare_token_candidates(q, tgt, wn_e, wn_t, n, args.k, n_rare=2)
                secs["LX2n"] = secs.get("LX2n", 0) + time.time() - t1
                t1 = time.time()
                d = lx.rare_token_candidates(q, tgt, wa_e, wa_t, n, args.k, n_rare=1, hn=hn)
                secs["LX2a"] = secs.get("LX2a", 0) + time.time() - t1
                for nm, r in (("LX1", a), ("LX2n", b), ("LX2a", d)):
                    for i in range(4):
                        res[nm][i].append(r[i])
                print(f"  country {c} S{s}: {q.numel():,} q x {tgt.numel():,} t -> LX1 {a[0].numel():,}, LX2n "
                      f"{b[0].numel():,}, LX2a {d[0].numel():,}", flush=True)
    info = {}
    for nm, (s1l, ml, scl, rkl) in res.items():
        s1, m, sc, rk = torch.cat(s1l), torch.cat(ml), torch.cat(scl).double(), torch.cat(rkl)
        new = ~torch.isin(s1 * n + m, base_key)
        pq.write_table(pa.table({"s1": s1[new].cpu().numpy(), "m": m[new].cpu().numpy(),
                                 "score": sc[new].cpu().numpy(), "rank": rk[new].cpu().numpy()}),
                       rrdir(cfg) / f"{scope}_src_{nm}.parquet")
        info[nm] = {"pairs": int(s1.numel()), "new": int(new.sum()), "gpu_s": round(secs.get(nm, 0), 1)}
    info["time"] = timer.summary()
    print(json.dumps(info), flush=True)
    (rrdir(cfg) / f"{scope}_lx_info.json").write_text(json.dumps(info, indent=1))


H2_VARIANTS = {"H2a3snc": ("addr", True), "H2a3onc": ("addr", False), "H2n3snc": ("name", True), "H2n3onc": ("name", False)}


def cmd_h2(args, cfg, timer) -> None:
    """Second hop (chosen variants, k = 3, no house-number / postcode conflict) for a split, with
    provenance. Anchors: dev = the shipped stage-1 OOF / holdout p > 0.80; test = subA's test stage-1 p
    > 0.80 (``<sub>/test_scores.parquet``). Labels are never used. Output per new (S1, record) pair:
    parent p (max over its anchors), parent source code, variant bit flags."""
    split = args.split
    scope = "dev" if split == "train" else "test"
    n = len(I.entity_table(cfg, split))
    with timer.host("io"):
        ent = I.entity_table(cfg, split)
        src_e = gl.to_dev(ent["src"].to_numpy(), DEV)
        cty_e = gl.to_dev(ent["country"].to_numpy(), DEV)
        at = EA.attrs(cfg, split)
        ok_e, hn_e, pc_e = gl.to_dev(at["addr_ok"], DEV), gl.to_dev(at["hn"], DEV), gl.to_dev(at["pc"], DEV)
        keys = gl.load_pairs(cfg, scope, split, [], DEV)
        if split == "train":
            sys.argv = sys.argv[:1]
            import run_p4_stage as S
            S.DEV = DEV
            D, _, _ = S.load_dev(cfg, timer)
            p = S.load_p(Path(cfg["abs"]["models_dir"]) / "p4" / f"{args.arm}_p.parquet", D)
            a_s1, a_m, a_p = D.s1, D.m, p
        else:
            t = pq.read_table(Path(args.out_root) / args.sub / "test_scores.parquet", columns=["s1", "m", "p"])
            a_s1, a_m, a_p = (gl.to_dev(t.column(c).to_numpy(), DEV) for c in ("s1", "m", "p"))
        emb = EA.EmbReader(cfg, split, DEV)
    base_key = torch.unique(keys["s1"] * n + keys["m"])
    anc = a_p > 0.80
    a_s1, a_m, a_p = a_s1[anc], a_m[anc], a_p[anc]
    order = torch.argsort(a_m, stable=True)
    am_sorted, as1_sorted, ap_sorted = a_m[order], a_s1[order], a_p[order]
    L1, L2, L3, L4, L5 = [], [], [], [], []
    t0 = time.time()
    with timer.gpu():
        recs = torch.unique(a_m)
        for field in ("name", "addr"):
            for c in torch.unique(cty_e[recs]).tolist():
                rc = recs[cty_e[recs] == c]
                qe = emb(field, rc)
                for s in (2, 3):
                    tgt = torch.nonzero((src_e == s) & (cty_e == c)).squeeze(1)
                    if field == "addr":
                        tgt = tgt[ok_e[tgt]]
                    te = emb(field, tgt)
                    qi, ti, sc, rk = retrieval.topk(qe, te, 4)
                    q_ent, t_ent = rc[qi], tgt[ti]
                    keep = q_ent != t_ent
                    q_ent, t_ent, rk = q_ent[keep], t_ent[keep], rk[keep]
                    rr = sg.rank_in_group(q_ent, -rk.to(torch.float64), t_ent)
                    conflict = ((hn_e[q_ent] != 0) & (hn_e[t_ent] != 0) & (hn_e[q_ent] != hn_e[t_ent])) | \
                               ((pc_e[q_ent] != 0) & (pc_e[t_ent] != 0) & (pc_e[q_ent] != pc_e[t_ent]))
                    same = src_e[q_ent] == s
                    for bit, (nm, (fld, want_same)) in enumerate(H2_VARIANTS.items()):
                        if fld != field:
                            continue
                        w = (rr <= 3) & ~conflict & (same == want_same)
                        q, tt = q_ent[w], t_ent[w]
                        lo = torch.searchsorted(am_sorted, q)
                        hi = torch.searchsorted(am_sorted, q, right=True)
                        cnt = hi - lo
                        rep = torch.repeat_interleave(torch.arange(q.numel(), device=DEV), cnt)
                        off = torch.arange(rep.numel(), device=DEV) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
                        idx = lo[rep] + off
                        L1.append(as1_sorted[idx]); L2.append(tt[rep]); L3.append(ap_sorted[idx])
                        L4.append(src_e[q[rep]].to(torch.int64)); L5.append(torch.full_like(idx, 1 << bit))
        s1, m, pp, ps, bits = torch.cat(L1), torch.cat(L2), torch.cat(L3), torch.cat(L4), torch.cat(L5)
        key = s1 * n + m
        new = ~torch.isin(key, base_key)
        key, pp, ps, bits = key[new], pp[new], ps[new], bits[new]
        uk, inv = torch.unique(key, return_inverse=True)
        pmax = torch.full((uk.numel(),), -1.0, device=DEV).scatter_reduce(0, inv, pp.float(), "amax")
        # parent source of the best anchor; variant flags = OR of bits (unique (pair, bit) then sum)
        best = sg.group_argmax(inv, pp.to(torch.float64), ps, uk.numel())
        psrc = ps[best]
        ub = torch.unique(inv * 16 + bits)
        flags = torch.zeros(uk.numel(), dtype=torch.int64, device=DEV).index_add_(0, ub // 16, ub % 16)
    timer.io(emb.io_wall, emb.io_cpu)
    s1u, mu = uk // n, uk % n
    pq.write_table(pa.table({"s1": s1u.cpu().numpy(), "m": mu.cpu().numpy(), "h2_parent_p": pmax.cpu().numpy(),
                             "h2_parent_src": psrc.cpu().numpy(), "h2_flags": flags.cpu().numpy()}),
                   rrdir(cfg) / f"{scope}_src_H2sel.parquet")
    st = timer.summary()
    print(f"[h2] {scope}: {int(anc.sum()):,} anchors, {uk.numel():,} new pairs in {time.time() - t0:.0f}s; {st}", flush=True)
    (rrdir(cfg) / f"{scope}_h2_info.json").write_text(json.dumps({"anchors": int(anc.sum()), "new": int(uk.numel()), **st}))


def cmd_lx3(args, cfg, timer) -> None:
    """GPU char-n-gram TF-IDF blockers (P1 B1 / B2 weighting): LX3n on the name key text (3-4 grams,
    max_df 0.02) and LX3a on the address (3-grams, max_df 0.02); top-``k`` per (S1, target source)."""
    from ber.gpu import lexical as lx
    from ber.io import text_codes as tc
    split = args.split
    scope = "dev" if split == "train" else "test"
    n = len(I.entity_table(cfg, split))
    with timer.host("io"):
        ent = I.entity_table(cfg, split)
        src_e = gl.to_dev(ent["src"].to_numpy(), DEV)
        cty_e = gl.to_dev(ent["country"].to_numpy(), DEV)
        nb, ab = tc.name_bytes(cfg, split), tc.addr_bytes(cfg, split)
        wb = tc.word_ids(cfg, split, "both")
        wb_e, wb_t = gl.to_dev(wb["ent"], DEV), gl.to_dev(wb["tok"].astype("int64"), DEV) % (1 << 22)  # 22-bit ids
        pos_e = torch.full((n,), -1, dtype=torch.int64, device=DEV)
        keys = gl.load_pairs(cfg, scope, split, [], DEV)
        q_all = gl.dev_meta(cfg, DEV)["s1"] if split == "train" else torch.nonzero(src_e == 1).squeeze(1)
    base_key = torch.unique(keys["s1"] * n + keys["m"])
    res = {"LX3n": ([], [], [], []), "LX3a": ([], [], [], []), "LX4": ([], [], [], [])}
    secs = {"LX3n": 0.0, "LX3a": 0.0, "LX4": 0.0}
    with timer.gpu():
        for c in torch.unique(cty_e[q_all]).tolist():
            q = q_all[cty_e[q_all] == c]
            for s in (2, 3):
                tgt = torch.nonzero((src_e == s) & (cty_e == c)).squeeze(1)
                for nm, arr, ns in ((("LX3a", ab, (3,)),) if args.skip_lx3n else (("LX3n", nb, (3, 4)), ("LX3a", ab, (3,)))):
                    with timer.host("io"):
                        qb, tb = gl.to_dev(arr[q.cpu().numpy()], DEV), gl.to_dev(arr[tgt.cpu().numpy()], DEV)
                    t1 = time.time()
                    qr, qgid = lx.char_grams(qb, ns)
                    tr, tgid = lx.char_grams(tb, ns)
                    r = lx.tfidf_candidates(q, qr, qgid, tgt, tr, tgid, args.k, q_top=24, post_cap=20_000, q_chunk=5_000)
                    secs[nm] += time.time() - t1
                    for i in range(4):
                        res[nm][i].append(r[i])
                    print(f"  country {c} S{s} {nm}: {q.numel():,} q x {tgt.numel():,} t -> {r[0].numel():,} "
                          f"[{time.time() - t1:.0f}s]", flush=True)
                    del qb, tb, qr, qgid, tr, tgid
                # LX4 = P1's B3: word TF-IDF over name + address in one vocabulary (max_df 0.05)
                t1 = time.time()
                pos_e[:] = -1
                pos_e[q] = torch.arange(q.numel(), device=DEV)
                wq = pos_e[wb_e] >= 0
                qr, qgid = pos_e[wb_e[wq]], wb_t[wq]
                pos_e[:] = -1
                pos_e[tgt] = torch.arange(tgt.numel(), device=DEV)
                wt = pos_e[wb_e] >= 0
                tr, tgid = pos_e[wb_e[wt]], wb_t[wt]
                r = lx.tfidf_candidates(q, qr, qgid, tgt, tr, tgid, args.k, max_df=0.05, q_top=16, post_cap=20_000,
                                        q_chunk=5_000)
                secs["LX4"] += time.time() - t1
                for i in range(4):
                    res["LX4"][i].append(r[i])
                print(f"  country {c} S{s} LX4: -> {r[0].numel():,} [{time.time() - t1:.0f}s]", flush=True)
    info = {}
    for nm, (s1l, ml, scl, rkl) in res.items():
        if not s1l:
            continue
        s1, m, sc, rk = torch.cat(s1l), torch.cat(ml), torch.cat(scl).double(), torch.cat(rkl)
        new = ~torch.isin(s1 * n + m, base_key)
        pq.write_table(pa.table({"s1": s1[new].cpu().numpy(), "m": m[new].cpu().numpy(),
                                 "score": sc[new].cpu().numpy(), "rank": rk[new].cpu().numpy()}),
                       rrdir(cfg) / f"{scope}_src_{nm}.parquet")
        info[nm] = {"pairs": int(s1.numel()), "new": int(new.sum()), "gpu_s": round(secs[nm], 1)}
    info["time"] = timer.summary()
    print(json.dumps(info), flush=True)
    (rrdir(cfg) / f"{scope}_lx3_info.json").write_text(json.dumps(info, indent=1))


PICK = {"LX4": 1, "LX3a": 2, "LX1": 4, "H2sel": 8}   # R1 pick; bit per source
PROV = ["rr_src_flags", "rr_n_sources", "rr_lx4_score", "rr_lx4_rank", "rr_lx3a_score", "rr_lx3a_rank",
        "rr_lx1_score", "rr_lx1_rank", "rr_h2_parent_p", "rr_h2_parent_src", "rr_h2_flags"]


def cmd_union(args, cfg, timer) -> None:
    """New candidate pairs = union of the picked sources (already de-duplicated against K15), with
    provenance columns; written to rr/<scope>_union.parquet (int ids + string ids + provenance)."""
    split = args.split
    scope = "dev" if split == "train" else "test"
    n = len(I.entity_table(cfg, split))
    parts = {}
    with timer.host("io"):
        for nm in PICK:
            t = pq.read_table(rrdir(cfg) / f"{scope}_src_{nm}.parquet")
            parts[nm] = {c: gl.to_dev(t.column(c).to_numpy(), DEV) for c in t.column_names}
    with timer.gpu():
        keys = torch.unique(torch.cat([p_["s1"] * n + p_["m"] for p_ in parts.values()]))
        N = keys.numel()
        col = {c: torch.full((N,), float("nan"), dtype=torch.float32, device=DEV) for c in PROV}
        flags = torch.zeros(N, dtype=torch.int64, device=DEV)
        for nm, bit in PICK.items():
            p_ = parts[nm]
            j = torch.searchsorted(keys, p_["s1"] * n + p_["m"])
            flags.index_add_(0, j, torch.full_like(j, bit))
            if nm == "H2sel":
                col["rr_h2_parent_p"][j] = p_["h2_parent_p"].float()
                col["rr_h2_parent_src"][j] = p_["h2_parent_src"].float()
                col["rr_h2_flags"][j] = p_["h2_flags"].float()
            else:
                tag = nm.lower()
                col[f"rr_{tag}_score"][j] = p_["score"].float()
                col[f"rr_{tag}_rank"][j] = p_["rank"].float()
        col["rr_src_flags"] = flags.float()
        col["rr_n_sources"] = sum(((flags >> b) & 1) for b in range(len(PICK))).float()
    with timer.host("io"):
        ids = I.entity_table(cfg, split)["entity_id"].to_numpy(object)
        s1h, mh = (keys // n).cpu().numpy(), (keys % n).cpu().numpy()
        src = I.entity_table(cfg, split)["src"].to_numpy()
        tab = {"s1_id": ids[s1h], "m_id": ids[mh], "s1": s1h, "m": mh, "source": ["S%d" % v for v in src[mh]]}
        tab.update({c: v.cpu().numpy() for c, v in col.items()})
        pq.write_table(pa.table(tab), rrdir(cfg) / f"{scope}_union.parquet")
    print(f"[union] {scope}: {N:,} new pairs; per source "
          + ", ".join(f"{nm} {int(((flags & bit) > 0).sum()):,}" for nm, bit in PICK.items()) + f"; {timer.summary()}", flush=True)


def cmd_features(args, cfg, timer) -> None:
    """f2 features of the union's new pairs via FeatureContext (rapidfuzz on CPU: the permitted
    exception for this phase's new pairs; count and wall logged). Blocker columns of the main K15
    blockers are NaN (not retrieved by them), as for any candidate a blocker missed."""
    import joblib
    feats = joblib.load(Path(cfg["abs"]["models_dir"]) / "e1b.joblib")["features"]
    split = args.split
    scope = "dev" if split == "train" else "test"
    u = pq.read_table(rrdir(cfg) / f"{scope}_union.parquet").to_pandas()
    ent = I.entity_table(cfg, split)
    lab = ent["country_label"].to_numpy(object)
    cand = u[["s1_id", "m_id", "source"]].copy()
    cand["country"] = lab[u["s1"].to_numpy()]
    for c in ("b6n_rank", "b6n_score", "b6a_rank", "b6a_score", "b7_rank", "b7_score"):
        cand[c] = float("nan")
    from ber.match.build import FeatureContext
    c2 = {**cfg, "versions": {**cfg["versions"], "feature_version": "f2"}}
    t0, c0 = time.time(), time.process_time()
    print(f"[features] {scope}: {len(cand):,} pairs, {compute.n_jobs()} string workers; "
          f"estimated wall {(360 + len(cand) * 0.00008) / 60:.1f} min from the dev run", flush=True)
    ctx = FeatureContext(c2, split, gpu_features=True)
    parts = []
    for a in range(0, len(cand), 400_000):
        part = cand.iloc[a:a + 400_000].reset_index(drop=True)
        F = ctx.features(part)
        parts.append(F.reset_index(drop=True)[feats])
        print(f"  {min(a + 400_000, len(cand)):,}/{len(cand):,} [{time.time() - t0:.0f}s]", flush=True)
    import pandas as pd
    out = pd.concat([u.reset_index(drop=True), pd.concat(parts, ignore_index=True)], axis=1)
    out.to_parquet(rrdir(cfg) / f"{scope}_feats.parquet", index=False)
    wall, cpu = time.time() - t0, time.process_time() - c0
    print(f"[features] {scope}: {len(out):,} new pairs; rapidfuzz/FeatureContext wall {wall / 60:.1f} min, "
          f"CPU {cpu / 60:.1f} min", flush=True)
    append_run(cfg["abs"]["runs_ledger"], f"p4_rr_features_{scope}",
               {"stage": "p4_rr_features", "model": "f2", "device": "cuda + cpu string features",
                "wall_s": round(wall, 1), "cpu_time_s": round(cpu, 1), "cpu_violation": False,
                "gpu_ids": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
                "peak_vram_gb": timer.summary()["peak_vram_gb"]},
               notes=f"R2 new pairs {scope}: {len(out):,}")


def cmd_sets(args, cfg, timer) -> None:
    """E1b p on the new pairs (exact GPU port of the P2 bundle) -> verifier scoring set (band [0.005, 0.99])."""
    import joblib
    from ber.gpu.xgb import from_sklearn, predict_mean
    e1b = joblib.load(Path(cfg["abs"]["models_dir"]) / "e1b.joblib")
    models = [from_sklearn(c) for c in e1b["folds"].values()]
    band = cfg["verifier"]["band"]
    split = args.split
    scope = "dev" if split == "train" else "test"
    f = rrdir(cfg) / f"{scope}_feats.parquet"
    with timer.host("io"):
        P = gl.pair_file(cfg, split, f, f"rr_{scope}_feats", e1b["features"], DEV)
    with timer.gpu():
        p = predict_mean(models, torch.stack([P[c] for c in e1b["features"]], 1)).to(torch.float64)
        inb = (p >= band[0]) & (p <= band[1])
    with timer.host("io"):
        ids = I.entity_table(cfg, split)["entity_id"].to_numpy(object)
        s1h, mh = P["s1"][inb].cpu().numpy(), P["m"][inb].cpu().numpy()
        pq.write_table(pa.table({"s1_id": ids[s1h], "m_id": ids[mh], "inband": [1] * len(s1h), "rescued": [0] * len(s1h),
                                 "p_e1b": p[inb].cpu().numpy()}), rrdir(cfg) / f"{scope}_rr_set.parquet")
    print(f"[sets] {scope}: {int(inb.sum()):,} of {p.numel():,} new pairs in the E1b band -> verifier; {timer.summary()}",
          flush=True)


def main() -> None:
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "lx", "h2", "lx3", "union", "features", "sets"])
    ap.add_argument("--sub", default="subA")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--skip-lx3n", action="store_true", help="lx3: only LX3a + LX4 (the R1 pick)")
    ap.add_argument("--saved", action="store_true", help="measure: load H2 / E7 / LX sources from rr/dev_src_*.parquet")
    ap.add_argument("--config", default="configs/p4_noaug.yaml")
    ap.add_argument("--arm", default="s1_r2na_norescue")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out_root is None:
        args.out_root = Path(cfg["abs"]["cache_dir"]).parent / "submissions"
    compute.limit_threads()
    from ber.embed import gpu_setup
    DEV = gpu_setup()
    from ber.verify import set_deterministic
    set_deterministic(cfg["seed"])
    timer = StageTimer(f"rr_{args.cmd}")
    with compute.RamMonitor(f"rr_{args.cmd}"):
        {"measure": cmd_measure, "lx": cmd_lx, "h2": cmd_h2, "lx3": cmd_lx3, "union": cmd_union,
         "features": cmd_features, "sets": cmd_sets}[args.cmd](args, cfg, timer)
    if args.cmd not in ("measure", "features"):
        st = timer.summary()
        append_run(cfg["abs"]["runs_ledger"], f"p4_rr_{args.cmd}_{args.split}",
                   {"stage": f"p4_rr_{args.cmd}", "model": "retrieval_union", "device": "cuda",
                    "gpu_ids": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
                    **{k: st[k] for k in ("wall_s", "gpu_time_s", "cpu_time_s", "cpu_violation", "peak_vram_gb")}},
                   notes=f"retrieval {args.cmd}, split={args.split}", extra=st)


if __name__ == "__main__":
    main()
