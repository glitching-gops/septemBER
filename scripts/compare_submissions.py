"""Compare two submissions row by row -> exit 0 if they agree closely enough.

    python scripts/compare_submissions.py --a submissions/sub1 --b submissions/sub1_repro

Reports identical-file hashes, per-S1 set agreement and the pair-level Jaccard. The
acceptance bar for the reproduction run is >= 99.9% of S1 rows with an identical predicted
set; any shortfall is printed with examples so it can be explained rather than hidden.

A bit-identical result is not guaranteed even with a fixed seed: the E5 embeddings are
computed in fp16 on GPU, where the order of cuBLAS reductions can move a cosine by ~1e-3
and flip a borderline candidate's rank. That is exactly why the bar is row agreement rather
than an md5 match.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ber.io_utils import read_id_list_file

FILES = ("matching_results.tsv", "candidate_pairs.tsv")
BAR = 0.999


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(a: Path, b: Path) -> dict:
    out = {"files": {}, "ok": True}
    for name in FILES:
        pa, pb = a / name, b / name
        ha, hb = md5(pa), md5(pb)
        if ha == hb:
            # Byte equality proves set equality; count rows with streaming I/O.
            with pa.open("rb") as fh:
                rows = sum(1 for _ in fh) - 1
            out["files"][name] = {"md5_a": ha, "md5_b": hb, "identical": True,
                                  "rows_a": rows, "rows_b": rows, "rows_shared": rows,
                                  "row_agreement": 1.0, "pair_jaccard": 1.0, "examples": {}}
            continue
        _, la = read_id_list_file(pa)
        _, lb = read_id_list_file(pb)
        shared = set(la) & set(lb)
        same = sum(1 for k in shared if set(la[k]) == set(lb[k]))
        inter = sum(len(set(la[k]) & set(lb[k])) for k in shared)
        union = sum(len(set(la[k]) | set(lb[k])) for k in shared)
        agree = same / len(shared) if shared else 0.0
        diff = [k for k in shared if set(la[k]) != set(lb[k])][:5]
        rec = {"md5_a": ha, "md5_b": hb, "identical": ha == hb,
               "rows_a": len(la), "rows_b": len(lb), "rows_shared": len(shared),
               "row_agreement": agree, "pair_jaccard": inter / union if union else 1.0,
               "examples": {k: {"a": sorted(set(la[k]))[:4], "b": sorted(set(lb[k]))[:4]} for k in diff}}
        out["files"][name] = rec
        if len(la) != len(lb) or agree < BAR:
            out["ok"] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    r = compare(Path(a.a), Path(a.b))
    for name, x in r["files"].items():
        flag = "IDENTICAL" if x["identical"] else f"row agreement {x['row_agreement']:.5%}"
        print(f"{name}: {flag}")
        print(f"  rows {x['rows_a']:,} vs {x['rows_b']:,} | pair Jaccard {x['pair_jaccard']:.6f}")
        print(f"  md5 {x['md5_a'][:12]} / {x['md5_b'][:12]}")
        for k, v in x["examples"].items():
            print(f"  differs {k}: a={v['a']} b={v['b']}")
    print(f"\n{'PASS' if r['ok'] else 'FAIL'} (bar: identical rows and >= {BAR:.1%} set agreement)")
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
