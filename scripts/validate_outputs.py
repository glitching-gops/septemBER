"""Validate a submission pair against the competition's output invariants -> exit 0 / 1.

    python scripts/validate_outputs.py --dir submissions/sub1
    python scripts/validate_outputs.py --matching out/matching_results.tsv \
        --candidate out/candidate_pairs.tsv [--split test] [--skip-official]

Checks, in order:
  1. headers, row counts and dtypes; one row per required S1, in order, no duplicates
  2. every predicted match is a SUBSET of that S1's scored candidate set
  3. every S2/S3 record is claimed by at most one S1 (exclusivity)
  4. both records of a predicted pair share the same country value
  5. every predicted/candidate id exists in the split's S2/S3 inputs
Then it runs the organisers' own validator (tests/validate_submission.py --check-ids)
unless --skip-official is given. Country values come from the data, never enumerated.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ber.io_utils import (CANDIDATE_HEADER, MATCHING_HEADER, load_config, read_id_list_file,
                          read_source)


class Failure(AssertionError):
    """An invariant was violated."""


def _read(path: Path, header: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Parse through the canonical reader so the separator is never re-guessed here."""
    got, lists = read_id_list_file(path)
    if got != header:
        raise Failure(f"{path.name}: header {got} != {header}")
    return list(lists), lists


def validate(matching: Path, candidate: Path, cfg: dict, split: str = "test",
             skip_official: bool = False) -> dict:
    checks: list[str] = []
    m_ids, m_lists = _read(matching, MATCHING_HEADER)
    c_ids, c_lists = _read(candidate, CANDIDATE_HEADER)

    s1, _ = read_source(cfg["abs"][f"{split}.source1"])
    required = list(s1["entity_id"])
    country_of = dict(zip(s1["entity_id"], s1["country"]))

    # 1. row counts, uniqueness, coverage
    for name, ids in (("matching_results", m_ids), ("candidate_pairs", c_ids)):
        if len(set(ids)) != len(ids):
            raise Failure(f"{name}: duplicate source1_entity_id rows")
        if ids != required:
            missing, extra = set(required) - set(ids), set(ids) - set(required)
            raise Failure(f"{name}: S1 id set/order mismatch "
                          f"({len(missing)} missing, {len(extra)} unexpected)")
    for k, v in m_lists.items():
        if len(set(v)) != len(v):
            raise Failure(f"matching_results: {k} lists a duplicate id")
    checks.append(f"row counts and S1 coverage: {len(m_ids):,} rows, one per required S1")

    msets = {k: set(v) for k, v in m_lists.items()}
    csets = {k: set(v) for k, v in c_lists.items()}

    # 2. matching subset of candidates
    bad = [k for k in msets if not msets[k] <= csets.get(k, set())]
    if bad:
        k = bad[0]
        raise Failure(f"matching_results: {len(bad):,} S1 predict ids not in their candidate set "
                      f"(e.g. {k}: {sorted(msets[k] - csets.get(k, set()))[:3]})")
    checks.append("every predicted match is a subset of that S1's candidate set")

    # 3. exclusivity: each target claimed at most once
    claimed: dict[str, str] = {}
    dupes = 0
    for k, ids in msets.items():
        for t in ids:
            if t in claimed:
                dupes += 1
            else:
                claimed[t] = k
    if dupes:
        raise Failure(f"matching_results: {dupes:,} S2/S3 ids claimed by more than one S1")
    checks.append(f"exclusivity: {len(claimed):,} predicted ids, none claimed twice")

    # 4/5. ids exist, and country matches within every predicted pair
    tgt_country: dict[str, str] = {}
    for src in ("source2", "source3"):
        t, _ = read_source(cfg["abs"][f"{split}.{src}"])
        tgt_country.update(zip(t["entity_id"], t["country"]))
    unknown = [t for t in claimed if t not in tgt_country]
    if unknown:
        raise Failure(f"matching_results: {len(unknown):,} ids absent from the {split} S2/S3 inputs "
                      f"(e.g. {unknown[:3]})")
    mismatch = [(t, k) for t, k in claimed.items() if tgt_country[t] != country_of[k]]
    if mismatch:
        t, k = mismatch[0]
        raise Failure(f"matching_results: {len(mismatch):,} cross-country pairs "
                      f"(e.g. {k} [{country_of[k]!r}] -> {t} [{tgt_country[t]!r}])")
    checks.append("all predicted ids exist and every pair is within one country")

    cand_ids = {t for ids in csets.values() for t in ids}
    unknown_c = [t for t in cand_ids if t not in tgt_country]
    if unknown_c:
        raise Failure(f"candidate_pairs: {len(unknown_c):,} ids absent from the {split} S2/S3 inputs")
    checks.append(f"candidate ids all exist ({len(cand_ids):,} distinct)")

    official = None
    if not skip_official:
        val = cfg["abs"]["validator"]  # organisers' script, located by configs/default.yaml
        r = subprocess.run([sys.executable, str(val), "--matching", str(matching),
                            "--candidate", str(candidate), "--test-dir", str(cfg["abs"]["test_dir"]),
                            "--check-ids"], capture_output=True, text=True)
        official = r.stdout.strip()
        if r.returncode != 0:
            raise Failure(f"organisers' validator failed:\n{official}\n{r.stderr.strip()}")

    return {"s1": len(m_ids), "predicted_pairs": sum(len(v) for v in msets.values()),
            "candidate_pairs": sum(len(v) for v in csets.values()),
            "empty": sum(1 for v in msets.values() if not v), "checks": checks, "official": official}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2.yaml")
    ap.add_argument("--dir", help="directory holding matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--matching")
    ap.add_argument("--candidate")
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-official", action="store_true")
    a = ap.parse_args()
    if a.dir:
        d = Path(a.dir)
        matching, candidate = d / "matching_results.tsv", d / "candidate_pairs.tsv"
    else:
        matching, candidate = Path(a.matching), Path(a.candidate)
    cfg = load_config(a.config)
    try:
        r = validate(matching, candidate, cfg, a.split, a.skip_official)
    except Failure as e:
        print(f"FAIL — {e}")
        raise SystemExit(1)
    for line in r["checks"]:
        print(f"  ok  {line}")
    print(f"\n{r['s1']:,} S1 | {r['predicted_pairs']:,} predicted pairs | "
          f"{r['candidate_pairs']:,} candidates | {r['empty']:,} empty predictions "
          f"({r['empty'] / r['s1']:.2%})")
    if r["official"]:
        print("\n--- organisers' validator ---\n" + r["official"])
    print("\nPASS")


if __name__ == "__main__":
    main()
