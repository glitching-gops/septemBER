"""Independent re-check of output/*.tsv against the challenge format rules
(does not use the organisers' validator or ber's writers)."""
from pathlib import Path

import pytest

from ber.io_utils import load_config, read_column

A = load_config()["abs"]
FILES = {
    "matching_results.tsv": "matched_entity_ids",
    "candidate_pairs.tsv": "candidate_entity_ids",
}


def _parse(path: Path, col: str):
    raw = path.read_bytes()
    text = raw.decode("utf-8")  # must be valid UTF-8
    assert "\r" not in text, "CR line endings"
    assert text.endswith("\n")
    lines = text.split("\n")[:-1]
    assert lines[0] == f"source1_entity_id\t{col}", lines[0]
    rows = {}
    for n, line in enumerate(lines[1:], 2):
        assert line.count("\t") == 1, f"line {n}: expected exactly one tab"
        s1, ids = line.split("\t")
        assert s1.startswith("S1-") and s1 == s1.strip(), f"line {n}: bad S1 id {s1!r}"
        assert s1 not in rows, f"duplicate row for {s1}"
        assert '"' not in ids and "'" not in ids and " " not in ids, f"line {n}: quoting/space in list"
        lst = ids.split(",") if ids else []
        assert all(lst), f"line {n}: empty element in list"
        assert len(lst) == len(set(lst)), f"line {n}: duplicate id in list"
        assert all(x.startswith(("S2-", "S3-")) for x in lst), f"line {n}: non S2/S3 id"
        rows[s1] = set(lst)
    return rows


@pytest.fixture(scope="module")
def outputs():
    paths = {name: A["output_dir"] / name for name in FILES}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("final output files are not available")
    return {name: _parse(p, FILES[name]) for name, p in paths.items()}


@pytest.fixture(scope="module")
def test_s1_ids():
    if not A["test.source1"].exists():
        pytest.skip("raw test data not available")
    return set(read_column(A["test.source1"], "entity_id"))


def test_one_row_per_test_s1(outputs, test_s1_ids):
    for name, rows in outputs.items():
        assert set(rows) == test_s1_ids, name


def test_matches_subset_of_candidates(outputs):
    m, c = outputs["matching_results.tsv"], outputs["candidate_pairs.tsv"]
    bad = [s1 for s1, ids in m.items() if not ids <= c.get(s1, set())]
    assert not bad, f"{len(bad)} rows with matches outside candidates, e.g. {bad[:3]}"
