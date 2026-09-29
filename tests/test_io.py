import pytest

from ber.io_utils import (
    DataValidationError,
    load_config,
    parse_id_list,
    read_ground_truth,
    read_id_list_file,
    read_source,
    write_candidates,
    write_matching,
)

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_scratch_root_can_be_relocated(tmp_path, monkeypatch):
    monkeypatch.setenv("BER_SCRATCH_ROOT", str(tmp_path))
    cfg = load_config("configs/p4_noaug.yaml")
    assert cfg["abs"]["cache_dir"] == tmp_path / "p4" / "cache"
    assert cfg["abs"]["p2_features_dir"] == tmp_path / "p2" / "features"
    assert cfg["abs"]["raw"].is_relative_to(cfg["abs"]["root"])


def test_read_source_quotes_and_empty_fields(tmp_path):
    p = _write(
        tmp_path / "x_source2.tsv",
        HEADER
        + 'S2-1\t"Joe\'s" Diner\t12 "A" St, Town\tAA\n'
        + "S2-2\tNo Addr Co\t\tBB\n"
        + 'S2-3\t"unterminated\t  \tAA\n',
    )
    df, rep = read_source(p)
    assert len(df) == 3 == rep["raw_rows"]
    assert df.loc[0, "business_name"] == '"Joe\'s" Diner'
    assert df.loc[0, "business_address"] == '12 "A" St, Town'
    assert df.loc[2, "business_name"] == '"unterminated'
    assert rep["empty"]["business_address"] == 1
    assert rep["whitespace_only"]["business_address"] == 1
    assert rep["prefix"] == "S2-"
    assert (df.dtypes == object).all() or all(str(t) in ("str", "string", "object") for t in df.dtypes)


def test_read_source_na_like_strings_stay_strings(tmp_path):
    p = _write(tmp_path / "x_source1.tsv", HEADER + "S1-1\tNA\tNULL\tnan\n")
    df, _ = read_source(p)
    assert df.iloc[0].tolist() == ["S1-1", "NA", "NULL", "nan"]


def test_read_source_bad_prefix(tmp_path):
    p = _write(tmp_path / "x_source3.tsv", HEADER + "S2-1\ta\tb\tc\n")
    with pytest.raises(DataValidationError, match="prefix"):
        read_source(p)


def test_read_source_duplicate_id(tmp_path):
    p = _write(tmp_path / "x_source1.tsv", HEADER + "S1-1\ta\tb\tc\nS1-1\td\te\tf\n")
    with pytest.raises(DataValidationError, match="duplicate"):
        read_source(p)


def test_read_source_extra_tab_rejected(tmp_path):
    p = _write(tmp_path / "x_source1.tsv", HEADER + "S1-1\ta\tb\tc\textra\n")
    with pytest.raises(Exception):
        read_source(p)


def test_read_source_wrong_columns(tmp_path):
    p = _write(tmp_path / "x_source1.tsv", "id\tname\taddr\tcountry\nS1-1\ta\tb\tc\n")
    with pytest.raises(DataValidationError, match="columns"):
        read_source(p)


def test_read_source_invalid_utf8(tmp_path):
    p = tmp_path / "x_source1.tsv"
    p.write_bytes(HEADER.encode() + b"S1-1\t\xff\tb\tc\n")
    with pytest.raises(UnicodeDecodeError):
        read_source(p)


def test_parse_id_list():
    assert parse_id_list("") == []
    assert parse_id_list("   ") == []
    assert parse_id_list("S2-1, S3-2 ,S2-3") == ["S2-1", "S3-2", "S2-3"]


def test_read_ground_truth_parsing_and_checks(tmp_path):
    p = _write(
        tmp_path / "gt.tsv",
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1, S3-1\n"
        "S1-2\t\n"
        "S1-3\t \n",
    )
    gt, rep = read_ground_truth(p, s1_ids=["S1-1", "S1-2", "S1-3", "S1-4"], s2_ids=["S2-1"], s3_ids=["S3-1"])
    assert gt["S1-1"] == frozenset({"S2-1", "S3-1"})
    assert gt["S1-2"] == frozenset() and gt["S1-3"] == frozenset()
    assert gt["S1-4"] == frozenset()  # missing S1 -> singleton
    assert rep["n_source1_ids_missing_from_gt"] == 1
    assert rep["n_matched_ids_not_found"] == 0


def test_read_ground_truth_violations(tmp_path):
    p = _write(
        tmp_path / "gt.tsv",
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-1\nS1-9\tS3-7\n",
    )
    with pytest.raises(DataValidationError):
        read_ground_truth(p, s1_ids=["S1-1"], s2_ids=["S2-1"], s3_ids=[])
    gt, rep = read_ground_truth(p, s1_ids=["S1-1"], s2_ids=["S2-1"], s3_ids=[], strict=False)
    assert rep["n_intra_list_duplicates"] == 1
    assert rep["n_gt_ids_not_in_source1"] == 1
    assert rep["n_matched_ids_not_found"] == 1


def test_read_ground_truth_wrong_source(tmp_path):
    p = _write(tmp_path / "gt.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-5\n")
    _, rep = read_ground_truth(p, s1_ids=["S1-1"], s2_ids=[], s3_ids=["S2-5"], strict=False)
    assert rep["n_matched_ids_wrong_source"] == 1


def test_writers_roundtrip(tmp_path):
    s1 = ["S1-b", "S1-a", "S1-c"]
    cands = {"S1-b": ["S3-2", "S2-1", "S2-1"], "S1-a": []}
    matches = {"S1-b": {"S2-1"}}
    write_candidates(tmp_path / "c.tsv", cands, s1)
    write_matching(tmp_path / "m.tsv", matches, s1, candidates=cands)
    text = (tmp_path / "m.tsv").read_text()
    assert text == "source1_entity_id\tmatched_entity_ids\nS1-b\tS2-1\nS1-a\t\nS1-c\t\n"
    header, c = read_id_list_file(tmp_path / "c.tsv")
    assert header == ["source1_entity_id", "candidate_entity_ids"]
    assert c == {"S1-b": ["S2-1", "S3-2"], "S1-a": [], "S1-c": []}


def test_writer_rejects_bad_ids(tmp_path):
    with pytest.raises(ValueError):
        write_matching(tmp_path / "m.tsv", {"S1-a": ["S1-z"]}, ["S1-a"])
    with pytest.raises(ValueError):
        write_matching(tmp_path / "m.tsv", {"S1-q": ["S2-1"]}, ["S1-a"])
    with pytest.raises(ValueError):
        write_matching(tmp_path / "m.tsv", {"S1-a": ["S2-1"]}, ["S1-a"], candidates={"S1-a": []})
    with pytest.raises(ValueError):
        write_candidates(tmp_path / "c.tsv", {}, ["S1-a", "S1-a"])


def test_config_resolves_paths():
    cfg = load_config()
    assert cfg["abs"]["train.source1"].name == "train_source1.tsv"
    assert cfg["abs"]["validator"].name == "validate_submission.py"
    assert isinstance(cfg["seed"], int) and cfg["n_folds"] >= 2


def test_project_root_auto_resolves_to_this_repo():
    cfg = load_config()
    root = cfg["abs"]["root"]
    assert (root / "src" / "ber").is_dir()
    assert cfg["abs"]["raw"].is_relative_to(root)
    assert (root / "data" / "raw").is_dir()


def test_ledger_migrates_old_header_and_stores_extra(tmp_path):
    import csv

    from ber.evaluate.ledger import LEDGER_COLUMNS, append_run

    p = tmp_path / "runs.csv"
    p.write_text("timestamp,git_hash,run_name,cv_macro_f05_mean,cv_macro_f05_std,singleton_mean,"
                 "nonsingleton_mean,full_train_macro_f05,notes\nT,h,old,0.1,0,1,0,0.1,n\n")
    append_run(p, "new", {"cv_macro_f05_mean": 0.5}, extra={"k": 1})
    rows = list(csv.DictReader(open(p)))
    assert list(rows[0].keys()) == LEDGER_COLUMNS
    assert rows[0]["run_name"] == "old" and rows[0]["metrics"] == ""
    assert rows[1]["metrics"] == '{"k": 1}'
    assert rows[0]["device"] == rows[0]["gpu_ids"] == rows[0]["peak_vram_gb"] == "unknown"
    append_run(p, "gpu", {"device": "cuda", "gpu_ids": [0, 1], "peak_vram_gb": 12.5})
    rows = list(csv.DictReader(open(p)))
    assert (rows[2]["device"], rows[2]["gpu_ids"], rows[2]["peak_vram_gb"]) == ("cuda", "0;1", "12.500000")
