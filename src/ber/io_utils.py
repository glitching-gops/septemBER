"""The single entry point for reading source/ground-truth TSVs and writing submissions.

All reads go through :func:`read_source` / :func:`read_ground_truth`, which parse
with a fixed, quote-agnostic configuration and validate the result. All writes go
through :func:`write_matching` / :func:`write_candidates`, which enforce the output
format rules of the challenge.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import yaml

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
TARGET_PREFIXES = ("S2-", "S3-")

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "default.yaml"
PROJECT_ROOT = PACKAGE_ROOT

_READ_KW = dict(
    sep="\t",
    dtype=str,
    keep_default_na=False,
    na_filter=False,
    quoting=csv.QUOTE_NONE,
    encoding="utf-8",
    encoding_errors="strict",
    on_bad_lines="error",
    engine="c",
)


class DataValidationError(ValueError):
    """Raised when an input file violates an expected invariant."""


# --------------------------------------------------------------------------- config

def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config and attach resolved absolute paths.

    Adds ``cfg["abs"]`` with keys ``root``, ``raw``, ``train.<k>``, ``test.<k>``,
    ``test_dir``, ``validator`` and every key of ``cfg["paths"]``.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = _load_yaml_chain(Path(path))
    if cfg.get("project_root", "auto") in ("auto", None):
        root = PROJECT_ROOT
    else:
        root = Path(cfg["project_root"]).expanduser()
    raw = _under(root, cfg["raw_root"])
    abs_ = {"root": root, "raw": raw}
    for split in ("train", "test"):
        for k, v in cfg["data"][split].items():
            abs_[f"{split}.{k}"] = _under(raw, v)
    for k in ("test_dir", "validator", "doc_template"):
        abs_[k] = _under(raw, cfg["data"][k])
    for k, v in cfg["paths"].items():
        abs_[k] = _under(root, v)
    cfg["abs"] = abs_
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    """``over`` wins; nested dicts merge key by key, everything else is replaced."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml_chain(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Load a config, first applying any parent named by its ``extends:`` key.

    ``extends`` is resolved relative to the child config's directory, so
    ``configs/p2.yaml`` can say ``extends: default.yaml`` and override a few keys
    instead of duplicating the whole file.
    """
    path = path.resolve()
    if path in _seen:
        raise DataValidationError(f"circular config extends chain at {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = _load_yaml_chain(_under(path.parent, parent), _seen + (path,))
        cfg = _deep_merge(base, cfg)
    return cfg


def _under(base: Path, p: str) -> Path:
    p = Path(p).expanduser()
    scratch = os.environ.get("BER_SCRATCH_ROOT")
    if scratch and p.is_absolute():
        try:
            return Path(scratch).expanduser() / p.relative_to("/scratch/user4")
        except ValueError:
            pass
    return p if p.is_absolute() else base / p


# --------------------------------------------------------------------------- readers

def count_data_lines(path: str | Path) -> int:
    """Number of newline-terminated lines minus the header (raw ``wc -l`` - 1)."""
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n - 1


def _infer_prefix(path: Path) -> str | None:
    m = re.search(r"source([123])", path.name)
    return f"S{m.group(1)}-" if m else None


def read_source(
    path: str | Path,
    expected_prefix: str | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Read one ``*_sourceN.tsv`` file as all-string columns.

    Returns ``(df, report)``. With ``validate=True`` raises
    :class:`DataValidationError` on: column mismatch, row-count mismatch against
    the raw line count, wrong ID prefix, duplicate IDs, or tab/newline/CR
    characters inside a field. Empty and whitespace-only field counts per column
    are reported, not raised.
    """
    path = Path(path)
    df = pd.read_csv(path, **_READ_KW)
    report: dict = {"file": path.name, "n_rows": len(df)}
    if not validate:
        return df, report

    if list(df.columns) != SOURCE_COLUMNS:
        raise DataValidationError(f"{path.name}: columns {list(df.columns)} != {SOURCE_COLUMNS}")

    raw_rows = count_data_lines(path)
    report["raw_rows"] = raw_rows
    if raw_rows != len(df):
        raise DataValidationError(f"{path.name}: parsed {len(df)} rows but file has {raw_rows} data lines")

    prefix = expected_prefix or _infer_prefix(path)
    report["prefix"] = prefix
    if prefix:
        bad = ~df["entity_id"].str.startswith(prefix)
        if bad.any():
            raise DataValidationError(
                f"{path.name}: {int(bad.sum())} ids lack prefix {prefix!r}, e.g. {df.loc[bad, 'entity_id'].head(3).tolist()}"
            )

    dup = df["entity_id"].duplicated()
    if dup.any():
        raise DataValidationError(
            f"{path.name}: {int(dup.sum())} duplicate entity_id(s), e.g. {df.loc[dup, 'entity_id'].head(3).tolist()}"
        )

    report["empty"] = {}
    report["whitespace_only"] = {}
    report["untrimmed"] = {}
    for c in SOURCE_COLUMNS:
        s = df[c]
        if s.str.contains(r"[\t\r\n]", regex=True).any():
            raise DataValidationError(f"{path.name}: control characters (tab/CR/LF) inside column {c}")
        stripped = s.str.strip()
        report["empty"][c] = int((s == "").sum())
        report["whitespace_only"][c] = int(((s != "") & (stripped == "")).sum())
        report["untrimmed"][c] = int(((stripped != s) & (stripped != "")).sum())
    return df, report


def read_column(path: str | Path, column: str) -> pd.Series:
    """Read a single column from a source TSV with the standard parse settings."""
    return pd.read_csv(path, usecols=[column], **_READ_KW)[column]


def parse_id_list(field: str) -> list[str]:
    """Split a comma-separated ID list; empty or whitespace-only -> []."""
    if not field.strip():
        return []
    return [x.strip() for x in field.split(",") if x.strip()]


def read_ground_truth(
    path: str | Path,
    s1_ids: Iterable[str] | None = None,
    s2_ids: Iterable[str] | None = None,
    s3_ids: Iterable[str] | None = None,
    strict: bool = True,
) -> tuple[dict[str, frozenset[str]], dict]:
    """Parse ``train_ground_truth.tsv`` into ``{s1_id: frozenset(matched ids)}``.

    When the id collections are provided, cross-checks are run and reported:
    GT ids missing from Source 1, Source-1 ids missing from the GT (to be treated
    as singletons), matched ids missing from Source 2/3, prefix violations and
    duplicates within a list. With ``strict=True`` any violation other than
    "S1 id missing from GT" raises :class:`DataValidationError`.
    """
    path = Path(path)
    df = pd.read_csv(path, **_READ_KW)
    if list(df.columns) != GT_COLUMNS:
        raise DataValidationError(f"{path.name}: columns {list(df.columns)} != {GT_COLUMNS}")
    raw_rows = count_data_lines(path)
    if raw_rows != len(df):
        raise DataValidationError(f"{path.name}: parsed {len(df)} rows but file has {raw_rows} data lines")

    gt: dict[str, frozenset[str]] = {}
    dup_rows: list[str] = []
    intra_dupes: list[str] = []
    bad_prefix: list[str] = []
    for s1, field in zip(df["source1_entity_id"].str.strip(), df["matched_entity_ids"]):
        ids = parse_id_list(field)
        if len(ids) != len(set(ids)):
            intra_dupes.append(s1)
        for m in ids:
            if not m.startswith(TARGET_PREFIXES):
                bad_prefix.append(m)
        if s1 in gt:
            dup_rows.append(s1)
        gt[s1] = frozenset(ids)

    report: dict = {
        "file": path.name,
        "n_rows": len(df),
        "raw_rows": raw_rows,
        "n_duplicate_rows": len(dup_rows),
        "n_intra_list_duplicates": len(intra_dupes),
        "n_bad_prefix": len(bad_prefix),
        "n_bad_s1_prefix": sum(not k.startswith("S1-") for k in gt),
    }
    problems = []
    if dup_rows:
        problems.append(f"duplicate source1 rows e.g. {dup_rows[:3]}")
    if intra_dupes:
        problems.append(f"duplicate ids within list for e.g. {intra_dupes[:3]}")
    if bad_prefix:
        problems.append(f"matched ids without S2-/S3- prefix e.g. {bad_prefix[:3]}")
    if report["n_bad_s1_prefix"]:
        problems.append("source1_entity_id without S1- prefix")

    if s1_ids is not None:
        s1_set = set(s1_ids)
        gt_not_in_s1 = [k for k in gt if k not in s1_set]
        s1_not_in_gt = [k for k in s1_set if k not in gt]
        report["n_gt_ids_not_in_source1"] = len(gt_not_in_s1)
        report["n_source1_ids_missing_from_gt"] = len(s1_not_in_gt)
        report["missing_policy"] = "treated as singleton (empty truth)"
        if gt_not_in_s1:
            problems.append(f"{len(gt_not_in_s1)} GT ids not in Source 1 e.g. {gt_not_in_s1[:3]}")
        for k in s1_not_in_gt:
            gt[k] = frozenset()

    if s2_ids is not None and s3_ids is not None:
        s2_set, s3_set = set(s2_ids), set(s3_ids)
        missing = []
        wrong_source = []
        for ids in gt.values():
            for m in ids:
                if m.startswith("S2-"):
                    if m not in s2_set:
                        (wrong_source if m in s3_set else missing).append(m)
                elif m.startswith("S3-"):
                    if m not in s3_set:
                        (wrong_source if m in s2_set else missing).append(m)
        report["n_matched_ids_not_found"] = len(missing)
        report["n_matched_ids_wrong_source"] = len(wrong_source)
        if missing:
            problems.append(f"{len(missing)} matched ids not in Source 2/3 e.g. {missing[:3]}")
        if wrong_source:
            problems.append(f"{len(wrong_source)} matched ids whose prefix disagrees with their file")

    report["problems"] = problems
    if strict and problems:
        raise DataValidationError(f"{path.name}: " + "; ".join(problems))
    return gt, report


# --------------------------------------------------------------------------- writers

def _clean_ids(s1: str, ids: Iterable[str]) -> list[str]:
    out = sorted(set(ids))
    for m in out:
        if not m.startswith(TARGET_PREFIXES):
            raise ValueError(f"{s1}: id {m!r} is not a Source-2/3 id")
        if any(ch in m for ch in ",\t\n\r\" "):
            raise ValueError(f"{s1}: id {m!r} contains a forbidden character")
    return out


def _write_id_lists(
    path: str | Path,
    header: list[str],
    mapping: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str],
) -> dict[str, list[str]]:
    s1_ids = list(s1_ids)
    if len(s1_ids) != len(set(s1_ids)):
        raise ValueError("duplicate S1 ids in s1_ids")
    extra = set(mapping) - set(s1_ids)
    if extra:
        raise ValueError(f"{len(extra)} ids in mapping are not in s1_ids, e.g. {sorted(extra)[:3]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = _clean_ids(s1, mapping.get(s1, ()))
            written[s1] = ids
            f.write(f"{s1}\t{','.join(ids)}\n")
    return written


def write_candidates(path, candidates: Mapping[str, Iterable[str]], s1_ids: Iterable[str]) -> None:
    """Write ``candidate_pairs.tsv``: one row per id in ``s1_ids`` (missing -> empty)."""
    _write_id_lists(path, CANDIDATE_HEADER, candidates, s1_ids)


def write_matching(
    path,
    matches: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str],
    candidates: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Write ``matching_results.tsv``; if ``candidates`` given, enforce matches ⊆ candidates."""
    if candidates is not None:
        bad = [s1 for s1, ids in matches.items() if set(ids) - set(candidates.get(s1, ()))]
        if bad:
            raise ValueError(f"{len(bad)} S1 ids have matches outside their candidates, e.g. {bad[:3]}")
    _write_id_lists(path, MATCHING_HEADER, matches, s1_ids)


def read_id_list_file(path: str | Path) -> tuple[list[str], dict[str, list[str]]]:
    """Parse a written matching/candidate file back to (header, {s1: [ids]}) (raw order kept)."""
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        out: dict[str, list[str]] = {}
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1] = rest.split(",") if rest else []
    return header, out
