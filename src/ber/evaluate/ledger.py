"""Append-only experiment ledger (runs/runs.csv)."""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LEDGER_COLUMNS = [
    # --- P0/P1 columns, kept so historical rows stay readable
    "timestamp",
    "git_hash",
    "run_name",
    "cv_macro_f05_mean",
    "cv_macro_f05_std",
    "singleton_mean",
    "nonsingleton_mean",
    "full_train_macro_f05",
    "notes",
    "metrics",  # JSON object with run-specific numbers (added in P1)
    # GPU-first compute policy (P2): logged for every run; rows predating these are "unknown".
    "device",        # "cuda" or "cpu"
    "gpu_ids",       # e.g. "1" or "0;1"; empty for cpu
    "peak_vram_gb",
    # --- P2 experiment-tracking schema
    "run_id",         # short unique id for this run
    "utc_time",       # == timestamp, kept under the P2 name
    "git_sha",        # == git_hash, kept under the P2 name
    "stage",          # e0 / e1 / e3 / e4 / test_inference / reproduction ...
    "config_hash",    # sha1 of the resolved config actually used
    "norm_version",
    "blocker_set",
    "K",
    "n_pairs_dev",
    "n_pairs_test",
    "feature_version",
    "model",
    "cv_macro_f05",
    "cv_india",
    "cv_us",
    "micro_p",
    "micro_r",
    "loco_us_to_in",
    "loco_in_to_us",
    "pair_completeness",
    "oracle_ceiling",
    "gpu_id",
    "wall_s",
    "peak_ram_gb",
    "peak_gpu_gb",
    "submission_id",
    "public_lb",      # filled in after the owner reports a leaderboard score
    # --- P3-P6 additions
    "verifier_model",  # HF id of the pair verifier feeding the model, if any
    "verifier_band",   # stage-1 probability band the verifier scores, e.g. "0.02-0.95"
    "bytes_freed",     # cleanup runs only
    # --- P4 GPU-only contract: per-stage time split; cpu_violation = CPU > 10% of wall outside I/O
    "gpu_time_s",
    "cpu_time_s",
    "cpu_violation",
]
COMPUTE_COLUMNS = ("device", "gpu_ids", "peak_vram_gb")


def git_hash(cwd: str | Path | None = None) -> str:
    """Short HEAD hash, suffixed with ``-dirty`` when tracked files are modified."""
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, capture_output=True, text=True
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def new_run_id(prefix: str = "run") -> str:
    """Short, sortable, unique run id: ``<prefix>-YYYYmmddTHHMMSS-xxxx``."""
    import uuid

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:4]}"


def config_hash(cfg: dict) -> str:
    """Stable sha1 of a resolved config (Paths and other objects via ``str``)."""
    import hashlib

    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def set_public_lb(ledger_path: str | Path, submission_id: str, public_lb: float) -> int:
    """Record a leaderboard score against every row carrying ``submission_id``.

    The owner uploads submissions and reports the score; this only writes it down.
    Returns the number of rows updated.
    """
    ledger_path = Path(ledger_path)
    with open(ledger_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n = 0
    for r in rows:
        if r.get("submission_id") == submission_id:
            r["public_lb"] = f"{public_lb:.6f}"
            n += 1
    if n:
        with open(ledger_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in LEDGER_COLUMNS})
    return n


def _migrate(ledger_path: Path) -> None:
    """Rewrite an existing ledger whose header predates new columns (values kept)."""
    with open(ledger_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        header = rows[0].keys() if rows else []
    if rows and list(header) != LEDGER_COLUMNS:
        with open(ledger_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "unknown" if c in COMPUTE_COLUMNS else "")
                            for c in LEDGER_COLUMNS})


def append_run(ledger_path: str | Path, run_name: str, metrics: dict, notes: str = "",
               extra: dict | None = None) -> dict:
    """Append one row; creates the file with a header if needed. Returns the row.

    ``metrics`` fills the fixed columns (including ``device``, ``gpu_ids``, ``peak_vram_gb``;
    any of those not given are written as "unknown"); ``extra`` is stored as JSON in ``metrics``.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists() and ledger_path.stat().st_size:
        _migrate(ledger_path)
    row = {c: "unknown" if c in COMPUTE_COLUMNS else "" for c in LEDGER_COLUMNS}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = git_hash(ledger_path.parent)
    row.update(
        timestamp=now,
        git_hash=sha,
        run_name=run_name,
        notes=notes,
        # P2 aliases so a single row is readable under either schema
        run_id=new_run_id(run_name),
        utc_time=now,
        git_sha=sha,
    )
    for k, v in metrics.items():
        if k not in row:
            raise KeyError(f"unknown ledger column {k!r}")
        if k == "gpu_ids" and isinstance(v, (list, tuple)):
            v = ";".join(str(g) for g in v)
        row[k] = f"{v:.6f}" if isinstance(v, float) else v
    if extra:
        row["metrics"] = json.dumps(extra, sort_keys=True, default=str)
    new = not ledger_path.exists() or ledger_path.stat().st_size == 0
    with open(ledger_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
    return row
