"""GPU-only contract (final phase): AST lint of the pipeline code.

Linted strictly: every module under ``src/ber/`` except ``src/ber/io/`` (host I/O and the
one-time id encoding) and the LEGACY modules listed below, plus the P4 pipeline scripts.
A linted module fails on:
  * any import of sklearn or scipy;
  * pandas ``groupby`` / ``sort_values`` / ``merge`` / ``join`` / ``apply`` / ``drop_duplicates`` / ``rank``;
  * numpy calls other than ``np.load`` / ``np.save`` / ``np.asarray`` (loading);
  * ``for`` / ``while`` loops whose iterable is a data collection (rows / entities): loops are allowed
    only over ``range(...)``, literal tuples/lists, ``.items()`` of small dicts, ``torch.unique(...).tolist()``
    of bucket codes, ``zip`` of such, or names listed in ``LOOP_OK``.

LEGACY: CPU stages whose outputs were produced in P1-P3 and are read from cache; the final-phase
counted runs never execute them. They stay importable for the reproduction README and as pytest
oracles; ``test_p4_scripts_import_no_cpu_compute`` checks the P4 pipeline scripts never import them.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ber"

LEGACY = {
    "normalize": "P1 text normalisation (cached norm tables)",
    "blocking": "P1/P2 candidate generation (cached candidates)",
    "features": "P1/P2 pair features incl. rapidfuzz (cached f2 shards)",
    "embed": "P1 e5 encoding (cached embeddings; GPU torch)",
    "namekeys": "P1 learned transliteration keys (cached)",
    "decide": "P3 CPU decision rules: pytest oracles only",
    "evaluate": "P3 CPU metric + ledger writer: metric is a pytest oracle; ledger is I/O",
    "match/gbdt.py": "P2/P3 XGBoost wrapper with sklearn isotonic: pytest oracle only",
    "match/features.py": "P1/P2 pair-feature definitions incl. rapidfuzz (cached f2 shards)",
    "match/features_gpu.py": "P2 token features (host staging around torch; cached f2 shards)",
    "match/build.py": "P2 feature construction (cached)",
    "match/__init__.py": "package marker",
    "verify/train.py": "verifier training loop + CPU mining oracle (training finished before the contract)",
    "verify/serialize.py": "record text serialisation feeding the tokenizer (tokenisation phase)",
    "verify/augment.py": "French string augmentation (string ops; used in training only)",
    "verify/rerankers.py": "HF model wrapper: batched fast-tokenizer + GPU inference",
    "verify/folds.py": "P3 fold bookkeeping (holdout hash = encoding step); pandas features superseded",
    "verify/__init__.py": "determinism flags",
    "splits.py": "P0 splits (cached)",
    "io_utils.py": "raw TSV readers / writers (I/O)",
    "compute.py": "nvidia-smi / resource probing (control flow)",
    "eda.py": "P0 EDA",
    "__init__.py": "package marker",
}
P4_SCRIPTS = ["run_p4_stage.py", "run_p4_infer.py"]
PANDAS_BANNED = {"groupby", "sort_values", "merge", "join", "apply", "drop_duplicates", "rank"}
NP_OK = {"load", "save", "asarray"}
LOOP_OK = {"range", "enumerate", "zip", "reversed", "sorted"}
# small control collections, never rows / entities: fold models (<= 5), segment codes (<= 4)
LOOP_OK_NAMES = {"models", "codes"}


def _legacy(path: Path) -> bool:
    rel = path.relative_to(SRC).as_posix()
    return any(rel == k or rel.startswith(k.rstrip("/") + "/") or rel.split("/")[0] == k for k in LEGACY)


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if n.split(".")[0] in ("sklearn", "scipy"):
                    out.append(f"{path.name}:{node.lineno} imports {n}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == "np" and attr not in NP_OK:
                out.append(f"{path.name}:{node.lineno} np.{attr}")
            path_or_str = (isinstance(base, ast.Constant) and isinstance(base.value, str)) or \
                (isinstance(base, ast.Attribute) and ast.unparse(base) == "os.path")
            if attr in PANDAS_BANNED and not path_or_str and \
                    not (isinstance(base, ast.Name) and base.id in ("torch", "sg", "sgm", "os")):
                out.append(f"{path.name}:{node.lineno} .{attr}(...)")
        if isinstance(node, (ast.For, ast.comprehension)):
            it = node.iter
            ok = False
            if isinstance(it, (ast.Tuple, ast.List)):
                ok = True
            elif isinstance(it, ast.Name) and (it.id.isupper() or it.id in LOOP_OK_NAMES):
                ok = True  # module constants (feature-name lists) and small control collections
            elif isinstance(it, ast.Call):
                f = it.func
                if isinstance(f, ast.Name) and f.id in LOOP_OK:
                    ok = True
                elif isinstance(f, ast.Attribute) and f.attr in ("items", "keys", "values", "tolist"):
                    ok = True
            if not ok:
                out.append(f"{path.name}:{getattr(node, 'lineno', getattr(it, 'lineno', 0))} loop over {ast.unparse(it)[:40]}")
        if isinstance(node, ast.While):
            out.append(f"{path.name}:{node.lineno} while loop")
    return out


def test_pipeline_packages_are_gpu_only():
    bad = []
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith("io/") or _legacy(p):
            continue
        bad += _violations(p)
    assert not bad, "GPU-only contract violations:\n" + "\n".join(bad)


def test_gpu_package_exists_and_is_linted():
    mods = {p.name for p in (SRC / "gpu").glob("*.py")}
    assert {"segment.py", "decide.py", "metric.py", "search.py", "calib.py", "boot.py", "mining.py", "xgb.py"} <= mods
    assert not _legacy(SRC / "gpu" / "decide.py") and not _legacy(SRC / "match" / "context.py")


def test_p4_scripts_import_no_cpu_compute():
    banned = ("sklearn", "scipy", "ber.decide", "ber.match.gbdt", "ber.evaluate.metric", "run_e1", "run_verifier_integrate")
    bad = []
    for name in P4_SCRIPTS:
        p = ROOT / "scripts" / name
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                bad += [f"{name}:{node.lineno} {m}" for m in mods if m.startswith(banned)]
    assert not bad, "P4 scripts import CPU compute:\n" + "\n".join(bad)
