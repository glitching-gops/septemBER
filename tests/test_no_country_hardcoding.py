"""src/ must not contain any country value found in the data as a string literal.

Only string literals are checked (``ast.Constant`` str nodes, including the literal
parts of f-strings); comments, docstrings-as-prose and identifiers are ignored, so
the English word "us" in a comment or a name like ``us_state`` is fine. YAML
dictionaries under ``src/ber/normalize/dictionaries/`` are data, not code, and are
not scanned (only ``*.py`` files are).
"""
import ast
import re
from pathlib import Path

import pytest

from ber.io_utils import load_config, read_column

SRC = Path(__file__).resolve().parents[1] / "src"
SOURCE_KEYS = [f"{split}.source{i}" for split in ("train", "test") for i in (1, 2, 3)]
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _country_values() -> set[str]:
    a = load_config()["abs"]
    paths = [a[k] for k in SOURCE_KEYS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"raw data not available: {missing[0]}")
    values: set[str] = set()
    for p in paths:
        values |= set(read_column(p, "country").unique())
    return {v.strip() for v in values if v.strip()}


def _string_literals(tree: ast.AST):
    """Yield (lineno, text) for every str constant, skipping module/class/def docstrings."""
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_nodes.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_nodes:
            yield node.lineno, node.value


def _contains_value(text: str, value_tokens: tuple[str, ...]) -> bool:
    toks = [t.casefold() for t in _TOKEN.findall(text)]
    n = len(value_tokens)
    return any(tuple(toks[i:i + n]) == value_tokens for i in range(len(toks) - n + 1))


def find_country_literals(root: Path, values) -> list[str]:
    targets = [tuple(t.casefold() for t in _TOKEN.findall(v)) for v in values]
    targets = [t for t in targets if t]
    hits = []
    for f in sorted(root.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for lineno, text in _string_literals(tree):
            for t in targets:
                if _contains_value(text, t):
                    hits.append(f"{f.relative_to(root)}:{lineno}: {' '.join(t)!r} in {text[:80]!r}")
    return hits


def test_detects_planted_literal(tmp_path):
    (tmp_path / "bad.py").write_text('X = df[df.country == "France"]\nY = f"{x} in france"\n')
    assert len(find_country_literals(tmp_path, {"France"})) == 2


def test_ignores_comments_identifiers_and_docstrings(tmp_path):
    (tmp_path / "ok.py").write_text(
        '"""Module docstring mentioning US and France."""\n'
        "# let us normalise; France-style addresses are handled generically\n"
        "us_state = 1\n"
        "def f():\n"
        '    """Docstring: works for us."""\n'
        '    return "status"  # substring of a value is not a token match\n'
    )
    assert find_country_literals(tmp_path, {"US", "France"}) == []


def test_multiword_values_match_as_phrase(tmp_path):
    (tmp_path / "m.py").write_text('A = "new zealand"\nB = "new things in zealand"\n')
    assert len(find_country_literals(tmp_path, {"New Zealand"})) == 1


def test_no_country_literals_in_src():
    values = _country_values()
    assert len(values) >= 2, values
    hits = find_country_literals(SRC, values)
    assert not hits, "country literals found in src/:\n" + "\n".join(hits)
