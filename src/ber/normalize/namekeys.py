"""Blocking keys for names (normalisation v1.1, added after the P1 recall gate).

Three additions, each country-agnostic:

* ``TranslitMap``: a token map learned from TRUE training pairs, from tokens of
  anyascii-transliterated non-Latin names to Latin tokens. anyascii drops Indic
  inherent vowels ("बिल्डर्स" -> "bildrs"), so these tokens share few n-grams with
  the English word ("builders"). A map entry needs support >= ``min_support``
  pairs and precision >= ``min_precision`` (the share of the token's pairs whose
  Latin side contains the target token). Learned from training-fold pairs only.
* ``skeleton``: a consonant skeleton per token. The first letter is kept, later
  vowels (and y) dropped, anusvara-style 'm' before a consonant other than b/p
  becomes 'n', and repeated letters are collapsed ("indian" and "imdiyn" -> "indn").
* ``Segmenter``: unigram word segmentation of long glued tokens from domain-like
  names ("servicesprivatemercantile" -> "services private mercantile"), using a
  vocabulary of S1 name tokens (label-free).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

_VOWELS = set("aeiouy")
_CONS_AFTER_M = set("cdfghjklnqrstvwxz")  # 'm' before these (not b/p/m) behaves like anusvara -> 'n'


def skeleton(tok: str) -> str:
    if len(tok) <= 2 or tok.isdigit():
        return tok
    out = [tok[0]]
    for i, ch in enumerate(tok[1:], start=1):
        if ch in _VOWELS:
            continue
        if ch == "m" and i + 1 < len(tok) and tok[i + 1] in _CONS_AFTER_M:
            ch = "n"
        if out[-1] != ch:
            out.append(ch)
    return "".join(out)


def skeleton_text(text: str) -> str:
    return " ".join(skeleton(t) for t in text.split())


class TranslitMap:
    def __init__(self, mapping: dict[str, str] | None = None, meta: dict | None = None):
        self.mapping = mapping or {}
        self.meta = meta or {}

    @classmethod
    def learn(cls, pairs: Iterable[tuple[str, str]], min_support: int = 5, min_precision: float = 0.8,
              exclude: set[str] | None = None) -> "TranslitMap":
        """``pairs``: (latin_name_tokens, transliterated_name_tokens) strings of TRUE pairs."""
        co: dict[str, Counter] = defaultdict(Counter)
        freq: Counter = Counter()
        n = 0
        for latin, translit in pairs:
            n += 1
            lt = set(latin.split())
            for t in set(translit.split()):
                if t.isdigit() or len(t) < 2:
                    continue
                freq[t] += 1
                co[t].update(lt)
        exclude = exclude or set()
        mapping = {}
        for t, f in freq.items():
            if f < min_support or t in exclude:
                continue
            best, c = co[t].most_common(1)[0]
            if best != t and c / f >= min_precision:
                mapping[t] = best
        return cls(mapping, {"n_pairs": n, "n_source_tokens": len(freq), "n_mapped": len(mapping),
                             "min_support": min_support, "min_precision": min_precision})

    def apply(self, text: str) -> str:
        m = self.mapping
        return " ".join(m.get(t, t) for t in text.split())

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"meta": self.meta, "mapping": self.mapping}, indent=0, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "TranslitMap":
        d = json.loads(Path(path).read_text())
        return cls(d["mapping"], d.get("meta"))


class Segmenter:
    """Viterbi unigram segmentation for glued tokens (e.g. domain names)."""

    def __init__(self, counts: Counter, min_count: int = 20, max_word: int = 20):
        vocab = {w: c for w, c in counts.items() if c >= min_count and w.isalpha() and len(w) >= 2}
        total = sum(vocab.values()) or 1
        self.logp = {w: math.log(c / total) for w, c in vocab.items()}
        self.max_word = max_word
        self.unk = math.log(1.0 / (total * 10))  # per-character penalty for unknown spans

    def segment(self, tok: str, min_len: int = 8) -> list[str]:
        if len(tok) < min_len or tok in self.logp or not tok.isalpha():
            return [tok]
        n = len(tok)
        best = [0.0] + [-math.inf] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_word), i):
                w = tok[j:i]
                lp = self.logp.get(w, self.unk * len(w) if len(w) == 1 else -math.inf)
                if lp == -math.inf:
                    continue
                s = best[j] + lp
                if s > best[i]:
                    best[i], back[i] = s, j
        if best[n] == -math.inf:
            return [tok]
        out, i = [], n
        while i > 0:
            out.append(tok[back[i]:i])
            i = back[i]
        out.reverse()
        # accept only if every piece is a known word (no single-letter fallbacks)
        return out if all(w in self.logp for w in out) and len(out) > 1 else [tok]


def name_key(core: str, script: str, is_domain: bool, tmap: TranslitMap | None, seg: Segmenter | None) -> str:
    """Name text for blocking: transliteration map (non-Latin), then segmentation (domains)."""
    text = core or ""
    if tmap is not None and script not in ("latin", "none"):
        text = tmap.apply(text)
    if seg is not None and is_domain:
        text = " ".join(w for t in text.split() for w in seg.segment(t))
    return text


_WS = re.compile(r"\s+")


def block_text(key: str) -> str:
    """B1 input: the name key plus its consonant skeleton."""
    key = _WS.sub(" ", key or "").strip()
    return f"{key} {skeleton_text(key)}".strip()
