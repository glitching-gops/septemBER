"""Low-level, country-agnostic text helpers shared by name and address normalisation."""
from __future__ import annotations

import re
import unicodedata

from anyascii import anyascii

# Unicode blocks for script detection (letters only).
_SCRIPT_RANGES = (
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
)
_TOKEN = re.compile(r"[a-z0-9]+")
_ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th|e|eme|er|ere|ieme)$")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def to_ascii(s: str) -> str:
    """Transliterate to ASCII with anyascii (ISC licence; never Unidecode, which is GPL)."""
    return anyascii(s)


def dominant_script(s: str) -> str:
    """Most frequent script among letters: latin, devanagari, telugu, ..., or other/none."""
    counts: dict[str, int] = {}
    for ch in s:
        o = ord(ch)
        if o < 0x80:
            if ch.isalpha():
                counts["latin"] = counts.get("latin", 0) + 1
            continue
        if not ch.isalpha() and not (0x0900 <= o <= 0x0D7F):
            continue
        if o <= 0x024F:
            name = "latin"
        else:
            name = "other"
            for lo, hi, n in _SCRIPT_RANGES:
                if lo <= o <= hi:
                    name = n
                    break
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda kv: (kv[1], kv[0] == "latin"))[0]


def tokens(s: str) -> list[str]:
    """Lower-case ASCII alphanumeric tokens (punctuation acts as a separator)."""
    return _TOKEN.findall(s)


def ordinal(tok: str) -> str:
    """'3Rd' / '2eme' / '1er' -> canonical English ordinal ('3rd', '2nd', '1st')."""
    m = _ORDINAL.match(tok)
    if not m:
        return tok
    n = int(m.group(1))
    if 10 <= n % 100 <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


class PhraseMapper:
    """Greedy longest-match replacement of token phrases.

    ``mapping`` maps a tuple of tokens to a tuple of replacement tokens (possibly
    empty, which deletes the phrase).
    """

    def __init__(self, mapping: dict[tuple[str, ...], tuple[str, ...]]):
        self.mapping = mapping
        self.max_len = max((len(k) for k in mapping), default=1)
        self.first = {k[0] for k in mapping}

    def apply(self, toks: list[str]) -> list[str]:
        out: list[str] = []
        i, n = 0, len(toks)
        while i < n:
            if toks[i] in self.first:
                for L in range(min(self.max_len, n - i), 0, -1):
                    rep = self.mapping.get(tuple(toks[i:i + L]))
                    if rep is not None:
                        out.extend(rep)
                        i += L
                        break
                else:
                    out.append(toks[i])
                    i += 1
            else:
                out.append(toks[i])
                i += 1
        return out

    def find(self, toks: list[str]) -> list[tuple[int, int, tuple[str, ...]]]:
        """Non-overlapping (start, end, replacement) matches, longest first."""
        hits = []
        i, n = 0, len(toks)
        while i < n:
            if toks[i] in self.first:
                for L in range(min(self.max_len, n - i), 0, -1):
                    rep = self.mapping.get(tuple(toks[i:i + L]))
                    if rep is not None:
                        hits.append((i, i + L, rep))
                        i += L
                        break
                else:
                    i += 1
            else:
                i += 1
        return hits
