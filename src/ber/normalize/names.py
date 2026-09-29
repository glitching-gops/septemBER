"""Business-name normalisation (country-agnostic).

Output fields (see :func:`normalize_name`):
    name_script  dominant script of the original name
    name_ascii   anyascii transliteration of the NFKC name (case kept)
    name_norm    canonical token string: casefolded, punctuation removed, aliases and
                 bracket contents kept as tokens, legal forms kept
    name_core    name_norm without legal forms and leading honorifics
    name_alt     list of core strings: full core, core without bracketed text, and
                 each alias (aka/dba/formerly/...) core; primary first, deduplicated
    legal_class  sorted list of canonical legal classes (PVT_LTD, LTD, LLC, SARL, ...)
    is_domain    name looked like a domain / handle ("foo.com", "www.foo.in", "@foo")
"""
from __future__ import annotations

import re
from functools import lru_cache

from ber.normalize import dictionaries
from ber.normalize.text import dominant_script, nfkc, to_ascii, tokens

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"})
_HAS_ALPHA = re.compile(r"[a-z]")
_STORE_NO = re.compile(r"#\s*\d+")
_MS = re.compile(r"\bm\s*/\s*s\b\.?")
_BRACKETS = re.compile(r"[\(\[\{]([^\)\]\}]*)[\)\]\}]")
_AMP = re.compile(r"[&+]")


@lru_cache(maxsize=1)
def _regexes():
    d = dictionaries.load()
    tld = "|".join(re.escape(t) for t in d.tlds)
    domain = re.compile(rf"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:{tld})\b(?:/\S*)?")
    handle = re.compile(r"(?:^|\s)@([a-z0-9_\.]{3,})")
    markers = sorted(d.alias_markers, key=len, reverse=True)
    alias = re.compile(r"(?:^|\s)(?:" + "|".join(re.escape(m).replace(r"\ ", r"\s+") for m in markers) + r")(?=\s|$|[:,\.])")
    return domain, handle, alias


def _fix_leet(tok: str) -> str:
    """Digit-for-letter typos inside alphabetic tokens: 'mcc0y' -> 'mccoy', '5ai' -> 'sai'."""
    if tok.isdigit() or not _HAS_ALPHA.search(tok):
        return tok
    if sum(c.isalpha() for c in tok) < 2:
        return tok
    fixed = tok.translate(_LEET)
    return fixed if fixed.isalpha() else tok


def _join_letter_runs(toks: list[str], lexicon) -> list[str]:
    """'l l c' -> 'llc', 's a r l' -> 'sarl' when the joined run is a known legal token."""
    out: list[str] = []
    i = 0
    while i < len(toks):
        if len(toks[i]) == 1 and toks[i].isalpha():
            j = i
            while j < len(toks) and len(toks[j]) == 1 and toks[j].isalpha():
                j += 1
            run = toks[i:j]
            joined = "".join(run)
            if len(run) >= 2 and joined in lexicon:
                out.append(joined)
            else:
                out.extend(run)
            i = j
        else:
            out.append(toks[i])
            i += 1
    return out


def _tokenize_part(s: str, lexicon) -> list[str]:
    s = _AMP.sub(" and ", s)
    toks = tokens(s)
    toks = _join_letter_runs(toks, lexicon)
    toks = ["and" if t == "et" else _fix_leet(t) for t in toks]
    return toks


def _strip_honorifics(toks: list[str], d) -> list[str]:
    changed = True
    while changed and toks:
        changed = False
        for i, j, _ in d.honorifics.find(toks[:3]):
            if i == 0 and j < len(toks):
                toks = toks[j:]
                changed = True
            break
    return toks


def _core(toks: list[str], d) -> tuple[list[str], set[str]]:
    """Remove legal-form tokens -> (core tokens, set of atoms)."""
    atoms = set()
    core = []
    for t in toks:
        a = d.legal_atom.get(t)
        if a is not None:
            atoms.add(a)
        else:
            core.append(t)
    if not core:  # e.g. a business literally named "Co" or "SA": keep ambiguous tokens
        core = [t for t in toks if t in d.legal_ambiguous] or []
    return core, atoms


def legal_classes(atoms: set[str]) -> list[str]:
    classes = set()
    if "PRIVATE" in atoms:
        classes.add("PVT_LTD")
    elif "LIMITED" in atoms:
        classes.add("LTD")
    # DESCRIPTOR words (societe, ets, groupe, ...) are stripped from the core but are not a
    # legal form, so they must never produce a class (and hence never a legal_conflict)
    classes |= {a for a in atoms if a not in ("PRIVATE", "LIMITED", "DESCRIPTOR")}
    return sorted(classes)


def normalize_name(raw: str) -> dict:
    d = dictionaries.load()
    domain_re, handle_re, alias_re = _regexes()
    s = nfkc(raw or "")
    script = dominant_script(s)
    ascii_ = to_ascii(s).strip()
    low = ascii_.casefold()
    low = _MS.sub(" ms ", low)
    low = _STORE_NO.sub(" ", low)

    is_domain = False
    if domain_re.search(low):
        is_domain = True
        low = domain_re.sub(lambda m: " " + m.group(1).replace("-", " ") + " ", low)
    if handle_re.search(low):
        is_domain = True
        low = handle_re.sub(lambda m: " " + m.group(1).replace(".", " ").replace("_", " ") + " ", low)

    # alias split on the bracket-free text; bracket contents become their own part
    bracket_parts = [m.group(1) for m in _BRACKETS.finditer(low)]
    no_brackets = _BRACKETS.sub(" ", low)
    alias_parts = [p for p in alias_re.split(no_brackets) if p and p.strip()]
    primary = alias_parts[0] if alias_parts else no_brackets
    aliases = alias_parts[1:]

    lex = d.legal_atom
    full_toks = _tokenize_part(alias_re.sub(" ", _BRACKETS.sub(r" \1 ", low)), lex)
    full_toks = _strip_honorifics(full_toks, d)
    core_full, atoms = _core(full_toks, d)

    alts: list[str] = []

    def add(ts: list[str]):
        c, a = _core(_strip_honorifics(ts, d), d)
        atoms.update(a)
        s_ = " ".join(c)
        if s_ and s_ not in alts:
            alts.append(s_)

    add(full_toks)
    add(_tokenize_part(primary, lex))
    for p in aliases:
        add(_tokenize_part(p, lex))
    for p in bracket_parts:
        t = _tokenize_part(p, lex)
        if len([x for x in t if x not in lex]) >= 2:
            add(t)

    return {
        "name_script": script,
        "name_ascii": ascii_,
        "name_norm": " ".join(full_toks),
        "name_core": " ".join(core_full),
        "name_alt": alts,
        "legal_class": legal_classes(atoms),
        "is_domain": is_domain,
    }
