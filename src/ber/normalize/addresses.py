"""Address normalisation (country-agnostic; EN/IN/FR vocabulary in one dictionary).

Output fields (see :func:`normalize_address`):
    addr_norm      canonical tokens of the whole address (landmarks included)
    addr_core      addr_norm without landmark phrases, placeholders and number markers
    landmark       tokens captured after a landmark trigger (near/opp/behind/...)
    nums           all digit runs, leading zeros stripped, as strings (order kept)
    house_no       leading house/door/plot number of the street segment ('' if none)
    postcode_like  5-6 digit runs (as written, leading zeros kept)
    region         canonical region codes found (space-joined)
    addr_missing   True when the address is empty or only placeholders -- downstream
                   this is MISSING, never a mismatch
"""
from __future__ import annotations

import re

from ber.normalize import dictionaries
from ber.normalize.text import nfkc, ordinal, to_ascii, tokens

_DEG = re.compile(r"\bn\s*[°º]\s*", re.IGNORECASE)
_DEGREE = re.compile(r"[°º]")
_NULLS = re.compile(r"<\s*null\s*>|\bn/a\b|\bnull\b", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_POSTCODE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
_HOUSE_MARKER = r"(?:h|house|door|d|plot|p|pl|flat|f|shop|sh|s|survey|sy|sr|kh|khasra|ward|wd|gat|no|n|num|number|#)"
_HOUSE = re.compile(
    rf"^\s*(?:{_HOUSE_MARKER}(?=[\s\.:/#-]|\d)[\s\.:/#-]*)*"
    r"(?P<hn>(?:[a-z]{1,2}[\s-]?)?\d+[a-z]?(?:\s*[-/]\s*\d+[a-z]?)*(?:\s*[-/]\s*[a-z]\b)?)"
    r"(?:\s+(?P<suf>bis|ter|quater)\b)?"
)
_ROMAN_FLOOR = {"ist": "1st", "iind": "2nd", "iird": "3rd", "iiird": "3rd", "ivth": "4th"}
_FLOOR_WORDS = {"floor", "fl", "flr", "etage"}
_FR_PARTICLES = {"de", "du", "des", "la", "le", "l", "d"}


_GLUED = re.compile(r"^([a-z]{3,})(\d+)$")


def _split_glued(toks: list[str]) -> list[str]:
    """'cour2' -> 'cour', '2' (a word glued to a number); short codes like 'b3' stay."""
    out = []
    for t in toks:
        m = _GLUED.match(t)
        out.extend(m.groups() if m else (t,))
    return out


def _house_no(segments: list[str]) -> str:
    for seg in segments:
        if re.fullmatch(r"\d{5,6}", seg):  # a bare PIN/ZIP segment is not a house number
            continue
        m = _HOUSE.match(seg)
        if m:
            hn = re.sub(r"\s+", "", m.group("hn"))
            hn = re.sub(r"^0+(?=\d)", "", hn)
            if m.group("suf"):
                hn += m.group("suf")
            return hn
    return ""


def _context(toks: list[str], d) -> list[str]:
    """Resolve st/ste/dr/r/bd/av by position within one comma segment."""
    out = []
    n = len(toks)
    units = d.unit_markers
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < n else None
        prev = toks[i - 1] if i > 0 else None
        last = nxt is None
        nxt_num = nxt is not None and nxt[0].isdigit()
        if t == "st":
            out.append("street" if (last or nxt_num or nxt in units) else "saint")
        elif t == "ste":
            out.append("suite" if nxt_num else "sainte")
        elif t == "dr":
            out.append("drive" if (last or nxt_num or nxt in units) else "doctor")
        elif t == "r":
            # "12 bis r. de la paix", "15 r francois madiot" -> rue; but keep Indian
            # initials such as "r k puram" / "5 r k road" untouched
            after_number = prev is not None and (prev[0].isdigit() or prev in d.house_suffixes)
            nxt_word = nxt is not None and len(nxt) > 1 and not nxt[0].isdigit()
            starts_fr = i == 0 and nxt in _FR_PARTICLES
            out.append("rue" if ((after_number and nxt_word) or starts_fr) else t)
        elif t in ("bd", "av"):
            out.append(d.context[t])
        elif t in _ROMAN_FLOOR and nxt in _FLOOR_WORDS:
            out.append(_ROMAN_FLOOR[t])
        else:
            out.append(ordinal(t) if t[0].isdigit() else t)
    return out


def normalize_address(raw: str) -> dict:
    d = dictionaries.load()
    s = nfkc(raw or "").strip()
    s = _DEG.sub(" no ", s)
    s = _DEGREE.sub(" ", s)
    s = _NULLS.sub(" ", s)
    low = to_ascii(s).casefold()

    segments = [seg.strip() for seg in low.split(",")]
    segments = [seg for seg in segments if seg]
    house_no = _house_no(segments)
    postcodes = _POSTCODE.findall(low)

    norm_tokens: list[str] = []
    core_tokens: list[str] = []
    landmark: list[str] = []
    regions: list[str] = []
    for seg in segments:
        toks = d.placeholders.apply(_split_glued(tokens(seg)))
        if not toks:
            continue
        toks = _context(toks, d)
        # a short segment ending in a region code ("IN", "VA 24153", "Hartford CT") keeps
        # that code verbatim: it must not be expanded as a street type (ct -> court)
        tail: list[str] = []
        p = len(toks) - 1
        while p >= 0 and toks[p].isdigit():
            p -= 1
        trailing = toks[p + 1:]
        zip_like = all(len(t) in (5, 6) for t in trailing)  # "fl 0" is a floor, "fl 33601" a state
        if p >= 0 and len(toks) <= 4 and toks[p] in d.region_codes and zip_like:
            regions.append(toks[p])
            tail = toks[p:]
            toks = toks[:p]
        toks = d.street.apply(toks)
        regions.extend(rep[0] for _, _, rep in d.regions.find(toks) if rep[0] in d.region_codes)
        toks = d.regions.apply(toks) + tail
        norm_tokens.extend(toks)
        # landmarks: trigger + up to N tokens, removed from core
        marked = d.landmark_triggers.apply(toks)
        i = 0
        while i < len(marked):
            if marked[i] == "<lm>":
                j = i + 1
                while j < len(marked) and j - i - 1 < d.landmark_max and marked[j] != "<lm>":
                    j += 1
                landmark.extend(marked[i + 1:j])
                i = j
            else:
                if marked[i] not in d.number_markers:
                    core_tokens.append(marked[i])
                i += 1

    nums = [n.lstrip("0") or "0" for n in _DIGITS.findall(" ".join(norm_tokens))]
    # a 5-digit leading house number ("16306 Edwards Rd") is not a postcode
    lead = re.match(r"\d+", house_no)
    if lead:
        postcodes = [p for p in postcodes if (p.lstrip("0") or "0") != lead.group(0)]
    missing = not core_tokens and not landmark
    return {
        "addr_norm": " ".join(norm_tokens),
        "addr_core": " ".join(core_tokens),
        "landmark": " ".join(landmark),
        "nums": nums,
        "house_no": house_no,
        "postcode_like": postcodes,
        "region": " ".join(dict.fromkeys(regions)),
        "addr_missing": missing,
    }
