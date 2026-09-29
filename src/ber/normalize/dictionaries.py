"""Load the YAML dictionaries under normalize/dictionaries/ into lookup structures.

Dictionaries are data, applied to every record regardless of country.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from ber.normalize.text import PhraseMapper, tokens

DICT_DIR = Path(__file__).resolve().parent / "dictionaries"


def _load(name: str) -> dict:
    with open(DICT_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _phrase(s: str) -> tuple[str, ...]:
    return tuple(tokens(str(s).lower()))


@dataclass(frozen=True)
class Dictionaries:
    legal_atom: dict[str, str]            # token -> atom
    legal_ambiguous: frozenset[str]
    honorifics: PhraseMapper              # leading phrases -> ()
    alias_markers: tuple[str, ...]
    tlds: tuple[str, ...]
    street: PhraseMapper                  # variants -> canonical
    street_canonical: frozenset[str]
    context: dict
    unit_markers: frozenset[str]
    house_suffixes: frozenset[str]
    placeholders: PhraseMapper
    number_markers: frozenset[str]
    landmark_triggers: PhraseMapper       # trigger phrase -> ("<lm>",)
    landmark_max: int
    regions: PhraseMapper                 # variant -> code
    region_codes: frozenset[str]
    stopwords: frozenset[str]

    def all_known_tokens(self) -> frozenset[str]:
        known = set(self.legal_atom) | set(self.street_canonical) | set(self.region_codes)
        for m in (self.street, self.regions, self.honorifics, self.placeholders, self.landmark_triggers):
            for k, v in m.mapping.items():
                known.update(k)
                known.update(v)
        known |= self.unit_markers | self.house_suffixes | self.number_markers | self.stopwords
        known |= set(self.context)
        return frozenset(known)


@lru_cache(maxsize=1)
def load() -> Dictionaries:
    """Load the shared normalisation dictionaries once."""
    legal = _load("legal_forms.yaml")
    atom = {}
    for a, toks in legal["atoms"].items():
        for t in toks:
            atom[str(t).lower()] = a
    hon = _load("honorifics.yaml")["prefixes"]
    honor = PhraseMapper({_phrase(p): () for p in hon if _phrase(p)})
    alias = tuple(str(m).lower() for m in _load("alias_markers.yaml")["markers"])
    tlds = tuple(sorted((str(t).lower() for t in _load("tlds.yaml")["tlds"]), key=len, reverse=True))

    st = _load("street_types.yaml")
    smap: dict[tuple[str, ...], tuple[str, ...]] = {}
    for canon, variants in st["expand"].items():
        for v in variants:
            k = _phrase(v)
            if k and k != (canon,):
                smap.setdefault(k, (canon,))
    ph = PhraseMapper({_phrase(p): () for p in st["placeholders"] if _phrase(p)})

    lm = _load("landmarks.yaml")
    lmap = {_phrase(t): ("<lm>",) for t in lm["triggers"]}
    # expanded canonical forms also trigger
    lmap[("near",)] = ("<lm>",)
    lmap[("opposite",)] = ("<lm>",)

    rg = _load("regions.yaml")
    rmap: dict[tuple[str, ...], tuple[str, ...]] = {}
    for code, variants in rg["regions"].items():
        for v in variants:
            rmap[_phrase(v)] = (code,)
    for canon, variants in rg["city_aliases"].items():
        for v in variants:
            rmap[_phrase(v)] = (canon,)

    return Dictionaries(
        legal_atom=atom,
        legal_ambiguous=frozenset(str(t).lower() for t in legal.get("ambiguous_atoms", [])),
        honorifics=honor,
        alias_markers=alias,
        tlds=tlds,
        street=PhraseMapper(smap),
        street_canonical=frozenset(st["expand"].keys()),
        context=st["context"],
        unit_markers=frozenset(str(u).lower() for u in st["unit_markers"]),
        house_suffixes=frozenset(str(u).lower() for u in st["house_suffixes"]),
        placeholders=ph,
        number_markers=frozenset(str(u).lower() for u in st["number_markers"]),
        landmark_triggers=PhraseMapper(lmap),
        landmark_max=int(lm["max_tokens"]),
        regions=PhraseMapper(rmap),
        region_codes=frozenset(rg["regions"].keys()),
        stopwords=frozenset(str(w).lower() for w in _load("stopwords.yaml")["words"]),
    )
