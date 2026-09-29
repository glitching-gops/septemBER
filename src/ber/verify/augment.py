"""D3: French-style rewrite of verifier training records (our dictionary v2, in reverse).

``to_french_style(name, addr, rng)`` rewrites English street types, legal forms and titles
into their French renderings (``augment_fr.yaml``, hand-written) and changes the house-number
presentation. It never touches the label or the country field: a pair whose one side is
rewritten keeps its label, so the verifier learns that "12 Main Street" and "12 RUE MAIN" can
be the same record. ``augment_pairs`` applies it to ONE side of a seeded ``frac`` of pairs.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

_ST_END = r"(?=\s*(?:,|$))"  # "st" only when it closes a street segment ("Main St," / "Main St")


@lru_cache(maxsize=1)
def rules() -> dict:
    d = yaml.safe_load((Path(__file__).parent / "augment_fr.yaml").read_text(encoding="utf-8"))
    comp = {}
    for group in ("street", "legal", "title"):
        items = sorted(d[group].items(), key=lambda kv: -len(kv[0]))  # longest first
        comp[group] = [(re.compile(r"(?<![\w.])" + re.escape(k) + (_ST_END if k == "st" else r"(?![\w])"),
                                   re.IGNORECASE), v) for k, v in items]
    d["_compiled"] = comp
    return d


def _case(src: str, tgt: str) -> str:
    return tgt.upper() if src.isupper() else tgt.capitalize() if src[:1].isupper() else tgt


def _sub(text: str, pats, rng: np.random.Generator, upper: bool = False) -> str:
    for rx, targets in pats:
        if rx.search(text):
            text = rx.sub(lambda m: (str(targets[rng.integers(len(targets))]).upper() if upper else
                                     _case(m.group(0), str(targets[rng.integers(len(targets))]))), text)
    return text


def _street(addr: str, pats, rng: np.random.Generator) -> str:
    """Rewrite street types; a type closing its segment moves in front ("Main St" -> "Rue Main")."""
    segs = addr.split(",")
    out = []
    for seg in segs:
        done = False
        for rx, targets in pats:
            m = rx.search(seg)
            if not m:
                continue
            fr = _case(m.group(0), str(targets[rng.integers(len(targets))]))
            head, tail = seg[:m.start()].rstrip(), seg[m.end():]
            num = re.match(r"^(\s*\d+[a-zA-Z]?\s+)(.*)$", head)
            if not tail.strip() and head.strip():  # type closes the segment: French order
                seg = (f"{num.group(1)}{fr} {num.group(2)}" if num and num.group(2) else
                       f"{seg[:len(seg) - len(seg.lstrip())]}{fr} {head.strip()}")
            else:
                seg = f"{seg[:m.start()]}{fr}{seg[m.end():]}"
            done = True
            break
        out.append(seg)
    return ",".join(out) if done or len(out) else addr


_NUM = re.compile(r"^\s*(\d+[a-zA-Z]?)\s+([^,]+)(.*)$")


def to_french_style(name: str, addr: str, rng: np.random.Generator) -> tuple[str, str]:
    r = rules()
    c = r["_compiled"]
    name = _sub(_sub(name, c["legal"], rng, upper=True), c["title"], rng)  # legal forms: SARL, SAS
    if addr:
        addr = _street(addr, c["street"], rng)
        m = _NUM.match(addr)
        u = rng.random()
        if m and u < r["p_number_after"]:
            addr = f"{m.group(2).strip()} {m.group(1)}{m.group(3)}"
        elif m and u < r["p_number_after"] + r["p_number_prefix"]:
            pre = r["number_prefix"][rng.integers(len(r["number_prefix"]))]
            addr = f"{pre}{addr.lstrip()}"
        if rng.random() < r["p_upper_addr"]:
            addr = addr.upper()
    return name, addr


def augment_pairs(rows: pd.DataFrame, recs: pd.DataFrame, fn, frac: float, seed: int
                  ) -> tuple[list[str], list[str], np.ndarray]:
    """Serialise ``rows[s1_id, m_id]`` with ``fn``; rewrite ONE random side of ``frac`` of them.

    Returns (left texts, right texts, augmented-side array: -1 none, 0 left, 1 right). A pair
    whose rewrite changes nothing (no English cue, no house number) counts as not augmented.
    """
    rng = np.random.default_rng(seed)
    need = pd.unique(np.concatenate([rows["s1_id"].to_numpy(), rows["m_id"].to_numpy()]))
    recs_n = recs.loc[need]
    txt = {eid: fn(rr) for eid, rr in zip(need, recs_n.itertuples(index=False))}
    L = [txt[s] for s in rows["s1_id"]]
    R = [txt[m] for m in rows["m_id"]]
    side = np.full(len(rows), -1, np.int8)
    pick = np.flatnonzero(rng.random(len(rows)) < frac)
    sides = rng.integers(0, 2, len(pick))
    ids = (rows["s1_id"].to_numpy(), rows["m_id"].to_numpy())
    for i, sd in zip(pick, sides):
        rr = recs_n.loc[ids[sd][i]]
        nm, ad = to_french_style(rr["name"], rr["addr"], rng)
        if nm == rr["name"] and ad == rr["addr"]:
            continue
        t = fn(SimpleNamespace(**{**rr.to_dict(), "name": nm, "addr": ad}))
        (L if sd == 0 else R)[i] = t
        side[i] = sd
    return L, R, side
