"""Record -> text for the verifier models.

Raw ``business_name`` / ``business_address`` are used (NFKC only): the rerankers were
pretrained on natural text, and our normalisation (legal-form stripping, street-type
canonicalisation) removes exactly the cues they can read. The learned transliteration key
is appended for non-Latin names, because it is the only Latin-script view the model gets of
a Devanagari/Tamil/... name.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ber.io_utils import read_source

EMPTY = "[EMPTY]"
LATIN = "latin"


def _clean(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).map(lambda x: unicodedata.normalize("NFKC", x).strip())
    return s.str.replace(r"\s+", " ", regex=True)


def load_records(cfg: dict, split: str, ids: set[str] | None = None) -> pd.DataFrame:
    """Raw text + script + transliteration key for every record of ``split`` (index entity_id).

    ``ids`` restricts the result (the raw TSVs are still read in full; filtering after the
    read keeps this simple and the read is ~1 min for a whole split).
    """
    a = cfg["abs"]
    norm_dir = Path(a["cache_dir"]) / "norm"
    out = []
    for src in ("source1", "source2", "source3"):
        raw, _ = read_source(a[f"{split}.{src}"], validate=False)
        if ids is not None:
            raw = raw[raw["entity_id"].isin(ids)]
        norm = pd.read_parquet(norm_dir / f"{split}_{src}.parquet", columns=["entity_id", "name_script"])
        keys = pd.read_parquet(norm_dir / f"{split}_{src}_keys.parquet", columns=["entity_id", "name_key"])
        df = (raw.merge(norm, on="entity_id", how="left").merge(keys, on="entity_id", how="left"))
        out.append(pd.DataFrame({
            "entity_id": df["entity_id"].to_numpy(),
            "name": _clean(df["business_name"]).to_numpy(),
            "addr": _clean(df["business_address"]).to_numpy(),
            "country": _clean(df["country"]).to_numpy(),
            "script": df["name_script"].fillna(LATIN).to_numpy(),
            "translit": df["name_key"].fillna("").astype(str).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True).set_index("entity_id")


def _translit_suffix(script: str, translit: str) -> str:
    return translit if script != LATIN and translit else ""


def plain(r) -> str:
    """Natural-language record for instruction-following rerankers / NLI."""
    s = f"name: {r.name}; address: {r.addr or EMPTY}; country: {r.country}"
    t = _translit_suffix(r.script, r.translit)
    return f"{s}; name transliterated: {t}" if t else s


def ditto(r) -> str:
    """Ditto-style ``COL attr VAL value`` serialisation for encoder cross-encoders."""
    s = f"COL name VAL {r.name} COL address VAL {r.addr or EMPTY} COL country VAL {r.country}"
    t = _translit_suffix(r.script, r.translit)
    return f"{s} COL name_translit VAL {t}" if t else s


def pair_texts(pairs: pd.DataFrame, recs: pd.DataFrame, style: str) -> tuple[list[str], list[str]]:
    """(left texts, right texts) for ``pairs[s1_id, m_id]`` in the given style."""
    fn = {"plain": plain, "ditto": ditto}[style]
    need = pd.unique(np.concatenate([pairs["s1_id"].to_numpy(), pairs["m_id"].to_numpy()]))
    txt = {eid: fn(r) for eid, r in zip(need, recs.loc[need].itertuples(index=False))}
    return [txt[s] for s in pairs["s1_id"]], [txt[m] for m in pairs["m_id"]]
