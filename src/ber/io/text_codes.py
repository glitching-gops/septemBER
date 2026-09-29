"""One-time text -> integer encodings for the GPU lexical blockers (cached; host encoding step only).

* ``name_bytes``: the transliterated name key + consonant skeleton (``b1_text``, P1's B1 input) of
  every entity, lower-case, as a zero-padded uint8 [n_entities, 64] array (the GPU computes the
  char 3-grams, MinHash and LSH bands).
* ``word_ids``: the words of ``name_core`` / ``addr_core`` dictionary-encoded to int32 ids, as
  (entity index per token, token id) arrays (the GPU computes df / IDF / postings / scores).
* ``house_codes``: house number as int64 (hash; 0 = missing) per entity.
Row order = the sorted entity index of ``ber.io.ids``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from ber.io import ids as I

SOURCES = ("source1", "source2", "source3")
L = 64


def _norm(cfg, split: str, cols: list[str], keys: bool = False) -> pd.DataFrame:
    d = Path(cfg["abs"]["cache_dir"]) / "norm"
    parts = []
    for s in SOURCES:
        f = d / (f"{split}_{s}_keys.parquet" if keys else f"{split}_{s}.parquet")
        parts.append(pd.read_parquet(f, columns=["entity_id"] + cols))
    return pd.concat(parts, ignore_index=True)


def name_bytes(cfg, split: str) -> np.ndarray:
    p = I.ids_dir(cfg) / f"{split}_name_bytes.npy"
    if p.exists():
        return np.load(p)
    n = len(I.entity_table(cfg, split))
    k = _norm(cfg, split, ["b1_text"], keys=True)
    e = I.encode(cfg, split, k["entity_id"])
    txt = k["b1_text"].fillna("").astype(str).str.lower().str.encode("ascii", "ignore")
    out = np.zeros((n, L), np.uint8)
    buf = np.frombuffer(b"".join(t[:L].ljust(L, b"\0") for t in txt), np.uint8).reshape(-1, L)
    out[e] = buf
    np.save(p, out)
    return out


def word_ids(cfg, split: str, field: str) -> dict:
    """{"ent": int64 [n_tok], "tok": int32 [n_tok], "vocab": int} for the words of name_core / addr_core,
    or ``field="both"``: name_core + " " + addr_core in ONE vocabulary (P1's B3 text)."""
    p = I.ids_dir(cfg) / f"{split}_{field}_words.npz"
    if p.exists():
        z = np.load(p)
        return {"ent": z["ent"], "tok": z["tok"], "vocab": int(z["vocab"])}
    if field == "both":
        d = _norm(cfg, split, ["name_core", "addr_core"])
        d["both"] = d["name_core"].fillna("") + " " + d["addr_core"].fillna("")
    else:
        d = _norm(cfg, split, [field])
    e = I.encode(cfg, split, d["entity_id"])
    arr = pa.array(d[field].fillna("").astype(str))
    lists = pc.split_pattern(pc.utf8_trim_whitespace(arr), " ")
    lens = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(lists)
    if isinstance(flat, pa.ChunkedArray):
        flat = flat.combine_chunks()
    enc = pc.dictionary_encode(flat)
    tok = enc.indices.to_numpy(zero_copy_only=False).astype(np.int32)
    words = enc.dictionary.to_numpy(zero_copy_only=False)
    ent = np.repeat(e, lens)
    keep = words[tok] != ""
    out = {"ent": ent[keep].astype(np.int64), "tok": tok[keep], "vocab": np.int64(len(words))}
    np.savez(p, **out)
    return {"ent": out["ent"], "tok": out["tok"], "vocab": int(out["vocab"])}


def addr_bytes(cfg, split: str, width: int = 96) -> np.ndarray:
    """``addr_core`` (ascii, lower) as zero-padded uint8 [n_entities, width] (P1's B2 input text)."""
    p = I.ids_dir(cfg) / f"{split}_addr_bytes.npy"
    if p.exists():
        return np.load(p)
    n = len(I.entity_table(cfg, split))
    d = _norm(cfg, split, ["addr_core"])
    e = I.encode(cfg, split, d["entity_id"])
    txt = d["addr_core"].fillna("").astype(str).str.lower().str.encode("ascii", "ignore")
    out = np.zeros((n, width), np.uint8)
    out[e] = np.frombuffer(b"".join(t[:width].ljust(width, b"\0") for t in txt), np.uint8).reshape(-1, width)
    np.save(p, out)
    return out
