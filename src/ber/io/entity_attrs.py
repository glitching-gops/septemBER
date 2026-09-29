"""Per-entity lookups for the c2c features, built once and cached (encoding / I/O only).

* embedding row of each entity in ``<cache>/emb/{split}_{source}_{field}.npy`` (+ which file);
* house-number / postcode as int64 codes (hash of the normalised string; 0 = missing);
* address-present flag.

``EmbReader`` returns the embeddings of a set of entity indices as a CUDA tensor: the index
order is computed on the GPU, the host only reads those rows (byte I/O) from the memory map.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber.io import ids as I
from ber.io.gpu_load import to_dev

SOURCES = ("source2", "source3", "source1")  # embedding file index k = position here


def attrs(cfg, split: str) -> dict:
    p = I.ids_dir(cfg) / f"{split}_attrs.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    n = len(I.entity_table(cfg, split))
    out = {"hn": np.zeros(n, np.int64), "pc": np.zeros(n, np.int64), "addr_ok": np.zeros(n, bool)}
    emb_dir = Path(cfg["abs"]["cache_dir"]) / "emb"
    norm = Path(cfg["abs"]["cache_dir"]) / "norm"
    for f in ("name", "addr"):
        out[f"{f}_file"] = np.full(n, -1, np.int8)
        out[f"{f}_row"] = np.full(n, -1, np.int64)
    for s in SOURCES:
        d = pd.read_parquet(norm / f"{split}_{s}.parquet", columns=["entity_id", "house_no", "postcode_like", "addr_missing"])
        e = I.encode(cfg, split, d["entity_id"])
        hn = d["house_no"].fillna("").astype(str).to_numpy(object)
        pc = d["postcode_like"].map(lambda x: x[0] if x is not None and len(x) else "").to_numpy(object)
        for key, v in (("hn", hn), ("pc", pc)):
            h = pd.util.hash_array(v).astype(np.int64)
            out[key][e] = np.where(v == "", 0, np.where(h == 0, 1, h))
        out["addr_ok"][e] = ~d["addr_missing"].astype(bool).to_numpy()
    for k, s in enumerate(SOURCES):
        for f in ("name", "addr"):
            ids = np.load(emb_dir / f"{split}_{s}_{f}_ids.npy", allow_pickle=True)
            e = I.encode(cfg, split, ids)
            out[f"{f}_file"][e] = k
            out[f"{f}_row"][e] = np.arange(len(ids), dtype=np.int64)
    np.savez(p, **out)
    return out


class EmbReader:
    def __init__(self, cfg, split: str, device):
        import torch
        a = attrs(cfg, split)
        emb_dir = Path(cfg["abs"]["cache_dir"]) / "emb"
        self.dev = device
        self.file = {f: to_dev(a[f"{f}_file"], device) for f in ("name", "addr")}
        self.row = {f: to_dev(a[f"{f}_row"], device) for f in ("name", "addr")}
        self.mm = {(f, k): np.load(emb_dir / f"{split}_{s}_{f}.npy", mmap_mode="r")
                   for k, s in enumerate(SOURCES) for f in ("name", "addr")}
        self.dim = self.mm[("name", 0)].shape[1]
        self.torch = torch
        self.io_wall = 0.0  # host byte reads (+ transfer), for the caller's StageTimer.io
        self.io_cpu = 0.0

    def __call__(self, field: str, ent):
        """Embeddings (float16, [len(ent), dim]) of entity indices ``ent`` (CUDA int64)."""
        torch = self.torch
        out = torch.zeros((ent.numel(), self.dim), dtype=torch.float16, device=self.dev)
        fi, ri = self.file[field][ent], self.row[field][ent]
        for k in range(len(SOURCES)):
            sel = torch.nonzero(fi == k).squeeze(1)
            if sel.numel() == 0:
                continue
            rows, inv = torch.unique(ri[sel], sorted=True, return_inverse=True)
            w, c = time.time(), time.process_time()
            host = np.asarray(self.mm[(field, k)][rows.cpu().numpy()])  # byte read of sorted rows
            dev_rows = to_dev(host, self.dev)
            self.io_wall += time.time() - w
            self.io_cpu += time.process_time() - c
            out[sel] = dev_rows[inv]
        return out
