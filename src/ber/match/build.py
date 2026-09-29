"""Assemble the pair-feature matrix for a candidate table.

Moved out of ``scripts/run_matching.py`` in P2 so the CV script, the sharded feature
stage (``scripts/run_features.py``) and test inference all build features through one
implementation. The P1 behaviour is unchanged: ``build_features`` returns exactly the
columns in :data:`ber.match.features.FEATURES`, in that order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ber.match.features import NORM_COLS, pair_cosines, pair_features_parallel
from ber.normalize.run import load_norm


def load_with_keys(cfg: dict, split: str, source: str, ids: set | None) -> pd.DataFrame:
    """Normalised rows plus ``name_key``; ``ids=None`` keeps every row."""
    df = load_norm(cfg, split, source, NORM_COLS)
    keys = pd.read_parquet(cfg["abs"]["cache_dir"] / "norm" / f"{split}_{source}_keys.parquet",
                           columns=["entity_id", "name_key"])
    df["name_key"] = keys["name_key"].to_numpy()
    return df if ids is None else df[df["entity_id"].isin(ids)]


def add_cosines(cfg: dict, cand: pd.DataFrame, split: str) -> pd.DataFrame:
    """Exact E5 name/address cosines for every candidate pair (vectorised)."""
    from ber.embed.encoder import load_emb

    cand = cand.copy()
    row = {}
    for s in ("source1", "source2", "source3"):
        ids = load_norm(cfg, split, s, ["entity_id"])["entity_id"].to_numpy()
        row[s] = pd.Series(np.arange(len(ids)), index=ids)
    s1_rows = cand["s1_id"].map(row["source1"]).to_numpy()
    for field in ("name", "addr"):
        E1 = load_emb(cfg, split, "source1", field)
        out = np.zeros(len(cand), np.float32)
        for src, s in (("S2", "source2"), ("S3", "source3")):
            m = (cand["source"] == src).to_numpy()
            if m.any():
                out[m] = pair_cosines(cand[m], s1_rows[m], cand.loc[m, "m_id"].map(row[s]).to_numpy(),
                                      E1, load_emb(cfg, split, s, field))
        cand[f"{field}_cos"] = out
    return cand


def build_features(cfg: dict, cand: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Feature matrix aligned row-for-row with ``cand``."""
    cand = add_cosines(cfg, cand, split)
    s1n = load_with_keys(cfg, split, "source1", set(cand["s1_id"]))
    ids = set(cand["m_id"])
    mn = pd.concat([load_with_keys(cfg, split, s, ids) for s in ("source2", "source3")])
    return pair_features_parallel(cand, s1n, mn)


class FeatureContext:
    """Normalised tables and embedding row maps, loaded once and reused for many shards.

    Building a shard on its own re-reads three normalised tables plus the keys tables,
    which dominates the cost when the shard is small. Holding them here turns a 61 s
    per-shard build into a few seconds, at the price of keeping the tables resident.
    Produces exactly the same features as :func:`build_features`.
    """

    def __init__(self, cfg: dict, split: str, gpu_features: bool = False):
        from ber.embed.encoder import load_emb

        self.cfg, self.split, self.gpu_features = cfg, split, gpu_features
        self.device, self.spaces, self.tgt_row = None, None, None
        self.row, self.emb = {}, {}
        for s in ("source1", "source2", "source3"):
            ids = load_norm(cfg, split, s, ["entity_id"])["entity_id"].to_numpy()
            self.row[s] = pd.Series(np.arange(len(ids)), index=ids)
        for s in ("source1", "source2", "source3"):
            for field in ("name", "addr"):
                self.emb[s, field] = load_emb(cfg, split, s, field)  # mmap, not resident
        self.s1n = load_with_keys(cfg, split, "source1", None).set_index("entity_id")
        self.mn = pd.concat([load_with_keys(cfg, split, s, None)
                             for s in ("source2", "source3")]).set_index("entity_id")
        if gpu_features:
            self._init_token_spaces()

    def _init_token_spaces(self) -> None:
        """Shared-vocabulary token spaces for the GPU features (see ber.match.features_gpu).

        The name space uses ``name_key`` (the transliteration-mapped key) rather than
        ``name_core``: the P1 audit found cross-script pairs share no ``name_core`` tokens
        at all, so IDF overlap on the raw core would be identically zero for them.
        """
        from ber.embed import gpu_setup
        from ber.match.features_gpu import MAX_TOKENS, TokenSpace

        self.device = gpu_setup()
        self.s1_tok_row = pd.Series(np.arange(len(self.s1n)), index=self.s1n.index)
        self.tgt_row = pd.Series(np.arange(len(self.mn)), index=self.mn.index)
        self.spaces = {
            "name": TokenSpace(self.s1n["name_key"].reset_index(drop=True),
                               self.mn["name_key"].reset_index(drop=True), MAX_TOKENS["name"]),
            "addr": TokenSpace(self.s1n["addr_core"].reset_index(drop=True),
                               self.mn["addr_core"].reset_index(drop=True), MAX_TOKENS["addr"]),
        }

    def add_cosines(self, cand: pd.DataFrame) -> pd.DataFrame:
        cand = cand.copy()
        s1_rows = cand["s1_id"].map(self.row["source1"]).to_numpy()
        for field in ("name", "addr"):
            out = np.zeros(len(cand), np.float32)
            for src, s in (("S2", "source2"), ("S3", "source3")):
                m = (cand["source"] == src).to_numpy()
                if m.any():
                    target_rows = cand.loc[m, "m_id"].map(self.row[s]).to_numpy()
                    if self.gpu_features:
                        import torch
                        # Gather cached vectors during host staging; multiply and reduce on CUDA.
                        left = torch.as_tensor(self.emb["source1", field][s1_rows[m]], device=self.device)
                        right = torch.as_tensor(self.emb[s, field][target_rows], device=self.device)
                        out[m] = (left.float() * right.float()).sum(1).cpu().numpy()
                        del left, right
                    else:
                        out[m] = pair_cosines(cand[m], s1_rows[m], target_rows,
                                              self.emb["source1", field], self.emb[s, field])
            cand[f"{field}_cos"] = out
        return cand

    def features(self, cand: pd.DataFrame) -> pd.DataFrame:
        withcos = self.add_cosines(cand)
        F = pair_features_parallel(withcos, self.s1n, self.mn)
        if not self.gpu_features:
            return F
        from ber.match.features_gpu import pair_token_features

        withcos = withcos.copy()
        withcos["addr_missing_pair"] = F["addr_missing"].to_numpy(np.float32)
        l_rows = withcos["s1_id"].map(self.s1_tok_row).to_numpy()
        r_rows = withcos["m_id"].map(self.tgt_row).to_numpy()
        G = pair_token_features(withcos, l_rows, r_rows, self.spaces, self.device)
        return pd.concat([F.reset_index(drop=True), G.reset_index(drop=True)], axis=1)
