"""Submission writer for the GPU pipeline: int pair arrays -> the two TSVs (host formatting = I/O).

The GPU side hands over pairs already sorted by S1 index; here they are split into per-S1 id
lists and written through ``ber.io_utils`` (which enforces matches subset of candidates).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ber.io import ids as I
from ber.io_utils import write_candidates, write_matching


def _lists(ids_arr, s1_sorted: np.ndarray, m_sorted: np.ndarray) -> dict:
    if len(s1_sorted) == 0:
        return {}
    cut = np.flatnonzero(s1_sorted[1:] != s1_sorted[:-1]) + 1
    keys = ids_arr[s1_sorted[np.r_[0, cut]]]
    return dict(zip(keys, (list(x) for x in np.split(ids_arr[m_sorted], cut))))


def write(cfg, outdir: Path, s1_order_ids, cand_s1, cand_m, sel_s1, sel_m) -> None:
    """``*_s1`` / ``*_m``: host int64 arrays sorted by s1 (GPU-sorted); ``s1_order_ids``: S1 strings in file order."""
    ids_arr = I.entity_table(cfg, "test")["entity_id"].to_numpy(object)
    cands = _lists(ids_arr, cand_s1, cand_m)
    matches = _lists(ids_arr, sel_s1, sel_m)
    outdir.mkdir(parents=True, exist_ok=True)
    write_candidates(outdir / "candidate_pairs.tsv", cands, s1_order_ids)
    write_matching(outdir / "matching_results.tsv", matches, s1_order_ids, candidates=cands)
