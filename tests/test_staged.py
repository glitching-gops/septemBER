"""Dense top-k (GPU vs NumPy), the reverse filter, fusion and resumable checkpoints."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ber.blocking.dense import dense_topk, dense_topk_numpy
from ber.blocking.staged import DenseConfig, dense_partition, fuse, partition_slug

pytestmark = pytest.mark.usefixtures("toy_data_cpu_fallback")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _unit(rng, n, d=16):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def test_numpy_topk_is_exact_brute_force():
    rng = np.random.default_rng(0)
    Q, T = _unit(rng, 50), _unit(rng, 300)
    q, t, s = dense_topk_numpy(Q, T, 7)
    S = Q.astype(np.float32) @ T.astype(np.float32).T
    for i in range(50):
        got = t[q == i]
        ref = np.argsort(-S[i], kind="stable")[:7]
        assert set(got) == set(ref)
        assert np.all(np.diff(s[q == i]) <= 1e-6)  # sorted by score


def test_gpu_topk_matches_numpy_or_falls_back():
    rng = np.random.default_rng(1)
    Q, T = _unit(rng, 200), _unit(rng, 2000)
    q1, t1, s1 = dense_topk(Q, T, 10, chunk=64)  # GPU when enabled, else the NumPy fallback
    q2, t2, s2 = dense_topk_numpy(Q, T, 10)
    a = set(zip(q1.tolist(), t1.tolist()))
    b = set(zip(q2.tolist(), t2.tolist()))
    assert len(a ^ b) <= 4  # fp16 accumulation may swap near-ties at the k-th position
    np.testing.assert_allclose(np.sort(s1)[-50:], np.sort(s2)[-50:], atol=3e-3)


def test_reverse_filter_keeps_only_query_hits_against_full_s1():
    rng = np.random.default_rng(2)
    s1_all = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(6)], "addr_missing": [False] * 6,
                           "emb_row": np.arange(6)})
    q = s1_all.iloc[[1, 4]].reset_index(drop=True)          # the "dev slice": 2 of 6 S1
    t = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(6)], "addr_missing": [False] * 6,
                      "emb_row": np.arange(6)})
    E = _unit(rng, 6)
    emb = {"S1": {"name": E, "addr": E}, "T": {"name": E, "addr": E}}  # target i is identical to S1 i
    out, _ = dense_partition(q, t, s1_all, emb, DenseConfig(k6=2, k7=1))
    rev = out[out["b7_rank"] > 0]
    # reverse top-1 of each target is its identical S1; only S1-1 and S1-4 are queries
    assert set(zip(rev["qi"], rev["ti"])) == {(0, 1), (1, 4)}


def test_address_blockers_skip_missing_addresses():
    rng = np.random.default_rng(3)
    s1_all = pd.DataFrame({"entity_id": ["S1-0", "S1-1"], "addr_missing": [False, True], "emb_row": [0, 1]})
    t = pd.DataFrame({"entity_id": ["S2-0", "S2-1"], "addr_missing": [True, False], "emb_row": [0, 1]})
    E = _unit(rng, 2)
    out, _ = dense_partition(s1_all, t, s1_all, {"S1": {"name": E, "addr": E}, "T": {"name": E, "addr": E}},
                             DenseConfig(k6=2, k7=2))
    a = out[out.get("b6a_rank", pd.Series(0, index=out.index)).fillna(0) > 0]
    assert set(zip(a["qi"], a["ti"])) <= {(0, 1)}


def test_fuse_ranks_within_s1_and_source_and_caps():
    c = pd.DataFrame({"s1_id": ["A"] * 4, "m_id": ["x", "y", "z", "w"], "source": ["S2", "S2", "S3", "S2"],
                      "b1_rank": [1, 0, 1, 2], "b6n_rank": [1, 3, 0, 0]})
    f = fuse(c)
    r = dict(zip(f["m_id"], f["cand_rank"]))
    assert r["x"] == 1 and r["z"] == 1 and r["w"] == 2 and r["y"] == 3
    assert set(fuse(c, k_max=1)["m_id"]) == {"x", "z"}


def test_partition_slug_is_data_driven():
    assert partition_slug("S2", "Some Place") == "S2__Some_Place"


def test_resumable_checkpoint(tmp_path):
    from run_blocking import is_done, mark_done

    p = tmp_path / "lexical_S2__X.parquet"
    key = {"k": [1, 2, 3], "n_q": 2}
    assert not is_done(p, key)
    pd.DataFrame({"s1_id": ["a", "b"], "m_id": ["x", "y"]}).to_parquet(p)
    assert not is_done(p, key)          # no marker yet (crash mid-write)
    mark_done(p, key, 2, {})
    assert is_done(p, key)
    assert not is_done(p, {**key, "n_q": 3})   # different config -> recompute
    json.loads(p.with_suffix(".done.json").read_text())
