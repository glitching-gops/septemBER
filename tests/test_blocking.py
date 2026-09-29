"""Blocking on a small synthetic fixture."""
import numpy as np
import pandas as pd

from ber.blocking.keys import key_pairs
from ber.blocking.pipeline import BlockingConfig, run_blocking
from ber.blocking.tfidf import SPECS, TfidfIndex, ranks_within_query


def _rows(prefix, items):
    return pd.DataFrame(
        [
            {"entity_id": f"{prefix}-{i}", "country": c, "name_core": n, "addr_core": a, "house_no": h}
            for i, (c, n, a, h) in enumerate(items)
        ]
    )


Q = _rows("S1", [
    ("AA", "acme widgets", "12 main street springfield", "12"),
    ("BB", "acme widgets", "12 main street springfield", "12"),
    ("AA", "zeta foods", "9 river road shelbyville", "9"),
])
T2 = _rows("S2", [
    ("AA", "acme widget", "12 main st springfield", "12"),
    ("BB", "acme widgets", "12 main street springfield", "12"),
    ("AA", "zeta food", "9 river road shelbyville", "9"),
    ("AA", "other thing", "77 elm lane ogdenville", "77"),
] + [("AA", f"filler {i} co", f"{100 + i} oak avenue capital city", str(100 + i)) for i in range(40)])
T3 = _rows("S3", [
    ("AA", "acme widgets inc", "springfield 12 main street", "12"),
    ("CC", "acme widgets", "12 main street springfield", "12"),
])


def _run(**kw):
    cfg = BlockingConfig(k1=5, k2=5, k3=5, k5=2, key_cap=500, threshold=0.0, **kw)
    return run_blocking(Q, {"S2": T2, "S3": T3}, cfg, log=lambda *a: None)


def test_country_gating():
    c = _run()
    s1_country = dict(zip(Q["entity_id"], Q["country"]))
    m_country = dict(zip(pd.concat([T2, T3])["entity_id"], pd.concat([T2, T3])["country"]))
    assert (c["s1_id"].map(s1_country) == c["m_id"].map(m_country)).all()
    assert "S3-1" not in set(c["m_id"])  # the CC record can never be reached


def test_true_pairs_found_in_both_sources():
    pairs = set(zip(_run()["s1_id"], _run()["m_id"]))
    assert {("S1-0", "S2-0"), ("S1-1", "S2-1"), ("S1-2", "S2-2"), ("S1-0", "S3-0")} <= pairs


def test_union_is_deduplicated_with_per_blocker_columns():
    c = _run()
    assert not c.duplicated(["s1_id", "m_id"]).any()
    row = c[(c["s1_id"] == "S1-0") & (c["m_id"] == "S2-0")].iloc[0]
    assert row["b1_rank"] >= 1 and row["b2_rank"] >= 1
    assert {"b1_score", "b2_score", "b3_score", "b4_hn", "b4_name", "b5_rank", "best_norm", "cand_rank"} <= set(c.columns)


def test_per_source_topk_and_kmax_cap():
    c = _run()
    # per-source ranks are independent: S1-0 has a rank-1 candidate in each source
    top = c[c["cand_rank"] == 1].groupby("s1_id")["source"].nunique()
    assert top.loc["S1-0"] == 2
    capped = _run(k_max=2)
    assert capped.groupby(["s1_id", "source"]).size().max() <= 2
    # B1 alone never returns more than k1 per query per source
    assert c[c["b1_rank"] > 0].groupby(["s1_id", "source"]).size().max() <= 5


def test_block_size_cap():
    t_addr = pd.Series(["5 common street"] * 30 + ["5 unique lane"])
    t_house = pd.Series(["5"] * 31)
    t_name = pd.Series(["shared"] * 30 + ["rare"])
    q = key_pairs(pd.Series(["5 common street"]), pd.Series(["5"]), pd.Series(["shared"]),
                  t_addr, t_house, t_name, cap=10)
    assert q.empty  # both keys hit a 30-record block > cap
    q = key_pairs(pd.Series(["5 unique lane"]), pd.Series(["5"]), pd.Series(["rare"]),
                  t_addr, t_house, t_name, cap=10)
    assert set(q["t"]) == {30} and bool(q["b4_hn"].iloc[0]) and bool(q["b4_name"].iloc[0])


def test_ranks_within_query():
    assert ranks_within_query(np.array([0, 0, 0, 2, 2, 5])).tolist() == [1, 2, 3, 1, 2, 1]


def test_tfidf_forward_and_reverse_agree_on_best_match():
    idx = TfidfIndex(SPECS["b1_name"]).fit(["acme widgets", "zeta foods", "other"])
    Qm = idx.transform(["acme widget", "zeta food"])
    q, t, s = idx.forward(Qm, k=1)
    assert t.tolist() == [0, 1]
    t2, q2, _ = idx.reverse(Qm, k=1)
    assert dict(zip(t2.tolist(), q2.tolist()))[0] == 0


def test_reverse_uses_full_s1_population_for_subset_queries():
    cfg = BlockingConfig(k1=5, k2=5, k3=5, k5=1, key_cap=500, threshold=0.0)
    full = run_blocking(Q, {"S2": T2}, cfg, log=lambda *a: None)
    sub_q = Q[Q["entity_id"] == "S1-0"]
    sub = run_blocking(sub_q, {"S2": T2}, cfg, log=lambda *a: None, s1_all=Q)
    assert set(sub["s1_id"]) == {"S1-0"}
    f5 = full[(full["s1_id"] == "S1-0") & (full["b5_rank"] > 0)]
    s5 = sub[sub["b5_rank"] > 0]
    assert set(f5["m_id"]) == set(s5["m_id"])  # identical reverse hits for the query


def test_dense_blocker_b6_on_gpu():
    import pytest

    torch = pytest.importorskip("torch")
    from ber import compute

    if not (torch.cuda.is_available() and compute.use_gpu()):
        pytest.skip("GPU not available or not enabled")
    rng = np.random.default_rng(0)
    t_emb = rng.normal(size=(len(T2), 16)).astype(np.float16)
    t_emb /= np.linalg.norm(t_emb.astype(np.float32), axis=1, keepdims=True).astype(np.float16)
    q_emb = t_emb[[0, 1, 2]].copy()  # each query's embedding equals one target's
    emb = {"S1": q_emb, "S2": t_emb}
    q = Q.assign(emb_row=[0, 1, 2])
    t = T2.assign(emb_row=np.arange(len(T2)))
    cfg = BlockingConfig(k1=5, k2=5, k3=5, k5=1, k6=3, threshold=0.0)
    c = run_blocking(q, {"S2": t}, cfg, log=lambda *a: None, emb=emb)
    top = c[c["b6_rank"] == 1].set_index("s1_id")["m_id"].to_dict()
    assert top.get("S1-0") == "S2-0" and top.get("S1-2") == "S2-2"
