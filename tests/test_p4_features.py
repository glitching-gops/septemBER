"""P4 GPU ports vs CPU reference oracles on small fixtures (torch CPU tensors; production = CUDA).

The CPU reference implementations are used ONLY here (and in scripts/gpu_parity.py).
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ber.gpu import boot, calib, mining, search  # noqa: E402
from ber.gpu import decide as gd  # noqa: E402
from ber.gpu import metric as gm  # noqa: E402
from ber.gpu import segment as sg  # noqa: E402
from ber.match.context import C2C_FEATURES, GROUP_FEATURES, c2c_features, group_features  # noqa: E402
from ber.verify.augment import augment_pairs, rules, to_french_style  # noqa: E402

T = torch.tensor


def test_order_and_ranks():
    g = T([1, 0, 1, 0, 1])
    s = T([0.5, 0.9, 0.5, 0.1, 0.7])
    tie = T([9, 1, 3, 2, 5])
    np.testing.assert_array_equal(sg.rank_in_group(g, s, tie).numpy(), [3, 1, 2, 2, 1])
    np.testing.assert_array_equal(sg.topk_mask(g, s, 1, tie).numpy(), [0, 1, 0, 0, 1])
    np.testing.assert_array_equal(sg.group_argmax(g, s, tie, 3).numpy(), [1, 4, -1])


def _random_pairs(seed=0, n_s1=60, n_m=80, per=5):
    rng = np.random.default_rng(seed)
    s1 = np.repeat(np.arange(n_s1), per)
    m = rng.integers(n_s1, n_s1 + n_m, len(s1))
    df = pd.DataFrame({"s1": s1, "m": m}).drop_duplicates().reset_index(drop=True)
    df["p"] = np.round(rng.random(len(df)), 2).astype(np.float32)  # many ties on purpose
    return df


def test_exclusive_matches_cpu_semantics_with_ties():
    df = _random_pairs()
    keep = gd.exclusive(T(df["s1"].to_numpy()), T(df["m"].to_numpy()), T(df["p"].to_numpy()))
    o = df.sort_values(["m", "p", "s1"], ascending=[True, False, True], kind="stable")
    assert set(np.flatnonzero(keep.numpy())) == set(o.drop_duplicates("m", keep="first").index)


def test_metric_uses_ground_truth_counts():
    s1, m = T([0, 2]), T([10, 12])
    sel = T([True, True])
    gt_s1, gt_m = T([0, 0]), T([10, 11])
    ntrue = gm.n_true(gt_s1, 20)
    y = gm.labels(s1, m, gt_s1, gt_m, 20)
    f = gm.per_s1_f(sel, s1, y, ntrue)
    assert float(f[0]) == pytest.approx(1.25 * 0.5 / (0.25 + 0.5))  # unretrieved truth 11 is a miss
    assert float(f[1]) == 1 and float(f[2]) == 0


def test_tau_search_matches_loop():
    df = _random_pairs(1)
    rng = np.random.default_rng(1)
    s1, m, p = T(df["s1"].to_numpy()), T(df["m"].to_numpy()), T(df["p"].to_numpy())
    y = T(rng.random(len(df)) < 0.3)
    ntrue = torch.zeros(200, dtype=torch.int64).index_add_(0, s1[y], torch.ones(int(y.sum()), dtype=torch.int64))
    keep = gd.exclusive(s1, m, p)
    ids = torch.arange(60)
    taus = torch.linspace(0.1, 0.9, 81, dtype=torch.float64)
    tb, fb = search.best_tau(s1, y, p, keep, ids, ntrue, taus)
    ref = [gm.macro(gm.per_s1_f(gd.threshold(keep, p, float(t)), s1, y, ntrue), ids) for t in taus]
    assert fb == pytest.approx(max(ref), abs=1e-12)
    assert tb == pytest.approx(float(taus[int(np.argmax(ref))]))


def test_expected_f_matches_cpu_oracle():
    from ber.decide.expected_f import best_prefix
    df = _random_pairs(2)
    s1, p = T(df["s1"].to_numpy()), T(df["p"].to_numpy())
    sel = gd.expected_f(s1, p, torch.ones_like(s1, dtype=torch.bool)).numpy()
    for _, g in df.groupby("s1"):
        k, _ = best_prefix(g["p"].to_numpy())
        assert sel[g.index].sum() == k


def test_platt_isotonic_ks_psi():
    from scipy.stats import ks_2samp
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    z = rng.normal(size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(1.7 * z - 0.4)))).astype(float)
    lr = LogisticRegression(C=1e4).fit(z[:, None], y)
    a, b = calib.platt_fit(T(z), T(y))
    assert a == pytest.approx(lr.coef_[0, 0], abs=1e-4) and b == pytest.approx(lr.intercept_[0], abs=1e-4)
    x = 1 / (1 + np.exp(-z))
    iso = calib.BinnedIsotonic(T(x), T(y), n_bins=4000)  # one bin per point = exact isotonic
    ref = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    q = np.linspace(x.min(), x.max(), 500)
    assert np.abs(iso(T(q)).numpy() - ref.predict(q)).max() < 1e-9
    assert calib.ks(T(z[:2000]), T(z[2000:])) == pytest.approx(ks_2samp(z[:2000], z[2000:]).statistic)
    edges = np.unique(np.quantile(z[:2000], np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    ha = np.histogram(z[:2000], edges)[0] / 2000 + 1e-4
    hb = np.histogram(z[2000:] + 0.3, edges)[0] / 2000 + 1e-4
    assert calib.psi(T(z[:2000]), T(z[2000:] + 0.3)) == pytest.approx(float(np.sum((hb - ha) * np.log(hb / ha))))


def test_bootstrap_ci_brackets_mean():
    d = torch.linspace(-1, 1, 1001, dtype=torch.float64) + 0.1
    lo, hi = boot.ci(d, seed=0)
    assert lo < float(d.mean()) < hi


def test_group_features_hand_checked():
    s1, m, src = T([0, 0, 0, 1, 1]), T([10, 11, 12, 10, 13]), T([2, 3, 2, 2, 2])
    p = T([0.9, 0.5, 0.1, 0.6, 0.7])
    g = {k: v.numpy() for k, v in group_features(s1, m, src, p, 0.55, 20).items()}
    assert set(g) == set(GROUP_FEATURES)
    np.testing.assert_array_equal(g["grp_rank"], [1, 2, 3, 2, 1])
    np.testing.assert_array_equal(g["grp_rank_src"], [1, 1, 2, 2, 1])
    np.testing.assert_allclose(g["grp_gap_top1"], [0, 0.4, 0.8, 0.1, 0], atol=1e-6)
    np.testing.assert_array_equal(g["grp_n_above"], [1, 1, 1, 2, 2])
    np.testing.assert_allclose(g["grp_m_best_oth"], [0.6, 0, 0, 0.9, 0], atol=1e-6)
    np.testing.assert_array_equal(g["grp_m_rank"], [1, 1, 1, 2, 1])
    np.testing.assert_array_equal(g["grp_m_n"], [2, 1, 1, 2, 1])


def test_c2c_hand_checked_and_chunking():
    e = T([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)
    s1, m = T([0, 0, 0]), T([0, 1, 2])
    p = T([0.2, 0.9, 0.5])
    r = {k: v.numpy() for k, v in c2c_features(s1, m, p, T([2, 2, 3]), lambda f, i: e[i], T([True, True, False]),
                                               T([5, 5, 0]), T([0, 0, 0])).items()}
    assert set(r) == set(C2C_FEATURES)
    assert r["c2c_name_max_same"][0] == pytest.approx(1.0) and r["c2c_name_max_cross"][0] == pytest.approx(0.0)
    assert r["c2c_n_close"][0] == 1 and r["c2c_n_close"][2] == 0
    assert np.isnan(r["c2c_addr_max_same"][2]) and np.isnan(r["c2c_addr_max_cross"][0])
    assert r["c2c_hn_agree"][0] == 1 and np.isnan(r["c2c_hn_agree"][1])
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 9, 40)
    s1 = T(np.repeat(np.arange(40), sizes))
    n = s1.numel()
    E = torch.nn.functional.normalize(torch.randn(n, 8), dim=1).half()
    m = torch.arange(n)
    p = T(rng.random(n).astype(np.float32))
    src = T(rng.integers(2, 4, n))
    ok, hn = T(rng.random(n) > 0.2), T(rng.integers(0, 3, n))
    a = c2c_features(s1, m, p, src, lambda f, i: E[i], ok, hn, hn)
    b = c2c_features(s1, m, p, src, lambda f, i: E[i], ok, hn, hn, row_budget=16, pair_budget=128)
    for k in C2C_FEATURES:
        np.testing.assert_allclose(a[k].numpy(), b[k].numpy(), rtol=1e-5, atol=1e-6, equal_nan=True)


def test_scoring_set_band_and_rescue():
    s1, m = T([0] * 5 + [1] * 2), T([10, 11, 12, 13, 14, 10, 15])
    pe = T([0.001, 0.5, 0.004, 0.003, 0.002, 0.999, 0.2])
    ps3 = T([0.1, 0.2, 0.1, 0.1, 0.1, 0.95, 0.4])
    empty = torch.zeros(20, dtype=torch.bool)
    r = mining.scoring_set(s1, m, pe, ps3, empty, [0.005, 0.99], 3, 0.30, 20)
    np.testing.assert_array_equal(r["inband"].numpy(), [0, 1, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(r["rescued"].numpy(), [0, 1, 1, 1, 0, 0, 0])
    empty[1] = True
    r2 = mining.scoring_set(s1, m, pe, ps3, empty, [0.005, 0.99], 3, 0.30, 20)
    assert bool(r2["rescued"][5]) and bool(r2["rescued"][6])


def test_mining_matches_cpu_oracle():
    from ber.verify.train import build_training_pairs_r2
    cand = pd.DataFrame({"s1_id": ["a"] * 4, "m_id": ["t", "n1", "n2", "n3"], "y": [1, 0, 0, 0],
                         "key": [0.2, 0.9, 0.5, 0.8], "fp": [0, 0, 1, 0], "fn": [1, 0, 0, 0]})
    ref = build_training_pairs_r2(cand, {"a": {"t", "u"}}, ["a"], set(), set(), n_hard=2, n_rand=0, fn_dup=2)
    names = ["a", "t", "n1", "n2", "n3", "u"]
    mask = torch.zeros(6, dtype=torch.bool)
    mask[0] = True
    r = mining.training_rows(T([0] * 4), T([1, 2, 3, 4]), T([1, 0, 0, 0]), T([0.2, 0.9, 0.5, 0.8]), T([0, 0, 1, 0]),
                             T([1, 0, 0, 0]), mask, T([0, 0]), T([1, 5]), torch.zeros(6, dtype=torch.bool),
                             n_hard=2, n_rand=0, fn_dup=2)
    got = sorted((names[int(a)], names[int(b)], int(c)) for a, b, c in zip(r["s1"], r["m"], r["y"]))
    assert got == sorted(zip(ref["s1_id"], ref["m_id"], ref["y"]))


def test_french_style_targets_are_in_our_dictionaries():
    import yaml
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "src" / "ber" / "normalize" / "dictionaries"
    vocab = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                vocab.add(str(k).lower())
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            vocab.add(str(x).lower())
    for f in d.glob("*.yaml"):
        walk(yaml.safe_load(f.read_text(encoding="utf-8")))
    for group in ("street", "legal", "title"):
        for targets in rules()[group].values():
            for t in targets:
                assert t.rstrip(".").lower() in vocab, t


def test_french_style_rewrite_and_labels_untouched():
    rng = np.random.default_rng(1)
    name, addr = to_french_style("Acme Pvt Ltd", "12 Main Street, Pune", rng)
    assert "Pvt" not in name and name.split()[-1] in {"SARL", "SAS", "SA", "EURL"}
    assert "street" not in addr.lower()
    recs = pd.DataFrame({"name": ["Acme Ltd", "Acme"], "addr": ["1 Oak Road", "1 Oak Road"], "country": ["X", "X"],
                         "script": ["latin", "latin"], "translit": ["", ""]}, index=["S1-a", "S2-b"])
    rows = pd.DataFrame({"s1_id": ["S1-a"] * 50, "m_id": ["S2-b"] * 50, "y": [1] * 50})
    from ber.verify.serialize import ditto
    L, R, side = augment_pairs(rows, recs, ditto, frac=0.3, seed=0)
    assert 0 < (side >= 0).sum() < 50
    assert all("COL country VAL X" in t for t in L + R)
    L2, R2, _ = augment_pairs(rows, recs, ditto, frac=0.3, seed=0)
    assert L == L2 and R == R2


def test_exact_isotonic_matches_sklearn_with_ties():
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(7)
    x = np.round(rng.random(20000), 3)  # heavy ties
    y = (rng.random(20000) < x ** 2).astype(float)
    iso = calib.Isotonic(T(x), T(y))
    ref = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    q = np.concatenate([x[:3000], np.linspace(-0.1, 1.1, 777)])
    assert np.abs(iso(T(q)).numpy() - ref.predict(q)).max() < 1e-12
