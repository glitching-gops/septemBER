"""Verifier fold bookkeeping: OOF alignment, holdout isolation, feature construction."""
import numpy as np
import pandas as pd

from ber.verify import folds as vf
from ber.verify.train import build_training_pairs


def test_verifier_fold_is_xgb_fold_mod_2():
    assert vf.verifier_fold([0, 1, 2, 3, 4]).tolist() == [0, 1, 0, 1, 0]


def test_holdout_is_whole_components_and_stable():
    comps = np.array([f"C{i}" for i in range(5000)])
    m1 = vf.holdout_mask(comps, 0.15)
    m2 = vf.holdout_mask(comps, 0.15)
    assert (m1 == m2).all()
    assert 0.12 < m1.mean() < 0.18
    # repeated component ids always land on the same side
    rep = np.repeat(comps[:100], 3)
    m = vf.holdout_mask(rep, 0.15).reshape(100, 3)
    assert (m == m[:, :1]).all()


def _toy():
    rng = np.random.default_rng(0)
    s1 = [f"S1-{i}" for i in range(40)]
    rows = []
    for s in s1:
        for k in range(12):
            rows.append((s, f"S2-{s}-{k}", "S2", float(rng.random()), int(k == 0)))
    oof = pd.DataFrame(rows, columns=["s1_id", "m_id", "source", "p", "y"])
    gt = {s: {f"S2-{s}-0"} for s in s1}
    dev = pd.DataFrame({"s1_id": s1, "fold": [i % 5 for i in range(40)],
                        "component_id": [f"C{i}" for i in range(40)]})
    return oof, gt, dev


def test_no_entity_scored_by_a_model_that_trained_on_it():
    oof, gt, dev = _toy()
    dev["holdout"] = vf.holdout_mask(dev["component_id"].to_numpy(), 0.15)
    dev["vfold"] = vf.verifier_fold(dev["fold"])
    hold = set(dev.loc[dev["holdout"], "s1_id"])
    for j in (0, 1):
        tr = dev.loc[~dev["holdout"] & (dev["vfold"] == j), "s1_id"]
        rows = build_training_pairs(oof, gt, tr, set(), set(), n_hard=3, n_rand=1)
        scored_oof = set(dev.loc[~dev["holdout"] & (dev["vfold"] != j), "s1_id"])
        assert not set(rows["s1_id"]) & scored_oof
        assert not set(rows["s1_id"]) & hold
    # every non-holdout S1 is scored out-of-fold by exactly one verifier model
    nh = dev[~dev["holdout"]]
    assert all(((nh["vfold"] != j).astype(int)).sum() >= 0 for j in (0, 1))
    assert ((nh["vfold"] != 0).astype(int) + (nh["vfold"] != 1).astype(int) == 1).all()


def test_training_pairs_composition():
    oof, gt, _ = _toy()
    ids = [f"S1-{i}" for i in range(5)]
    rows = build_training_pairs(oof, gt, ids, nonlatin={"S1-0"}, empty_addr=set(), n_hard=3, n_rand=2,
                                oversample=2)
    per = rows[rows["s1_id"] == "S1-1"]
    assert per["y"].sum() == 1 and (per["y"] == 0).sum() == 5
    # the non-Latin S1 is repeated oversample times
    assert (rows["s1_id"] == "S1-0").sum() == 2 * 6
    # hard negatives are the top-p negatives
    neg = oof[(oof["s1_id"] == "S1-1") & (oof["y"] == 0)].nlargest(3, "p")["m_id"]
    assert set(neg) <= set(per["m_id"])


def test_verifier_features_nan_outside_band_and_rank():
    pairs = pd.DataFrame({"s1_id": ["a", "a", "a", "b"], "m_id": ["x", "y", "z", "w"]})
    scored = pd.DataFrame({"s1_id": ["a", "a"], "m_id": ["x", "y"], "logit": [1.0, 3.0], "p": [0.7, 0.9]})
    V = vf.verifier_features(pairs, scored)
    assert V["v_inband"].tolist() == [1, 1, 0, 0]
    assert np.isnan(V.loc[2, "v_logit"]) and np.isnan(V.loc[3, "v_p"])
    assert V.loc[1, "v_rank"] == 1 and V.loc[0, "v_rank"] == 2
    assert V.loc[1, "v_margin"] == 0 and V.loc[0, "v_margin"] == -2
