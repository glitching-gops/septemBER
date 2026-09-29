import numpy as np
import pandas as pd

from ber.evaluate.splits import build_components, check_no_positive_leak, make_cv_folds, make_loco


def test_components_link_shared_targets():
    gt = {"a": frozenset({"S2-1"}), "b": frozenset({"S2-1", "S3-9"}), "c": frozenset({"S3-9"}),
          "d": frozenset(), "e": frozenset({"S2-5"})}
    comp = build_components(list("abcde"), gt)
    assert comp[0] == comp[1] == comp[2]
    assert len({comp[0], comp[3], comp[4]}) == 3
    assert comp[0] == 0  # relabelled by first appearance


def test_folds_keep_components_together_and_no_leak():
    rng = np.random.default_rng(0)
    ids = [f"S1-{i}" for i in range(400)]
    gt = {s: frozenset({f"S2-{i // 2}"}) if i % 3 else frozenset() for i, s in enumerate(ids)}
    comp = build_components(ids, gt)
    strata = rng.choice(["x|m", "y|s"], size=len(ids))
    folds = make_cv_folds(comp, strata, n_folds=5, seed=1)
    assert set(folds) == set(range(5))
    assert pd.Series(folds).groupby(comp).nunique().max() == 1
    assert check_no_positive_leak(folds, ids, gt) == 0
    assert (make_cv_folds(comp, strata, 5, 1) == folds).all()  # deterministic


def test_loco_is_data_driven():
    loco = make_loco(pd.Series(["p", "q", "p", "r"]))
    assert list(loco.columns) == ["loco:p", "loco:q", "loco:r"]
    assert loco["loco:p"].tolist() == ["valid", "train", "valid", "train"]
