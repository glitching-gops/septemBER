"""Hand-computed checks for the per-entity F0.5 and its macro average."""
import pytest

from ber.evaluate.metric import f05_entity, macro_f05


# --- per-entity -------------------------------------------------------------

def test_problem_statement_example():
    # P = 2/3, R = 1 -> 1.25 * (2/3) / (0.25 * (2/3) + 1) = 0.8333 / 1.1667 = 0.714285...
    assert f05_entity({"a", "b", "c"}, {"a", "c"}) == pytest.approx(0.7143, abs=5e-5)


def test_empty_empty_is_one():
    assert f05_entity(set(), set()) == 1.0


def test_empty_pred_nonempty_truth_is_zero():
    assert f05_entity(set(), {"a"}) == 0.0


def test_nonempty_pred_empty_truth_is_zero():
    assert f05_entity({"a"}, set()) == 0.0


def test_one_of_two_found_no_fp():
    # P = 1, R = 0.5 -> 1.25 * 0.5 / (0.25 + 0.5) = 0.625 / 0.75 = 0.8333...
    assert f05_entity({"a"}, {"a", "b"}) == pytest.approx(0.8333, abs=5e-5)


def test_two_found_plus_one_fp():
    # Same shape as the problem example: P = 2/3, R = 1.
    assert f05_entity({"a", "b", "x"}, {"a", "b"}) == pytest.approx(0.7143, abs=5e-5)


def test_disjoint_nonempty_is_zero():
    assert f05_entity({"x"}, {"a"}) == 0.0


def test_perfect_is_one():
    assert f05_entity({"a", "b"}, {"a", "b"}) == 1.0


def test_accepts_any_iterable():
    assert f05_entity(["a", "a"], frozenset({"a"})) == 1.0


# --- macro ------------------------------------------------------------------

def _mix():
    truth = {
        "s1": frozenset(),            # singleton, predicted empty      -> 1.0
        "s2": frozenset(),            # singleton, false positive       -> 0.0
        "s3": frozenset({"a", "b"}),  # one of two found                 -> 0.8333
        "s4": frozenset({"a", "c"}),  # two found + one FP               -> 0.7143
        "s5": frozenset({"d"}),       # missed everything                -> 0.0
        "s6": frozenset({"e"}),       # perfect                          -> 1.0
    }
    pred = {
        "s1": set(),
        "s2": {"z"},
        "s3": {"a"},
        "s4": {"a", "b", "c"},
        "s5": set(),
        "s6": {"e"},
    }
    return pred, truth


def test_macro_mix():
    pred, truth = _mix()
    r = macro_f05(pred, truth, list(truth))
    expected = (1.0 + 0.0 + 0.625 / 0.75 + (1.25 * 2 / 3) / (0.25 * 2 / 3 + 1) + 0.0 + 1.0) / 6
    assert r["macro"] == pytest.approx(expected)
    assert r["macro"] == pytest.approx(0.5913, abs=5e-5)
    assert r["n"] == 6
    assert r["n_singleton"] == 2 and r["n_nonsingleton"] == 4
    assert r["singleton_mean"] == pytest.approx(0.5)
    assert r["nonsingleton_mean"] == pytest.approx((0.625 / 0.75 + (1.25 * 2 / 3) / (0.25 * 2 / 3 + 1) + 0 + 1) / 4)
    assert r["n_singleton_fp"] == 1
    assert r["n_missed_all"] == 1
    assert r["n_perfect"] == 1        # among non-singletons: s6
    assert r["n_partial"] == 2        # s3, s4


def test_macro_missing_prediction_counts_as_empty_and_warns():
    truth = {"s1": frozenset(), "s2": frozenset({"a"})}
    with pytest.warns(UserWarning, match="missing"):
        r = macro_f05({}, truth, ["s1", "s2"])
    assert r["macro"] == pytest.approx(0.5)
    assert r["n_missed_all"] == 1


def test_macro_restricts_to_given_ids():
    pred, truth = _mix()
    r = macro_f05(pred, truth, ["s1", "s6"])
    assert r["macro"] == 1.0 and r["n"] == 2


def test_macro_all_empty_equals_singleton_fraction():
    _, truth = _mix()
    r = macro_f05({k: set() for k in truth}, truth, list(truth))
    assert r["macro"] == pytest.approx(2 / 6)


def test_macro_empty_id_list_raises():
    with pytest.raises(ValueError):
        macro_f05({}, {}, [])


def test_fast_macro_equals_reference():
    import numpy as np

    from ber.evaluate.metric import macro_f05_fast

    pred, truth = _mix()
    ps = [(s, m) for s, ms in pred.items() for m in ms]
    ts = [(s, m) for s, ms in truth.items() for m in ms]
    fast = macro_f05_fast([a for a, _ in ps], [b for _, b in ps], [a for a, _ in ts], [b for _, b in ts], list(truth))
    assert fast == pytest.approx(macro_f05(pred, truth, list(truth))["macro"])
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(60)]
    tr = {s: frozenset(f"m{j}" for j in rng.choice(8, rng.integers(0, 4), replace=False)) for s in ids}
    pr = {s: set(f"m{j}" for j in rng.choice(8, rng.integers(0, 4), replace=False)) for s in ids}
    ps = [(s, m) for s, ms in pr.items() for m in ms]
    ts = [(s, m) for s, ms in tr.items() for m in ms]
    assert macro_f05_fast([a for a, _ in ps], [b for _, b in ps], [a for a, _ in ts], [b for _, b in ts], ids) == \
        pytest.approx(macro_f05(pr, tr, ids)["macro"])
