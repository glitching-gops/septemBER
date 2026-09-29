"""Decision layer: exact expected F0.5 vs brute force, exclusivity, empty-set choice."""
import itertools

import numpy as np
import pandas as pd
import pytest

from ber.decide.expected_f import best_prefix, expected_f_prefixes, poisson_binomial
from ber.decide.sets import decide_expected_f, decide_threshold, exclusive, to_sets
from ber.evaluate.metric import f05_entity


def brute_force(p):
    """E[F0.5] of each prefix by enumerating all 2^n truth outcomes."""
    n = len(p)
    out = np.zeros(n + 1)
    for outcome in itertools.product([0, 1], repeat=n):
        prob = np.prod([pi if o else 1 - pi for pi, o in zip(p, outcome)])
        truth = {i for i, o in enumerate(outcome) if o}
        for k in range(n + 1):
            out[k] += prob * f05_entity(set(range(k)), truth)
    return out


@pytest.mark.parametrize("seed", range(12))
def test_expected_f_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 9))
    p = np.sort(rng.random(n))[::-1]
    np.testing.assert_allclose(expected_f_prefixes(p), brute_force(p), atol=1e-12)


def test_poisson_binomial_sums_to_one():
    d = poisson_binomial(np.array([0.1, 0.5, 0.9]))
    assert d.sum() == pytest.approx(1.0) and d[0] == pytest.approx(0.9 * 0.5 * 0.1)


def test_empty_set_chosen_for_low_p_singleton():
    k, ef = best_prefix(np.array([0.1, 0.05, 0.02]))
    assert k == 0 and ef == pytest.approx(0.9 * 0.95 * 0.98)


def test_confident_matches_all_taken():
    k, _ = best_prefix(np.array([0.99, 0.98, 0.97, 0.01]))
    assert k == 3


def test_tail_beyond_max_n_still_counts_in_expectation():
    p = np.array([0.9, 0.9, 0.9])
    k_full, ef_full = best_prefix(p, max_n=3)
    k_cut, ef_cut = best_prefix(p, max_n=2)
    assert k_cut == 2 and ef_cut < ef_full


def _pairs():
    return pd.DataFrame({
        "s1_id": ["A", "A", "B", "B", "C"],
        "m_id": ["x", "y", "x", "z", "w"],
        "source": ["S2", "S3", "S2", "S2", "S3"],
        "p": [0.9, 0.2, 0.6, 0.95, 0.05],
    })


def test_exclusivity_keeps_highest_p_owner():
    ex = exclusive(_pairs())
    assert set(zip(ex["s1_id"], ex["m_id"])) == {("A", "x"), ("A", "y"), ("B", "z"), ("C", "w")}


def test_threshold_and_cap():
    sel = decide_threshold(_pairs(), 0.5)
    assert set(zip(sel["s1_id"], sel["m_id"])) == {("A", "x"), ("B", "x"), ("B", "z")}
    capped = decide_threshold(_pairs(), 0.5, per_source_cap=1)
    assert set(zip(capped["s1_id"], capped["m_id"])) == {("A", "x"), ("B", "z")}


def test_expected_f_decision_sets():
    sel = decide_expected_f(exclusive(_pairs()), processes=1)
    sets = to_sets(sel, ["A", "B", "C"])
    assert sets["A"] == {"x"} and sets["B"] == {"z"} and sets["C"] == set()
