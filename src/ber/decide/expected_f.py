"""Expected-F0.5 set selection.

For one S1 entity with candidate match probabilities p_1 >= ... >= p_n (treated as
independent Bernoulli events), evaluate every prefix {1..k}, k = 0..n, including the
EMPTY set, by its exact expected F0.5 and return the best k.

With A = true matches inside the prefix and B = true matches outside it,
    F0.5 = 1.25 A / (0.25 (A + B) + k)      for k > 0
    F0.5 = 1{A + B = 0}                      for k = 0 (empty prediction)
A and B are independent Poisson-binomial variables, so
    E[F] = sum_a sum_b P(A = a) P(B = b) F(a, b, k).
True matches that blocking never retrieved are not modelled (unknowable here).
"""
from __future__ import annotations

import numpy as np

BETA2 = 0.25


def poisson_binomial(p: np.ndarray) -> np.ndarray:
    """P(sum of independent Bernoulli(p_i) = j), j = 0..len(p)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for i, pi in enumerate(p):
        dist[1:i + 2] = dist[1:i + 2] * (1 - pi) + dist[0:i + 1] * pi
        dist[0] *= 1 - pi
    return dist


def expected_f_prefixes(p_sorted: np.ndarray) -> np.ndarray:
    """E[F0.5] of each prefix k = 0..n of the (descending) probabilities."""
    p = np.asarray(p_sorted, dtype=np.float64)
    n = len(p)
    # suffix distributions P(B = b) for the items k..n-1
    suffix = [None] * (n + 1)
    suffix[n] = np.array([1.0])
    for k in range(n - 1, -1, -1):
        prev = suffix[k + 1]
        cur = np.zeros(len(prev) + 1)
        cur[:-1] += prev * (1 - p[k])
        cur[1:] += prev * p[k]
        suffix[k] = cur
    out = np.empty(n + 1)
    out[0] = suffix[0][0]  # empty prediction is right only if nothing is a match
    pin = np.array([1.0])
    for k in range(1, n + 1):
        nxt = np.zeros(len(pin) + 1)
        nxt[:-1] += pin * (1 - p[k - 1])
        nxt[1:] += pin * p[k - 1]
        pin = nxt
        a = np.arange(len(pin))[:, None]
        b = np.arange(len(suffix[k]))[None, :]
        F = (1 + BETA2) * a / (BETA2 * (a + b) + k)
        out[k] = pin @ F @ suffix[k]
    return out


def best_prefix(p: np.ndarray, max_n: int = 40, min_p: float = 1e-4) -> tuple[int, float]:
    """(k, expected F0.5) of the best prefix after sorting ``p`` in descending order.

    Candidates below ``min_p`` or beyond ``max_n`` are treated as outside the prefix
    for B only (they cannot be selected); their mass still enters the expectation.
    """
    p = np.sort(np.asarray(p, dtype=np.float64))[::-1]
    sel = p[:max_n]
    sel = sel[sel >= min_p]
    rest = p[len(sel):]
    if len(rest):
        # fold the unselectable tail into the suffix distribution
        ef = _expected_with_tail(sel, rest)
    else:
        ef = expected_f_prefixes(sel)
    k = int(np.argmax(ef))
    return k, float(ef[k])


def _expected_with_tail(sel: np.ndarray, rest: np.ndarray) -> np.ndarray:
    tail = poisson_binomial(rest)
    n = len(sel)
    out = np.empty(n + 1)
    # suffix distributions of sel[k:] convolved with the tail
    suffix = [None] * (n + 1)
    suffix[n] = tail
    for k in range(n - 1, -1, -1):
        prev = suffix[k + 1]
        cur = np.zeros(len(prev) + 1)
        cur[:-1] += prev * (1 - sel[k])
        cur[1:] += prev * sel[k]
        suffix[k] = cur
    out[0] = suffix[0][0]
    pin = np.array([1.0])
    for k in range(1, n + 1):
        nxt = np.zeros(len(pin) + 1)
        nxt[:-1] += pin * (1 - sel[k - 1])
        nxt[1:] += pin * sel[k - 1]
        pin = nxt
        a = np.arange(len(pin))[:, None]
        b = np.arange(len(suffix[k]))[None, :]
        out[k] = pin @ ((1 + BETA2) * a / (BETA2 * (a + b) + k)) @ suffix[k]
    return out
