"""TF-IDF cosine top-k blockers (B1 name, B2 address, B3 name+address).

Vectorisers are fitted on the TARGET partition (one country x one source), so IDF
reflects the distractor pool the query is searched against. Query and target
matrices are float32 CSR with L2-normalised rows; top-k cosine comes from
``sparse_dot_topn.sp_matmul_topn`` (Apache-2.0), in query chunks.

``max_df`` prunes n-grams present in more than that fraction of target records:
such posting lists dominate the sparse-product cost while their IDF weight is
near zero. It is a tuned parameter (see reports/p1_blocking.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from ber import compute


@dataclass
class TfidfSpec:
    """One or more vectoriser configs whose outputs are h-stacked and re-normalised."""

    parts: tuple[dict, ...]
    max_df: float = 0.05
    min_df: int = 1


SPECS = {
    # B1: character n-grams inside word boundaries, robust to typos/transliteration noise
    "b1_name": TfidfSpec(parts=({"analyzer": "char_wb", "ngram_range": (3, 4)},), max_df=0.02),
    # B2: address words (1-2 grams) + character 3-grams
    "b2_addr": TfidfSpec(
        parts=({"analyzer": "word", "ngram_range": (1, 2), "token_pattern": r"(?u)\b\w+\b"},
               {"analyzer": "char_wb", "ngram_range": (3, 3)}),
        max_df=0.02,
    ),
    # B3: name + address words, IDF-weighted
    "b3_both": TfidfSpec(parts=({"analyzer": "word", "ngram_range": (1, 1), "token_pattern": r"(?u)\b\w+\b"},), max_df=0.05),
}


N_FEATURES = 2 ** 22


def _hash_chunk(args) -> sp.csr_matrix:
    texts, part = args
    hv = HashingVectorizer(lowercase=False, n_features=N_FEATURES, alternate_sign=False, norm=None,
                           dtype=np.float32, **part)
    return hv.transform(texts).tocsr()


def _hash_counts(texts: list[str], part: dict, chunk: int = 50_000) -> sp.csr_matrix:
    """Term counts via the stateless HashingVectorizer, parallel over text chunks."""
    jobs = [(texts[i:i + chunk], part) for i in range(0, len(texts), chunk)]
    if len(jobs) <= 1:
        mats = [_hash_chunk(j) for j in jobs] or [sp.csr_matrix((0, N_FEATURES), dtype=np.float32)]
    else:
        with compute.pool(min(compute.n_jobs(), len(jobs))) as p:
            mats = p.map(_hash_chunk, jobs, chunksize=1)
    return sp.vstack(mats, format="csr")


class TfidfIndex:
    """Fitted on one target partition; supports forward (query->target) and reverse
    (target->query) top-k in the same vector space.

    Implementation: hashed n-gram counts (2^22 buckets per part, parallel over
    chunks), sublinear tf (1 + log tf), smooth IDF from TARGET document frequencies,
    n-grams with df above ``max_df`` dropped (idf set to 0), parts h-stacked, rows
    L2-normalised. Equivalent to TfidfVectorizer(sublinear_tf=True) up to rare hash
    collisions, but ~n_jobs times faster to fit on millions of records.
    """

    def __init__(self, spec: TfidfSpec):
        self.spec = spec
        self.idfs: list[np.ndarray] = []
        self.T: sp.csr_matrix | None = None      # targets x features
        self.T_t: sp.csr_matrix | None = None    # features x targets

    @property
    def n_targets(self) -> int:
        return 0 if self.T is None else self.T.shape[0]

    @staticmethod
    def _weight(X: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
        X = X.copy()
        X.data = (1.0 + np.log(X.data)) * idf[X.indices]
        X.eliminate_zeros()
        return X

    def fit(self, target_texts: list[str]) -> "TfidfIndex":
        n = len(target_texts)
        # fraction of targets, with an absolute floor so tiny partitions keep their vocabulary
        max_df = max(int(self.spec.max_df * n), min(n, 50))
        mats = []
        self.idfs = []
        for part in self.spec.parts:
            C = _hash_counts(target_texts, part)
            df = np.bincount(C.indices, minlength=N_FEATURES)
            idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
            idf[(df > max_df) | (df < self.spec.min_df)] = 0.0
            self.idfs.append(idf)
            mats.append(self._weight(C, idf))
        self.T = normalize(sp.hstack(mats, format="csr").astype(np.float32))
        self.T_t = self.T.T.tocsr()
        return self

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        mats = [self._weight(_hash_counts(texts, part), idf) for part, idf in zip(self.spec.parts, self.idfs)]
        return normalize(sp.hstack(mats, format="csr").astype(np.float32))

    @staticmethod
    def _topk(A: sp.csr_matrix, B_t: sp.csr_matrix, k: int, threshold: float, chunk: int):
        qi, ti, sc = [], [], []
        k = max(1, min(k, B_t.shape[1]))
        for start in range(0, A.shape[0], chunk):
            C = sp_matmul_topn(A[start:start + chunk], B_t, top_n=k, threshold=threshold, sort=True,
                               n_threads=compute.n_jobs()).tocoo()
            qi.append(C.row.astype(np.int64) + start)
            ti.append(C.col.astype(np.int64))
            sc.append(C.data.astype(np.float32))
        if not qi:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
        q, t, s = np.concatenate(qi), np.concatenate(ti), np.concatenate(sc)
        order = np.lexsort((t, -s, q))  # by row, then score desc, then id for determinism
        return q[order], t[order], s[order]

    def forward(self, Q: sp.csr_matrix, k: int, threshold: float = 0.0, chunk: int = 50_000):
        """(query_idx, target_idx, cosine) with each query's hits sorted by score."""
        return self._topk(Q, self.T_t, k, threshold, chunk)

    def reverse(self, Q: sp.csr_matrix, k: int, threshold: float = 0.0, chunk: int = 200_000):
        """(target_idx, query_idx, cosine): each target's top-k queries."""
        return self._topk(self.T, Q.T.tocsr(), k, threshold, chunk)


def ranks_within_query(q: np.ndarray) -> np.ndarray:
    """1-based rank of each row within its (already score-sorted) query group."""
    if len(q) == 0:
        return np.empty(0, np.int16)
    starts = np.r_[0, np.flatnonzero(np.diff(q)) + 1]
    counts = np.diff(np.r_[starts, len(q)])
    return (np.arange(len(q)) - np.repeat(starts, counts) + 1).astype(np.int16)
