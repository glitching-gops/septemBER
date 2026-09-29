"""GPU lexical blockers (final push): char-3-gram MinHash + LSH (LX1) and a rare-token index (LX2).

LX1: 3-grams of the zero-padded name bytes (unfold), 64 MinHash functions (a*x + b mod p, fixed
seed), 16 bands x 4 rows (collision probability ~0.5 at Jaccard 0.5). Query and target band keys are
joined by sort + searchsorted; buckets larger than ``cap`` are skipped (uninformative, e.g. very
common names). Score = fraction of equal MinHash values (Jaccard estimate); top-k per query.

LX2: postings of (token, entity). For each query, its ``n_name`` rarest name tokens and its rarest
address token (df <= cap) retrieve every target sharing the token; score = sum of the IDF of the
query's selected tokens that the pair shares; top-k per query. Address tokens can be required to
come with an equal house number (``hn``).

Queries / targets are entity-index tensors; every partition filter (country, target source) is the
caller's. Deterministic: fixed hash seeds, stable sorts, ties by smaller target index.
"""
from __future__ import annotations

import torch

from ber.gpu import segment as sg

P_MOD = (1 << 31) - 1


def minhash(nb: torch.Tensor, n_hash: int = 64, seed: int = 20260927, chunk: int = 20_000) -> torch.Tensor:
    """[N, L] uint8 name bytes -> [N, n_hash] int32 MinHash signature of the char 3-gram set."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(1, P_MOD, (n_hash,), generator=g).to(nb.device)
    b = torch.randint(0, P_MOD, (n_hash,), generator=g).to(nb.device)
    out = torch.empty((nb.shape[0], n_hash), dtype=torch.int32, device=nb.device)
    for s in range(0, nb.shape[0], chunk):
        x = nb[s:s + chunk].to(torch.int64)
        g3 = x[:, :-2] * 65536 + x[:, 1:-1] * 256 + x[:, 2:]          # [n, L-2] 3-gram codes
        valid = (x[:, :-2] != 0) & (x[:, 1:-1] != 0) & (x[:, 2:] != 0)
        h = (g3.unsqueeze(2) * a + b) % P_MOD                        # [n, L-2, n_hash]
        h = torch.where(valid.unsqueeze(2), h, torch.full_like(h, P_MOD))
        out[s:s + chunk] = h.amin(1).to(torch.int32)
    return out


def _band_keys(sig: torch.Tensor, bands: int, rows: int) -> torch.Tensor:
    """[N, bands] int64 key per band (polynomial hash of its rows, band index mixed in)."""
    s = sig[:, :bands * rows].reshape(sig.shape[0], bands, rows).to(torch.int64)
    k = torch.zeros(sig.shape[0], bands, dtype=torch.int64, device=sig.device)
    for r in range(rows):
        k = (k * 1_000_003 + s[:, :, r]) % P_MOD  # < 2^31, so k * 1e6 stays inside int64
    return k * 64 + torch.arange(bands, device=sig.device)


def _join(qkey: torch.Tensor, qrow: torch.Tensor, tkey: torch.Tensor, trow: torch.Tensor, cap: int):
    """All (qrow, trow) with equal keys, skipping target buckets larger than ``cap``."""
    tk, o = torch.sort(tkey, stable=True)
    tr = trow[o]
    lo = torch.searchsorted(tk, qkey)
    hi = torch.searchsorted(tk, qkey, right=True)
    cnt = hi - lo
    ok = (cnt > 0) & (cnt <= cap)
    lo, cnt, qr = lo[ok], cnt[ok], qrow[ok]
    rep = torch.repeat_interleave(torch.arange(qr.numel(), device=qr.device), cnt)
    off = torch.arange(rep.numel(), device=qr.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    return qr[rep], tr[lo[rep] + off]


def lsh_candidates(q_ent, q_sig, t_ent, t_sig, k: int, bands: int = 16, rows: int = 4, cap: int = 200,
                   q_chunk: int = 100_000, n_ent: int | None = None):
    """(s1, m, jaccard_estimate, rank) top-k per query among LSH-colliding targets."""
    tb = _band_keys(t_sig, bands, rows)
    tkey = tb.reshape(-1)
    trow = torch.arange(t_ent.numel(), device=t_ent.device).repeat_interleave(bands)
    o1, o2, o3 = [], [], []
    for s in range(0, q_ent.numel(), q_chunk):
        qs = q_sig[s:s + q_chunk]
        qb = _band_keys(qs, bands, rows).reshape(-1)
        qrow = torch.arange(qs.shape[0], device=qs.device).repeat_interleave(bands)
        qi, ti = _join(qb, qrow, tkey, trow, cap)
        if qi.numel() == 0:
            continue
        key = torch.unique(qi * t_ent.numel() + ti)
        qi, ti = key // t_ent.numel(), key % t_ent.numel()
        est = (qs[qi] == t_sig[ti]).float().mean(1)
        s1, m = q_ent[s + qi], t_ent[ti]
        keep = sg.rank_in_group(s1, est.to(torch.float64), m) <= k
        o1.append(s1[keep]); o2.append(m[keep]); o3.append(est[keep])
    if not o1:
        z = torch.zeros(0, dtype=torch.int64, device=q_ent.device)
        return z, z, z.float(), z
    s1, m, sc = torch.cat(o1), torch.cat(o2), torch.cat(o3)
    return s1, m, sc, sg.rank_in_group(s1, sc.to(torch.float64), m)


def rare_token_candidates(q_ent, t_ent, tok_ent, tok_id, n_ent: int, k: int, n_rare: int = 2, cap: int = 200,
                          hn=None, q_chunk: int = 200_000):
    """(s1, m, idf_score, rank): targets sharing one of the query's ``n_rare`` rarest tokens (df <= cap
    among the targets). ``hn`` (per-entity int64, 0 = missing): if given, a pair also needs equal
    house numbers (address blocker)."""
    dev = q_ent.device
    is_t = torch.zeros(n_ent, dtype=torch.bool, device=dev)
    is_t[t_ent] = True
    is_q = torch.zeros(n_ent, dtype=torch.bool, device=dev)
    is_q[q_ent] = True
    tt = is_t[tok_ent]
    V = int(tok_id.max()) + 1 if tok_id.numel() else 1
    # unique (token, entity) postings among targets
    tkey = torch.unique(tok_id[tt].to(torch.int64) * n_ent + tok_ent[tt])
    t_tok, t_e = tkey // n_ent, tkey % n_ent
    df = torch.bincount(t_tok, minlength=V)
    idf = torch.log((1.0 + t_e.unique().numel()) / (1.0 + df.double())) + 1.0
    # query tokens: unique (entity, token), usable if 0 < df <= cap; pick the n_rare rarest per query
    qq = is_q[tok_ent]
    qkey = torch.unique(tok_ent[qq] * V + tok_id[qq].to(torch.int64))
    q_e, q_tok = qkey // V, qkey % V
    use = (df[q_tok] > 0) & (df[q_tok] <= cap)
    q_e, q_tok = q_e[use], q_tok[use]
    r = sg.rank_in_group(q_e, -df[q_tok].double(), q_tok)
    sel = r <= n_rare
    q_e, q_tok = q_e[sel], q_tok[sel]
    o1, o2, o3 = [], [], []
    for s in range(0, q_e.numel(), q_chunk):
        qe, qt = q_e[s:s + q_chunk], q_tok[s:s + q_chunk]
        qi, ti = _join(qt, torch.arange(qt.numel(), device=dev), t_tok, torch.arange(t_tok.numel(), device=dev), cap)
        s1, m, w = qe[qi], t_e[ti], idf[qt[qi]]
        if hn is not None:
            ok = (hn[s1] != 0) & (hn[s1] == hn[m])
            s1, m, w = s1[ok], m[ok], w[ok]
        o1.append(s1); o2.append(m); o3.append(w)
    if not o1:
        z = torch.zeros(0, dtype=torch.int64, device=dev)
        return z, z, z.double(), z
    s1, m, w = torch.cat(o1), torch.cat(o2), torch.cat(o3)
    key = s1 * n_ent + m
    uk, inv = torch.unique(key, return_inverse=True)
    score = torch.zeros(uk.numel(), dtype=torch.float64, device=dev).index_add_(0, inv, w)
    s1, m = uk // n_ent, uk % n_ent
    rk = sg.rank_in_group(s1, score, m)
    keep = rk <= k
    return s1[keep], m[keep], score[keep], rk[keep]


def char_grams(nb: torch.Tensor, ns=(3, 4), n_buckets: int = 1 << 22, chunk: int = 200_000):
    """char_wb n-grams of zero-padded byte strings (each word padded by one space, like sklearn's
    ``char_wb``: an n-gram may start / end with a space but never contains an inner space).
    Returns (row, gram id in [0, n_buckets)) for every n-gram occurrence (duplicates kept = tf)."""
    rows, gids = [], []
    for s in range(0, nb.shape[0], chunk):
        x = nb[s:s + chunk].to(torch.int64)
        n_, L = x.shape
        sp = torch.full((n_, 1), 32, dtype=torch.int64, device=x.device)
        x = torch.cat([sp, torch.where(x == 0, torch.full_like(x, 0), x), sp], 1)  # leading / trailing space
        # the byte after the last real char becomes a space so the final word is closed
        last = (nb[s:s + chunk] != 0).sum(1).to(torch.int64) + 1  # first position after the text
        x[torch.arange(n_, device=x.device), last.clamp(max=L + 1)] = 32
        for n in sorted(ns):
            win = x.unfold(1, n, 1)                                   # [n_, L+2-n+1, n]
            ok = (win != 0).all(2)
            inner = (win[:, :, 1:-1] == 32).any(2) if n > 2 else torch.zeros_like(ok)
            allsp = (win == 32).all(2)
            ok = ok & ~inner & ~allsp
            code = torch.zeros(win.shape[:2], dtype=torch.int64, device=x.device)
            for j in range(n):
                code = code * 257 + win[:, :, j]
            code = (code * 2654435761 + n) % n_buckets
            r = torch.arange(n_, device=x.device).unsqueeze(1).expand_as(code)
            rows.append(r[ok] + s); gids.append(code[ok])
    return torch.cat(rows), torch.cat(gids)


def tfidf_candidates(q_ent, q_rows, q_gid, t_ent, t_rows, t_gid, k: int, max_df: float = 0.02,
                     q_top: int = 12, post_cap: int = 4000, q_chunk: int = 20_000, n_ent: int | None = None,
                     max_entries: int = 120_000_000):
    """Char-n-gram TF-IDF cosine top-k (P1's B1 weighting: sublinear tf, idf = ln((1+n)/(1+df)) + 1,
    grams with df > max_df * n dropped, L2 norm), scored through an inverted index over each query's
    ``q_top`` highest-weight grams whose target df <= ``post_cap`` (partial cosine over those grams).
    ``*_rows`` index into ``*_ent``; returns (s1, m, score, rank)."""
    dev = q_ent.device
    nt = t_ent.numel()
    # target tf-idf
    tk, tcnt = torch.unique(t_rows * (1 << 22) + t_gid, return_counts=True)
    tr, tg = tk >> 22, tk & ((1 << 22) - 1)
    df = torch.bincount(tg, minlength=1 << 22)
    maxdf = max(int(max_df * nt), min(nt, 50))
    idf = torch.log((1.0 + nt) / (1.0 + df.double())) + 1.0
    idf = torch.where(df > maxdf, torch.zeros_like(idf), idf)
    tw = (1.0 + torch.log(tcnt.double())) * idf[tg]
    tn = torch.zeros(nt, dtype=torch.float64, device=dev).index_add_(0, tr, tw * tw).sqrt().clamp(min=1e-12)
    tw = tw / tn[tr]
    ts, to = torch.sort(tg, stable=True)           # postings sorted by gram
    p_row, p_w = tr[to], tw[to]
    # query tf-idf with the target idf
    qk, qcnt = torch.unique(q_rows * (1 << 22) + q_gid, return_counts=True)
    qr, qg = qk >> 22, qk & ((1 << 22) - 1)
    qw = (1.0 + torch.log(qcnt.double())) * idf[qg]
    nq = q_ent.numel()
    qn = torch.zeros(nq, dtype=torch.float64, device=dev).index_add_(0, qr, qw * qw).sqrt().clamp(min=1e-12)
    qw = qw / qn[qr]
    use = (qw > 0) & (df[qg] <= post_cap)
    qr, qg, qw = qr[use], qg[use], qw[use]
    rk = sg.rank_in_group(qr, qw, qg)
    sel = rk <= q_top
    qr, qg, qw = qr[sel], qg[sel], qw[sel]
    o1, o2, o3 = [], [], []
    order = torch.argsort(qr, stable=True)
    qr, qg, qw = qr[order], qg[order], qw[order]
    # chunk boundaries on query boundaries, each chunk expanding to <= max_entries postings
    plen = torch.searchsorted(ts, qg, right=True) - torch.searchsorted(ts, qg)
    per_q = torch.zeros(nq, dtype=torch.int64, device=dev).index_add_(0, qr, plen)
    cq = torch.cumsum(per_q, 0)
    targets = torch.arange(0, int(cq[-1]) + max_entries, max_entries, device=dev) if nq else cq[:0]
    qcut = torch.unique(torch.cat([torch.zeros(1, dtype=torch.int64, device=dev),
                                   torch.searchsorted(cq, targets, right=True).clamp(max=nq),
                                   torch.tensor([nq], device=dev)]))
    bounds = torch.searchsorted(qr, qcut)
    for i in range(bounds.numel() - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if a == b:
            continue
        cr, cg, cw = qr[a:b], qg[a:b], qw[a:b]
        lo = torch.searchsorted(ts, cg)
        hi = torch.searchsorted(ts, cg, right=True)
        cnt = hi - lo
        rep = torch.repeat_interleave(torch.arange(cg.numel(), device=dev), cnt)
        off = torch.arange(rep.numel(), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        pi = lo[rep] + off
        key = cr[rep] * nt + p_row[pi]
        val = cw[rep] * p_w[pi]
        uk, inv = torch.unique(key, return_inverse=True)
        sc = torch.zeros(uk.numel(), dtype=torch.float64, device=dev).index_add_(0, inv, val)
        qq, tt = uk // nt, uk % nt
        keep = sg.rank_in_group(qq, sc, tt) <= k
        o1.append(q_ent[qq[keep]]); o2.append(t_ent[tt[keep]]); o3.append(sc[keep])
    if not o1:
        z = torch.zeros(0, dtype=torch.int64, device=dev)
        return z, z, z.double(), z
    s1, m, sc = torch.cat(o1), torch.cat(o2), torch.cat(o3)
    return s1, m, sc, sg.rank_in_group(s1, sc, m)
