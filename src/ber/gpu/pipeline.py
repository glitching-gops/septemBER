"""Stage-1 / stage-2 / decision building blocks on CUDA tensors (P4 counted runs).

``Dev`` holds the dev pair table as tensors (int entity ids, features, labels, folds). ``run_arm``
is P3's ``run_verifier_integrate.run_arm`` protocol on the GPU: 5 component-grouped folds on the
non-holdout rows; fold k's tau is searched on the OOF p of the OTHER folds (exclusivity over those
rows); the S1 of fold k are decided with exclusivity over fold k's rows; the holdout uses the mean
of the fold models, exclusivity over holdout rows and the median fold tau.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ber.gpu import boot, search
from ber.gpu import decide as gd
from ber.gpu import metric as gm
from ber.gpu import segment as sg
from ber.gpu.xgb import fit_calibrated, predict_mean

BUCKETS = ["correct", "missed all", "under-predicted", "over-predicted", "wrong entity (FP and FN)",
           "singleton false positive"]


@dataclass
class Dev:
    s1: torch.Tensor
    m: torch.Tensor
    src: torch.Tensor
    y: torch.Tensor
    fold: torch.Tensor           # xgb fold per pair
    hold: torch.Tensor           # holdout pair
    country: torch.Tensor        # per entity
    ntrue: torch.Tensor          # per entity (ground truth counts)
    cv_ids: torch.Tensor         # non-holdout dev S1
    hold_ids: torch.Tensor
    ids_by_fold: dict
    n_ent: int
    labels: list = field(default_factory=list)


def verifier_features(d_s1, d_m, sc_s1, sc_m, logit, p_cal, rescued, n_ent: int) -> torch.Tensor:
    """[n, 6] = v_logit, v_p, v_inband (scored), v_rank, v_margin, v_rescued; NaN outside the scored set.

    Rank / margin are among the S1's scored pairs by logit (ties: smaller record index)."""
    rank = sg.rank_in_group(sc_s1, logit.to(torch.float64), sc_m).to(torch.float32)
    mx = sg.seg_max(logit.to(torch.float32), sc_s1, n_ent)
    margin = logit.to(torch.float32) - mx[sc_s1]
    j = sg.lookup_pairs(d_s1, d_m, sc_s1, sc_m, n_ent)
    hit = j >= 0
    js = j.clamp(min=0)
    nan = torch.full(d_s1.shape, float("nan"), dtype=torch.float32, device=d_s1.device)
    cols = [torch.where(hit, logit.to(torch.float32)[js], nan), torch.where(hit, p_cal.to(torch.float32)[js], nan),
            hit.to(torch.float32), torch.where(hit, rank[js], nan), torch.where(hit, margin[js], nan),
            torch.where(hit, rescued.to(torch.float32)[js], torch.zeros_like(nan))]
    return torch.stack(cols, 1)


V_NAMES = ["v_logit", "v_p", "v_inband", "v_rank", "v_margin", "v_rescued"]


def select_cv(D: Dev, p, taus: dict, rule: str = "threshold") -> torch.Tensor:
    sel = torch.zeros_like(D.y)
    for k in sorted(D.ids_by_fold):
        rows = (~D.hold) & (D.fold == k)
        keep = gd.exclusive(D.s1, D.m, p, rows, D.n_ent)
        sel |= gd.threshold(keep, p, taus[k]) if rule == "threshold" else gd.expected_f(D.s1, p, keep)
    return sel


def select_hold(D: Dev, p, tau: float, rule: str = "threshold") -> torch.Tensor:
    keep = gd.exclusive(D.s1, D.m, p, D.hold, D.n_ent)
    return gd.threshold(keep, p, tau) if rule == "threshold" else gd.expected_f(D.s1, p, keep)


def run_arm(D: Dev, X: torch.Tensor, seed: int, cfg_xgb: dict | None, taus_grid: torch.Tensor, log=print) -> dict:
    cv = ~D.hold
    oof = torch.zeros(D.y.numel(), dtype=torch.float32, device=X.device)
    models = {}
    for k in sorted(D.ids_by_fold):
        tr, te = cv & (D.fold != k), cv & (D.fold == k)
        models[k] = fit_calibrated(X[tr], D.y[tr], D.s1[tr], seed=seed, cfg_xgb=cfg_xgb)
        oof[te] = models[k].predict(X[te])
        log(f"  fold {k} fitted")
    taus = {}
    for k in sorted(D.ids_by_fold):
        tr = cv & (D.fold != k)
        keep = gd.exclusive(D.s1, D.m, oof, tr, D.n_ent)
        ids = torch.cat([D.ids_by_fold[j] for j in sorted(D.ids_by_fold) if j != k])
        taus[k], _ = search.best_tau(D.s1, D.y, oof, keep, ids, D.ntrue, taus_grid)
    f_thr = gm.per_s1_f(select_cv(D, oof, taus), D.s1, D.y, D.ntrue)
    f_ef = gm.per_s1_f(select_cv(D, oof, taus, "expected_f"), D.s1, D.y, D.ntrue)
    m_thr, m_ef = gm.macro(f_thr, D.cv_ids), gm.macro(f_ef, D.cv_ids)
    winner = "threshold" if m_thr > m_ef else "expected_f"
    tau_med = float(torch.quantile(torch.tensor(list(taus.values()), dtype=torch.float64), 0.5))
    p_hold = predict_mean(list(models.values()), X[D.hold])
    ph = torch.zeros_like(oof)
    ph[D.hold] = p_hold
    f_hold = gm.per_s1_f(select_hold(D, ph, tau_med, winner), D.s1, D.y, D.ntrue)
    f_cv = f_thr if winner == "threshold" else f_ef
    log(f"  CV thr {m_thr:.6f} EF {m_ef:.6f} -> {winner}; holdout {gm.macro(f_hold, D.hold_ids):.6f}")
    return {"models": models, "oof": oof, "p_all": torch.where(D.hold, ph, oof), "taus": taus, "tau_median": tau_med,
            "winner": winner, "f_cv": f_cv, "f_thr": f_thr, "f_ef": f_ef, "f_hold": f_hold}


def paired(f_new, f_ref, ids, seed: int = 0) -> tuple[float, list]:
    d = (f_new[ids] - f_ref[ids]).to(torch.float64)
    lo, hi = boot.ci(d, seed=seed)
    return 100 * float(d.mean()), [100 * lo, 100 * hi]


def buckets(D: Dev, sel, f, ids) -> dict:
    """Counts and macro points lost per P1-audit bucket over ``ids`` (from per-S1 counts)."""
    n = D.n_ent
    tp = sg.seg_count(D.s1[sel], n, D.y[sel])
    npred = sg.seg_count(D.s1[sel], n)
    nt = D.ntrue
    fp, fn = npred - tp, nt - tp
    code = torch.where(nt == 0, torch.where(npred > 0, 5, 0),
                       torch.where(tp == 0, 1, torch.where((fp == 0) & (fn == 0), 0,
                                   torch.where((fp > 0) & (fn > 0), 4, torch.where(fp > 0, 3, 2)))))
    c = code[ids]
    cnt = torch.bincount(c, minlength=6)
    loss = torch.zeros(6, dtype=torch.float64, device=c.device).index_add_(0, c, 1 - f[ids].to(torch.float64))
    return {BUCKETS[i]: (int(cnt[i]), float(loss[i]) / ids.numel()) for i in range(6)}


def loco_plain(D: Dev, X, cser_pair, tau: float, seed: int, cfg_xgb) -> dict:
    """Stage-1 LOCO: fit on all CV pairs of country A, decide B's S1 with ``tau``."""
    cv = ~D.hold
    out = {}
    codes = torch.unique(cser_pair[cv]).tolist()
    for a in codes:
        for b in codes:
            if a == b:
                continue
            ra, rb = cv & (cser_pair == a), cv & (cser_pair == b)
            mdl = fit_calibrated(X[ra], D.y[ra], D.s1[ra], seed=seed, cfg_xgb=cfg_xgb)
            p = torch.zeros(D.y.numel(), dtype=torch.float32, device=X.device)
            p[rb] = mdl.predict(X[rb])
            sel = gd.threshold(gd.exclusive(D.s1, D.m, p, rb, D.n_ent), p, tau)
            ids_b = D.cv_ids[D.country[D.cv_ids] == b]
            out[f"{D.labels[a]}->{D.labels[b]}"] = gm.macro(gm.per_s1_f(sel, D.s1, D.y, D.ntrue), ids_b)
    return out


def oof_within(D: Dev, X, rows, seed, cfg_xgb, X_other_rows=None) -> tuple[torch.Tensor, list]:
    """5-fold OOF p within ``rows`` (by xgb fold); fold models also score ``X_other_rows``."""
    p = torch.zeros(D.y.numel(), dtype=torch.float32, device=X.device)
    models = []
    for k in sorted(D.ids_by_fold):
        tr, te = rows & (D.fold != k), rows & (D.fold == k)
        mdl = fit_calibrated(X[tr], D.y[tr], D.s1[tr], seed=seed, cfg_xgb=cfg_xgb)
        p[te] = mdl.predict(X[te])
        models.append(mdl)
    if X_other_rows is not None:
        p[X_other_rows] = predict_mean(models, X[X_other_rows])
    return p, models
