"""XGBoost on device arrays + GPU isotonic calibration (item 11; replaces ``ber.match.gbdt`` in P4).

Features stay on the GPU: torch tensor -> cupy view (DLPack, zero copy) -> ``QuantileDMatrix``;
predictions come back as a cupy array -> torch (DLPack). Calibration contract of E1b / sub3:
the booster is fitted on a grouped 75% of the training rows and an exact isotonic map on the
other 25%; the split is a fixed hash of the S1 group index (deterministic, no sklearn).
"""
from __future__ import annotations

import torch

import time

from ber.gpu.calib import BinnedIsotonic, Isotonic

# host time of the XGBoost driver (the thread that launches the GPU tree kernels), reported
# separately by StageTimer: it is kernel launching, not host computation on data
DRIVER = {"wall": 0.0, "cpu": 0.0}


class _Driver:
    def __enter__(self):
        self.w, self.c = time.time(), time.process_time()

    def __exit__(self, *exc):
        DRIVER["wall"] += time.time() - self.w
        DRIVER["cpu"] += time.process_time() - self.c

DEFAULTS = dict(objective="binary:logistic", device="cuda", tree_method="hist", n_estimators=600, max_depth=8,
                learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, max_bin=256,
                eval_metric="aucpr")


def _cp(t: torch.Tensor):
    import cupy
    return cupy.from_dlpack(t.contiguous())


def _params(cfg_xgb: dict | None, seed: int) -> tuple[dict, int, int]:
    p = {**DEFAULTS, **(cfg_xgb or {})}
    p.pop("early_stopping_rounds", None)
    if p.get("device") != "cuda":
        raise RuntimeError(f"GPU-only: xgb device must be 'cuda', got {p.get('device')!r}")
    n = int(p.pop("n_estimators"))
    max_bin = int(p.pop("max_bin"))
    p["seed"] = seed
    p["verbosity"] = 0
    return p, n, max_bin


def calib_split(groups: torch.Tensor) -> torch.Tensor:
    """True for the ~25% of S1 groups used to fit the isotonic map (fixed hash)."""
    h = (groups.to(torch.int64) * 2654435761) >> 7
    return (h % 4) == 0


class CalibratedXGB:
    def __init__(self, booster, iso: BinnedIsotonic):
        self.booster, self.iso = booster, iso

    def raw(self, X: torch.Tensor, batch: int = 4_000_000) -> torch.Tensor:
        outs = []
        with _Driver():
            for a in range(0, X.shape[0], batch):
                outs.append(torch.from_dlpack(self.booster.inplace_predict(_cp(X[a:a + batch]))).to(torch.float32))
        return torch.cat(outs) if outs else torch.zeros(0, device=X.device)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.iso(self.raw(X)).to(torch.float32)

    def importances(self, names: list[str]) -> dict:
        s = self.booster.get_score(importance_type="gain")
        v = torch.tensor([s.get(f"f{i}", 0.0) for i in range(len(names))], dtype=torch.float64)
        v = v / (v.sum() if float(v.sum()) > 0 else 1.0)
        return dict(zip(names, [round(x, 5) for x in v.tolist()]))

    def __getstate__(self):
        return {"booster": self.booster.save_raw("ubj"), "x": self.iso.x.cpu(), "f": self.iso.f.cpu()}

    def __setstate__(self, st):
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(bytearray(st["booster"]))
        b.set_param({"device": "cuda"})
        iso = BinnedIsotonic.__new__(BinnedIsotonic)
        dev = torch.device("cuda")
        iso.x, iso.f = st["x"].to(dev), st["f"].to(dev)
        self.booster, self.iso = b, iso


def fit_calibrated(X: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, seed: int = 42,
                   cfg_xgb: dict | None = None) -> CalibratedXGB:
    import cupy
    import xgboost as xgb
    params, n_rounds, max_bin = _params(cfg_xgb, seed)
    cal = calib_split(groups)
    fit = ~cal
    with _Driver():
        d = xgb.QuantileDMatrix(_cp(X[fit]), label=_cp(y[fit].to(torch.float32)), max_bin=max_bin)
        booster = xgb.train(params, d, num_boost_round=n_rounds)
    del d
    cupy.get_default_memory_pool().free_all_blocks()
    m = CalibratedXGB(booster, None)
    m.iso = Isotonic(m.raw(X[cal]), y[cal])  # exact PAV (binned failed parity, reports/gpu_parity.md)
    return m


def predict_mean(models, X: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros(X.shape[0], dtype=torch.float64, device=X.device)
    for mm in models:
        acc += mm.predict(X).to(torch.float64)
    return (acc / len(models)).to(torch.float32)


def from_sklearn(clf) -> CalibratedXGB:
    """Exact GPU port of a P2/P3 ``CalibratedClassifierCV(XGBClassifier, isotonic, one split)``.

    sklearn's isotonic ``predict`` is linear interpolation between ``X_thresholds_`` /
    ``y_thresholds_`` clipped at the ends, which is what ``BinnedIsotonic.__call__`` computes
    from (x, f) - so the thresholds are loaded as its knots.
    """
    cc = clf.calibrated_classifiers_
    assert len(cc) == 1, "expected the one-split calibration of ber.match.gbdt.fit_calibrated"
    est = cc[0].estimator
    iso_sk = cc[0].calibrators[0]
    booster = est.get_booster()
    booster.set_param({"device": "cuda"})
    iso = BinnedIsotonic.__new__(BinnedIsotonic)
    dev = torch.device("cuda")
    iso.x = torch.as_tensor(iso_sk.X_thresholds_, dtype=torch.float64, device=dev)
    iso.f = torch.as_tensor(iso_sk.y_thresholds_, dtype=torch.float64, device=dev)
    return CalibratedXGB(booster, iso)
