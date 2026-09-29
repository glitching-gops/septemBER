"""GPU OOF verifier scoring for the retained retrieval-rescue pilot."""
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

import pandas as pd
import torch
from ber import compute
from ber.embed import gpu_setup
from ber.io import gpu_load as gl, ids as I
from ber.io_utils import load_config
from run_p4_name_ambiguity import freeze_check
from run_p4_verifier import cmd_score
import inspect
import textwrap
from ber.verify import rerankers

_score_source = textwrap.dedent(
    inspect.getsource(rerankers.Verifier.score)
)
_old = "1.0 / (1.0 + np.exp(-logit))"
_new = (
    "torch.sigmoid(torch.as_tensor(logit, "
    "device=self.device)).cpu().numpy()"
)
assert _score_source.count(_old) == 1
_score_source = _score_source.replace(_old, _new)
_score_namespace = dict(vars(rerankers))
exec(compile(_score_source, "<cuda_verifier_probability>", "exec"),
     _score_namespace)
rerankers.Verifier.score = _score_namespace["score"]

freeze_check()
cfg = load_config("configs/p4_noaug.yaml")
compute.limit_threads()
dev = gpu_setup()
assert dev.type == "cuda", "No CPU fallback."

root = Path(cfg["abs"]["cache_dir"]).parent
report = json.loads(
    (root / "reports/empty_retrieval_prefilter_cv.json").read_text()
)
assert report["gate_pass"], "Prefilter did not pass."
src = root / "reports/empty_retrieval_prefilter_cv_retained_pairs.parquet"
frame = pd.read_parquet(src, columns=["s1", "m"])
table = I.entity_table(cfg, "train")
meta = gl.dev_meta(cfg, dev)

s1 = gl.to_dev(frame.s1.to_numpy(), dev).long()
m = gl.to_dev(frame.m.to_numpy(), dev).long()
folds = torch.full((len(table),), -1, dtype=torch.long, device=dev)
hold = torch.zeros(len(table), dtype=torch.bool, device=dev)
folds[meta["s1"]] = meta["fold"]
hold[meta["s1"]] = meta["holdout"]

assert len(frame) == report["retained_pairs"] and len(frame) > 0
assert bool((folds[s1] >= 0).all())
assert not bool(hold[s1].any()), "Holdout contamination."
assert torch.unique(torch.stack((s1, m), dim=1), dim=0).shape[0] == len(frame)
sources = gl.to_dev(table.src.to_numpy(), dev)
assert bool((sources[s1] == 1).all())
assert bool(((sources[m] == 2) | (sources[m] == 3)).all())
countries = gl.to_dev(table.country.to_numpy(), dev)
assert bool((countries[s1] == countries[m]).all())

pilot = root / "empty_rescue_cv"
pilot.mkdir(parents=True, exist_ok=True)
digest = hashlib.sha256()
with src.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)

manifest = {
    "input_sha256": digest.hexdigest(),
    "pairs": len(frame),
    "verifier_key": cfg["verifier"]["key"],
}
manifest_path = pilot / "input.json"
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest, "Input changed."
else:
    manifest_path.write_text(json.dumps(manifest, indent=2))

ids = table.entity_id.to_numpy()
pairs = pd.DataFrame({
    "s1_id": ids[s1.cpu().numpy()],
    "m_id": ids[m.cpu().numpy()],
})
target = pilot / "dev_empty_rescue_cv_set.parquet"
if target.exists():
    assert pd.read_parquet(target).equals(pairs), "Existing pair set differs."
else:
    pairs.to_parquet(target, index=False)

print(f"V0 | input checked | {len(pairs):,} CV-only pairs | next: OOF scoring",
      flush=True)
del frame, pairs, table, meta, s1, m, folds, hold, sources, countries
torch.cuda.empty_cache()
os.environ["P4_SCORE_ROOT"] = str(pilot / "verifier")

for fold in (0, 1):
    freeze_check()
    cmd_score(SimpleNamespace(
        fold=fold, part="oof", extra="empty_rescue_cv",
        reverse=False, middle=False, det_batch=128,
    ), cfg)
    torch.cuda.empty_cache()

print(f"V1 | complete | {pilot / 'verifier'} | next: GPU matcher evaluation",
      flush=True)
