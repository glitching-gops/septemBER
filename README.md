# Business entity resolution

This repository is the final hackathon project for matching noisy business records across three sources. For each Source 1 record, it predicts zero or more matching Source 2/3 IDs. The challenge uses macro-averaged per-entity F0.5, including entities with no matches. The pipeline treats `country` as an open-set string and uses only the supplied challenge data.

## Final result

`output/matching_results.tsv` and `output/candidate_pairs.tsv` are the **Sub D** submission chosen for this repository. The candidate file contains the final pair set scored by the matcher, including pairs considered by the empty-entity rescue. Both TSVs passed the organiser's validator with `--check-ids`: 1,732,544 Source 1 rows, 5,698,072 selected pairs, and 67,064,553 candidate pairs.

Sub D scored **0.9803 on the leaderboard**, as reported by the project owner. It improved the frozen Sub C baseline on internal cross-validation (+0.100 percentage points) and the held-out split (+0.139 percentage points). Leave-one-country-out checks for Sub D were run later, but their historical baseline did not reproduce exactly; treat those checks as exploratory. See [METHODOLOGY.md](METHODOLOGY.md) for the model and evaluation details.

## Repository layout

```text
configs/          Pipeline settings and artifact paths
src/ber/          Normalisation, blocking, features, models, decisions, I/O
scripts/          Training, inference, and validation entry points
tests/            Focused unit and pipeline checks
output/           Final matching and candidate TSVs (Git LFS)
runs/runs.csv     Experiment ledger
METHODOLOGY.md    Method and measured results
LICENSES.md       Model and dependency licence notes
```

The challenge data, cached embeddings, and trained weights are excluded from Git. Download the embeddings, aligned ID arrays, base E5 encoder, and final Sub D verifier, blend, and prefilter weights from the [Business entity resolution artifact dataset on Hugging Face](https://huggingface.co/datasets/glitching-gops/Business-entity-resolution). The server copy is at `/scratch/user4/ber/hf_export`. Its README, manifest, and restoration script explain the original cache paths. A full rerun also needs generated P2/P4 intermediate caches, which are **not** in that export.

## Setup

Use Python 3.11 and an NVIDIA CUDA GPU for the full pipeline. The pinned environment is in `requirements.txt`. Install the CUDA PyTorch wheel for your GPU first; the commands below match the original CUDA 13 environment:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
pytest -q
```

The supplied dataset should be unpacked into `data/raw/student_resource/`, preserving its `dataset/train/`, `dataset/test/`, and `utils/` directories. `data/` is ignored by Git. `data/DATA_MANIFEST.md` records the input hashes. Read TSV files with an explicit tab separator; `ber.io_utils` centralises pipeline I/O.

Configs use the original scratch layout by default. The Hugging Face export uses clear folder names; run its `restore_pipeline_layout.py` after downloading to recreate the cache paths, then set `BER_SCRATCH_ROOT` to the resulting `pipeline_layout/` directory. For example, `/scratch/user4/p4/...` maps to `$BER_SCRATCH_ROOT/p4/...`. Set `TMPDIR=$BER_SCRATCH_ROOT/ber/tmp` for large jobs, and create that directory first.

## Reproduction

The saved final files are ready to inspect and validate without a GPU:

```bash
python data/raw/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/raw/student_resource/dataset/test --check-ids
```

The pipeline stages and their entry points are:

| Stage | Command or entry point | Main output |
| --- | --- | --- |
| Split and normalise | `scripts/make_splits.py`, `scripts/run_normalize.py`, `scripts/run_name_keys.py` | Grouped folds and normalised records |
| Dense blocking | `scripts/run_embed.py`, `scripts/run_blocking.py` | E5 vectors and initial candidates |
| Pair model | `scripts/run_features.py`, `scripts/run_e1.py` | Pair features and GPU XGBoost model |
| Pair verifier | `scripts/run_verifier_ft.py`, `scripts/run_verifier_integrate.py`, `scripts/run_p4_verifier.py` | Two round-2 BGE verifier folds |
| P4 blend | `scripts/run_p4_stage.py`, `scripts/run_p4_name_ambiguity.py`, `scripts/run_p4_ambiguity_blend.py`, `scripts/infer_p4_ambiguity_blend.py` | Frozen Sub C scores and blend |
| Empty-entity rescue | `scripts/pilot_p4_retrieval_prefilter.py`, `scripts/build_p4_empty_rescue_submission.py` | Sub D candidates and matches |

Source files live under `src/ber/`; `configs/p4_noaug.yaml` selects the shipped P4 verifier. Source 1 is searched against both other sources within its country. Training uses grouped folds; the unseen test country is processed with the same normalization, retrieval, and scoring code. `scripts/p4_env.sh` sets the offline model and scratch environment for P4 jobs. Run `python scripts/build_p4_empty_rescue_submission.py --help` for the final staged invocation (`prepare`, each verifier `score` fold, then `finish`). The script checks frozen inputs and refuses stale checkpoints.

The final run consumes intermediate P2/P3/P4 tables and Sub C scores in addition to the Hugging Face export. Reconstructing those caches from raw data is a multi-stage GPU run; the table above identifies the producing scripts and `configs/` fixes their paths and settings. The supplied TSVs and checksums are the easiest way to verify this archived result. No external business lookup was used.

## GitHub and artifact download

The 887 MB candidate TSV is tracked through Git LFS. Install Git LFS before adding, committing, or cloning this repository, then run `git lfs install`. The two final TSVs are selected by `.gitattributes`; other generated output files remain ignored.

The [Hugging Face artifact dataset](https://huggingface.co/datasets/glitching-gops/Business-entity-resolution) is uploaded. Download it to a local directory and verify every file:

```bash
hf download glitching-gops/Business-entity-resolution --repo-type dataset --local-dir /path/to/ber-artifacts
cd /path/to/ber-artifacts && sha256sum -c SHA256SUMS
python restore_pipeline_layout.py
export BER_SCRATCH_ROOT=/path/to/ber-artifacts/pipeline_layout
```
