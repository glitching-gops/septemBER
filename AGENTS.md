# Project instructions

## Purpose and score

Match business records across three supplied TSV sources with columns `entity_id`, `business_name`, `business_address`, and `country`. Source 1 (`S1-…`) is deduplicated; each Source 1 entity may match zero or more Source 2/3 IDs. Training covers two countries and test adds an unseen country. The final archived Sub D files are in `output/`.

The score is macro-averaged per-entity F0.5. Empty truth and prediction score 1; exactly one empty set scores 0. The implementation is `ber.evaluate.metric`, with hand-checked cases in `tests/test_metric.py`. The organiser's original specification is `data/raw/student_resource/README.md` when the untracked dataset is present.

## Data, licences, and conventions

- Use only the supplied dataset. Never call external identity, geocoding, registry, web lookup, or entity-resolution services from `src/`. Network use for the project is limited to downloading provided data, packages, and published project artifacts.
- Treat `country` as an open set of strings. Never hard-code, filter, or enumerate country values in `src/`. `tests/test_no_country_hardcoding.py` checks case-insensitive whole words, including words in comments.
- `data/raw/` is read-only and `data/` is ignored by Git. Test records may provide unlabelled statistics and inference inputs, never labels or label-like training signals.
- Use `ber.io_utils.read_source`, `read_ground_truth`, and `load_config` for TSV input and configuration; use `write_matching` and `write_candidates` for submissions.
- Keep paths, seeds, fold counts, and compute limits in `configs/default.yaml`. Source lives under `src/ber/`; plots and reports go to `reports/`, submissions to `output/`, and splits and experiment ledger to `runs/`.
- Every training experiment appends to `runs/runs.csv` through `ber.evaluate.ledger.append_run`. Run `pytest -q` before every commit.
- Final models must have MIT or Apache 2.0 licences and at most 8B parameters. Dependencies must be MIT, Apache 2.0, BSD, or ISC; flag other licences and never add GPL/AGPL code. Use `anyascii` rather than `Unidecode`; track licences in `LICENSES.md`.

## Compute and storage

- CUDA is the default for embeddings and encoding (torch fp16), similarity search and top-k (chunked torch matmul), sparse or TF-IDF similarity when reused (GPU sparse implementation), XGBoost training and inference (`device="cuda"`), verifier training and scoring, and large vectorised scoring, calibration, and decision stages (torch).
- CPU is for file I/O, parsing, orchestration, text normalisation, and RapidFuzz string features until a validated GPU equivalent exists. Vectorise or parallelise CPU work within `compute.n_jobs` and print an estimated finish time. A CPU stage projected over 20 minutes requires owner approval and a proposed GPU alternative.
- Never silently fall back to CPU. If CUDA is unavailable or out of memory, stop and report. CPU paths for toy tests or documented reproducibility may remain, but are off by default for production runs.
- Before heavy work check `uptime`, `free -h`, `df -h /`, and `nvidia-smi`. Before each GPU run select the least-loaded GPU, respect `compute.max_vram_gb`, and use both GPUs only if both are idle and work is embarrassingly parallel. Log device, GPU IDs, and peak VRAM in `runs/runs.csv`. Never disturb processes you did not start.
- The original shared server has 96 CPUs, about 500 GB RAM, and two 96 GB RTX PRO 6000 GPUs. `configs/default.yaml` caps jobs at 32 (`BER_N_JOBS` may only lower this), RAM at 200 GB, and VRAM at 40 GB; GPU use is enabled and CPU fallback disabled.
- Large artifacts live under `/scratch/user4/{ber,p2,p3,p4}` because the root disk filled on 2026-09-25. `data/cache` links to `/scratch/user4/ber/cache`. Set `BER_SCRATCH_ROOT` to remap the scratch prefix elsewhere and use `TMPDIR=/scratch/user4/ber/tmp` (or its remapped equivalent).
- Cached embeddings and final inference weights are staged in `/scratch/user4/ber/hf_export` and published at [Hugging Face](https://huggingface.co/datasets/glitching-gops/Business-entity-resolution). Its manifest and checksums describe the files. The export does not include every intermediate cache needed for a full rerun.

## Normalisation and validation details

- YAML dictionaries in `src/ber/normalize/dictionaries/` are hand-written from domain knowledge and tokens observed in supplied data. Apply them to every record regardless of country. Quote `no`, `on`, and `null` to avoid PyYAML coercion; a test guards this.
- Learned transliteration JSON maps under `cache/translit/` use true training pairs only. The `nondev` map excludes the dev slice and is used for train files; `alltrain` is used for test.
- `runs/dev_slice.parquet` contains 10% of training Source 1 queried against the full training Source 2/3 pool. The dense reverse blocker queries all Source 1. Exclusivity on this slice applies only among sampled Source 1 records; it is exact at test time because full-train confirmation was skipped in P1.

## Commands

Run from the repository root in Python 3.11. The old nested `code/business_entity_resolution/` layout and its `run_baseline_empty.py` and `run_eda.py` entry points were removed during publication cleanup; current stage entry points are in `README.md`.

```bash
source .venv/bin/activate
pytest -q
python scripts/make_splits.py
python scripts/run_normalize.py
python data/raw/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/raw/student_resource/dataset/test --check-ids
```

See `README.md` for setup and stage entry points, and `METHODOLOGY.md` for the model and evaluation.
