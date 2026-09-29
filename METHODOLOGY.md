# Methodology

## Task and data

Source 1 is a deduplicated reference. For each Source 1 record, the model emits any matching Source 2/3 IDs or an empty list. The supplied training labels alone define positives. Records may have abbreviated or transliterated names, reordered or incomplete addresses, and multiple matches. The test set contains an unseen country. The pipeline treats country labels as free text: country is used to partition searches and check pair consistency, never to select country-specific code or external data.

The score is per-entity F0.5, macro-averaged. Empty prediction against empty truth scores 1; a false merge on a singleton scores 0. Training and evaluation use component-grouped folds to reduce entity leakage, plus a separate 15% grouped holdout. Test data contributes only unlabelled statistics and inference inputs.

## Candidate generation

Names and addresses are normalized with Unicode NFKC, transliteration, punctuation and legal-form handling, address abbreviations, number extraction, and shared YAML dictionaries. A learned transliteration map is fitted from true training pairs only. The same dictionaries and rules apply to every country.

`intfloat/multilingual-e5-small` (MIT, 118M parameters) embeds names in their original script and normalized addresses. GPU top-k search runs per country and source in both directions. Name, address, and reverse results are combined by reciprocal-rank fusion, then capped at 15 candidates per Source 1 record per target source for the base model. The final Sub D candidate set includes additional pairs retrieved for records the frozen Sub C model predicted empty. `candidate_pairs.tsv` is the exact set scored by the final matcher, after the rescue prefilter.

For the rescue, empty Source 1 records receive a deeper E5 search with k=100. A frozen GPU prefilter retains the highest-scoring 5% of these new pairs. That set is scored by the same round-2 verifier and blended matcher as the existing candidates. Retrieval and the prefilter use no test labels.

## Matching model

Pair features include RapidFuzz name/address similarity, exact E5 cosine, legal-form and numeric agreement, missingness, address landmarks, inverse-document-frequency token overlap, and blocking rank/provenance. GPU XGBoost learns the primary pair probabilities. The IDF statistics are fitted from each split's unlabelled input records, including test input records at inference; no test labels or pseudo-labels are used.

Two folds of `BAAI/bge-reranker-v2-m3` (Apache 2.0, about 568M parameters) were fully fine-tuned on training pairs. The verifier reads both raw records and supplies calibrated logit, probability, rank, and margin features to the P4 XGBoost matcher. Each training pair's verifier features are out of fold; holdout and test pairs use both folds. The final matcher is a frozen 50/50 blend of the round-2 model and a model with additional name-frequency ambiguity features. Its global threshold is 0.752. A target record may be claimed by at most one Source 1 record; Source 1 may retain many targets. The decision is recomputed across old and rescued candidates, so newly retrieved pairs can also displace existing selections.

The base E5 model, the two final verifier folds, the blend bundle, and the rescue prefilter are in `/scratch/user4/ber/hf_export`. Its manifest maps the named upload folders to the original pipeline paths. `LICENSES.md` records model and package licences. All shipped models are MIT or Apache 2.0 and below the 8B-parameter cap. There are no external identity lookups, geocoding requests, hosted ER APIs, or external business databases in the pipeline.

## Evaluation and final files

| Evaluation | Frozen Sub C | Sub D with rescue | Gain |
| --- | ---: | ---: | ---: |
| Grouped CV macro F0.5 | 0.981916 | 0.982916 | +0.100 percentage points |
| Grouped holdout macro F0.5 | 0.982198 | 0.983587 | +0.139 percentage points |

The rescue evaluation selected 296 additional true and 5 false pairs on CV, and 76 true and 1 false pair on holdout, with no old true pairs removed in either run. These are internal comparisons with the frozen Sub C model. Later leave-one-country-out screens showed positive gains, but their historical baseline scores drifted on rerun, so they do not establish exact historical parity. The project owner reports a **0.9803 leaderboard score for Sub D**, the files in `output/`.

The shipped files contain 1,732,544 Source 1 rows, 67,064,553 final candidates, and 5,698,072 matches. The organiser's validator passed with ID checking: every Source 1 ID appears once, every output ID is a valid Source 2/3 test ID, lists contain no duplicates, and selected matches are a subset of candidates. `README.md` has the exact validation command and the pipeline entry points.

## Reproduction limits

The Git repository excludes raw challenge data, large intermediate caches, and model weights. The Hugging Face export covers embeddings, aligned ID arrays, the base encoder, and the final inference weights, but not all intermediate P2/P4 tables or Sub C scores. A full rerun starts with the organiser's raw TSVs and runs the staged scripts in `README.md`; the final inference script verifies hashes of its frozen inputs and writes both TSVs. CUDA is required for the full run. The final TSVs are kept in Git LFS so the archived prediction itself remains inspectable without regenerating all intermediate state.
