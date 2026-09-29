# Third-party licence notes

The final pipeline uses these pretrained models, both within the challenge's 8B-parameter limit:

| Model | Licence | Parameters | Use |
| --- | --- | ---: | --- |
| `intfloat/multilingual-e5-small` | MIT | 117,654,272 | Name/address embeddings |
| `BAAI/bge-reranker-v2-m3` | Apache 2.0 | 567,755,777 | Two fine-tuned final verifier folds |

The BGE verifier weights in the separate artifact export are modified versions trained only on the challenge data. Attribute the original model to BAAI. The E5 encoder weights are unmodified.

Core dependencies include NumPy (BSD), pandas (BSD), PyArrow (Apache 2.0), PyTorch (BSD), XGBoost (Apache 2.0), CuPy (MIT), scikit-learn (BSD), RapidFuzz (MIT), anyascii (ISC), PyYAML (MIT), and Transformers (Apache 2.0). Exact installed versions are pinned in `requirements.txt`. `anyascii` replaces GPL-licensed `Unidecode`.

The CUDA runtime wheels installed with GPU PyTorch have NVIDIA licences outside the project's permissive package allowlist; they are runtime dependencies, not project source. Some transitive utilities also have non-allowlist permissive or weak-copyleft licences, including `certifi` (MPL 2.0), `tqdm` (MPL 2.0/MIT), and Matplotlib/Pillow for optional reports. These packages are not vendored here. Review your deployment environment's full dependency notices before redistribution.
