# Data manifest

## Archive
| File | Bytes | SHA-256 |
|---|---:|---|
| `student_resource.zip` | 1,094,823,222 | `2aefd2f8eb6f132b8933fccc1cbb98fa6356756a7581fed5914a5f26704a8bc5` |

- **Source:** copied from `/home/user4/venu/amzn/student_resource.zip` on 2026-09-25. The SHA-256 of the copy matches the original.
- **Unzip:** `unzip -q student_resource.zip -d raw -x '__MACOSX/*' '*.DS_Store'`. The excluded entries are macOS metadata only.
- **Permissions:** `raw/` was made read-only (`chmod -R a-w`).
- **Location (2026-09-27):** the zip was moved to `/scratch/user4/archive/data/student_resource.zip` to free `/`
  (SHA-256 re-verified after the copy). Every data file in it is in `raw/` with identical size.

## Extracted files (`raw/student_resource/`)
| Path | Bytes | Raw lines (`wc -l`, incl. header) | Data rows | Parsed rows (`ber.io_utils`) | SHA-256 |
|---|---:|---:|---:|---:|---|
| dataset/train/train_source1.tsv | 210,069,713 | 2,206,822 | 2,206,821 | 2,206,821 ✓ | `591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973` |
| dataset/train/train_source2.tsv | 489,301,488 | 5,034,617 | 5,034,616 | 5,034,616 ✓ | `6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed` |
| dataset/train/train_source3.tsv | 503,705,637 | 5,285,604 | 5,285,603 | 5,285,603 ✓ | `67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9` |
| dataset/train/train_ground_truth.tsv | 127,015,583 | 2,206,822 | 2,206,821 | 2,206,821 ✓ | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` |
| dataset/test/test_source1.tsv | 175,022,086 | 1,732,545 | 1,732,544 | 1,732,544 ✓ | `3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5` |
| dataset/test/test_source2.tsv | 509,456,422 | 4,887,274 | 4,887,273 | 4,887,273 ✓ | `79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd` |
| dataset/test/test_source3.tsv | 506,002,772 | 5,082,317 | 5,082,316 | 5,082,316 ✓ | `850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035` |
| utils/validate_submission.py | 13,687 | 349 | – | – | `f96f59934383a15095914f507620c078c474e8f9c60e864b2d051ed173a22dfc` |
| Documentation_template.md | 2,175 | 74 | – | – | `599b661af92a8d5c239ba283ee9b2ffed050c0bc86360964430a2f9857e8c599` |
| README.md | 14,130 | 263 | – | – | `be67cd7d9dbc48a58f93e1f61402360655debac64214115bf9d27d3301939285` |

## Encoding and format checks
- **Encoding:** every TSV decodes as strict UTF-8, with no BOM and no `\r`. Every file ends with a newline.
  - `train_source1.tsv` and `train_ground_truth.tsv` are pure ASCII.
  - The other files contain non-ASCII characters, including Devanagari script in S2/S3 names and accented Latin in test.
- **Field counts:** every line has exactly 4 tab-separated fields (source files) or 2 (ground truth), so there are no stray tabs.
- **Quote characters:** lines containing `"` number 4, 6 and 0 in train S1, S2 and S3, and 134, 349 and 330 in test S1, S2 and S3. Files are therefore parsed with `csv.QUOTE_NONE`.
- **Headers:** source files use `entity_id, business_name, business_address, country`, and the ground truth uses `source1_entity_id, matched_entity_ids`. Both match the organisers' README.

## Parsed validation (`ber.io_utils`, 2026-09-25)
- **Row counts:** for all 7 files, parsed rows equal raw data lines (✓ above). Columns match the expected headers, every ID has the right prefix, and IDs are unique within each file. No tab, CR or LF characters appear inside fields.
- **Empty fields:** only `business_address` has any. Train S2: 168,967 (3.4%). Train S3: 175,916 (3.3%). Test S2: 129,408. Test S3: 136,098. S1 has none in either split. No field is whitespace-only or has leading or trailing whitespace.
- **Ground truth:**
  - no duplicate rows, no duplicate IDs within a list, and no prefix violations
  - every GT S1 ID is in `train_source1`, and every train S1 ID has a GT row (0 missing)
  - every matched ID exists in the S2/S3 file that its prefix names
