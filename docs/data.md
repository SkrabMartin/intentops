# BANKING77 provenance and preparation

Original source: [PolyAI task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets).
Pinned revision: [`57ec275d8078af65b7731c2a98be812d844a6d6b`](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data).
Expected identity comes from [PolyAI's Hugging Face metadata](https://huggingface.co/datasets/PolyAI/banking77/raw/main/dataset_infos.json).
`ml/data_config.py` stores immutable raw URLs, approved hashes, counts and the
77 exact category strings from the pinned upstream category inventory.

| Original file | Rows | Approved and observed SHA-256 |
| --- | ---: | --- |
| train.csv | 10,003 | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| test.csv | 3,080 | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

Both were fetched and verified during Phase 1. Original files use UTF-8 CSV
with columns `text,category`; both fields remain strings with their original
spelling and whitespace. Literal strings such as `NA` are not converted to null.
Hugging Face's numeric label encoding is not the original file format.

## Validation and EDA

Both original splits contain 77 categories. Training classes contain 35–187 rows;
each official test category contains 40. No null/blank fields, duplicate texts,
duplicate labeled records, conflicting-label identical texts, or identical-text
overlap between official train/test were observed under exact string comparison.

| Character lengths | Official train | Official test |
| --- | ---: | ---: |
| Minimum | 13 | 13 |
| Median | 47 | 45 |
| Mean | 59.474 | 54.232 |
| Maximum | 433 | 368 |

These describe the dataset, not model performance. CLI summaries contain no raw
message examples. Overlap reports contain row positions rather than message text.
Matching is exact and does not detect normalized, near-duplicate or semantic overlap.

## Reproducible membership

`python -m ml.dataset prepare --seed 42 --val-fraction 0.20` uses scikit-learn's
stratified `train_test_split` on **official train only**. It yields 8,002 training
and 2,001 validation examples, with every category represented. The 3,080 official
test rows never enter either derived partition and are not copied to processed/.

Derived CSVs contain `source_row_id,text,category`, sorted by source ID. Each ID
is the zero-based original training data-record index. JSON is sorted, UTF-8,
timestamp-free and newline-terminated; CSV records use LF. The manifest includes
the scikit-learn version, seed/fraction, membership, original hashes, output hashes
and category frequencies. Manifest publication is last, after per-file atomic
writes; output hashes allow detection of an interrupted multi-file rerun.
Use the same dependency versions to reproduce exact bytes across installations.

Default preparation was rerun with identical output bytes and zero exact-text
overlap between derived partitions. Every original training row is retained;
no messages or labels are normalized or deduplicated. If future approved sources
contain duplicate text, overlap is flagged rather than silently dropping rows.
The benchmark-preserving policy can leave leakage in such data; grouping and
its balancing trade-offs require an explicit later decision.

## Offline use and licensing

Only `fetch` opens external HTTP. Manually supply official files under data/raw/
or select `--raw-dir`, then run verify, prepare and summary entirely offline.
Importing modules, tests, notebook cells, API/dashboard startup and Docker builds
never download BANKING77. The local data directory is ignored by Git and Docker.

Original IntentOps code is MIT. BANKING77 is PolyAI third-party data under CC BY 4.0;
see [third-party notices](../THIRD_PARTY_NOTICES.md). Local train/validation files
are a repartitioning with added row IDs, not MIT-licensed dataset content.
Redistribution would require preserving attribution, linking CC BY 4.0 and
indicating modifications; this repository intentionally distributes no data rows.

Scope: English banking customer-support, one intent per message. The benchmark
does not establish generalization to other languages/domains, multi-intent input,
or deployment quality. No classifier, accuracy, F1, calibration or thresholds
are implemented or claimed by Phase 1.
