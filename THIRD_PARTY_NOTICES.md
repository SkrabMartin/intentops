# Third-party notices

The MIT license covers original IntentOps code. Dependencies and datasets retain
their own licenses; installing them does not relicense them as MIT.

## BANKING77 (local Phase-1 dataset preparation)

BANKING77 is a third-party dataset from PolyAI. The original
[PolyAI source repository](https://github.com/PolyAI-LDN/task-specific-datasets)
distributes its datasets under
[Creative Commons Attribution 4.0 International](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE).
See the [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).
Future use or redistribution must give appropriate credit, link the license,
and indicate changes. Dataset contents are not covered by IntentOps' MIT license.
Phase 1 explicitly downloads verified original files into Git-ignored local storage
only when `fetch` is requested. No dataset rows are bundled in this repository or
Docker images. Preparation repartitions official training rows and adds source row
IDs; original source files, text, categories and official test data remain unchanged.
Derived data retain the dataset's separate license and attribution obligations.
See [data provenance](docs/data.md) for the pinned revision, approved hashes and
transformation details. Code licensing does not override these obligations.
The exact category inventory in `ml/data_config.py` reproduces upstream dataset
metadata and is attributed to PolyAI under CC BY 4.0; surrounding original code
remains MIT-licensed.

Phase 2 generates local TF-IDF vocabularies, coefficients, validation reports and
aggregate metrics using these separately licensed inputs. Serialized artifacts
may contain learned source terms and remain Git-ignored. No model, vocabulary,
customer-message text or generated experiment bundle is published. Aggregate
validation numbers in docs/model.md are attributed to BANKING77 and describe a
repartitioning of official training data, not the reserved official test split.

Phase 3 serves the operator's explicitly trusted local bundle in memory. Models
are mounted at runtime rather than embedded or published. The approved data
identity is pinned by hash for dataset-free serving; no dataset content is copied
into the application. Serving does not change the separate data/metadata license.

Upstream citation: Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson,
and Ivan Vulić (2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807),
Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020.
