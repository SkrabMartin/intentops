# IntentOps — Open-Source Project Plan

**Status:** Planned, not yet implemented  
**License:** MIT for original source code  
**Development principle:** local-first, offline inference, CPU-compatible, no paid model APIs, optional self-hosting  
**Core task:** route customer messages to one of 77 BANKING77 intent classes and defer uncertain cases to human review  
**Estimated MVP effort:** 11–14 focused workdays, adjustable according to learning and testing needs

## 0. Project contract

The complete application will consist of a scikit-learn model, a FastAPI backend, a Streamlit demonstration interface, reproducible training/evaluation scripts, and optional Docker Compose deployment. The GitHub repository is the distribution channel, not a required production host. The trained model must work offline after installation; no request to a paid or hosted AI service may be necessary.

### Non-goals for v1.0

- No generative LLM, GPU, vector database, LangChain, autonomous agent, Kubernetes cluster, or paid API is required.
- No live banking system connections, customer account operations, automated financial decisions, or claims of deployment readiness for regulated banking services.
- No account registration, real customer private information, persistent collection of user inputs, or mandatory hosted telemetry.
- No unsupported claim that classifier confidence is a guaranteed measure of correctness on out-of-distribution examples.

## 1. Phase 1 — Repository foundation and licensing (Day 1)

**Objective:** establish a clear, reusable, legally coherent, public GitHub project.

**Implementation tasks:**

1. Create the Python project layout and install reproducible dependencies using `pyproject.toml` and a lockfile.
2. Add the standard MIT `LICENSE` with the proper project author's copyright notice.
3. Add `README.md`, `.gitignore`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` if useful, and `THIRD_PARTY_NOTICES.md`.
4. Document local-first constraints, supported Python versions, test commands, and the benchmark methodology planned for later phases.
5. Create GitHub issues or a milestone checklist for phases 2–7, keeping each issue testable and limited in scope.
6. Document third-party software/data licensing independently; the MIT license covers only code whose copyright the project controls.

**Acceptance gate:** a fresh machine can create the virtual environment, import the app package, run a smoke test, and understand the project's license and scope.

## 2. Phase 2 — Dataset and exploratory data science (Day 2)

**Objective:** obtain a reproducible, validated labeled dataset while preventing test leakage.

**Data:** BANKING77 contains 13,083 customer queries across 77 banking intents. The upstream split consists of 10,003 train and 3,080 test examples.

**Implementation tasks:**

1. Use official PolyAI dataset sources with pinned revision/location and integrity verification where possible.
2. Add a manual download/import fallback so future changes in hosted dataset loaders cannot block reproducing the experiment.
3. Preserve original train and test splits, and reserve a seeded stratified validation subset from the training split.
4. Analyze class counts, text lengths, label consistency, duplicates, and potential near-duplicate overlap.
5. Record dataset provenance and citation in `THIRD_PARTY_NOTICES.md` and in the experiment report.
6. Prefer keeping raw training data out of the code repository unless redistribution and documentation have been deliberately reviewed; a one-time data download is acceptable for training.

**Acceptance gate:** a documented command reproduces the same data statistics, labels, and train/validation split without using the official test split during model tuning.

## 3. Phase 3 — Baseline ML model and honest evaluation (Days 3–4)

**Objective:** implement a reliable multi-class classifier with measurable quality and realistic limits.

**Implementation tasks:**

1. Fit a trivial majority-class baseline, or another simple baseline, and record its test-independent validation metrics.
2. Train an scikit-learn `Pipeline` containing a `TfidfVectorizer` and multinomial-compatible `LogisticRegression`.
3. Tune hyperparameters on training/validation only, with a small reproducible search budget.
4. Evaluate macro F1, accuracy, per-label precision/recall, top-3 accuracy, class confusion, model size, and CPU response time.
5. Investigate incorrect predictions using a fixed error-analysis notebook or report.
6. Compare raw `predict_proba` scores with calibrated confidence, if appropriate, and document limitations of confidence on unseen or ambiguous requests.
7. Use validation examples to select any automatic-routing threshold; measure *coverage versus risk* rather than assuming a single threshold is universally safe.
8. Freeze the best configuration, refit according to the documented protocol, and use the official test set for the final held-out report. Do not continue changing the model based on repeated test-set examination.
9. Save model artifact, metadata, label mapping, dependency versions, evaluation summary, and checksums. Load serialized models only from trusted project artifacts.

**Acceptance gate:** one command trains the model; another reproduces evaluation. Results are compared with a baseline and not invented or selected using the test data.

## 4. Phase 4 — FastAPI serving layer (Days 5–6)

**Objective:** deliver predictions through a typed, tested API without external inference calls.

**Endpoints:**

- `GET /health`: readiness and liveness with a meaningful error if no model artifact is present.
- `GET /model/info`: model version, labels, training metadata, and artifact provenance.
- `POST /predict`: predicted category, top candidates, confidence/score description, review recommendation, and model version.
- `POST /predict/batch`: bounded batch inference without uploading to a cloud provider.

**Implementation tasks:**

1. Define strict Pydantic request and response schemas; reject empty or oversized texts.
2. Load the model only once at startup and fail clearly when expected artifacts are absent.
3. Separate model inference from HTTP handling, enabling unit tests without running a server.
4. Expose model metadata and clear information about threshold limitations.
5. Add unit tests, HTTP integration tests, and error-path tests.
6. Do not log raw user text by default. Keep network bindings local unless an operator explicitly changes deployment settings.

**Acceptance gate:** the API serves requests with no internet access, handles invalid inputs, and passes the automated test suite.

## 5. Phase 5 — Local interactive Streamlit interface (Days 7–8)

**Objective:** give reviewers a polished, self-explanatory demonstration requiring no paid services.

**Screens:**

1. **Try a message:** write or paste a customer message and inspect the top-3 predictions.
2. **Review decision:** show whether the example meets the validation-calibrated routing threshold, including limitations.
3. **Evaluation dashboard:** show actual confusion patterns, macro F1, class-level metrics, and coverage-risk trade-off.
4. **Batch CSV:** upload a local CSV and download predicted categories and review indicators without an external server.
5. **Model information:** show provenance, version, input-language scope, limitations, and third-party attribution.

**Implementation tasks:**

1. Use a minimal, responsive interface with straightforward navigation.
2. Connect it to the local FastAPI service, or clearly document the direct inference mode for offline development.
3. Handle no-model, no-server, invalid CSV, and unsupported-label cases gracefully.
4. Keep input content ephemeral unless the user explicitly opts into feedback storage.

**Acceptance gate:** a developer can launch the dashboard and understand what the classifier can and cannot do within a minute.

## 6. Phase 6 — Offline packaging, test automation, and deployment (Days 9–10)

**Objective:** let a new user clone, install, run, and test IntentOps independently.

**Implementation tasks:**

1. Provide a native Python installation path requiring no container runtime.
2. Provide a Dockerfile and optional Compose configuration for consistent local deployment.
3. Make all inference use a bundled or locally generated model artifact; no model download may occur at runtime.
4. Document that initial `git clone`, pip installation, Docker base-image pulls, and optional dataset acquisition require a network connection unless dependencies and images were previously cached or bundled.
5. Add GitHub Actions for linting, type checks, unit tests, and a small smoke test, within any applicable hosted CI limits.
6. Pin or lock dependencies, set deterministic seeds where practical, and check fresh-machine reproducibility.
7. Document Docker Desktop as separately licensed software: free for many personal, educational, open-source, and qualifying small-business uses, but not all enterprise use. Provide native Python and Docker Engine alternatives.
8. Run the application with networking disabled to verify the **post-install runtime** genuinely works offline.
9. Check that local files remain local, no telemetry service is contacted, and raw inputs are not emitted to hosted services.

**Acceptance gate:** native Python and container instructions work; the app starts without model inference APIs; inference succeeds without runtime internet access after prerequisites are installed.

## 7. Phase 7 — Portfolio release and measurable results (Days 11–14)

**Objective:** turn a working ML demo into a high-quality open-source engineering portfolio project.

**Implementation tasks:**

1. Produce an architecture overview, screenshots, a CLI demo, and a short reproducible walkthrough.
2. Publish the held-out metrics, evaluation methodology, baseline comparison, and honest failure examples.
3. Report CPU inference latency with test hardware information and clear measurement conditions.
4. Document model limitations: banking-domain English queries, single-intent labels, out-of-distribution inputs, threshold uncertainty, and no validated safety guarantee.
5. Run a final release checklist: clean clone, installation, offline inference, tests, dataset attribution, accessible UI, missing-model behavior.
6. Publish an annotated release such as `v1.0.0` with a changelog and exact reproduction instructions.
7. Add 2–4 good-first-issue ideas or contribution ideas so other GitHub users can extend the project.

**Acceptance gate:** a stranger can clone the repository, run the app, reproduce the documented evaluation, and understand why the design choices were made.

## Optional extensions — only after MVP is complete

- **Embedding comparison:** evaluate a small sentence-transformer model against TF-IDF; do not make it mandatory for the CPU-only core experience.
- **Feedback review loop:** allow users to propose corrected labels and review them before any retraining.
- **Drift simulation:** test unfamiliar query phrasing and demonstrate the need for retraining or human fallback.
- **Optional SQLite feedback store:** keep observations locally with clear deletion behavior and no hidden telemetry.
- **MLflow experiment tracking:** add an optional local tracking server if it helps reproducibility rather than increasing mandatory setup burden.
- **Model export and CLI:** add `intentops predict` and `intentops benchmark` commands for integration with external tools.

## Suggested repository layout

```text
intentops/
├── LICENSE
├── README.md
├── THIRD_PARTY_NOTICES.md
├── CONTRIBUTING.md
├── pyproject.toml
├── uv.lock                         # if uv is chosen
├── .gitignore
├── .github/workflows/ci.yml
├── src/intentops/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── schemas.py
│   ├── api.py
│   └── config.py
├── dashboard/
│   └── streamlit_app.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_inference.py
│   └── test_api.py
├── artifacts/
│   ├── model.joblib               # only trusted, project-produced artifact
│   ├── model_metadata.json
│   └── evaluation.json
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── offline_usage.md
├── Dockerfile
└── compose.yaml
```

## Release definition of done

- [ ] MIT license accurately covers original project code, with intact copyright/permission notice.
- [ ] BANKING77 credit and CC BY 4.0 terms are documented separately from MIT code licensing.
- [ ] No paid model API, account, or public hosting required for running the application.
- [ ] Prediction works offline after the first local setup; first-install network requirements are disclosed.
- [ ] Training and testing follow a documented reproducible protocol with proper split separation.
- [ ] Real metrics and limitations are published; no fabricated accuracy or benchmark results.
- [ ] CI, local tests, native installation, and Docker-based installation are documented.
- [ ] Human-review decision is evaluated, clearly labeled, and not described as a guarantee.
- [ ] No raw user text is transmitted or stored by default without explicit user control.
- [ ] A newcomer can clone, install, run, and understand the system without asking the author.

## Primary upstream references

- Dataset: https://huggingface.co/datasets/PolyAI/banking77
- Dataset source: https://github.com/PolyAI-LDN/task-specific-datasets
- Dataset research: https://arxiv.org/abs/2003.04807
- MIT license: https://choosealicense.com/licenses/mit/
- scikit-learn: https://scikit-learn.org/stable/
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://docs.streamlit.io/
- Docker Desktop licensing: https://docs.docker.com/subscription-billing/desktop-license
