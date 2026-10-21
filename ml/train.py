"""Fit one CPU TF-IDF/logistic-regression model and a train-only dummy baseline."""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ml import dataset as ds
from ml.evaluate import evaluate_models, report_csv, save_confusion
from ml.model_support import DEFAULT_ARTIFACTS, load_prepared, versions

CONFIGURATION = {
    "seed": 42,
    "tfidf": {
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 1.0,
        "max_features": 60000,
        "lowercase": True,
        "strip_accents": None,
        "analyzer": "word",
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "binary": False,
        "token_pattern": r"(?u)\b\w\w+\b",
        "dtype": "float64",
    },
    "classifier": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "tol": 0.0001,
        "random_state": 42,
        "fit_intercept": True,
        "class_weight": None,
        "regularization": "L2 via lbfgs default; no deprecated penalty keyword",
    },
    "baseline": {
        "strategy": "most_frequent",
        "representation": "one constant numeric feature; ignores text",
    },
    "threadpool_limit": 1,
    "trials": 1,
}


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_df=1.0,
                    max_features=60000,
                    lowercase=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4, random_state=42),
            ),
        ]
    )


def train_models(data) -> tuple:
    model = build_model()
    baseline = DummyClassifier(strategy="most_frequent")
    with threadpool_limits(limits=1), warnings.catch_warnings():
        # Do not publish an unconverged estimator silently.
        warnings.simplefilter("error", ConvergenceWarning)
        start = time.perf_counter()
        baseline.fit(np.zeros((len(data.train_labels), 1)), data.train_labels)
        baseline_seconds = time.perf_counter() - start
        start = time.perf_counter()
        model.fit(data.train_text, data.train_labels)
        model_seconds = time.perf_counter() - start
    return (
        model,
        baseline,
        {"model_fit_seconds": model_seconds, "baseline_fit_seconds": baseline_seconds},
    )


def configuration_for(model) -> dict:
    """Record exact estimator defaults without fitting or accessing datasets."""
    parameters = {
        key: (np.dtype(value).name if key == "tfidf__dtype" else value)
        for key, value in model.get_params(deep=True).items()
        if key not in ("steps", "tfidf", "classifier")
    }
    # JSON normalizes tuples to lists; retain every actual estimator parameter.
    return json.loads(ds.json_bytes({**CONFIGURATION, "pipeline_parameters": parameters}))


def publish_run(root: Path, data, model, baseline, fit_measurements: dict) -> tuple[dict, Path]:
    """Stage an entire new run, rename it atomically, publish pointer last; keep old runs."""
    metrics, inference_timing = evaluate_models(model, baseline, data)
    configuration = configuration_for(model)
    identity = {
        "configuration": configuration,
        "data_provenance": data.provenance,
        "labels": data.labels,
        "versions": versions(),
    }
    version = "tfidf-lr-" + ds.sha256(ds.json_bytes(identity))[:16]
    root.mkdir(parents=True, exist_ok=True)
    run_name = "run-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=root) as temporary:
        stage = Path(temporary) / run_name
        stage.mkdir()
        joblib.dump(model, stage / "intent_classifier.joblib", compress=3)
        joblib.dump(baseline, stage / "most_frequent_baseline.joblib", compress=3)
        (stage / "validation_metrics.json").write_bytes(ds.json_bytes(metrics))
        (stage / "validation_classification_report.csv").write_bytes(report_csv(metrics))
        save_confusion(metrics, stage / "validation_confusion_matrix.png")
        measurements = {
            **fit_measurements,
            "prediction": inference_timing,
            "environment": {
                "os": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "threadpool_limit": 1,
            },
            "model_artifact_bytes": (stage / "intent_classifier.joblib").stat().st_size,
            "classifier_iterations": model.named_steps["classifier"].n_iter_.tolist(),
        }
        (stage / "measurements.json").write_bytes(ds.json_bytes(measurements))
        hashes = {}
        for path in sorted(stage.iterdir()):
            with path.open("rb") as handle:
                hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        metadata = {
            "format_version": 1,
            "name": "Word TF-IDF + LogisticRegression",
            "model_version": version,
            "identity": identity,
            **identity,
            "artifact_sha256": hashes,
            "metrics_file": "validation_metrics.json",
            "license_note": "Original code MIT; BANKING77 and derived vocabulary/metadata have separate CC BY 4.0 provenance obligations.",
            "serving_ready": False,
            "probabilities_calibrated": False,
            "held_out_test_evaluated": False,
        }
        metadata_bytes = ds.json_bytes(metadata)
        (stage / "model_metadata.json").write_bytes(metadata_bytes)
        destination = root / run_name
        stage.rename(destination)
    ds.atomic_write(
        root / "current.json",
        ds.json_bytes(
            {
                "run": run_name,
                "model_version": version,
                "metadata_sha256": ds.sha256(metadata_bytes),
            }
        ),
    )
    return metrics, destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ds.DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=ds.DEFAULT_RAW)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args(argv)
    try:
        output = args.artifact_dir.resolve()
        for protected in (args.processed_dir.resolve(), args.raw_dir.resolve()):
            if output == protected or output in protected.parents or protected in output.parents:
                raise ds.DataError(
                    "Artifact and data directories must be separate, non-nested locations."
                )
        data = load_prepared(args.processed_dir, args.raw_dir)
        model, baseline, timing = train_models(data)
        metrics, folder = publish_run(output, data, model, baseline, timing)
        print(
            ds.json_bytes(
                {
                    "run": folder.name,
                    "accuracy": metrics["model"]["accuracy"],
                    "macro_f1": metrics["model"]["macro_f1"],
                    "baseline_accuracy": metrics["baseline"]["accuracy"],
                    "model_fit_seconds": timing["model_fit_seconds"],
                }
            ).decode(),
            end="",
        )
    except (ds.DataError, OSError, ValueError, ConvergenceWarning) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
