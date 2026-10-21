"""Prepared-data integrity and trusted local artifact storage shared by ML CLIs."""

import csv
import io
import json
import platform
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from ml import dataset as ds
from ml.data_config import REVISION, SOURCE, SOURCES, SourceSpec

DEFAULT_ARTIFACTS = ds.ROOT / "artifacts"


@dataclass
class PreparedData:
    train_text: list[str]
    train_labels: list[str]
    validation_text: list[str]
    validation_labels: list[str]
    labels: list[str]
    provenance: dict


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def decode_json(content: bytes, name: str) -> dict:
    try:
        result = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ds.DataError(f"{name}: invalid JSON.") from exc
    if not isinstance(result, dict):
        raise ds.DataError(f"{name}: expected a JSON object.")
    return result


def load_json(path: Path) -> dict:
    return decode_json(ds.read_bytes(path), path.name)


def load_prepared(
    processed_dir: Path = ds.DEFAULT_OUTPUT,
    raw_dir: Path = ds.DEFAULT_RAW,
    specs: tuple[SourceSpec, ...] = SOURCES,
) -> PreparedData:
    """Verify Phase-1 files; NEVER open raw/test.csv (metadata reference only).

    Reuses Phase-1 hashing, parsing, schema, split and CSV serialization. No downloads
    or regeneration. Injectable fixture contracts are unavailable from the CLI.
    """
    manifest_bytes = ds.read_bytes(processed_dir / "manifest.json")
    manifest = decode_json(manifest_bytes, "manifest.json")
    try:
        labels = sorted(specs[0].categories)
        if (
            manifest["format_version"] != 1
            or manifest["dataset"] != "BANKING77"
            or manifest["source_repository"] != SOURCE
            or manifest["source_revision"] != REVISION
            or manifest["license"] != "CC-BY-4.0"
            or manifest["categories"] != labels
        ):
            raise ds.DataError("Manifest upstream identity or category inventory mismatch.")
        # Only this raw source is ever opened. Test data cannot be selected by CLI.
        source = ds.read_bytes(raw_dir / "train.csv")
        ds.check_hash(source, specs[0])
        rows = ds.parse_csv(source)
        ds.validate_rows(rows, specs[0])
        expected_source = {
            "sha256": specs[0].sha256,
            "source_url": specs[0].url,
            **ds.describe(rows),
        }
        if manifest["files"]["train.csv"] != expected_source:
            raise ds.DataError("Manifest original training-file metadata mismatch.")
        test_reference = manifest["files"]["test.csv"]
        if (
            test_reference["sha256"] != specs[1].sha256
            or test_reference["rows"] != specs[1].rows
            or test_reference["source_url"] != specs[1].url
            or test_reference["category_count"] != len(labels)
            or sorted(test_reference["category_counts"]) != labels
            or sum(test_reference["category_counts"].values()) != specs[1].rows
        ):
            raise ds.DataError(
                "Reserved test provenance metadata mismatch; test bytes are never read."
            )
        preparation = manifest["preparation"]
        if (
            type(preparation["seed"]) is not int
            or preparation["seed"] != 42
            or preparation["validation_fraction"] != 0.20
            or preparation["sklearn_version"] != sklearn.__version__
            or preparation["source"] != "official train.csv only"
            or preparation["source_row_id"] != "zero-based CSV data-record index"
            or preparation["algorithm"]
            != "sklearn.model_selection.train_test_split(stratify=category); sorted source IDs"
        ):
            raise ds.DataError(
                "Expected Phase-1 seed 42, fraction 0.20 and compatible split version. Re-prepare explicitly if needed."
            )
        train_ids, val_ids = ds.split_rows(rows, seed=42, val_fraction=0.20)
        partitions = {}
        hashes = {}
        for name, ids, filename in (
            ("train", train_ids, "train.csv"),
            ("validation", val_ids, "validation.csv"),
        ):
            content = ds.read_bytes(processed_dir / filename)
            expected = ds.partition_bytes(rows, ids)
            recorded = preparation[name]
            counts = dict(sorted(Counter(rows[i]["category"] for i in ids).items()))
            if (
                content != expected
                or recorded["sha256"] != ds.sha256(content)
                or recorded["rows"] != len(ids)
                or recorded["row_ids"] != ids
                or recorded["category_counts"] != counts
            ):
                raise ds.DataError(
                    f"Prepared {name} hash, schema, membership or counts mismatch. Run dataset prepare explicitly after investigation."
                )
            partitions[name] = list(
                csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
            )
            hashes[name] = ds.sha256(content)
        expected_overlap = ds.overlap([rows[i] for i in train_ids], [rows[i] for i in val_ids])
        if preparation["train_validation_overlap"] != expected_overlap:
            raise ds.DataError("Manifest partition-overlap metadata mismatch.")
        provenance = {
            "dataset": "BANKING77",
            "source_repository": SOURCE,
            "source_revision": REVISION,
            "raw_train_sha256": specs[0].sha256,
            "reserved_test_sha256_from_phase1_manifest": specs[1].sha256,
            "reserved_test_content_read": False,
            "manifest_sha256": ds.sha256(manifest_bytes),
            "processed_sha256": hashes,
            "train_rows": len(train_ids),
            "validation_rows": len(val_ids),
            "seed": 42,
            "validation_fraction": 0.20,
        }
        return PreparedData(
            [r["text"] for r in partitions["train"]],
            [r["category"] for r in partitions["train"]],
            [r["text"] for r in partitions["validation"]],
            [r["category"] for r in partitions["validation"]],
            labels,
            provenance,
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise ds.DataError(
            "Manifest structure is incomplete or inconsistent; no training performed."
        ) from exc


def load_artifacts(root: Path, data: PreparedData, *, trusted: bool = False) -> tuple:
    """Checks integrity before joblib; hashes do NOT authenticate an untrusted producer."""
    if not trusted:
        raise ds.DataError(
            "Joblib can execute code. Use --trust-local-artifacts only for your own trusted training output."
        )
    pointer = load_json(root / "current.json")
    run = pointer.get("run", "")
    if not isinstance(run, str) or not re.fullmatch(r"run-[0-9a-f]{32}", run):
        raise ds.DataError("Invalid local artifact run identifier.")
    folder = (root / run).resolve()
    if folder.parent != root.resolve():
        raise ds.DataError("Artifact run must remain within its selected local directory.")
    metadata_bytes = ds.read_bytes(folder / "model_metadata.json")
    if ds.sha256(metadata_bytes) != pointer.get("metadata_sha256"):
        raise ds.DataError("Model metadata checksum mismatch.")
    metadata = decode_json(metadata_bytes, "model_metadata.json")
    if (
        metadata.get("format_version") != 1
        or metadata.get("labels") != data.labels
        or metadata.get("data_provenance") != data.provenance
        or metadata.get("versions") != versions()
        or metadata.get("model_version") != pointer.get("model_version")
    ):
        raise ds.DataError(
            "Incompatible model metadata, versions or input provenance; retrain explicitly."
        )
    try:
        expected_identity = ds.sha256(ds.json_bytes(metadata["identity"]))
        if (
            metadata["identity"]
            != {k: metadata[k] for k in ("configuration", "data_provenance", "labels", "versions")}
            or metadata["model_version"] != "tfidf-lr-" + expected_identity[:16]
        ):
            raise ds.DataError("Model reproducibility identity mismatch.")
        names = {
            "intent_classifier.joblib",
            "most_frequent_baseline.joblib",
            "validation_metrics.json",
            "validation_classification_report.csv",
            "validation_confusion_matrix.png",
            "measurements.json",
        }
        if set(metadata["artifact_sha256"]) != names:
            raise ds.DataError("Artifact file inventory mismatch.")
        verified_models = {}
        for name in sorted(names):
            # Read once: the deserialized bytes must be the bytes actually checked.
            with (folder / name).open("rb") as handle:
                content = handle.read(64 * 1024 * 1024 + 1)
            if len(content) > 64 * 1024 * 1024:
                raise ds.DataError(f"{name}: exceeds local artifact size limit.")
            if ds.sha256(content) != metadata["artifact_sha256"][name]:
                raise ds.DataError(f"{name}: checksum mismatch; refusing deserialization.")
            if name.endswith(".joblib"):
                verified_models[name] = content
    except (KeyError, TypeError) as exc:
        raise ds.DataError("Incomplete artifact identity/checksum metadata.") from exc
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(io.BytesIO(verified_models["intent_classifier.joblib"]))
            baseline = joblib.load(io.BytesIO(verified_models["most_frequent_baseline.joblib"]))
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        if (
            type(model) is not Pipeline
            or list(model.named_steps) != ["tfidf", "classifier"]
            or type(model.named_steps["tfidf"]) is not TfidfVectorizer
            or type(model.named_steps["classifier"]) is not LogisticRegression
            or type(baseline) is not DummyClassifier
            or list(model.classes_) != data.labels
            or list(baseline.classes_) != data.labels
        ):
            raise ds.DataError("Unexpected trained estimator types or label order.")
    except Exception as exc:
        raise ds.DataError(
            "Trusted model cannot be loaded with this environment; retrain rather than using incompatible artifacts."
        ) from exc
    return model, baseline, metadata, folder
