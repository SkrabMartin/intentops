"""Known validation metrics, label order, provenance and trusted artifact safeguards."""

import json

import numpy as np
import pytest

from ml import dataset as ds
from ml import evaluate, train
from ml.model_support import load_artifacts

pytestmark = pytest.mark.usefixtures("block_model_network")


def test_known_metrics_and_zero_division():
    result = evaluate.score_predictions(["a", "a", "b", "b"], ["a"] * 4, ["a", "b"])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["weighted_f1"] == pytest.approx(1 / 3)
    assert result["macro_precision"] == 0.25
    assert result["macro_recall"] == 0.5
    assert result["per_class"]["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}
    assert result["confusion_matrix"] == [[2, 0], [2, 0]]
    assert result["predicted_label_distribution"] == {"a": 4}


def test_top_three_explicit_order_and_stable_ties():
    labels = ["A?", "b_case", "c_case", "d_case"]
    probabilities = np.array(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.7, 0.1, 0.15, 0.05],
            [0.4, 0.3, 0.05, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    metrics = evaluate.score_predictions(labels, ["A?"] * 4, labels, probabilities, labels)
    assert metrics["top_3_accuracy"] == 0.5
    assert metrics["top_k_actual"] == 3
    assert metrics["label_order"] == labels


def test_top_three_small_class_fixture():
    labels = ["A?", "b_case"]
    metrics = evaluate.score_predictions(labels, labels, labels, [[0.9, 0.1], [0.1, 0.9]], labels)
    assert metrics["top_k_actual"] == 2 and metrics["top_3_accuracy"] == 1


@pytest.mark.parametrize(
    "probabilities,order",
    [
        ([[0.5, 0.5]], ["b", "a"]),
        ([[1]], ["a", "b"]),
        ([[-0.2, 1.2]], ["a", "b"]),
        ([[0.2, 0.2]], ["a", "b"]),
        ([[float("nan"), 0.1]], ["a", "b"]),
    ],
)
def test_invalid_probability_contract(probabilities, order):
    with pytest.raises(ds.DataError, match="Probability"):
        evaluate.score_predictions(["a"], ["a"], ["a", "b"], probabilities, order)


@pytest.fixture
def local_run(model_data, tmp_path, monkeypatch):
    data, *_ = model_data
    monkeypatch.setattr(
        train, "save_confusion", lambda metrics, path: path.write_bytes(b"fictional fixture image")
    )
    model, baseline, timing = train.train_models(data)
    root = tmp_path / "artifacts"
    metrics, folder = train.publish_run(root, data, model, baseline, timing)
    return root, folder, data, metrics


def test_explicit_trust_required_before_deserialization(local_run, monkeypatch):
    root, _, data, _ = local_run

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not deserialize untrusted data")

    monkeypatch.setattr("ml.model_support.joblib.load", forbidden)
    with pytest.raises(ds.DataError, match="Joblib can execute code"):
        load_artifacts(root, data)


@pytest.mark.parametrize(
    "filename",
    [
        "intent_classifier.joblib",
        "most_frequent_baseline.joblib",
        "model_metadata.json",
        "validation_metrics.json",
    ],
)
def test_corruption_rejected_before_deserialization(local_run, filename, monkeypatch):
    root, folder, data, _ = local_run
    path = folder / filename
    path.write_bytes(path.read_bytes() + b"corrupt")

    def forbidden(*args, **kwargs):
        raise AssertionError("Corrupt file must not deserialize")

    monkeypatch.setattr("ml.model_support.joblib.load", forbidden)
    with pytest.raises(ds.DataError, match="checksum mismatch"):
        load_artifacts(root, data, trusted=True)


def test_incompatible_metadata_rejected(local_run, monkeypatch):
    root, folder, data, _ = local_run
    path = folder / "model_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["versions"]["sklearn"] = "incompatible"
    path.write_bytes(ds.json_bytes(metadata))
    pointer = json.loads((root / "current.json").read_text())
    pointer["metadata_sha256"] = ds.sha256(path.read_bytes())
    (root / "current.json").write_bytes(ds.json_bytes(pointer))
    with pytest.raises(ds.DataError, match="Incompatible model metadata"):
        load_artifacts(root, data, trusted=True)


def test_pointer_path_cannot_escape(local_run):
    root, _, data, _ = local_run
    (root / "current.json").write_bytes(ds.json_bytes({"run": "../external"}))
    with pytest.raises(ds.DataError, match="run identifier"):
        load_artifacts(root, data, trusted=True)


def test_evaluation_recomputes_without_retraining(local_run, model_data, monkeypatch, capsys):
    root, folder, data, original = local_run
    monkeypatch.setattr(evaluate, "load_prepared", lambda *args: data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not fit")

    monkeypatch.setattr("sklearn.pipeline.Pipeline.fit", forbidden)
    assert evaluate.main(["--artifact-dir", str(root), "--trust-local-artifacts"]) == 0
    assert json.loads((folder / "validation_recomputed_metrics.json").read_text()) == original
    output = capsys.readouterr().out
    assert "fictional" not in output
    assert json.loads(output)["accuracy"] == original["model"]["accuracy"]


def test_missing_artifacts_fail_cleanly(model_data, tmp_path, monkeypatch, capsys):
    data, *_ = model_data
    monkeypatch.setattr(evaluate, "load_prepared", lambda *args: data)
    assert (
        evaluate.main(["--artifact-dir", str(tmp_path / "absent"), "--trust-local-artifacts"]) == 1
    )
    assert "Evaluation error:" in capsys.readouterr().err
