"""Training integrity and repeatability on fictional tiny datasets only."""

import json

import pytest

from ml import dataset as ds
from ml import train
from ml.model_support import load_artifacts, load_prepared

pytestmark = pytest.mark.usefixtures("block_model_network")


def test_train_only_vocabulary_and_repeatability(model_data):
    data, *_ = model_data
    model, baseline, _ = train.train_models(data)
    assert "validationonlytoken" not in model.named_steps["tfidf"].vocabulary_
    assert list(model.classes_) == data.labels
    assert model.predict(data.validation_text).tolist() == data.validation_labels
    another, other_baseline, _ = train.train_models(data)
    a, _ = train.evaluate_models(model, baseline, data)
    b, _ = train.evaluate_models(another, other_baseline, data)
    assert a == b


def test_loader_never_reads_reserved_test(model_data, monkeypatch):
    data, raw, processed, specs = model_data
    (raw / "test.csv").unlink()
    original = ds.read_bytes

    def guarded(path):
        assert path.name != "test.csv", "Official test access forbidden"
        return original(path)

    monkeypatch.setattr(ds, "read_bytes", guarded)
    assert load_prepared(processed, raw, specs).provenance == data.provenance


@pytest.mark.parametrize("missing", ["manifest.json", "train.csv", "validation.csv"])
def test_missing_prepared_input(model_data, missing):
    _, raw, processed, specs = model_data
    (processed / missing).unlink()
    with pytest.raises((OSError, ds.DataError)):
        load_prepared(processed, raw, specs)


@pytest.mark.parametrize(
    "field", ["revision", "seed", "fraction", "labels", "count", "ids", "raw_hash", "version"]
)
def test_manifest_tampering_rejected(model_data, field):
    _, raw, processed, specs = model_data
    path = processed / "manifest.json"
    m = json.loads(path.read_text())
    if field == "revision":
        m["source_revision"] = "wrong"
    elif field == "seed":
        m["preparation"]["seed"] = 43
    elif field == "fraction":
        m["preparation"]["validation_fraction"] = 0.3
    elif field == "labels":
        m["categories"] = list(reversed(m["categories"]))
    elif field == "count":
        m["preparation"]["train"]["rows"] += 1
    elif field == "ids":
        m["preparation"]["train"]["row_ids"][0] = 999
    elif field == "raw_hash":
        m["files"]["train.csv"]["sha256"] = "0" * 64
    elif field == "version":
        m["preparation"]["sklearn_version"] = "incompatible"
    path.write_bytes(ds.json_bytes(m))
    with pytest.raises(ds.DataError):
        load_prepared(processed, raw, specs)


@pytest.mark.parametrize("update_hash", [False, True])
def test_processed_tampering_rejected_even_with_new_hash(model_data, update_hash):
    _, raw, processed, specs = model_data
    path = processed / "train.csv"
    path.write_bytes(path.read_bytes().replace(b"fictional", b"altered", 1))
    if update_hash:
        manifest = json.loads((processed / "manifest.json").read_text())
        manifest["preparation"]["train"]["sha256"] = ds.sha256(path.read_bytes())
        (processed / "manifest.json").write_bytes(ds.json_bytes(manifest))
    with pytest.raises(ds.DataError):
        load_prepared(processed, raw, specs)


def test_reserved_source_cannot_be_fitting_input(model_data):
    _, raw, processed, specs = model_data
    (raw / "train.csv").write_bytes((raw / "test.csv").read_bytes())
    with pytest.raises(ds.DataError, match="SHA-256 mismatch"):
        load_prepared(processed, raw, specs)


def test_cli_invalid_inputs_create_no_artifacts(tmp_path, capsys):
    output = tmp_path / "artifacts"
    assert (
        train.main(
            [
                "--raw-dir",
                str(tmp_path / "raw"),
                "--processed-dir",
                str(tmp_path / "processed"),
                "--artifact-dir",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
    assert "Training error:" in capsys.readouterr().err


def test_atomic_artifacts_and_trusted_round_trip(model_data, tmp_path, capsys):
    data, *_ = model_data
    model, baseline, timing = train.train_models(data)
    root = tmp_path / "artifacts"
    metrics, folder = train.publish_run(root, data, model, baseline, timing)
    pointer = json.loads((root / "current.json").read_text())
    assert pointer["run"] == folder.name
    restored, restored_baseline, metadata, location = load_artifacts(root, data, trusted=True)
    assert (
        restored.predict(data.validation_text).tolist()
        == model.predict(data.validation_text).tolist()
    )
    assert metadata["data_provenance"] == data.provenance
    assert metadata["serving_ready"] is False and metadata["held_out_test_evaluated"] is False
    assert location == folder
    assert train.evaluate_models(restored, restored_baseline, data)[0] == metrics
    assert capsys.readouterr().out == ""


def test_failed_publish_keeps_previous_pointer(model_data, tmp_path, monkeypatch):
    data, *_ = model_data
    model, baseline, timing = train.train_models(data)
    root = tmp_path / "artifacts"
    train.publish_run(root, data, model, baseline, timing)
    before = (root / "current.json").read_bytes()

    def fail(*args):
        raise OSError("fictional plot failure")

    monkeypatch.setattr(train, "save_confusion", fail)
    with pytest.raises(OSError):
        train.publish_run(root, data, model, baseline, timing)
    assert (root / "current.json").read_bytes() == before
    assert not list(root.glob(".staging-*"))
