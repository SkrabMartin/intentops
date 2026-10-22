"""Hand-computable fictional statistics and separation of development/audit."""

from types import SimpleNamespace

import numpy as np
import pytest

from ml import calibrate
from ml import calibration as cal
from ml import dataset as ds
from ml import review_policy as store


@pytest.fixture
def experiment():
    labels = ["alpha", "beta", "gamma"]
    rows = [
        {"source_row_id": str(i), "text": f"fictional {i}", "category": labels[i % 3]}
        for i in range(180)
    ]
    scores = np.random.default_rng(7).normal(size=(180, 3))
    scores[np.arange(180), np.arange(180) % 3] += 2
    calls = []

    class Model:
        classes_ = labels

        def decision_function(self, texts):
            ids = [int(t.split()[-1]) for t in texts]
            calls.append(ids)
            return scores[ids]

        def predict_proba(self, texts):
            return cal.scaled(self.decision_function(texts), 1.0)

    metadata = {
        "model_version": "fictional-v1",
        "labels": labels,
        "versions": {"python": "fictional"},
        "data_provenance": {"validation_rows": 180},
    }
    return Model(), metadata, rows, calls


def test_partition_reproducible_disjoint_stratified(experiment):
    _, _, rows, _ = experiment
    parts, manifest = calibrate.split_manifest(rows)
    assert parts == calibrate.split_manifest(rows)[0]
    sets = [set(p) for p in parts.values()]
    assert len(set.union(*sets)) == 180
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert [len(p) for p in parts.values()] == [72, 54, 54]
    assert all(set(m["class_support"]) == {"alpha", "beta", "gamma"} for m in manifest.values())


def test_no_audit_predictions_or_labels_used_for_development(experiment):
    model, metadata, rows, calls = experiment
    frozen = calibrate.develop(model, metadata, rows)
    audit_ids = set(frozen.manifest.audit.source_ids)
    assert not audit_ids & {i for call in calls for i in call}
    parts, _ = calibrate.split_manifest(rows)
    # Fitting receives only calibration scores; selection is separate, audit is explicit.
    assert set(calls[0]) == set(parts["calibration"])
    before = ds.json_bytes(frozen.model_dump())
    completed = calibrate.audit(model, metadata, rows, frozen)
    assert ds.json_bytes(completed.frozen.model_dump()) == before
    assert set(calls[-1]) == audit_ids


def test_scaling_equivalence_order_ties_and_optimizer():
    scores = np.array([[3.0, 1.0, -2.0], [0.0, 0.0, 0.0], [-2.0, 1.0, 3.0]])
    for temperature in [0.05, 0.5, 1.0, 20.0]:
        p = cal.scaled(scores, temperature)
        assert p.sum(axis=1) == pytest.approx([1, 1, 1])
        assert np.array_equal(p.argmax(axis=1), scores.argmax(axis=1))
    assert cal.scaled(scores, 1)[0] == pytest.approx(np.exp(scores[0]) / np.exp(scores[0]).sum())
    t = cal.fit_temperature(scores, np.array([0, 1, 2]))
    assert t == cal.fit_temperature(scores, np.array([0, 1, 2]))
    assert 0.05 <= t <= 20


def test_optimizer_failure(monkeypatch):
    monkeypatch.setattr(cal, "minimize_scalar", lambda *a, **k: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="optimizer failed"):
        cal.fit_temperature([[1, 0]], np.array([0]))


@pytest.mark.parametrize(
    "values", [[], [1, 2], [[float("nan"), 0]], [[-1, 2]], [[0.2, 0.2]], [[1.1, -0.1]]]
)
def test_invalid_probability_inputs(values):
    with pytest.raises(ValueError):
        cal.metrics(values, np.array([0]))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), 21.0])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        cal.scaled([[1, 0]], temperature)


def test_hand_computed_metrics_empty_bins():
    result = cal.metrics([[0.75, 0.25], [0.25, 0.75]], np.array([0, 0]))
    assert result["log_loss"] == pytest.approx(-(np.log(0.75) + np.log(0.25)) / 2)
    assert result["brier"] == pytest.approx(0.625)
    assert result["ece"] == pytest.approx(0.25)
    assert result["accuracy"] == 0.5
    assert result["bins"][7] == {"bin": 7, "count": 2, "accuracy": 0.5, "mean_score": 0.75}
    assert result["bins"][0]["accuracy"] is None


def test_wilson_and_zero_accepted():
    z2 = cal.PROTOCOL["wilson_z"] ** 2
    assert cal.wilson_upper(0, 30) == pytest.approx(z2 / (30 + z2))
    assert cal.wilson_upper(10, 10) == 1
    assert cal.wilson_upper(0, 0) is None
    risk = cal.selective([[0.5, 0.5]], np.array([0]), None)
    assert risk["selective_error"] is None and risk["wilson_upper"] is None
    assert risk["review_rate"] == 1 and risk["accepted"] == 0
    assert cal.selective([[0.9, 0.1]], np.array([0]), 0.9)["accepted"] == 1
    with pytest.raises(ValueError):
        cal.wilson_upper(2, 1)


def test_selection_rule_and_deterministic_ties_review_all():
    values = np.tile([0.9, 0.1], (40, 1))
    selected, curve = cal.select_policy(values, np.zeros(40, dtype=int))
    assert selected["threshold"] == 0 and selected["accepted"] == 40
    assert len(curve) == 101 and [c["threshold"] for c in curve] == cal.PROTOCOL["thresholds"]
    failed, _ = cal.select_policy(values, np.ones(40, dtype=int))
    assert failed["threshold"] is None and failed["selective_error"] is None
    tiny, _ = cal.select_policy(values[:29], np.zeros(29, dtype=int))
    assert tiny["accepted"] == 0


def test_validation_reader_only_opens_two_approved_files(tmp_path, monkeypatch):
    import csv
    import io

    rows = [{"source_row_id": i, "text": f"fictional {i}", "category": "alpha"} for i in range(4)]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=["source_row_id", "text", "category"], lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode()
    manifest = ds.json_bytes(
        {
            "preparation": {
                "train": {"row_ids": [5, 6]},
                "validation": {"row_ids": [0, 1, 2, 3], "category_counts": {"alpha": 4}},
            }
        }
    )
    (tmp_path / "manifest.json").write_bytes(manifest)
    (tmp_path / "validation.csv").write_bytes(content)
    contract = SimpleNamespace(
        labels=["alpha"],
        provenance={
            "manifest_sha256": ds.sha256(manifest),
            "processed_sha256": {"validation": ds.sha256(content)},
            "validation_rows": 4,
        },
    )
    original = ds.read_bytes
    calls = []

    def guarded(path):
        assert path.name in {"manifest.json", "validation.csv"}
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(ds, "read_bytes", guarded)
    assert len(calibrate.load_validation(tmp_path, contract)) == 4
    assert calls == ["manifest.json", "validation.csv"]
    (tmp_path / "validation.csv").write_bytes(b"tampered fictional")
    with pytest.raises(ValueError, match="identity mismatch"):
        calibrate.load_validation(tmp_path, contract)


def test_cli_audit_once_and_no_redevelopment(experiment, tmp_path, monkeypatch):
    model, metadata, rows, _ = experiment
    monkeypatch.setattr(calibrate, "load_artifacts", lambda *a, **k: (model, None, metadata, None))
    monkeypatch.setattr(calibrate, "load_validation", lambda *a: rows)
    monkeypatch.setattr(calibrate, "plot_development", lambda *a: None)
    args = ["--policy-dir", str(tmp_path / "policy"), "--trust-local-artifacts"]
    assert calibrate.main(["develop", *args]) == 0
    frozen, _ = store.load_record(tmp_path / "policy", "draft.json", "draft", store.Frozen)
    assert calibrate.main(["audit", *args]) == 0
    assert calibrate.main(["audit", *args]) == 1
    assert calibrate.main(["develop", *args]) == 1
    completed, _ = store.load_policy(tmp_path / "policy", metadata)
    assert completed.frozen == frozen
