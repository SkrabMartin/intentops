"""Synthetic, explicitly fictional ML inputs; never use local BANKING77 in CI."""

import csv
import io
import socket
from pathlib import Path

import pytest

from ml import dataset as ds
from ml.data_config import SourceSpec
from ml.model_support import load_prepared


@pytest.fixture
def block_model_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("ML fixture tests may not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def model_data(tmp_path: Path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    topics = {
        "alpha_case": "orchard apple fruit",
        "Beta?": "spaceship galaxy rocket",
        "gamma_case": "symphony violin music",
        "delta_case": "ocean coral reef",
    }
    rows = [
        {"text": f"fictional {words} example {i}", "category": label}
        for label, words in topics.items()
        for i in range(12)
    ]
    _, val_ids = ds.split_rows(rows)
    for i in val_ids:
        rows[i]["text"] += " validationonlytoken"
    specs = []
    for filename, values in [
        ("train.csv", rows),
        (
            "test.csv",
            [
                {"text": f"fictional reserved {label} {i}", "category": label}
                for label in topics
                for i in range(2)
            ],
        ),
    ]:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["text", "category"])
        writer.writerows((r["text"], r["category"]) for r in values)
        content = stream.getvalue().encode()
        (raw / filename).write_bytes(content)
        specs.append(SourceSpec(filename, ds.sha256(content), len(values), frozenset(topics)))
    specs = tuple(specs)
    ds.prepare(raw, processed, specs=specs)
    return load_prepared(processed, raw, specs), raw, processed, specs
