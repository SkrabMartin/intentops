"""Data contracts use fictional local CSVs; no BANKING77 download or service needed."""

import ast
import csv
import io
import json
import socket
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from ml import dataset as ds
from ml.data_config import CATEGORIES, SOURCES, SourceSpec


def csv_bytes(rows: list[tuple[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["text", "category"])
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args, **kwargs):
        raise AssertionError("Dataset tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def source(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    specs = []
    for name, count in (("train.csv", 40), ("test.csv", 8)):
        rows = [
            (f"fictional {name} {label} {i}", label)
            for label in ("A?", "Mixed_case")
            for i in range(count)
        ]
        content = csv_bytes(rows)
        (raw / name).write_bytes(content)
        specs.append(
            SourceSpec(name, ds.sha256(content), len(rows), frozenset(("A?", "Mixed_case")))
        )
    return raw, tuple(specs)


def test_parser_preserves_original_strings() -> None:
    content = csv_bytes([(' NA, "fictional"\n text ', "Mixed_case?")])
    assert ds.parse_csv(content) == [
        {"text": ' NA, "fictional"\n text ', "category": "Mixed_case?"}
    ]


@pytest.mark.parametrize(
    "content",
    [
        b"text,label\nx,a\n",
        b"category,text\na,x\n",
        b"text,category,extra\nx,a,z\n",
        b"text,category\n,a\n",
        b"text,category\nx, \n",
        b"text,category\nx\n",
        b"text,category\nx,a,z\n",
        b'text,category\n"unterminated,a\n',
        b"text,category\n\xff,a\n",
        b"text,category\n",
        b"text,category\n\n",
    ],
)
def test_bad_csv(content: bytes) -> None:
    with pytest.raises(ds.DataError):
        ds.parse_csv(content)


def test_verify_fixture(source) -> None:
    raw, specs = source
    data, manifest = ds.verify(raw, specs)
    assert len(data["train.csv"]) == 80
    assert manifest["files"]["test.csv"]["rows"] == 16
    assert manifest["categories"] == ["A?", "Mixed_case"]
    assert manifest["official_train_test_overlap"]["identical_text_count"] == 0


def test_missing_file(source) -> None:
    raw, specs = source
    (raw / "test.csv").unlink()
    with pytest.raises(FileNotFoundError):
        ds.verify(raw, specs)


def test_bad_hash_preserved(source) -> None:
    raw, specs = source
    (raw / "train.csv").write_bytes(b"unapproved fixture")
    with pytest.raises(ds.DataError, match="SHA-256 mismatch"):
        ds.fetch(raw, specs)
    assert (raw / "train.csv").read_bytes() == b"unapproved fixture"


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("rows", 999, "expected 999 rows"),
        ("categories", frozenset(("unexpected",)), "category inventory mismatch"),
    ],
)
def test_contract_rejects_wrong_counts_or_inventory(source, field, value, error) -> None:
    raw, specs = source
    bad = (replace(specs[0], **{field: value}), specs[1])
    with pytest.raises(ds.DataError, match=error):
        ds.verify(raw, bad)


def test_production_contract_cannot_accept_fixture(source) -> None:
    raw, _ = source
    assert len(CATEGORIES) == 77
    assert "Refund_not_showing_up" in CATEGORIES and "reverted_card_payment?" in CATEGORIES
    assert SOURCES[0].rows == 10003 and SOURCES[1].rows == 3080
    with pytest.raises(ds.DataError, match="SHA-256 mismatch"):
        ds.verify(raw)


def mock_download(monkeypatch, content: bytes, status: int = 200) -> None:
    original = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=content))
    monkeypatch.setattr(ds.httpx, "Client", lambda **kwargs: original(transport=transport))


def test_failed_fetch_preserves_verified_file(source, monkeypatch) -> None:
    raw, specs = source
    before = (raw / "train.csv").read_bytes()
    (raw / "test.csv").unlink()
    mock_download(monkeypatch, b"fake content")
    with pytest.raises(ds.DataError, match="SHA-256 mismatch"):
        ds.fetch(raw, specs)
    assert (raw / "train.csv").read_bytes() == before
    assert not (raw / "test.csv").exists()
    assert not list(raw.glob(".download-*"))


def test_valid_mock_download_is_idempotent(source, monkeypatch) -> None:
    raw, specs = source
    content = (raw / "test.csv").read_bytes()
    (raw / "test.csv").unlink()
    mock_download(monkeypatch, content)
    ds.fetch(raw, specs)
    ds.fetch(raw, specs)
    assert (raw / "test.csv").read_bytes() == content
    assert not list(raw.glob(".download-*"))


@pytest.mark.parametrize(
    "status, content, limit", [(503, b"unavailable", 1000), (200, b"x" * 101, 100)]
)
def test_fetch_http_and_size_failures(source, monkeypatch, status, content, limit) -> None:
    raw, specs = source
    (raw / "test.csv").unlink()
    # Size limit must still accommodate the pre-existing training fixture.
    target = raw / "missing"
    mock_download(monkeypatch, content, status)
    monkeypatch.setattr(ds, "MAX_FILE_BYTES", limit)
    with pytest.raises((ds.DataError, httpx.HTTPStatusError)):
        ds.fetch(target, (specs[1],))
    assert not (target / "test.csv").exists()
    assert not list(target.glob(".download-*"))


def test_preparation_reproducible_and_separate(source, tmp_path) -> None:
    raw, specs = source
    before = {p.name: p.read_bytes() for p in raw.iterdir()}
    output = tmp_path / "processed"
    result = ds.prepare(raw, output, specs=specs)
    first = {p.name: p.read_bytes() for p in output.iterdir()}
    assert ds.prepare(raw, output, specs=specs) == result
    assert {p.name: p.read_bytes() for p in output.iterdir()} == first
    assert {p.name: p.read_bytes() for p in raw.iterdir()} == before
    split = result["preparation"]
    a, b = set(split["train"]["row_ids"]), set(split["validation"]["row_ids"])
    assert not a & b and a | b == set(range(80))
    assert split["train"]["rows"] == 64 and split["validation"]["rows"] == 16
    assert split["validation"]["category_counts"] == {"A?": 8, "Mixed_case": 8}
    assert (
        split["validation"]["row_ids"]
        != ds.prepare(raw, tmp_path / "other", seed=43, specs=specs)["preparation"]["validation"][
            "row_ids"
        ]
    )
    assert not (output / "test.csv").exists()
    for name in ("train", "validation"):
        content = (output / f"{name}.csv").read_bytes()
        assert ds.sha256(content) == split[name]["sha256"]
        assert "fictional test.csv" not in content.decode()
        assert b"\r\n" not in content


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True, "42"])
def test_invalid_seed(seed) -> None:
    with pytest.raises(ds.DataError, match="Seed"):
        ds.split_rows([], seed=seed)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_fraction(fraction) -> None:
    with pytest.raises(ds.DataError, match="Validation fraction"):
        ds.split_rows([], val_fraction=fraction)


def test_insufficient_class_size() -> None:
    with pytest.raises(ds.DataError, match="Cannot stratify"):
        ds.split_rows([{"text": "fictional", "category": "a"}, {"text": "other", "category": "b"}])


def test_duplicate_and_conflict_counts() -> None:
    a = {"text": "fictional duplicate", "category": "a"}
    b = {"text": "fictional duplicate", "category": "b"}
    summary = ds.describe([a, a, b])
    assert summary["duplicate_text_excess_rows"] == 2
    assert summary["duplicate_record_excess_rows"] == 1
    assert summary["conflicting_label_text_count"] == 1
    assert ds.overlap([a], [b])["identical_text_count"] == 1


def test_no_raw_output_location(source) -> None:
    raw, specs = source
    for output in (raw, raw / "processed", raw.parent):
        with pytest.raises(ds.DataError, match="separate"):
            ds.prepare(raw, output, specs=specs)


def test_offline_cli_verify_prepare_summary(source, tmp_path, monkeypatch, capsys) -> None:
    raw, specs = source
    original_verify = ds.verify
    monkeypatch.setattr(ds, "verify", lambda path, selected=specs: original_verify(path, selected))
    original_prepare = ds.prepare
    monkeypatch.setattr(
        ds,
        "prepare",
        lambda path, out, seed, fraction: original_prepare(path, out, seed, fraction, specs),
    )
    for command in ("verify", "summary", "prepare"):
        args = [command, "--raw-dir", str(raw)]
        if command == "prepare":
            args += ["--output-dir", str(tmp_path / "processed")]
        assert ds.main(args) == 0
        assert json.loads(capsys.readouterr().out)["files"]["train.csv"]["rows"] == 80


def test_cli_missing_data_nonzero(tmp_path, capsys) -> None:
    assert ds.main(["verify", "--raw-dir", str(tmp_path)]) == 1
    assert "Supply official files" in capsys.readouterr().err


def test_duplicate_overlap_is_flagged_in_preparation(source, tmp_path) -> None:
    raw, specs = source
    # Fictional repeated rows intentionally model upstream duplicates.
    content = csv_bytes(
        [("fictional duplicate", "A?")] * 40 + [("other duplicate", "Mixed_case")] * 40
    )
    (raw / "train.csv").write_bytes(content)
    specs = (replace(specs[0], sha256=ds.sha256(content)), specs[1])
    manifest = ds.prepare(raw, tmp_path / "output", specs=specs)
    assert manifest["preparation"]["train_validation_overlap"]["identical_text_count"] == 2
    assert (
        manifest["preparation"]["train"]["rows"] + manifest["preparation"]["validation"]["rows"]
        == 80
    )


def test_failed_second_download_does_not_publish_first(source, tmp_path, monkeypatch) -> None:
    raw, specs = source
    original = httpx.Client

    def response(request):
        content = (
            (raw / "train.csv").read_bytes()
            if request.url.path.endswith("train.csv")
            else b"invalid"
        )
        return httpx.Response(200, content=content)

    monkeypatch.setattr(
        ds.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(response))
    )
    target = tmp_path / "download"
    with pytest.raises(ds.DataError, match="SHA-256 mismatch"):
        ds.fetch(target, specs)
    assert list(target.iterdir()) == []


def test_bad_source_does_not_modify_existing_processed(source, tmp_path) -> None:
    raw, specs = source
    output = tmp_path / "processed"
    ds.prepare(raw, output, specs=specs)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    (raw / "test.csv").write_bytes(b"unapproved")
    with pytest.raises(ds.DataError, match="SHA-256 mismatch"):
        ds.prepare(raw, output, specs=specs)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_notebook_is_valid_code_without_saved_output() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks" / "exploration.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
