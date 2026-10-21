"""Explicit BANKING77 fetch and offline validation, preparation, and EDA.

Importing this module never accesses data or the network. Only fetch opens HTTP.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from ml.data_config import METADATA, REVISION, SOURCE, SOURCES, SourceSpec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
MAX_FILE_BYTES = 10 * 1024 * 1024
Row = dict[str, str]


class DataError(ValueError):
    """Actionable data failure; error messages never include customer text."""


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_bytes(path: Path) -> bytes:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DataError(f"{path.name}: exceeds {MAX_FILE_BYTES} byte limit.")
    with path.open("rb") as handle:
        content = handle.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise DataError(f"{path.name}: file exceeded size limit while reading.")
    return content


def check_hash(content: bytes, spec: SourceSpec) -> str:
    observed = sha256(content)
    if observed != spec.sha256:
        raise DataError(
            f"{spec.filename}: SHA-256 mismatch; expected {spec.sha256}, observed {observed}. "
            "File preserved. Inspect and move it aside manually before retrying fetch; "
            "never accept a revision whose bytes conflict with the approved hash."
        )
    return observed


def parse_csv(content: bytes) -> list[Row]:
    """Parse strings verbatim (including literal NA); detect rather than trim blanks."""
    try:
        reader = csv.reader(io.StringIO(content.decode("utf-8"), newline=""), strict=True)
        if next(reader, None) != ["text", "category"]:
            raise DataError("CSV header must be exactly text,category (original PolyAI format).")
        rows = []
        for line, values in enumerate(reader, start=2):
            if len(values) != 2 or any(not value.strip() for value in values):
                raise DataError(f"Invalid CSV record {line}: expected two nonempty string fields.")
            rows.append(dict(zip(("text", "category"), values, strict=True)))
        if not rows:
            raise DataError("CSV contains no data records.")
        return rows
    except (UnicodeError, csv.Error) as exc:
        raise DataError("Invalid UTF-8 encoding or malformed CSV quoting.") from exc


def validate_rows(rows: list[Row], spec: SourceSpec) -> None:
    if len(rows) != spec.rows:
        raise DataError(f"{spec.filename}: expected {spec.rows} rows, observed {len(rows)}.")
    categories = frozenset(row["category"] for row in rows)
    if categories != spec.categories:
        raise DataError(
            f"{spec.filename}: category inventory mismatch "
            f"({len(spec.categories - categories)} missing, {len(categories - spec.categories)} unexpected)."
        )


def describe(rows: list[Row]) -> dict:
    texts = [row["text"] for row in rows]
    counts = Counter(row["category"] for row in rows)
    text_counts = Counter(texts)
    pairs = Counter((r["text"], r["category"]) for r in rows)
    text_labels = defaultdict(set)
    for row in rows:
        text_labels[row["text"]].add(row["category"])
    lengths = pd.Series([len(text) for text in texts], dtype="int64")
    return {
        "rows": len(rows),
        "schema": {"text": "string", "category": "string"},
        "null_or_blank_fields": 0,
        "category_count": len(counts),
        "category_counts": dict(sorted(counts.items())),
        "class_size_min": min(counts.values()),
        "class_size_max": max(counts.values()),
        "class_imbalance_ratio": max(counts.values()) / min(counts.values()),
        "duplicate_text_excess_rows": sum(c - 1 for c in text_counts.values()),
        "duplicate_record_excess_rows": sum(c - 1 for c in pairs.values()),
        "conflicting_label_text_count": sum(len(labels) > 1 for labels in text_labels.values()),
        "text_length_characters": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "p95": float(lengths.quantile(0.95)),
        },
    }


def overlap(left: list[Row], right: list[Row]) -> dict:
    common = {r["text"] for r in left} & {r["text"] for r in right}
    return {
        "identical_text_count": len(common),
        "left_row_ids": [i for i, row in enumerate(left) if row["text"] in common],
        "right_row_ids": [i for i, row in enumerate(right) if row["text"] in common],
    }


def verify(
    raw_dir: Path = DEFAULT_RAW, specs: tuple[SourceSpec, ...] = SOURCES
) -> tuple[dict, dict]:
    """Verify exact bytes, schema and inventory. No HTTP calls; specs injectable for tests."""
    datasets, files = {}, {}
    for spec in specs:
        content = read_bytes(raw_dir / spec.filename)
        digest = check_hash(content, spec)
        rows = parse_csv(content)
        validate_rows(rows, spec)
        datasets[spec.filename] = rows
        files[spec.filename] = {"sha256": digest, "source_url": spec.url, **describe(rows)}
    manifest = {
        "format_version": 1,
        "dataset": "BANKING77",
        "license": "CC-BY-4.0",
        "source_repository": SOURCE,
        "source_revision": REVISION,
        "hash_metadata": METADATA,
        "files": files,
        "categories": sorted(specs[0].categories),
        "official_train_test_overlap": overlap(datasets["train.csv"], datasets["test.csv"]),
    }
    return datasets, manifest


def fetch(raw_dir: Path = DEFAULT_RAW, specs: tuple[SourceSpec, ...] = SOURCES) -> None:
    """Only network entry point. Stage and hash all missing files before publication."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if (raw_dir / spec.filename).exists():
            check_hash(read_bytes(raw_dir / spec.filename), spec)
    staged = []
    try:
        with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as client:
            for spec in specs:
                destination = raw_dir / spec.filename
                if destination.exists():
                    continue
                with tempfile.NamedTemporaryFile(
                    dir=raw_dir, prefix=".download-", delete=False
                ) as temp:
                    temp_path = Path(temp.name)
                    staged.append((temp_path, destination))
                    size = 0
                    with client.stream("GET", spec.url) as response:
                        response.raise_for_status()
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > MAX_FILE_BYTES:
                                raise DataError(f"{spec.filename}: download exceeds size limit.")
                            temp.write(chunk)
                check_hash(read_bytes(temp_path), spec)
                validate_rows(parse_csv(read_bytes(temp_path)), spec)
        for temp_path, destination in staged:
            # Atomic no-clobber publication on NTFS/Linux; never replace a concurrent file.
            try:
                os.link(temp_path, destination)
            except FileExistsError:
                spec = next(s for s in specs if s.filename == destination.name)
                check_hash(read_bytes(destination), spec)
    finally:
        for temp_path, _ in staged:
            temp_path.unlink(missing_ok=True)


def split_rows(
    rows: list[Row], seed: int = 42, val_fraction: float = 0.20
) -> tuple[list[int], list[int]]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise DataError("Seed must be an integer between 0 and 4294967295.")
    if (
        isinstance(val_fraction, bool)
        or not math.isfinite(val_fraction)
        or not 0 < val_fraction < 1
    ):
        raise DataError("Validation fraction must be finite and strictly between 0 and 1.")
    try:
        train_ids, val_ids = train_test_split(
            list(range(len(rows))),
            test_size=val_fraction,
            random_state=seed,
            stratify=[row["category"] for row in rows],
        )
    except ValueError as exc:
        raise DataError(
            "Cannot stratify: each category needs enough rows and space in both partitions."
        ) from exc
    expected = {row["category"] for row in rows}
    if any({rows[i]["category"] for i in ids} != expected for ids in (train_ids, val_ids)):
        raise DataError("Requested split does not represent every category in both partitions.")
    return sorted(train_ids), sorted(val_ids)


def json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prepare-", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        temp_path.write_bytes(content)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def partition_bytes(rows: list[Row], ids: list[int]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["source_row_id", "text", "category"])
    writer.writerows((i, rows[i]["text"], rows[i]["category"]) for i in ids)
    return output.getvalue().encode("utf-8")


def prepare(
    raw_dir: Path = DEFAULT_RAW,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = 42,
    val_fraction: float = 0.20,
    specs: tuple[SourceSpec, ...] = SOURCES,
) -> dict:
    raw, output = raw_dir.resolve(), output_dir.resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        raise DataError("Raw and processed directories must be separate, non-nested locations.")
    datasets, manifest = verify(raw, specs)
    rows = datasets["train.csv"]
    train_ids, val_ids = split_rows(rows, seed, val_fraction)
    train_content, val_content = partition_bytes(rows, train_ids), partition_bytes(rows, val_ids)
    manifest["preparation"] = {
        "algorithm": "sklearn.model_selection.train_test_split(stratify=category); sorted source IDs",
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "validation_fraction": val_fraction,
        "source": "official train.csv only",
        "source_row_id": "zero-based CSV data-record index",
        "train": {
            "rows": len(train_ids),
            "row_ids": train_ids,
            "sha256": sha256(train_content),
            "category_counts": dict(
                sorted(Counter(rows[i]["category"] for i in train_ids).items())
            ),
        },
        "validation": {
            "rows": len(val_ids),
            "row_ids": val_ids,
            "sha256": sha256(val_content),
            "category_counts": dict(sorted(Counter(rows[i]["category"] for i in val_ids).items())),
        },
        "train_validation_overlap": overlap(
            [rows[i] for i in train_ids], [rows[i] for i in val_ids]
        ),
        "duplicate_policy": "Retain all rows, flag exact-text overlap; no grouping or normalization.",
        "official_test_policy": "Verified separately; never included in derived partitions or copied.",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "train.csv", train_content)
    atomic_write(output / "validation.csv", val_content)
    # Manifest published last; hashes detect a partially interrupted rerun.
    atomic_write(output / "manifest.json", json_bytes(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "verify", "prepare", "summary"):
        sub = commands.add_parser(name)
        sub.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
        if name == "prepare":
            sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch(args.raw_dir)
            print("Official files fetched or already verified. Run verify next.")
        else:
            if args.command == "prepare":
                result = prepare(args.raw_dir, args.output_dir, args.seed, args.val_fraction)
            else:
                _, result = verify(args.raw_dir)
            print(json_bytes(result).decode("utf-8"), end="")
    except (DataError, OSError, httpx.HTTPError) as exc:
        print(
            f"Data error: {exc}\nSupply official files locally or explicitly run fetch.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
