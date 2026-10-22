"""Bounded JSON-only policy storage. Integrity is not producer authentication."""

import json
import math
import re
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml import calibration as cal
from ml import dataset as ds

DEFAULT_POLICIES = ds.ROOT / "policies"
MAX_POLICY_BYTES = 2 * 1024 * 1024


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Bin(Record):
    bin: int = Field(ge=0, le=9)
    count: int = Field(ge=0)
    accuracy: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    mean_score: float | None = Field(ge=0, le=1, allow_inf_nan=False)


class Metrics(Record):
    rows: int = Field(gt=0)
    accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    log_loss: float = Field(ge=0, allow_inf_nan=False)
    brier: float = Field(ge=0, le=2, allow_inf_nan=False)
    ece: float = Field(ge=0, le=1, allow_inf_nan=False)
    bins: list[Bin] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def consistent(self):
        if sum(b.count for b in self.bins) != self.rows or [b.bin for b in self.bins] != list(
            range(10)
        ):
            raise ValueError("Invalid reliability table")
        for b in self.bins:
            if (b.count == 0) != (b.accuracy is None and b.mean_score is None):
                raise ValueError("Invalid empty bin")
            if b.count and (b.accuracy is None or b.mean_score is None):
                raise ValueError("Missing supported bin statistics")
        ece = sum(
            b.count / self.rows * abs(b.accuracy - b.mean_score) for b in self.bins if b.count
        )
        accuracy = sum(b.count / self.rows * b.accuracy for b in self.bins if b.count)
        if not math.isclose(ece, self.ece, abs_tol=1e-12) or not math.isclose(
            accuracy, self.accuracy, abs_tol=1e-12
        ):
            raise ValueError("Reliability statistics mismatch")
        return self


class Risk(Record):
    threshold: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    rows: int = Field(gt=0)
    accepted: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    review_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    errors: int = Field(ge=0)
    selective_error: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    wilson_upper: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent(self):
        if self.accepted + self.reviewed != self.rows or self.errors > self.accepted:
            raise ValueError("Invalid selective counts")
        expected = {
            "coverage": self.accepted / self.rows,
            "review_rate": 1 - self.accepted / self.rows,
            "selective_error": self.errors / self.accepted if self.accepted else None,
            "wilson_upper": cal.wilson_upper(self.errors, self.accepted),
        }
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("Invalid selective statistics")
        if self.threshold is None and self.accepted:
            raise ValueError("Review-all cannot accept examples")
        return self


class Membership(Record):
    source_ids: list[int] = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    class_support: dict[str, int]

    @model_validator(mode="after")
    def consistent(self):
        if (
            self.source_ids != sorted(set(self.source_ids))
            or min(self.source_ids) < 0
            or self.sha256 != ds.sha256(ds.json_bytes(self.source_ids))
            or sum(self.class_support.values()) != len(self.source_ids)
            or any(v <= 0 for v in self.class_support.values())
        ):
            raise ValueError("Invalid split membership")
        return self


class Manifest(Record):
    calibration: Membership
    selection: Membership
    audit: Membership

    @model_validator(mode="after")
    def consistent(self):
        groups = [set(getattr(self, n).source_ids) for n in ("calibration", "selection", "audit")]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Overlapping development partitions")
        if not (
            self.calibration.class_support.keys()
            == self.selection.class_support.keys()
            == self.audit.class_support.keys()
        ):
            raise ValueError("Stratified labels missing")
        return self


class Binding(Record):
    model_version: str
    class_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    versions: dict[str, str]
    provenance: dict


def binding(metadata: dict) -> dict:
    return {
        "model_version": metadata["model_version"],
        "class_order_sha256": ds.sha256(ds.json_bytes(metadata["labels"])),
        "versions": metadata["versions"],
        "provenance": metadata["data_provenance"],
    }


class Comparison(Record):
    raw: Metrics
    calibrated: Metrics

    @model_validator(mode="after")
    def consistent(self):
        if self.raw.rows != self.calibrated.rows or self.raw.accuracy != self.calibrated.accuracy:
            raise ValueError("Temperature scaling changed top category")
        return self


class Frozen(Record):
    format_version: Literal[1]
    protocol: dict
    binding: Binding
    manifest: Manifest
    temperature: float = Field(ge=0.05, le=20, allow_inf_nan=False)
    calibration: Comparison
    selection: Comparison
    selected: Risk
    curve: list[Risk] = Field(min_length=101, max_length=101)

    @model_validator(mode="after")
    def consistent(self):
        if self.protocol != cal.PROTOCOL:
            raise ValueError("Unsupported calibration protocol")
        if [r.threshold for r in self.curve] != cal.PROTOCOL["thresholds"]:
            raise ValueError("Wrong threshold grid")
        for name in ("calibration", "selection"):
            if getattr(self, name).raw.rows != len(getattr(self.manifest, name).source_ids):
                raise ValueError("Evidence count differs from membership")
        if any(r.rows != self.selection.raw.rows for r in [*self.curve, self.selected]):
            raise ValueError("Selection count mismatch")
        if any(r.accuracy != self.selection.raw.accuracy for r in [*self.curve, self.selected]):
            raise ValueError("Selection accuracy mismatch")
        if any(a.accepted < b.accepted for a, b in zip(self.curve, self.curve[1:])):
            raise ValueError("Coverage must decrease with threshold")
        eligible = [
            r
            for r in self.curve
            if r.accepted >= cal.PROTOCOL["min_accepted"]
            and r.wilson_upper <= cal.PROTOCOL["risk_budget"]
        ]
        if eligible:
            expected = min(eligible, key=lambda r: (-r.accepted, r.threshold))
            if self.selected != expected:
                raise ValueError("Policy differs from declared selection rule")
        elif self.selected.threshold is not None or self.selected.accepted:
            raise ValueError("No eligible candidate requires review-all")
        return self


class Completed(Record):
    format_version: Literal[1]
    frozen_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen: Frozen
    audit: Comparison
    audit_risk: Risk

    @model_validator(mode="after")
    def consistent(self):
        if (
            self.frozen_sha256 != ds.sha256(ds.json_bytes(self.frozen.model_dump()))
            or self.audit.raw.rows != len(self.frozen.manifest.audit.source_ids)
            or self.audit_risk.rows != self.audit.raw.rows
            or self.audit_risk.accuracy != self.audit.raw.accuracy
            or self.audit_risk.threshold != self.frozen.selected.threshold
        ):
            raise ValueError("Audit differs from frozen development policy")
        return self


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON")
    if isinstance(value, dict):
        for v in value.values():
            _finite(v)
    if isinstance(value, list):
        for v in value:
            _finite(v)


def read_json(path: Path):
    if path.is_symlink():
        raise ValueError("Expected regular policy file")
    if not path.exists():
        raise FileNotFoundError("Policy file absent")
    if not path.is_file():
        raise ValueError("Expected regular policy file")
    with path.open("rb") as handle:
        content = handle.read(MAX_POLICY_BYTES + 1)
    if len(content) > MAX_POLICY_BYTES:
        raise ValueError("Policy size limit exceeded")
    value = json.loads(content, object_pairs_hook=_pairs)
    _finite(value)
    if not isinstance(value, dict):
        raise TypeError("Expected JSON object")
    return value, content


def load_record(root: Path, pointer_name: str, prefix: str, record_type):
    if root.is_symlink():
        raise ValueError("Policy root cannot be a symlink")
    pointer, _ = read_json(root / pointer_name)
    if set(pointer) != {"run", "sha256"} or not re.fullmatch(
        prefix + r"-[0-9a-f]{32}", pointer["run"]
    ):
        raise ValueError("Invalid policy pointer")
    folder = root / pointer["run"]
    if folder.is_symlink() or folder.resolve().parent != root.resolve():
        raise ValueError("Policy run outside selected root")
    allowed = {"record.json"} | ({"audit.claim"} if prefix == "draft" else set())
    if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in folder.iterdir()):
        raise ValueError("Unexpected policy inventory")
    record, content = read_json(folder / "record.json")
    if ds.sha256(content) != pointer["sha256"]:
        raise ValueError("Policy checksum mismatch")
    return record_type.model_validate(record), folder


def publish(root: Path, record: Frozen | Completed) -> str:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("Policy root cannot be a symlink")
    content = ds.json_bytes(record.model_dump())
    if len(content) > MAX_POLICY_BYTES:
        raise ValueError("Policy size limit exceeded")
    prefix = "draft" if isinstance(record, Frozen) else "policy"
    name = prefix + "-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=root) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        (stage / "record.json").write_bytes(content)
        stage.rename(root / name)
    # Pointer is last: interrupted staging never replaces a valid current policy.
    ds.atomic_write(
        root / ("draft.json" if prefix == "draft" else "current.json"),
        ds.json_bytes({"run": name, "sha256": ds.sha256(content)}),
    )
    return "review-" + ds.sha256(content)[:16]


def load_policy(root: Path, metadata: dict) -> tuple[Completed, str]:
    completed, _ = load_record(root, "current.json", "policy", Completed)
    if completed.frozen.binding.model_dump() != binding(metadata):
        raise ValueError("Policy/model identity mismatch")
    memberships = [
        getattr(completed.frozen.manifest, name) for name in ("calibration", "selection", "audit")
    ]
    if sum(len(m.source_ids) for m in memberships) != metadata["data_provenance"][
        "validation_rows"
    ] or any(sorted(m.class_support) != metadata["labels"] for m in memberships):
        raise ValueError("Policy validation inventory mismatch")
    version = "review-" + ds.sha256(ds.json_bytes(completed.model_dump()))[:16]
    return completed, version
