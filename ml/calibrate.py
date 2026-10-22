"""Explicit develop -> audit CLI. Never refit the base model or open official test data."""

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from ml import calibration as cal
from ml import dataset as ds
from ml import review_policy as store
from ml.model_support import DEFAULT_ARTIFACTS, load_artifacts, serving_contract


def load_validation(processed: Path, contract):
    """Only two explicit files: pinned Phase-1 manifest and validation CSV."""
    manifest_bytes = ds.read_bytes(processed / "manifest.json")
    content = ds.read_bytes(processed / "validation.csv")
    if (
        ds.sha256(manifest_bytes) != contract.provenance["manifest_sha256"]
        or ds.sha256(content) != contract.provenance["processed_sha256"]["validation"]
    ):
        raise ValueError("Phase-1 validation identity mismatch")
    manifest = json.loads(manifest_bytes)
    reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    if reader.fieldnames != ["source_row_id", "text", "category"]:
        raise ValueError("Invalid prepared validation schema")
    rows = list(reader)
    ids = [int(r["source_row_id"]) for r in rows]
    labels = [r["category"] for r in rows]
    prep = manifest["preparation"]
    if (
        ids != sorted(set(ids))
        or ids != prep["validation"]["row_ids"]
        or set(ids) & set(prep["train"]["row_ids"])
        or len(ids) != contract.provenance["validation_rows"]
        or dict(sorted(Counter(labels).items())) != prep["validation"]["category_counts"]
        or sorted(set(labels)) != contract.labels
        or any(not r["text"].strip() for r in rows)
    ):
        raise ValueError("Validation membership mismatch")
    return rows


def split_manifest(rows):
    ids = [int(r["source_row_id"]) for r in rows]
    parts = cal.partition(ids, [r["category"] for r in rows])
    manifest = {}
    for name, indices in parts.items():
        source_ids = [ids[i] for i in indices]
        manifest[name] = {
            "source_ids": source_ids,
            "sha256": ds.sha256(ds.json_bytes(source_ids)),
            "class_support": dict(sorted(Counter(rows[i]["category"] for i in indices).items())),
        }
    return parts, manifest


def scores_for(model, rows, indices, labels):
    """Transform only the designated subset; no fitting and no audit access in develop."""
    texts = [rows[i]["text"] for i in indices]
    with threadpool_limits(limits=1):
        scores = cal.matrix(model.decision_function(texts))
        raw = cal.probabilities(model.predict_proba(texts))
    if (
        scores.shape != (len(indices), len(labels))
        or list(model.classes_) != labels
        or not np.allclose(cal.scaled(scores, 1.0), raw, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("Decision scores/class order do not match multinomial probabilities")
    y = np.array([labels.index(rows[i]["category"]) for i in indices], dtype=int)
    return scores, raw, y


def compare(raw, calibrated, y):
    if not np.array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1)):
        raise ValueError("Temperature changed deterministic top category")
    return {"raw": cal.metrics(raw, y), "calibrated": cal.metrics(calibrated, y)}


def develop(model, metadata, rows) -> store.Frozen:
    parts, manifest = split_manifest(rows)
    scores, raw, y = scores_for(model, rows, parts["calibration"], metadata["labels"])
    temperature = cal.fit_temperature(scores, y)
    fit_metrics = compare(raw, cal.scaled(scores, temperature), y)
    scores, raw, y = scores_for(model, rows, parts["selection"], metadata["labels"])
    calibrated = cal.scaled(scores, temperature)
    selected, curve = cal.select_policy(calibrated, y)
    return store.Frozen.model_validate(
        {
            "format_version": 1,
            "protocol": cal.PROTOCOL,
            "binding": store.binding(metadata),
            "manifest": manifest,
            "temperature": temperature,
            "calibration": fit_metrics,
            "selection": compare(raw, calibrated, y),
            "selected": selected,
            "curve": curve,
        }
    )


def audit(model, metadata, rows, frozen: store.Frozen) -> store.Completed:
    parts, manifest = split_manifest(rows)
    if (
        store.binding(metadata) != frozen.binding.model_dump()
        or manifest != frozen.manifest.model_dump()
    ):
        raise ValueError("Frozen development identity mismatch")
    scores, raw, y = scores_for(model, rows, parts["audit"], metadata["labels"])
    calibrated = cal.scaled(scores, frozen.temperature)
    return store.Completed.model_validate(
        {
            "format_version": 1,
            "frozen_sha256": ds.sha256(ds.json_bytes(frozen.model_dump())),
            "frozen": frozen.model_dump(),
            "audit": compare(raw, calibrated, y),
            "audit_risk": cal.selective(calibrated, y, frozen.selected.threshold),
        }
    )


def plot_development(root: Path, frozen):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    curve = frozen.curve
    axes[0].plot([r.coverage for r in curve], [r.selective_error for r in curve])
    axes[0].set(
        xlabel="Selection coverage",
        ylabel="Accepted-item error",
        title=f"Selection only (n={frozen.selection.raw.rows})",
    )
    axes[1].plot([r.threshold for r in curve], [r.review_rate for r in curve])
    axes[1].set(xlabel="Threshold", ylabel="Human-review fraction", title="Development trade-off")
    fig.savefig(root / "development.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["develop", "audit"])
    parser.add_argument("--processed-dir", type=Path, default=ds.DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--policy-dir", type=Path, default=store.DEFAULT_POLICIES)
    parser.add_argument("--trust-local-artifacts", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.policy_dir
        # Prevent accidental overwrites/retuning of this experiment, including after audit.
        if args.step == "develop" and (
            (root / "draft.json").exists() or (root / "current.json").exists()
        ):
            raise ValueError("Development already frozen; preserve this experiment")
        for protected in (args.processed_dir, args.artifact_dir):
            a, b = root.resolve(), protected.resolve()
            if a == b or a in b.parents or b in a.parents:
                raise ValueError("Policy storage must be separate from models and input data")
        contract = serving_contract()
        model, _, metadata, _ = load_artifacts(
            args.artifact_dir, contract, trusted=args.trust_local_artifacts
        )
        rows = load_validation(args.processed_dir, contract)
        if args.step == "develop":
            record = develop(model, metadata, rows)
            version = store.publish(root, record)
            plot_development(root, record)
            summary = {
                "state": "frozen, audit not yet run",
                "temperature": record.temperature,
                "selection": record.selected.model_dump(),
            }
        else:
            frozen, folder = store.load_record(root, "draft.json", "draft", store.Frozen)
            # Exclusive durable claim BEFORE computing audit outcomes. Failures remain claimed;
            # no retries/retuning masquerading as an untouched one-time audit.
            with (folder / "audit.claim").open("x", encoding="utf-8") as handle:
                handle.write("Audit attempted; preserve frozen experiment.\n")
                handle.flush()
            record = audit(model, metadata, rows, frozen)
            version = store.publish(root, record)
            summary = {"state": "audited, no retuning", "audit": record.audit_risk.model_dump()}
        print(ds.json_bytes({"version": version, **summary}).decode(), end="")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # CLI messages are operator-only; never print model input texts or exceptions from inference.
        print(
            f"Calibration step failed ({type(exc).__name__}); inspect trusted inputs and frozen state.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
