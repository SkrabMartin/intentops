"""Validation-only metrics, no training, no held-out test evaluation or routing policy."""

import argparse
import csv
import io
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from ml import dataset as ds
from ml.model_support import DEFAULT_ARTIFACTS, load_artifacts, load_prepared


def score_predictions(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    probabilities=None,
    probability_labels=None,
) -> dict:
    """Explicit full category order and zero_division=0; no source text exported."""
    if (
        not labels
        or len(set(labels)) != len(labels)
        or not y_true
        or len(y_true) != len(y_pred)
        or not set(y_true + y_pred) <= set(labels)
    ):
        raise ds.DataError(
            "Metric labels/predictions must be nonempty, aligned and in the declared inventory."
        )
    per = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "label_order": labels,
        "zero_division": 0,
        "predicted_label_distribution": dict(sorted(Counter(y_pred).items())),
        "predicted_class_count": len(set(y_pred)),
        "per_class": {
            label: {
                "precision": float(per[0][i]),
                "recall": float(per[1][i]),
                "f1": float(per[2][i]),
                "support": int(per[3][i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        result.update(
            {
                f"{average}_precision": float(precision),
                f"{average}_recall": float(recall),
                f"{average}_f1": float(f1),
            }
        )
    if probabilities is not None:
        values = np.asarray(probabilities)
        if (
            probability_labels != labels
            or values.shape != (len(y_true), len(labels))
            or not np.isfinite(values).all()
            or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-6)
        ):
            raise ds.DataError(
                "Probability columns must match the explicit label order and valid normalized rows."
            )
        k = min(3, len(labels))
        # Deterministic ties: lower index in the declared label order wins.
        top = np.argsort(-values, axis=1, kind="stable")[:, :k]
        indices = {label: i for i, label in enumerate(labels)}
        result["top_3_accuracy"] = float(
            np.mean([indices[label] in top[i] for i, label in enumerate(y_true)])
        )
        result["top_k_actual"] = k
        result["top_k_tie_policy"] = "stable declared label order"
    pairs = [
        (int(result["confusion_matrix"][i][j]), labels[i], labels[j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and result["confusion_matrix"][i][j]
    ]
    result["most_frequent_confusions"] = [
        {"count": count, "actual": actual, "predicted": predicted}
        for count, actual, predicted in sorted(pairs, key=lambda p: (-p[0], p[1], p[2]))[:15]
    ]
    return result


def evaluate_models(model, baseline, data) -> tuple[dict, dict]:
    if list(model.classes_) != data.labels or list(baseline.classes_) != data.labels:
        raise ds.DataError("Trained model label order differs from verified data.")
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        probability = model.predict_proba(data.validation_text)
        model_seconds = time.perf_counter() - start
        prediction = [data.labels[i] for i in probability.argmax(axis=1)]
        start = time.perf_counter()
        baseline_prediction = baseline.predict(np.zeros((len(data.validation_labels), 1))).tolist()
        baseline_seconds = time.perf_counter() - start
    main = score_predictions(
        data.validation_labels, prediction, data.labels, probability, list(model.classes_)
    )
    dummy = score_predictions(data.validation_labels, baseline_prediction, data.labels)
    keys = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")
    metrics = {
        "evaluation_partition": "validation only; official test not read",
        "validation_rows": len(data.validation_labels),
        "model": main,
        "baseline": dummy,
        "model_minus_baseline": {key: main[key] - dummy[key] for key in keys},
    }
    timing = {
        "method": "one complete validation batch, no warmup; TF-IDF transform plus predict_proba; threadpool limit 1",
        "validation_rows": len(data.validation_labels),
        "model_batch_seconds": model_seconds,
        "model_amortized_ms_per_message": model_seconds * 1000 / len(prediction),
        "baseline_batch_seconds": baseline_seconds,
        "warning": "Amortized batch timing is not single-request API latency or a hardware-independent guarantee.",
    }
    return metrics, timing


def report_csv(metrics: dict) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["model", "category", "precision", "recall", "f1", "support"])
    for name in ("baseline", "model"):
        for label, values in metrics[name]["per_class"].items():
            writer.writerow(
                [
                    name,
                    label,
                    values["precision"],
                    values["recall"],
                    values["f1"],
                    values["support"],
                ]
            )
    return stream.getvalue().encode("utf-8")


def save_confusion(metrics: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matrix = np.asarray(metrics["model"]["confusion_matrix"])
    labels = metrics["model"]["label_order"]
    fig, ax = plt.subplots(figsize=(20, 18), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Validation rows")
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="Actual category",
        title="BANKING77 validation confusion matrix — TF-IDF + logistic regression",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ds.DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=ds.DEFAULT_RAW)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--trust-local-artifacts",
        action="store_true",
        help="Acknowledge that these are your own trusted joblib files (can execute code).",
    )
    args = parser.parse_args(argv)
    try:
        data = load_prepared(args.processed_dir, args.raw_dir)
        model, baseline, metadata, folder = load_artifacts(
            args.artifact_dir, data, trusted=args.trust_local_artifacts
        )
        metrics, timing = evaluate_models(model, baseline, data)
        # Never overwrite the immutable training run's original evidence.
        ds.atomic_write(folder / "validation_recomputed_metrics.json", ds.json_bytes(metrics))
        ds.atomic_write(folder / "validation_recomputed_measurements.json", ds.json_bytes(timing))
        print(
            ds.json_bytes(
                {
                    "model_version": metadata["model_version"],
                    "accuracy": metrics["model"]["accuracy"],
                    "macro_f1": metrics["model"]["macro_f1"],
                    "baseline_accuracy": metrics["baseline"]["accuracy"],
                    "metrics_file": f"{folder.name}/validation_recomputed_metrics.json",
                }
            ).decode(),
            end="",
        )
    except (ds.DataError, OSError, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
