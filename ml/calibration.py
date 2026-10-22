"""Pure temperature scaling and predeclared internal-development statistics."""

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split

PROTOCOL = {
    "method": "temperature_scaling_v1",
    "seed": 43,
    "fractions": [0.4, 0.3, 0.3],
    "temperature_bounds": [0.05, 20.0],
    "optimizer": "bounded_log_temperature",
    "xatol": 1e-10,
    "maxiter": 500,
    "ece_bins": 10,
    "risk_budget": 0.10,
    "min_accepted": 30,
    "thresholds": [i / 100 for i in range(101)],
    "wilson_z": 1.6448536269514722,
    "tie_break": "maximum accepted count, then smallest threshold",
}


def matrix(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[0] or values.shape[1] < 2:
        raise ValueError("Expected a nonempty multiclass matrix")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite matrix")
    return values


def targets(y, values):
    y = np.asarray(y)
    if (
        y.shape != (len(values),)
        or not np.issubdtype(y.dtype, np.integer)
        or (y < 0).any()
        or (y >= values.shape[1]).any()
    ):
        raise ValueError("Invalid aligned class indices")
    return y


def probabilities(values):
    values = matrix(values)
    if (
        (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("Invalid probability distribution")
    return values


def scaled(scores, temperature: float):
    scores = matrix(scores)
    if not math.isfinite(temperature) or not 0.05 <= temperature <= 20:
        raise ValueError("Temperature outside predeclared bounds")
    logits = scores / temperature
    return probabilities(np.exp(logits - logsumexp(logits, axis=1, keepdims=True)))


def fit_temperature(scores, y) -> float:
    scores = matrix(scores)
    y = targets(y, scores)

    def objective(log_t):
        logits = scores / math.exp(log_t)
        return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]))

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20)),
        method="bounded",
        options={"xatol": PROTOCOL["xatol"], "maxiter": PROTOCOL["maxiter"]},
    )
    if not result.success or not np.isfinite([result.x, result.fun]).all():
        raise ValueError("Temperature optimizer failed")
    temperature = math.exp(float(result.x))
    scaled(scores, temperature)
    return temperature


def metrics(values, y) -> dict:
    values = probabilities(values)
    y = targets(y, values)
    correct = values.argmax(axis=1) == y
    confidence = values.max(axis=1)
    # [left,right), last bin includes 1. Empty bins contribute zero to ECE.
    index = np.minimum((confidence * 10).astype(int), 9)
    bins = []
    ece = 0.0
    for i in range(10):
        mask = index == i
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean)
        bins.append({"bin": i, "count": count, "accuracy": accuracy, "mean_score": mean})
    return {
        "rows": len(y),
        "accuracy": float(correct.mean()),
        "log_loss": float(-np.log(np.maximum(values[np.arange(len(y)), y], 1e-300)).mean()),
        "brier": float(np.square(values - np.eye(values.shape[1])[y]).sum(axis=1).mean()),
        "ece": float(ece),
        "bins": bins,
    }


def wilson_upper(errors: int, count: int) -> float | None:
    if type(errors) is not int or type(count) is not int or not 0 <= errors <= count:
        raise ValueError("Invalid binomial counts")
    if not count:
        return None
    z = PROTOCOL["wilson_z"]
    p = errors / count
    return min(
        1.0,
        (p + z * z / (2 * count) + z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)))
        / (1 + z * z / count),
    )


def selective(values, y, threshold: float | None) -> dict:
    values = probabilities(values)
    y = targets(y, values)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid threshold")
    accepted = (
        np.zeros(len(y), dtype=bool) if threshold is None else values.max(axis=1) >= threshold
    )
    count = int(accepted.sum())
    errors = int(((values.argmax(axis=1) != y) & accepted).sum())
    return {
        "threshold": threshold,
        "rows": len(y),
        "accepted": count,
        "reviewed": len(y) - count,
        "coverage": count / len(y),
        "review_rate": 1 - count / len(y),
        "errors": errors,
        "selective_error": errors / count if count else None,
        "wilson_upper": wilson_upper(errors, count),
        "accuracy": float((values.argmax(axis=1) == y).mean()),
    }


def select_policy(values, y) -> tuple[dict, list[dict]]:
    curve = [selective(values, y, threshold) for threshold in PROTOCOL["thresholds"]]
    eligible = [
        c
        for c in curve
        if c["accepted"] >= PROTOCOL["min_accepted"]
        and c["wilson_upper"] <= PROTOCOL["risk_budget"]
    ]
    selected = (
        min(eligible, key=lambda c: (-c["accepted"], c["threshold"]))
        if eligible
        else selective(values, y, None)
    )
    return selected, curve


def partition(ids: list[int], labels: list[str]) -> dict[str, list[int]]:
    if len(ids) != len(labels) or len(set(ids)) != len(ids) or len(ids) < 6:
        raise ValueError("Invalid validation membership")
    indices = list(range(len(ids)))
    fit, remaining = train_test_split(indices, test_size=0.6, random_state=43, stratify=labels)
    selection, audit = train_test_split(
        remaining, test_size=0.5, random_state=43, stratify=[labels[i] for i in remaining]
    )
    return {
        name: sorted(values, key=lambda i: ids[i])
        for name, values in (("calibration", fit), ("selection", selection), ("audit", audit))
    }
