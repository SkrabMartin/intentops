"""Load a trusted local classifier once and predict without files or network calls."""

import logging
import os
import threading
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    PolicyInfoResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionScore,
)
from ml import calibration as cal
from ml import review_policy as policies
from ml.model_support import DEFAULT_ARTIFACTS, load_artifacts, serving_contract

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """No verified in-memory classifier is available."""


class InferenceError(RuntimeError):
    """A loaded classifier failed; request text must never appear in errors."""


class ModelService:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.reason = "disabled"
        self.policy = None
        self.policy_version = None
        self.policy_status = "disabled"
        self._lock = threading.Lock()

    @classmethod
    def load(cls, root: Path, *, trusted: bool = False, contract=None):
        service = cls()
        if not trusted:
            return service
        try:
            model, _, metadata, _ = load_artifacts(
                root, contract if contract is not None else serving_contract(), trusted=True
            )
            if not callable(getattr(model, "predict_proba", None)) or len(model.classes_) < 3:
                raise ValueError("Invalid classifier interface")
            service.model = model
            service.metadata = metadata
            service.reason = None
        except Exception:  # noqa: BLE001 -- startup boundary must sanitize all artifact failures.
            # No exception strings/tracebacks: they may contain paths or serialized content.
            service.reason = "invalid_artifacts"
            logger.warning(
                "Local model unavailable: invalid_artifacts; inspect the trusted bundle and exact environment."
            )
        return service

    @classmethod
    def from_environment(cls):
        trusted = os.environ.get("INTENTOPS_TRUST_LOCAL_ARTIFACTS") == "1"
        service = cls.load(
            Path(os.environ.get("INTENTOPS_ARTIFACT_DIR", str(DEFAULT_ARTIFACTS))), trusted=trusted
        )
        service.load_policy(
            Path(os.environ.get("INTENTOPS_POLICY_DIR", str(policies.DEFAULT_POLICIES))),
            enabled=os.environ.get("INTENTOPS_ENABLE_REVIEW_POLICY") == "1",
        )
        return service

    def load_policy(self, root: Path, *, enabled: bool = False):
        """Called only during explicit initialization, never by a request."""
        self.policy = None
        self.policy_version = None
        self.policy_status = "disabled" if not enabled else "unavailable"
        if not enabled or not self.ready:
            return
        try:
            self.policy, self.policy_version = policies.load_policy(root, self.metadata)
            self.policy_status = "ready"
        except FileNotFoundError:
            self.policy_status = "unavailable"
        except Exception:  # noqa: BLE001 -- sanitize all operator-selected sidecar errors.
            self.policy_status = "invalid"
            logger.warning("Review policy invalid; raw predictions require human review.")

    def policy_info(self):
        if self.policy is None or not self.ready:
            return PolicyInfoResponse(status=self.policy_status if self.ready else "unavailable")
        threshold = self.policy.frozen.selected.threshold
        return PolicyInfoResponse(
            status="ready",
            policy_version=self.policy_version,
            method="temperature_scaling_v1",
            temperature=self.policy.frozen.temperature,
            threshold=threshold,
            mode="review_all" if threshold is None else "threshold",
            demo_error_budget=0.1,
            audit_rows=self.policy.audit_risk.rows,
            audit_accepted=self.policy.audit_risk.accepted,
            audit_upper_error_bound=self.policy.audit_risk.wilson_upper,
        )

    @property
    def ready(self):
        return self.model is not None and self.metadata is not None

    def info(self) -> ModelInfoResponse:
        if not self.ready:
            return ModelInfoResponse(reason=self.reason)
        return ModelInfoResponse(
            status="loaded",
            model_ready=True,
            model_version=self.metadata["model_version"],
            classifier_type="TfidfVectorizer + LogisticRegression",
            category_count=len(self.model.classes_),
            reason=None,
        )

    def predict(self, text: str) -> PredictionResponse:
        # Single and batch share normalization, serialization and recommendation logic.
        return self.predict_batch([PredictionRequest(text=text).text]).predictions[0]

    def predict_batch(self, messages: list[str]) -> BatchPredictionResponse:
        messages = BatchPredictionRequest(messages=messages).messages
        # One bounded vectorized operation; readiness and result assembly share the lock.
        with self._lock:
            if not self.ready:
                raise ModelNotReadyError("No trained model is ready.")
            try:
                with threadpool_limits(limits=1):
                    values = np.asarray(self.model.predict_proba(messages), dtype=float)
                    if self.policy is not None:
                        raw = cal.probabilities(values)
                        decision = self.model.decision_function(messages)
                        if not np.allclose(cal.scaled(decision, 1.0), raw, atol=1e-12, rtol=1e-12):
                            raise ValueError("Inconsistent decision scores")
                        values = cal.scaled(decision, self.policy.frozen.temperature)
                        if not np.array_equal(values.argmax(axis=1), raw.argmax(axis=1)):
                            raise ValueError("Changed ranking")
                labels = list(self.model.classes_)
                if (
                    values.shape != (len(messages), len(labels))
                    or not np.isfinite(values).all()
                    or (values < 0).any()
                    or (values > 1).any()
                    or not np.allclose(values.sum(axis=1), 1, atol=1e-6)
                ):
                    raise ValueError("Invalid probabilities")
                policy_info = self.policy_info()
                results = []
                for row in values:
                    order = np.argsort(-row, kind="stable")[:3]
                    scores = [
                        PredictionScore(intent=str(labels[i]), score=float(row[i])) for i in order
                    ]
                    automatic = (
                        self.policy is not None
                        and policy_info.threshold is not None
                        and scores[0].score >= policy_info.threshold
                    )
                    reason = (
                        "policy_not_available"
                        if self.policy is None
                        else "meets_configured_threshold"
                        if automatic
                        else "review_all_policy"
                        if policy_info.threshold is None
                        else "below_configured_threshold"
                    )
                    results.append(
                        PredictionResponse(
                            intent=scores[0].intent,
                            model_version=self.metadata["model_version"],
                            top_predictions=scores,
                            score_type="temperature_scaled_probability"
                            if self.policy
                            else "uncalibrated_model_probability",
                            recommendation="suggest_automatic" if automatic else "needs_review",
                            policy_status=policy_info.status,
                            policy_version=policy_info.policy_version,
                            policy_threshold=policy_info.threshold,
                            reason=reason,
                        )
                    )
                return BatchPredictionResponse(predictions=results)
            except Exception:  # noqa: BLE001 -- no input text or model exception details may escape.
                self.model = None
                self.metadata = None
                self.reason = "internal_error"
                logger.error(
                    "Local inference failed; model readiness disabled. Restart after operator inspection."
                )
                raise InferenceError("Local inference failed.") from None
