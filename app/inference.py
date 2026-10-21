"""Load a trusted local classifier once and predict without files or network calls."""

import logging
import os
import threading
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from app.schemas import ModelInfoResponse, PredictionRequest, PredictionResponse, PredictionScore
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
        return cls.load(
            Path(os.environ.get("INTENTOPS_ARTIFACT_DIR", str(DEFAULT_ARTIFACTS))), trusted=trusted
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
        text = PredictionRequest(text=text).text
        if not self.ready:
            raise ModelNotReadyError("No trained model is ready.")
        try:
            # Serial local inference bounds concurrent threadpool changes and CPU use.
            with self._lock, threadpool_limits(limits=1):
                values = np.asarray(self.model.predict_proba([text]), dtype=float)
            labels = list(self.model.classes_)
            if (
                values.shape != (1, len(labels))
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-6)
            ):
                raise ValueError("Invalid probabilities")
            order = np.argsort(-values[0], kind="stable")[:3]
            scores = [
                PredictionScore(intent=str(labels[i]), score=float(values[0, i])) for i in order
            ]
            return PredictionResponse(
                intent=scores[0].intent,
                model_version=self.metadata["model_version"],
                top_predictions=scores,
            )
        except Exception:  # noqa: BLE001 -- model errors must never expose submitted text.
            # Fail closed after a severe inference failure; restart required.
            self.model = None
            self.metadata = None
            self.reason = "internal_error"
            logger.error(
                "Local inference failed; model readiness disabled. Restart after operator inspection."
            )
            raise InferenceError("Local inference failed.") from None
