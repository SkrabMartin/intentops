"""Strict local serving contracts; probability scores are not calibrated confidence."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_TEXT_LENGTH = 5000
MessageText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH),
]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PredictionRequest(StrictSchema):
    text: MessageText


class HealthResponse(StrictSchema):
    status: Literal["ok"] = "ok"
    model_ready: bool = False


class ModelInfoResponse(StrictSchema):
    phase: Literal[3] = 3
    status: Literal["unavailable", "loaded"] = "unavailable"
    model_ready: bool = False
    model_version: str | None = None
    classifier_type: Literal["TfidfVectorizer + LogisticRegression"] | None = None
    category_count: int = Field(default=0, ge=0)
    reason: Literal["disabled", "invalid_artifacts", "internal_error"] | None = "disabled"

    @model_validator(mode="after")
    def consistent(self):
        if self.model_ready:
            if (
                self.status != "loaded"
                or not self.model_version
                or self.category_count < 3
                or self.classifier_type is None
                or self.reason is not None
            ):
                raise ValueError("Loaded model metadata must be complete.")
        elif (
            self.status != "unavailable"
            or self.model_version is not None
            or self.category_count != 0
            or self.classifier_type is not None
            or self.reason is None
        ):
            raise ValueError("Unavailable model metadata must not claim a loaded classifier.")
        return self


class PredictionScore(StrictSchema):
    intent: str = Field(min_length=1)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class PredictionResponse(StrictSchema):
    intent: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    top_predictions: list[PredictionScore] = Field(min_length=3, max_length=3)
    score_type: Literal["uncalibrated_model_probability"] = "uncalibrated_model_probability"

    @model_validator(mode="after")
    def coherent(self):
        if (
            self.intent != self.top_predictions[0].intent
            or len({p.intent for p in self.top_predictions}) != 3
            or any(
                a.score < b.score for a, b in zip(self.top_predictions, self.top_predictions[1:])
            )
        ):
            raise ValueError("Prediction ranking is inconsistent.")
        return self


class ErrorDetail(StrictSchema):
    code: Literal["model_not_ready", "inference_failed"] = "model_not_ready"
    message: str = (
        "No verified model is loaded. Enable trusted local artifacts and restart the API."
    )


class ErrorResponse(StrictSchema):
    error: ErrorDetail
