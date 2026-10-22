"""Strict local serving contracts; probability scores are not calibrated confidence."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_TEXT_LENGTH = 5000
MAX_BATCH_MESSAGES = 100
MessageText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH),
]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PredictionRequest(StrictSchema):
    text: MessageText


class BatchPredictionRequest(StrictSchema):
    messages: list[MessageText] = Field(
        min_length=1,
        max_length=MAX_BATCH_MESSAGES,
        examples=[["When will my new card arrive?", "I was charged twice for a card payment."]],
    )


class HealthResponse(StrictSchema):
    status: Literal["ok"] = "ok"
    model_ready: bool = False


class ModelInfoResponse(StrictSchema):
    phase: Literal[3, 4] = 4
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


PolicyStatus = Literal["ready", "disabled", "unavailable", "invalid"]
PolicyReason = Literal[
    "policy_not_available",
    "review_all_policy",
    "below_configured_threshold",
    "meets_configured_threshold",
]


class PolicyInfoResponse(StrictSchema):
    status: PolicyStatus = "disabled"
    policy_version: str | None = None
    method: Literal["temperature_scaling_v1"] | None = None
    temperature: float | None = Field(default=None, ge=0.05, le=20, allow_inf_nan=False)
    threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    mode: Literal["threshold", "review_all"] | None = None
    demo_error_budget: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    audit_upper_error_bound: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    audit_rows: int | None = Field(default=None, gt=0)
    audit_accepted: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "ready":
            if (
                not self.policy_version
                or self.method is None
                or self.temperature is None
                or self.demo_error_budget != 0.1
                or self.mode is None
                or self.audit_rows is None
                or self.audit_accepted is None
                or self.audit_accepted > self.audit_rows
                or (self.audit_accepted == 0) != (self.audit_upper_error_bound is None)
                or (self.mode == "review_all") != (self.threshold is None)
            ):
                raise ValueError("Incomplete ready policy")
        elif any(v is not None for k, v in self.model_dump().items() if k != "status"):
            raise ValueError("Inactive policy cannot claim calibration")
        return self


class PredictionResponse(StrictSchema):
    intent: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    top_predictions: list[PredictionScore] = Field(min_length=3, max_length=3)
    score_type: Literal["uncalibrated_model_probability", "temperature_scaled_probability"] = (
        "uncalibrated_model_probability"
    )
    recommendation: Literal["suggest_automatic", "needs_review"] = "needs_review"
    policy_status: PolicyStatus = "disabled"
    policy_version: str | None = None
    policy_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reason: PolicyReason = "policy_not_available"

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
        if self.policy_status == "ready":
            if not self.policy_version or self.score_type != "temperature_scaled_probability":
                raise ValueError("Ready policy requires calibrated scores")
            automatic = (
                self.policy_threshold is not None
                and self.top_predictions[0].score >= self.policy_threshold
            )
            reason = (
                "meets_configured_threshold"
                if automatic
                else "review_all_policy"
                if self.policy_threshold is None
                else "below_configured_threshold"
            )
            if (self.recommendation == "suggest_automatic") != automatic or self.reason != reason:
                raise ValueError("Recommendation differs from policy")
        elif (
            self.recommendation != "needs_review"
            or self.policy_version is not None
            or self.policy_threshold is not None
            or self.reason != "policy_not_available"
            or self.score_type != "uncalibrated_model_probability"
        ):
            raise ValueError("Inactive policy must require review and raw scores")
        return self


class ErrorDetail(StrictSchema):
    code: Literal[
        "model_not_ready", "inference_failed", "payload_too_large", "unsupported_encoding"
    ] = "model_not_ready"
    message: str = (
        "No verified model is loaded. Enable trusted local artifacts and restart the API."
    )


class ErrorResponse(StrictSchema):
    error: ErrorDetail


class BatchPredictionResponse(StrictSchema):
    predictions: list[PredictionResponse] = Field(min_length=1, max_length=MAX_BATCH_MESSAGES)

    @model_validator(mode="after")
    def consistent(self):
        identities = {
            (p.model_version, p.policy_status, p.policy_version, p.score_type, p.policy_threshold)
            for p in self.predictions
        }
        if len(identities) != 1:
            raise ValueError("A batch must use one model and policy snapshot.")
        return self


class ValidationIssue(StrictSchema):
    loc: list[str | int]
    type: str
    msg: str


class ValidationErrorResponse(StrictSchema):
    detail: list[ValidationIssue]
