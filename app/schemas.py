"""Strict Phase-0 service contracts; no fabricated prediction schema."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

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
    model_ready: Literal[False] = False


class ModelInfoResponse(StrictSchema):
    phase: Literal[0] = 0
    status: Literal["untrained"] = "untrained"
    model_ready: Literal[False] = False
    model_version: None = None


class ErrorDetail(StrictSchema):
    code: Literal["model_not_ready"] = "model_not_ready"
    message: str = "No trained model is available. Classification is deferred to a later phase."


class ErrorResponse(StrictSchema):
    error: ErrorDetail
