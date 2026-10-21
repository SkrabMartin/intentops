"""Local service liveness and truthful untrained-model endpoints."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.inference import ModelNotReadyError, predict
from app.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
)

app = FastAPI(title="IntentOps", version="0.1.0", description="Phase 0: untrained scaffold")


@app.exception_handler(ModelNotReadyError)
async def model_not_ready(request: Request, exc: ModelNotReadyError) -> JSONResponse:
    return JSONResponse(status_code=503, content=ErrorResponse(error=ErrorDetail()).model_dump())


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Omit Pydantic's input/context fields so messages are not echoed in errors.
    details = [{"loc": err["loc"], "type": err["type"], "msg": err["msg"]} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness is independent of future inference readiness."""
    return HealthResponse()


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse()


@app.post("/predict", responses={503: {"model": ErrorResponse}})
def prediction(payload: PredictionRequest) -> None:
    """Reserved endpoint: all valid requests receive model-unavailable (503)."""
    predict(payload.text)
