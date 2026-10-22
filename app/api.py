"""Local model service: separate liveness/readiness and sanitized prediction errors."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.inference import InferenceError, ModelNotReadyError, ModelService
from app.request_limits import BatchBodyLimit
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PolicyInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ValidationErrorResponse,
    ValidationIssue,
)


def create_app(service_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        application.state.model_service = (service_factory or ModelService.from_environment)()
        yield
        application.state.model_service = ModelService()

    application = FastAPI(
        title="IntentOps",
        version="1.0.0",
        description="Local single and bounded batch predictions; optional audited review recommendations",
        lifespan=lifespan,
    )
    application.state.model_service = ModelService()
    application.add_middleware(BatchBodyLimit)

    @application.exception_handler(ModelNotReadyError)
    async def model_not_ready(request: Request, exc: ModelNotReadyError):
        return JSONResponse(
            status_code=503, content=ErrorResponse(error=ErrorDetail()).model_dump()
        )

    @application.exception_handler(InferenceError)
    async def inference_failed(request: Request, exc: InferenceError):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="inference_failed",
                    message="Local inference failed. Restart after operator inspection.",
                )
            ).model_dump(),
        )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            location = ["body"]
            if err["loc"] == ("body", "text"):
                location.append("text")
            elif err["loc"][:2] == ("body", "messages"):
                location.append("messages")
                if len(err["loc"]) > 2 and type(err["loc"][2]) is int:
                    location.append(err["loc"][2])
            details.append(ValidationIssue(loc=location, type=err["type"], msg=err["msg"]))
        return JSONResponse(
            status_code=422, content=ValidationErrorResponse(detail=details).model_dump()
        )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request):
        return HealthResponse(model_ready=request.app.state.model_service.ready)

    @application.get(
        "/ready", response_model=ModelInfoResponse, responses={503: {"model": ErrorResponse}}
    )
    def ready(request: Request):
        service = request.app.state.model_service
        if not service.ready:
            raise ModelNotReadyError()
        return service.info()

    @application.get("/model/info", response_model=ModelInfoResponse)
    def model_info(request: Request):
        return request.app.state.model_service.info()

    @application.get("/policy/info", response_model=PolicyInfoResponse)
    def policy_info(request: Request):
        return request.app.state.model_service.policy_info()

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def prediction(payload: PredictionRequest, request: Request):
        return request.app.state.model_service.predict(payload.text)

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        description="1–100 messages, 1–5000 trimmed characters each. All-or-nothing validation; ordered results. 8 MiB JSON body cap. No routing is executed.",
        responses={
            422: {"model": ValidationErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def batch_prediction(payload: BatchPredictionRequest, request: Request):
        return request.app.state.model_service.predict_batch(payload.messages)

    return application


app = create_app()
