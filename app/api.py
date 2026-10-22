"""Local model service: separate liveness/readiness and sanitized prediction errors."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.inference import InferenceError, ModelNotReadyError, ModelService
from app.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PolicyInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


def create_app(service_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        application.state.model_service = (service_factory or ModelService.from_environment)()
        yield
        application.state.model_service = ModelService()

    application = FastAPI(
        title="IntentOps",
        version="0.1.0",
        description="Phase 4: local calibration and human-review recommendations",
        lifespan=lifespan,
    )
    application.state.model_service = ModelService()

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
        details = [
            {
                "loc": ["body", "text"] if err["loc"] == ("body", "text") else ["body"],
                "type": err["type"],
                "msg": err["msg"],
            }
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

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

    return application


app = create_app()
