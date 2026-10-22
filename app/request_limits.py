"""Bound the batch body before JSON parsing; no bodies or exception values logged."""

import json

from starlette.responses import JSONResponse

from app.schemas import ErrorDetail, ErrorResponse, ValidationErrorResponse, ValidationIssue

MAX_BATCH_BODY_BYTES = 8 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-standard JSON constant")


class BatchBodyLimit:
    """ASGI stream cap applies only to the new batch POST (including slash redirect)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"].rstrip("/") != "/predict/batch"
        ):
            await self.app(scope, receive, send)
            return

        async def invalid_json():
            response = ValidationErrorResponse(
                detail=[
                    ValidationIssue(
                        loc=["body"], type="json_invalid", msg="Invalid batch JSON body."
                    )
                ]
            )
            await JSONResponse(status_code=422, content=response.model_dump())(scope, receive, send)

        async def too_large():
            response = ErrorResponse(
                error=ErrorDetail(
                    code="payload_too_large", message="Batch request exceeds the 8 MiB body limit."
                )
            )
            await JSONResponse(status_code=413, content=response.model_dump())(scope, receive, send)

        headers = scope.get("headers", [])
        encodings = [v.lower() for k, v in headers if k.lower() == b"content-encoding"]
        if encodings and encodings != [b"identity"]:
            response = ErrorResponse(
                error=ErrorDetail(
                    code="unsupported_encoding",
                    message="Compressed batch bodies are not supported.",
                )
            )
            await JSONResponse(status_code=415, content=response.model_dump())(scope, receive, send)
            return
        lengths = [v for k, v in headers if k.lower() == b"content-length"]
        if lengths:
            if len(lengths) != 1 or not lengths[0].isdigit():
                await invalid_json()
                return
            if len(lengths[0]) > 12 or int(lengths[0]) > MAX_BATCH_BODY_BYTES:
                await too_large()
                return
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunk = event.get("body", b"")
            if len(body) + len(chunk) > MAX_BATCH_BODY_BYTES:
                await too_large()
                return
            body.extend(chunk)
            if not event.get("more_body", False):
                break
        try:
            # Reject ambiguous duplicate keys and non-JSON constants before framework parsing.
            json.loads(body, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (ValueError, UnicodeError, RecursionError):
            await invalid_json()
            return

        async def replay():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)
