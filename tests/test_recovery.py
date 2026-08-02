from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from pm_coder import (
    ContextLimitReachedError,
    EndpointRequestError,
    TruncatedModelOutputError,
    _endpoint_failure,
    _generation_boundary_failure,
    _transient_retry_delay,
)

SUPPORTS_HTTP_ERROR_HEADERS = "headers" in inspect.signature(ModelHTTPError).parameters


def _length_response(output_tokens: int) -> list[dict[str, object]]:
    return [{
        "kind": "response",
        "finish_reason": "length",
        "usage": {
            "input_tokens": 90_000,
            "output_tokens": output_tokens,
            "total_tokens": 90_000 + output_tokens,
        },
        "parts": [],
    }]


def test_length_at_response_budget_is_recoverable() -> None:
    failure = _generation_boundary_failure(8_192, _length_response(8_192))

    assert failure is not None
    assert failure[0] is TruncatedModelOutputError


def test_length_before_response_budget_is_a_real_context_failure() -> None:
    failure = _generation_boundary_failure(8_192, _length_response(1_000))

    assert failure is not None
    assert failure[0] is ContextLimitReachedError
    assert "before it could use the 8,192-token response limit" in failure[1]


def test_known_http_errors_get_a_short_explanation() -> None:
    failure = _endpoint_failure(ModelHTTPError(
        401,
        "qwen",
        {"error": {"message": "bad key"}},
    ))

    assert failure is not None
    assert failure[0] is EndpointRequestError
    assert "credentials" in failure[1]


def test_transient_retry_supports_legacy_http_error_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(ModelHTTPError, "retry_after", raising=False)
    failure = ModelHTTPError(503, "qwen", {"error": "busy"})

    assert _transient_retry_delay(failure, 1) == 1.0


@pytest.mark.skipif(
    not SUPPORTS_HTTP_ERROR_HEADERS,
    reason="installed pydantic-ai does not expose HTTP response headers",
)
def test_transient_retry_uses_numeric_retry_after_header() -> None:
    failure = ModelHTTPError(
        429,
        "qwen",
        {"error": "busy"},
        headers={"Retry-After": "9"},
    )

    assert _transient_retry_delay(failure, 1) == 9.0


@pytest.mark.skipif(
    not SUPPORTS_HTTP_ERROR_HEADERS,
    reason="installed pydantic-ai does not expose HTTP response headers",
)
def test_transient_retry_uses_http_date_retry_after_header() -> None:
    retry_at = datetime.now(UTC) + timedelta(seconds=20)
    failure = ModelHTTPError(
        503,
        "qwen",
        {"error": "restarting"},
        headers={"Retry-After": retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT")},
    )

    delay = _transient_retry_delay(failure, 1)
    assert 15.0 <= delay <= 20.0
