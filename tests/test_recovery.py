from __future__ import annotations

from pydantic_ai.exceptions import ModelHTTPError

from pm_coder import (
    ContextLimitReachedError,
    EndpointRequestError,
    TruncatedModelOutputError,
    _endpoint_failure,
    _generation_boundary_failure,
)


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
