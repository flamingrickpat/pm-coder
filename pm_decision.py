"""Typed System One requests without an agent session or workflow policy.

The client sends one request and never retries. Callers own fallback decisions
and receipt storage. Async cancellation closes the HTTP connection. It does
not prove that the server stopped inference.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from typing import Any, Generic, Literal, Mapping, Protocol, TypeVar, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, model_validator


class _Typed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ChoiceQuestion(_Typed):
    """Ask for one of 2 to 255 named options with concrete descriptions."""

    type: Literal["choice"] = "choice"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    criteria: dict[str, str | None] = Field(min_length=2, max_length=255)


class ScoreQuestion(_Typed):
    """Ask for an expected index among 2 to 10 ordered level descriptions."""

    type: Literal["score"] = "score"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    criteria: list[str] = Field(min_length=2, max_length=10)


class ProbabilityQuestion(_Typed):
    """Ask for a probability of true through the provider's `noul` primitive."""

    type: Literal["noul"] = "noul"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    criteria: dict[Literal["true", "false"], str] | None = None

    @model_validator(mode="after")
    def _both_labels(self) -> ProbabilityQuestion:
        if self.criteria is not None and set(self.criteria) != {"true", "false"}:
            raise ValueError("Probability criteria must describe both true and false.")
        return self


Question = ChoiceQuestion | ScoreQuestion | ProbabilityQuestion
State = str | dict[str, JsonValue] | list[JsonValue]


class ChoiceAnswer(_Typed):
    """Preserve the provider's option, probability distribution, and confidence."""

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)


class ScoreAnswer(_Typed):
    """Preserve the provider's expected index, legend, and distribution."""

    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)


class ProbabilityAnswer(_Typed):
    """Preserve a probability of true in the closed interval from zero to one."""

    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


Answer = ChoiceAnswer | ScoreAnswer | ProbabilityAnswer
AnswerT = TypeVar("AnswerT", bound=Answer)


class _Usage(_Typed):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0, le=0)


class _Response(_Typed):
    model: str
    answers: dict[str, Answer]
    usage: _Usage


@dataclass(frozen=True)
class DecisionReceipt:
    """Describe one request without credentials or source state.

    Digests use compact UTF-8 JSON with sorted keys. Question definitions retain
    their transmitted option order. `provider_identity` is caller-supplied provenance, not
    an identity discovered by this client. Timing includes connection setup.
    The caller decides where to store this receipt with `to_dict()`.
    """

    started_at: str
    endpoint: str
    expected_model: str
    provider_identity: dict[str, JsonValue]
    state_sha256: str
    request_sha256: str
    request_bytes: int
    questions: dict[str, Any]
    outcome: str = "pending"
    seconds: float = 0.0
    http_status: int | None = None
    response_sha256: str | None = None
    response: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy for instance-local evidence storage."""
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class DecisionResult(Generic[AnswerT]):
    """Return validated answers and their receipt without committing state."""

    answers: dict[str, AnswerT]
    receipt: DecisionReceipt

    @property
    def answer(self) -> AnswerT:
        """Return the only answer. Raise ValueError for a multi-question result."""
        if len(self.answers) != 1:
            raise ValueError("Select an answer by question ID for a multi-question result.")
        return next(iter(self.answers.values()))


class DecisionError(RuntimeError):
    """Base error with a receipt when request validation completed."""

    def __init__(self, message: str, receipt: DecisionReceipt | None = None):
        super().__init__(message)
        self.receipt = receipt


class DecisionInputError(DecisionError):
    """Reject invalid configuration, input, or a client byte limit."""


class DecisionHTTPError(DecisionError):
    """Report an HTTP rejection. The receipt contains its status and body digest."""


class DecisionTransportError(DecisionError):
    """Report connection loss or timeout without retrying the request."""


class DecisionSchemaError(DecisionError):
    """Reject incomplete, inconsistent, non-finite, or wrong-model answers."""


class DecisionCancelled(asyncio.CancelledError):
    """Preserve async cancellation semantics and the uncertain request receipt."""

    def __init__(self, receipt: DecisionReceipt):
        super().__init__("Decision request cancelled. Server completion is unknown.")
        self.receipt = receipt


class CancellationSignal(Protocol):
    """Accept asyncio.Event or threading.Event without owning its lifecycle."""

    def is_set(self) -> bool:
        """Return whether the caller requested cancellation."""
        ...


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonfinite(value: str) -> None:
    raise ValueError("The response contains a non-finite JSON number.")


def _distribution(values: dict[str, float], keys: set[str]) -> None:
    """Reject missing options and invalid mass without normalizing provider data."""
    if set(values) != keys or any(not 0 <= p <= 1 for p in values.values()):
        raise ValueError("The probability keys or values do not match the question.")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-5):
        raise ValueError("The probability distribution does not sum to one.")


def _validate_response(raw: bytes, questions: dict[str, Any], model: str) -> _Response:
    """Validate the entire response before the caller receives any answer."""
    value = _Response.model_validate(json.loads(raw, parse_constant=_nonfinite))
    if value.model != model:
        raise ValueError("The returned model does not match the configured decision model.")
    if set(value.answers) != set(questions):
        raise ValueError("The response question IDs do not match the request.")
    for name, question in questions.items():
        answer = value.answers[name]
        if answer.type != question["type"]:
            raise ValueError("The response answer type does not match the question.")
        if isinstance(answer, ChoiceAnswer):
            _distribution(answer.probabilities, set(question["criteria"]))
            if answer.choice not in answer.probabilities:
                raise ValueError("The chosen option is not in the question.")
            if answer.probabilities[answer.choice] < max(answer.probabilities.values()):
                raise ValueError("The chosen option does not have the highest probability.")
        elif isinstance(answer, ScoreAnswer):
            legend = {str(i): label for i, label in enumerate(question["criteria"])}
            _distribution(answer.probabilities, set(legend))
            if answer.legend != legend or not 0 <= answer.score <= len(legend) - 1:
                raise ValueError("The score or legend does not match the question.")
            expected = sum(int(i) * p for i, p in answer.probabilities.items())
            if not math.isclose(answer.score, expected, abs_tol=1e-5):
                raise ValueError("The score does not match its distribution.")
    return value


class _Client:
    def __init__(self, endpoint: str = "http://127.0.0.1:8082/v1/systemone", *,
                 model: str = "laya", api_key: str = "", timeout: float = 30.0,
                 max_request_bytes: int = 1_048_576,
                 provider_identity: Mapping[str, JsonValue] | None = None):
        try:
            url = httpx.URL(endpoint)
            if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
                    or url.query or url.fragment or url.path != "/v1/systemone"):
                raise ValueError("Use an absolute /v1/systemone URL without credentials or query parameters.")
            if (not isinstance(model, str) or not model.strip()
                    or type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("Specify a decision model and a positive finite timeout in seconds.")
            if type(max_request_bytes) is not int or max_request_bytes < 1:
                raise ValueError("max_request_bytes must be a positive integer.")
            if not isinstance(api_key, str) or not api_key.isascii() or "\r" in api_key or "\n" in api_key:
                raise ValueError("api_key must be an ASCII string without line breaks.")
            self.provider_identity = json.loads(_json_bytes(dict(provider_identity or {})))
        except (TypeError, ValueError, httpx.InvalidURL) as error:
            raise DecisionInputError(str(error)) from error
        self.endpoint, self.model, self.api_key = str(url), model, api_key
        self.timeout, self.max_request_bytes = timeout, max_request_bytes

    def _prepare(self, state: str | dict[str, JsonValue] | list[JsonValue],
                 questions: Mapping[str, Question]) -> tuple[bytes, DecisionReceipt]:
        """Snapshot mutable inputs before cancellation or the first network await."""
        try:
            if not isinstance(state, (str, dict, list)) or not state:
                raise ValueError("State must be nonempty text, an object, or an array.")
            if isinstance(state, str) and not state.strip():
                raise ValueError("Text state must contain a non-whitespace character.")
            TypeAdapter(JsonValue).validate_python(state, strict=True)
            if not questions or len(questions) > 64:
                raise ValueError("Specify between one and 64 independent questions.")
            definitions = {}
            for name, question in questions.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Question IDs must be nonempty strings.")
                if not isinstance(question, (ChoiceQuestion, ScoreQuestion, ProbabilityQuestion)):
                    raise ValueError("Use typed question objects.")
                # Revalidate because a frozen model can contain mutable dictionaries.
                question = type(question).model_validate(question.model_dump())
                if not question.instructions:
                    raise ValueError("Question instructions must be nonempty.")
                if isinstance(question.instructions, str) and not question.instructions.strip():
                    raise ValueError("Text instructions must contain a non-whitespace character.")
                if isinstance(question, ChoiceQuestion) and any(not k.strip() for k in question.criteria):
                    raise ValueError("Option names must be nonempty strings.")
                definitions[name] = question.model_dump(exclude_none=True)
                # Null option descriptions are valid and must retain their keys.
                if isinstance(question, ChoiceQuestion):
                    definitions[name]["criteria"] = dict(question.criteria)
            body = _json_bytes({"model": self.model, "state": state, "questions": definitions})
            if len(body) > self.max_request_bytes:
                raise ValueError("The request exceeds max_request_bytes. No input was truncated.")
            snapshot = json.loads(body)
            receipt = DecisionReceipt(datetime.now(timezone.utc).isoformat(), self.endpoint,
                self.model, json.loads(_json_bytes(self.provider_identity)), _digest(_json_bytes(snapshot["state"])),
                _digest(body), len(body), snapshot["questions"])
            return body, receipt
        except (TypeError, ValueError, ValidationError) as error:
            raise DecisionInputError(str(error)) from error

    async def _decide(self, state: Any, questions: Mapping[str, Question],
                      cancel: CancellationSignal | None) -> DecisionResult[Answer]:
        body, receipt = self._prepare(state, questions)
        started = time.monotonic()
        response: httpx.Response | None = None

        async def send() -> httpx.Response:
            # A private connection prevents one caller's cancellation from closing another.
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                return await client.post(self.endpoint, content=body, headers=headers)

        async def wait_for_cancel() -> None:
            while cancel is not None and not cancel.is_set():
                await asyncio.sleep(0.01)

        request: asyncio.Task | None = None
        watcher: asyncio.Task | None = None
        try:
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError()
            request = asyncio.create_task(send())
            if cancel is not None:
                watcher = asyncio.create_task(wait_for_cancel())
                done, _ = await asyncio.wait((request, watcher), return_when=asyncio.FIRST_COMPLETED)
                if watcher in done:
                    raise asyncio.CancelledError()
            response = await request
            receipt = replace(receipt, http_status=response.status_code,
                              response_sha256=_digest(response.content))
            if not response.is_success:
                raise DecisionHTTPError(f"Decision endpoint returned HTTP {response.status_code}.")
            try:
                value = _validate_response(response.content, receipt.questions, self.model)
            except (ValueError, TypeError, ValidationError) as error:
                # Exclude echoed response values from failure diagnostics and receipts.
                detail = (json.dumps(error.errors(include_input=False, include_context=False, include_url=False))
                          if isinstance(error, ValidationError) else str(error))
                raise DecisionSchemaError("Decision response failed schema validation: " + detail) from error
            receipt = replace(receipt, outcome="success", seconds=time.monotonic() - started,
                              response=value.model_dump())
            return DecisionResult(value.answers, receipt)
        except asyncio.CancelledError as error:
            receipt = replace(receipt, outcome="cancelled", seconds=time.monotonic() - started,
                              error="Server completion is unknown after cancellation.")
            raise DecisionCancelled(receipt) from error
        except (httpx.RequestError, DecisionError) as error:
            kind = ("transport_error" if isinstance(error, httpx.RequestError) else
                    "http_error" if isinstance(error, DecisionHTTPError) else "schema_error")
            detail = type(error).__name__ if isinstance(error, httpx.RequestError) else str(error)
            receipt = replace(receipt, outcome=kind, seconds=time.monotonic() - started, error=detail)
            if isinstance(error, httpx.RequestError):
                raise DecisionTransportError("Decision connection failed: " + type(error).__name__, receipt) from error
            error.receipt = receipt
            raise
        finally:
            # Join both tasks so a cancelled synchronous call leaves no HTTP worker behind.
            pending = [task for task in (request, watcher) if task is not None]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


class DecisionClient(_Client):
    """Make ordinary Python calls with optional threading.Event cancellation.

    Each call owns an event loop and one HTTP connection. Use
    `AsyncDecisionClient` inside an existing event loop. The full endpoint is
    separate from chat. `timeout` is an HTTP phase timeout in seconds.
    `max_request_bytes` limits encoded bytes, not model tokens. The client
    never truncates input, retries requests, or stores receipts automatically.
    """

    def decide(self, state: str | dict[str, JsonValue] | list[JsonValue],
               questions: Mapping[str, Question], *, cancel: CancellationSignal | None = None) -> DecisionResult[Answer]:
        """Answer all questions atomically or raise a typed decision error.

        Cancellation raises DecisionCancelled with a receipt. HTTP, transport,
        and schema errors retain receipts. Invalid input raises DecisionInputError
        before the network call. The caller owns any fallback or durable commit.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._decide(state, questions, cancel))
        raise DecisionInputError("Use AsyncDecisionClient inside an existing event loop.")

    def choice(self, state: State, instructions: str, criteria: dict[str, str | None],
               *, cancel: CancellationSignal | None = None) -> DecisionResult[ChoiceAnswer]:
        """Return one ChoiceAnswer under the question ID `choice` and its receipt."""
        return cast(DecisionResult[ChoiceAnswer], self.decide(state, {"choice": ChoiceQuestion(instructions=instructions, criteria=criteria)}, cancel=cancel))

    def score(self, state: State, instructions: str, criteria: list[str],
              *, cancel: CancellationSignal | None = None) -> DecisionResult[ScoreAnswer]:
        """Return one ScoreAnswer under the question ID `score` and its receipt."""
        return cast(DecisionResult[ScoreAnswer], self.decide(state, {"score": ScoreQuestion(instructions=instructions, criteria=criteria)}, cancel=cancel))

    def probability(self, state: State, instructions: str, criteria: dict[Literal["true", "false"], str] | None = None,
                    *, cancel: CancellationSignal | None = None) -> DecisionResult[ProbabilityAnswer]:
        """Return one ProbabilityAnswer under `probability`. Its wire type is noul."""
        return cast(DecisionResult[ProbabilityAnswer], self.decide(state, {"probability": ProbabilityQuestion(instructions=instructions, criteria=criteria)}, cancel=cancel))


class AsyncDecisionClient(_Client):
    """Make async decisions with task cancellation or an explicit event.

    Configuration and validation match DecisionClient. Concurrent calls own
    separate connections. Cancelling one call does not cancel another call.
    """

    async def decide(self, state: str | dict[str, JsonValue] | list[JsonValue],
                     questions: Mapping[str, Question], *, cancel: CancellationSignal | None = None) -> DecisionResult[Answer]:
        """Return complete typed answers. Cancellation raises DecisionCancelled.

        Failure receipts and caller-owned fallback match DecisionClient.decide.
        Cancellation closes local HTTP resources without a server-stop guarantee.
        """
        return await self._decide(state, questions, cancel)

    async def choice(self, state: State, instructions: str, criteria: dict[str, str | None],
                     *, cancel: CancellationSignal | None = None) -> DecisionResult[ChoiceAnswer]:
        """Return a ChoiceAnswer and receipt through an async request."""
        return cast(DecisionResult[ChoiceAnswer], await self.decide(state, {"choice": ChoiceQuestion(instructions=instructions, criteria=criteria)}, cancel=cancel))

    async def score(self, state: State, instructions: str, criteria: list[str],
                    *, cancel: CancellationSignal | None = None) -> DecisionResult[ScoreAnswer]:
        """Return a ScoreAnswer and receipt through an async request."""
        return cast(DecisionResult[ScoreAnswer], await self.decide(state, {"score": ScoreQuestion(instructions=instructions, criteria=criteria)}, cancel=cancel))

    async def probability(self, state: State, instructions: str, criteria: dict[Literal["true", "false"], str] | None = None,
                          *, cancel: CancellationSignal | None = None) -> DecisionResult[ProbabilityAnswer]:
        """Return a ProbabilityAnswer and receipt through an async request."""
        return cast(DecisionResult[ProbabilityAnswer], await self.decide(state, {"probability": ProbabilityQuestion(instructions=instructions, criteria=criteria)}, cancel=cancel))
