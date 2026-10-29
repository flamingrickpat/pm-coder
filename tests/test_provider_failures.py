"""Permanent endpoint rejections surface once instead of reconnecting forever.

The reconnect loop is correct for a cold or flaky transport. It is wrong for
an authentication, billing, or configuration error: the identical request
will be rejected identically every ``RETRY_DELAY_SECONDS``. These checks use
an in-process exception shape, not model inference, to prove the policy.
"""
import asyncio

from pm_coder import (
    PermanentProviderError,
    SessionStore,
    Settings,
    is_permanent_failure,
    run_turn,
)


class StatusError(Exception):
    """The shape the OpenAI SDK raises: a message plus an HTTP status."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class FailingAgent:
    def __init__(self, error: BaseException):
        self.error = error

    async def run(self, *args, **kwargs):
        raise self.error


def make_settings(tmp_path) -> Settings:
    return Settings(
        cwd=tmp_path,
        base_url="http://127.0.0.1:9/v1",
        api_key="local",
        model="qwen",
        mcp_config=None,
        shell_kind="bash",
        shell_executable="/bin/bash",
        shell_timeout=5,
        disable_thinking=True,
        skill=None,
        verbose=False,
        context_window=4096,
        enable_write=True,
        workspace_discovery=False,
        live_test=False,
    )


def test_permanent_classifier_covers_auth_billing_and_config():
    assert is_permanent_failure(StatusError("invalid api key", 401))
    assert is_permanent_failure(StatusError("payment required", 402))
    assert is_permanent_failure(StatusError("forbidden", 403))
    assert is_permanent_failure(Exception("Insufficient quota for this model"))
    assert is_permanent_failure(Exception("model `ghost` not found"))
    assert is_permanent_failure(Exception("invalid_request_error: bad field"))


def test_permanent_classifier_leaves_transient_errors_retryable():
    assert not is_permanent_failure(ConnectionError("connection refused"))
    assert not is_permanent_failure(TimeoutError("read timed out"))
    assert not is_permanent_failure(StatusError("internal server error", 500))
    assert not is_permanent_failure(StatusError("rate limited", 429))


def test_permanent_failure_raises_without_entering_the_reconnect_loop(tmp_path):
    settings = make_settings(tmp_path)
    session = SessionStore.open(tmp_path, "permanent", log_root=tmp_path / "logs")
    agent = FailingAgent(StatusError("invalid api key", 401))
    try:
        asyncio.run(run_turn(agent, settings, session, "unused"))
    except PermanentProviderError as exc:
        assert "401" in str(exc) or "invalid api key" in str(exc)
    else:  # pragma: no cover - the policy must fail the turn
        raise AssertionError("a permanent failure was absorbed by run_turn")