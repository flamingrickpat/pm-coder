# Failure and cancellation

## Summary

Requests fail once with explicit errors.
The client never retries or selects a fallback.
Cancellation closes local HTTP resources and retains an uncertain-outcome receipt.
The caller decides whether another request or an LLM fallback is appropriate.

Parent: [decision client](../index.md).
Contracts: [requests and receipts](requests.md).
Live procedure: [operation and limits](operation.md).

## Error paths

| Error | Meaning and recovery |
| --- | --- |
| [`pm_decision.DecisionInputError`][] | Invalid input or configuration. Correct the request before another attempt |
| [`pm_decision.DecisionHTTPError`][] | Non-success HTTP status. Inspect status, body digest, and server logs |
| [`pm_decision.DecisionTransportError`][] | Connection failure or HTTP timeout. Verify service health before another attempt |
| [`pm_decision.DecisionSchemaError`][] | Wrong model, missing answer, malformed JSON, invalid distribution, or inconsistent typed answer |
| [`pm_decision.DecisionCancelled`][] | Caller cancellation. Server completion remains unknown |

`DecisionError.receipt` is absent when request validation fails before dispatch.
Other documented request failures attach a receipt.
HTTP error messages omit response bodies and credentials.
Schema errors never return partial answers.
Callers can use `receipt.to_dict()` to retain diagnostics.

The full endpoint must end in `/v1/systemone`.
Credentials in URLs, query parameters, and fragments are rejected.
An API key goes only in the authorization header.
HTTPX environment proxies are disabled.
The caller must provide a directly reachable service.

## Cancellation boundaries

Async callers can cancel the request task or supply an `asyncio.Event`.
Both clients also accept a `threading.Event` through the `cancel` argument.
The cancellation watcher polls explicit events every 10 milliseconds.
A pre-set event prevents dispatch after request validation.

The synchronous client owns an event loop for each call.
Inside an existing event loop, it raises `DecisionInputError` with the async API instruction.
The async client uses the caller's loop.
Each request owns a separate HTTP connection.
One request's cancellation cannot close another request's connection.

Cancellation joins the local request and watcher tasks before returning control.
`DecisionCancelled` inherits `asyncio.CancelledError`, so ordinary `except Exception` handlers do not erase cancellation semantics.
Its receipt records the request digest and elapsed time.
The server can already finish inference before the socket closes.
The receipt therefore makes no server-stop claim.

## Timeout and recovery

`timeout` is a positive finite HTTP phase timeout in seconds.
It is not a total request deadline.
Cancellation can impose a caller-owned deadline.
A timeout or dropped connection can leave an unknown server outcome.
This client sends read-only inference requests, so it has no tool effect to roll back.

A02 tested cancellation after a real server slot became active.
A later real request succeeded on the same owned service.
Another active request failed with `RemoteProtocolError` after the owned process terminated.
The live runner stores slot observations, client receipts, and the process log.
The shared chat, embedding, and decision services remained available.

The client has no built-in uncertainty threshold.
Low confidence and semantically wrong answers can pass protocol validation.
An application must establish its own classes, evidence limits, and fallback policy before adoption.
PM retains LLM workflows unless a later item explicitly chooses another path.
