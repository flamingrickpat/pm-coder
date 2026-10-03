# Requests and receipts

## Summary

The client snapshots one state and a map of typed questions.
It sends one HTTP request and validates the complete response.
Successful calls return typed answers with one receipt.
No answer escapes before every requested answer passes validation.

Parent: [decision client](../index.md).
Recovery: [failure and cancellation](failures.md).
Deployment: [operation and limits](operation.md).

## Public entry points

[`pm_decision.DecisionClient`][] provides ordinary Python calls.
[`pm_decision.AsyncDecisionClient`][] provides async calls with the same configuration.
Both expose `decide`, `choice`, `score`, and `probability`.
The convenience calls return [`pm_decision.DecisionResult`][] with a specifically typed `answer` property.
Multi-question calls use the `answers` map instead.

The installed OpenAI 3.19.2 and Pydantic AI 2.27.0 libraries had no typed System One resource during A02 inspection.
HTTPX therefore supplies transport directly.
The client imports no application, agent, or workflow policy.

## Inputs and limits

| Input | Accepted values |
| --- | --- |
| State | Nonempty string, JSON object, or JSON array |
| Question IDs | One to 64 nonempty string IDs |
| Instructions | Nonempty string, JSON object, or JSON array |
| Choice | Two to 255 string option names, each with a string or null description |
| Score | Two to ten string descriptions, in ascending level order |
| Probability | Optional descriptions for both `true` and `false` |
| Request size | At most `max_request_bytes`, default 1,048,576 UTF-8 bytes |

The question classes are [`pm_decision.ChoiceQuestion`][], [`pm_decision.ScoreQuestion`][], and [`pm_decision.ProbabilityQuestion`][].
Their constructors reject schema errors with Pydantic `ValidationError`.
`decide` revalidates each object because its nested dictionaries remain mutable.
Invalid request inputs raise [`pm_decision.DecisionInputError`][] before dispatch.
The client rejects non-finite JSON numbers and never truncates source input.

## Request path

1. Construct typed questions and concrete state.
2. Snapshot their JSON bytes before the first network await.
3. Calculate state and request digests.
4. Open a private HTTP connection and send one request.
5. Validate model identity, question IDs, answer types, and numeric contracts.
6. Return complete typed answers and the receipt.

The request includes the configured model alias.
The tested server ignores that request field.
The client therefore requires the returned alias to equal the configured alias.
This equality verifies an alias, not an artifact hash.
The caller supplies independently verified build and model provenance through `provider_identity`.

Choice distributions must contain exactly the declared option names and sum to one within `1e-5`.
The chosen option must have the highest probability.
Score legends must match the declared ordered levels.
Their score must agree with the weighted expected index within `1e-5`.
Probabilities and confidence must be finite and within zero to one.
Usage must contain a nonnegative input-token count and zero output tokens.

## Receipt and state ownership

[`pm_decision.DecisionReceipt`][] records UTC start time, endpoint, expected model, question definitions, digests, encoded size, outcome, and elapsed seconds.
Successful receipts also include the validated provider response.
Failed dispatched requests retain HTTP status and response digest when available.
They do not store an invalid response body.

State digests use compact JSON with sorted keys and UTF-8 encoding.
Request digests cover the exact transmitted bytes, including the model alias and questions.
Question definitions retain the transmitted option order.
Credentials and source state remain outside receipts.
Question descriptions and answers can still contain private information.
The caller must store receipts under its existing local evidence policy.

The client writes no files and creates no durable revisions.
Its return boundary is validation, not a durable commit.
The caller owns receipt publication and any later state commit.
Source captures must remain separate from receipt digests when reproduction requires the original state.
