# Optional decision client

## Summary

The pm-coder distribution includes `pm_decision` from version 0.10.0.
This module sends typed System One requests without an agent session.
It preserves answers and returns diagnostic receipts.
The caller owns uncertainty policy, fallback, and storage.

The existing coding agent remains in `pm_coder`.
Importing `pm_decision` loads neither that agent nor `pm-workflows`.
The distribution still installs the agent's dependencies.
The decision module itself uses Python, Pydantic, and HTTPX.

## Find the mechanism

| Page | Purpose |
| --- | --- |
| [Requests and receipts](decisions/requests.md) | Concrete inputs, validation, answers, and evidence ownership |
| [Failure and cancellation](decisions/failures.md) | Error recovery, cancellation boundaries, and concurrent calls |
| [Operation and limits](decisions/operation.md) | Real deployment, measured limits, and installation |
| [Generated API](reference/pm_decision.md) | Exact Python objects, signatures, and source |
| [MCP request timeouts](mcp-timeouts.md) | Native transport deadlines and cancellation boundaries |

PM's engineering manual records the A02 application receipts.
Laya is optional.
Existing PM workflows retain their LLM models.
No workflow must adopt Laya after this delivery.
