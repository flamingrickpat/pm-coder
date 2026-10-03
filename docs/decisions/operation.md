# Operation and limits

## Summary

A02 verified the Laya BF16 artifact with llama.cpp build `b11374-b92761a51`.
The decision endpoint uses port 8082, separate from chat and embeddings.
Protocol success establishes valid typed answers.
It does not establish reliable task classification or calibrated probabilities.

Parent: [decision client](../index.md).
Contracts: [requests and receipts](requests.md).
Recovery: [failure and cancellation](failures.md).

## Installation and public imports

```powershell
uv pip install "pm-coder @ git+https://github.com/flamingrickpat/pm-coder.git@main"
```

Use the exact tested revision from the application lock for reproducible installation.
The distribution installs `pm_decision` beside `pm_coder` and `pm_bash_machine`.
No agent session starts when the decision module imports.

The real A01 report remains in PM's ignored evidence directory.
This example reads that captured report without inventing a replacement.

```python
from pathlib import Path
from pm_decision import DecisionClient

report = Path(
    "C:/source/pm/pm_next_v2/.runtime/alpha/a01/console-spamuel/instances/"
    "fd71e4655da44430b9779eb6739d45f5/managed_tasks/"
    "dd8a80caec16462b97e0e85cbdf87346/outbox/report.yaml"
).read_text(encoding="utf-8")
result = DecisionClient().choice(
    report, "What is the task status?",
    {"COMPLETE": "The task completed.", "FAILED": "The task failed.",
     "INVALID": "The report cannot establish a result."},
)
print(result.answer.choice)
print(result.receipt.to_dict())
```

Async use imports `AsyncDecisionClient` and awaits the same methods.
`probability` exposes the provider's `noul` field without converting it to a boolean.
The caller must not treat model probabilities as reviewed truth labels.

## Verified service startup

If port 8082 is free, start the verified binary with the selected model.
Use a hidden process for background operation on Windows.

```powershell
$decisionArgs = @(
  '--host', '127.0.0.1', '--port', '8082', '--alias', 'laya',
  '--model', 'C:/source/ai/Laya-BF16.gguf', '--device', 'CUDA1',
  '--n-gpu-layers', '99', '--jinja', '--parallel', '1',
  '--ctx-size', '8192', '--batch-size', '8192', '--ubatch-size', '8192',
  '--metrics', '--slots'
)
Start-Process -FilePath 'C:/workspace/llama-systemone-b11374/llama-server.exe' `
  -ArgumentList $decisionArgs -WindowStyle Hidden
Invoke-RestMethod 'http://127.0.0.1:8082/props'
Invoke-RestMethod 'http://127.0.0.1:8082/v1/models'
```

`/props` must identify the build, artifact path, context, and `laya` alias.
`/v1/models` alone establishes discovery, not inference.
Use an actual client call to establish typed inference.
Chat remains on 8080 and embeddings remain on 8081.
Port 8093 belongs to the console smoke.

## Effective limits and observed quality

The model declares an 8192-token architectural context and a 192-token decision head budget.
The server truncates long question and option text to that head budget.
It does not expose this truncation in the typed response.
Keep instructions and descriptions short.
The client byte limit does not guarantee that the model sees every instruction token.

A02 accepted a 7,951-token request with real Spamuel history and the captured completed task report at its end.
It rejected the longer 8,300-token history prefix plus report with HTTP 400.
The completed label remained the highest probability in the measured long examples.
Its confidence fell to 0.154 with the report at the end of the longest accepted input.
This observation cannot establish dependable long-input classification.

The publisher describes the base English checkpoint's trained context as 512 tokens.
It also documents calibration and `noul` limitations in the [model card](https://huggingface.co/convaiinnovations/laya).
The runtime's larger architectural context does not establish equivalent trained behavior.
See the [System One API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api) for wire schemas.

## Reproduce the real gates

From the PM checkout with its tested dependencies installed:

```powershell
uv run --locked python scripts/alpha_decision_gate.py --out .runtime/alpha/a02-repeat/decisions
```

The output directory must be new and inside that checkout's `.runtime`.
The runner reads the real seed and A01 report, then measures short and long inputs.
It stores all source layouts separately from receipts.
It reserves port 8084 for an owned CPU process with the same binary and artifact.
It cancels active inference and terminates only that process during the service-loss gate.
It also measures three warm chat samples with and without real Laya load.
These small workstation measurements are not a stable performance guarantee.

`test_decision.py` uses captured protocol responses plus explicit corruptions.
Those tests establish validation contracts, not model quality.
The A02 application manual retains the full live receipts and known companion failures.
