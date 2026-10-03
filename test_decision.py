"""Check the boundary with captured responses and explicit corruptions.

These checks establish protocol validation only. The A02 live runner records
actual inference, cancellation, timeout, and process loss separately.
"""
import asyncio
import copy
import json
from pathlib import Path
import sys
import threading

import pytest

from pm_decision import (
    AsyncDecisionClient, ChoiceQuestion, DecisionCancelled, DecisionClient,
    DecisionInputError, ProbabilityQuestion, ScoreQuestion, _validate_response,
)

FIXTURE = Path(__file__).parent / "tests/fixtures/systemone"
REQUEST = json.loads((FIXTURE / "captured-request.json").read_text())
RESPONSE = json.loads((FIXTURE / "captured-response.json").read_text())


def validate(value):
    return _validate_response(json.dumps(value).encode(), REQUEST["questions"], "laya")


def test_captured_three_primitive_response():
    value = validate(RESPONSE)
    assert value.answers["route"].choice == "COMPLETE"
    assert value.answers["quality"].score == RESPONSE["answers"]["quality"]["score"]
    assert value.answers["replied"].noul == RESPONSE["answers"]["replied"]["noul"]


@pytest.mark.parametrize("mutation", [
    "missing", "extra", "model", "type", "nan", "infinity", "range",
    "mass", "keys", "unknown_choice", "wrong_choice", "legend", "score",
    "score_range", "confidence", "usage", "bool_probability",
])
def test_corrupted_capture_is_rejected(mutation):
    value = copy.deepcopy(RESPONSE)
    answers = value["answers"]
    if mutation == "missing": del answers["replied"]
    elif mutation == "extra": answers["extra"] = answers["replied"]
    elif mutation == "model": value["model"] = "qwen"
    elif mutation == "type": answers["replied"]["type"] = "score"
    elif mutation == "nan": answers["replied"]["noul"] = float("nan")
    elif mutation == "infinity": answers["replied"]["noul"] = float("inf")
    elif mutation == "range": answers["replied"]["noul"] = 1.1
    elif mutation == "mass": answers["route"]["probabilities"]["COMPLETE"] = 0.1
    elif mutation == "keys": del answers["route"]["probabilities"]["INVALID"]
    elif mutation == "unknown_choice": answers["route"]["choice"] = "invented"
    elif mutation == "wrong_choice": answers["route"]["choice"] = "FAILED"
    elif mutation == "legend": answers["quality"]["legend"]["0"] = "wrong level"
    elif mutation == "score": answers["quality"]["score"] = 1.0
    elif mutation == "score_range": answers["quality"]["score"] = -0.1
    elif mutation == "confidence": answers["route"]["confidence"] = float("nan")
    elif mutation == "usage": value["usage"]["input_tokens"] = -1
    elif mutation == "bool_probability": answers["replied"]["noul"] = True
    with pytest.raises(ValueError): validate(value)


@pytest.mark.parametrize("state", [None, 1, True, "", "   ", {}, [], {"x": float("nan")}, {"x": object()}])
def test_bad_state_fails_before_network(state):
    with pytest.raises(DecisionInputError):
        DecisionClient().probability(state, "Is the task done?")


def test_input_snapshot_and_null_descriptions():
    question = ChoiceQuestion(instructions="What happened?", criteria={"yes": None, "no": None})
    client = DecisionClient(api_key="secret")
    body, receipt = client._prepare({"task": "actual report"}, {"status": question})
    question.criteria["yes"] = "changed later"
    assert json.loads(body)["questions"]["status"]["criteria"]["yes"] is None
    assert "secret" not in json.dumps(receipt.to_dict())
    assert "actual report" not in json.dumps(receipt.to_dict())
    assert receipt.request_bytes == len(body)


def test_invalid_question_mutation_fails_before_network():
    question = ChoiceQuestion(instructions="What happened?", criteria={"yes": None, "no": None})
    question.criteria.clear()
    with pytest.raises(DecisionInputError):
        DecisionClient().decide("report", {"status": question})


@pytest.mark.parametrize("kwargs", [
    {"endpoint": "http://user:secret@localhost/v1/systemone"},
    {"endpoint": "http://localhost/v1/chat/completions"},
    {"endpoint": "http://localhost/v1/systemone?key=secret"},
    {"timeout": float("nan")}, {"timeout": 0}, {"model": ""},
    {"max_request_bytes": 0}, {"max_request_bytes": 1.5},
    {"api_key": "secret\ninvalid"},
])
def test_bad_connection_configuration(kwargs):
    with pytest.raises(DecisionInputError): DecisionClient(**kwargs)


def test_request_byte_limit_does_not_truncate():
    with pytest.raises(DecisionInputError):
        DecisionClient(max_request_bytes=50).probability("report", "Is it done?")


def test_cancelled_call_retains_digest_and_starts_no_request():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DecisionCancelled) as info:
        DecisionClient(endpoint="http://127.0.0.1:9/v1/systemone").probability("report", "Is it done?", cancel=cancel)
    assert info.value.receipt.outcome == "cancelled"
    assert info.value.receipt.http_status is None


def test_sync_call_inside_event_loop_explains_async_api():
    async def run():
        with pytest.raises(DecisionInputError, match="AsyncDecisionClient"):
            DecisionClient().probability("report", "Is it done?")
    asyncio.run(run())


def test_import_does_not_load_agent_or_workflow_engine():
    import subprocess

    code = "import sys, pm_decision; assert not any(n in sys.modules for n in ('pm_coder','pm_workflows','ghost','pydantic_ai'))"
    subprocess.run([sys.executable, "-c", code], check=True)
