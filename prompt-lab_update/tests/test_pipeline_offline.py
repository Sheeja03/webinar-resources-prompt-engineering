"""Pipeline and evaluation tests with a fake model, so they run without an API key."""
import json

import pytest

import promptlab.llm as llm
import promptlab.runlog as runlog
from promptlab.cases import load_cases
from promptlab.evaluate import run_eval, summary, to_frame
from promptlab.pipeline import run_pipeline

CASES = {c.id: c for c in load_cases()}


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(runlog, "RUN_LOG", tmp_path / "log.jsonl")


def fake(outputs):
    """Patch the provider call to return the given outputs in order."""
    it = iter(outputs)

    def _complete(prompt, cfg, **kw):
        return llm.LLMResult(next(it), 0.01, len(prompt) // 4, 50, "fake", "fake-model")
    return _complete


def obj(**over):
    base = {"category": "card_dispute", "priority": "P3", "language": "en", "escalate_to_human": False,
            "escalation_reason": None, "missing_info": [], "injection_detected": False,
            "reply_draft": "A specialist will review your dispute."}
    base.update(over)
    return json.dumps(base)


def test_pipeline_redacts_before_model(monkeypatch):
    seen = {}

    def _complete(prompt, cfg, **kw):
        seen["prompt"] = prompt
        return llm.LLMResult(obj(category="fraud_report"), 0.01, 1, 1, "fake", "m")
    monkeypatch.setattr("promptlab.pipeline.complete", _complete)
    res = run_pipeline(CASES["c2"], "engineered_v1", llm.LLMConfig(provider="fake"))
    assert "4539" not in seen["prompt"]
    assert res.final.priority == "P1" and res.final.escalate_to_human  # enforced in code


def test_pipeline_repairs_invalid_output(monkeypatch):
    monkeypatch.setattr("promptlab.pipeline.complete", fake(["not json", obj()]))
    res = run_pipeline(CASES["c5"], "engineered_v1", llm.LLMConfig(provider="fake"))
    assert len(res.raw_outputs) == 2 and res.final is not None
    assert res.final.injection_detected and res.final.escalate_to_human  # input guardrail flag applied


def test_pipeline_blocks_unsafe_reply(monkeypatch):
    monkeypatch.setattr("promptlab.pipeline.complete",
                        fake([obj(reply_draft="Your refund has been approved.")]))
    res = run_pipeline(CASES["c5"], "engineered_v1", llm.LLMConfig(provider="fake"))
    assert res.reply_to_send is None


def test_evaluation_summary(monkeypatch):
    monkeypatch.setattr("promptlab.runner.complete", fake(["free text"] * 8))
    outcomes, errors = run_eval(list(CASES.values()), ["naive_v1"], llm.LLMConfig(provider="fake"), workers=1)
    s = summary(to_frame(outcomes)).iloc[0]
    assert not errors and s.emails_fully_correct == 0
