"""Check the OpenAI request parameters without calling the API."""
from types import SimpleNamespace

import openai
import pytest

from promptlab.llm import LLMConfig, LLMError, complete


class FakeClient:
    calls = []
    reject_first = False
    content = '{"ok": true}'
    finish = "stop"

    def __init__(self, **kw):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        FakeClient.calls.append(kw)
        if FakeClient.reject_first and len(FakeClient.calls) == 1:
            raise openai.BadRequestError("unsupported parameter", response=SimpleNamespace(
                status_code=400, headers={}, request=None), body=None)
        msg = SimpleNamespace(content=FakeClient.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=FakeClient.finish)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch):
    FakeClient.calls, FakeClient.reject_first = [], False
    FakeClient.content, FakeClient.finish = '{"ok": true}', "stop"
    monkeypatch.setattr(openai, "OpenAI", FakeClient)


def test_reasoning_model_uses_reasoning_effort():
    r = complete("hi", LLMConfig(model="gpt-5.4-mini"))
    kw = FakeClient.calls[0]
    assert "reasoning_effort" in kw and "temperature" not in kw and r.output_tokens == 5


def test_classic_model_uses_temperature():
    complete("hi", LLMConfig(model="gpt-4.1-mini", temperature=0.0))
    assert FakeClient.calls[0]["temperature"] == 0.0


def test_falls_back_when_parameters_rejected():
    FakeClient.reject_first = True
    complete("hi", LLMConfig(model="llama3.1"))
    assert "max_tokens" in FakeClient.calls[1] and "temperature" not in FakeClient.calls[1]


def test_empty_answer_from_token_budget_is_explained():
    FakeClient.content, FakeClient.finish = "", "length"
    with pytest.raises(LLMError, match="token budget"):
        complete("hi", LLMConfig(model="gpt-5.4-mini"))
