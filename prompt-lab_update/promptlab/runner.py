"""Run one email through one prompt version, score it and log it."""
from __future__ import annotations

from dataclasses import dataclass

from .cases import Case
from .checks import Check, run_checks, score
from .llm import LLMConfig, LLMResult, complete
from .prompts import load_prompt, render
from .runlog import log_run


@dataclass
class RunOutcome:
    case: Case
    prompt_id: str
    result: LLMResult
    checks: list[Check]

    @property
    def passed(self) -> int:
        return score(self.checks)[0]

    @property
    def total(self) -> int:
        return score(self.checks)[1]


def run_case(case: Case, prompt_id: str, cfg: LLMConfig, template: str | None = None) -> RunOutcome:
    template = template if template is not None else load_prompt(prompt_id)
    prompt = render(template, case.as_email())
    result = complete(prompt, cfg, case_id=case.id, prompt_id=prompt_id)
    checks = run_checks(case, result.text)
    if result.provider != "replay":
        log_run({
            "kind": "triage", "case_id": case.id, "prompt_id": prompt_id,
            "provider": result.provider, "model": result.model,
            "latency_s": round(result.latency_s, 3),
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "passed": score(checks)[0], "total": score(checks)[1],
            "failed_checks": [c.key for c in checks if not c.passed],
            "output": result.text,
        })
    return RunOutcome(case, prompt_id, result, checks)
