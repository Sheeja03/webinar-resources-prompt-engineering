"""The prompt is one component. This pipeline adds the controls that surround it in production:

1. Input guardrails   - redact card numbers, flag likely prompt injection (deterministic code)
2. Prompt assembly    - versioned template + sanitised email
3. Model call         - via the provider-neutral client
4. Validate + repair  - schema check; on failure, one retry with the validation error
5. Policy enforcement - business rules applied in code, not left to the model
6. Output guardrails  - block replies that leak data or promise outcomes
7. Route              - send to a team queue, or to human review as the safe fallback
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .cases import Case
from .checks import APPROVED_RE, CARD_RE
from .llm import LLMConfig, LLMError, complete
from .prompts import load_prompt, render
from .runlog import log_run
from .schema import TriageResult, parse_json, validate

INJECTION_PATTERNS = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions|system note|you are now|"
    r"disregard (your|the) (rules|instructions)|act as (an? )?(admin|system)",
    re.IGNORECASE,
)

ALWAYS_ESCALATE = {"fraud_report", "card_dispute", "complaint"}
QUEUES = {
    "card_services": "Cards team", "card_dispute": "Disputes team", "fraud_report": "Fraud desk (24x7)",
    "loan_enquiry": "Lending team", "account_access": "Digital support", "payment_issue": "Payments operations",
    "complaint": "Complaints and resolution", "general_enquiry": "Customer service", "out_of_scope": "Customer service",
}


@dataclass
class Step:
    name: str
    status: str          # ok | warn | fail
    detail: str
    ms: int = 0


@dataclass
class PipelineResult:
    steps: list[Step] = field(default_factory=list)
    final: TriageResult | None = None
    queue: str = "Human review"
    reply_to_send: str | None = None
    raw_outputs: list[str] = field(default_factory=list)


def redact(text: str) -> tuple[str, int]:
    return CARD_RE.subn("[CARD_REDACTED]", text)


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION_PATTERNS.search(text))


def run_pipeline(case: Case, prompt_id: str, cfg: LLMConfig, max_repairs: int = 1) -> PipelineResult:
    out = PipelineResult()
    t = time.perf_counter
    ms = lambda s: int((t() - s) * 1000)

    # 1. Input guardrails
    s = t()
    email, n_redacted = redact(case.as_email())
    injection_flag = looks_like_injection(email)
    detail = f"Redacted {n_redacted} card number(s)." if n_redacted else "No card numbers found."
    detail += " Possible prompt injection flagged." if injection_flag else " No injection patterns found."
    out.steps.append(Step("Input guardrails", "warn" if (n_redacted or injection_flag) else "ok", detail, ms(s)))

    # 2. Prompt assembly
    s = t()
    prompt = render(load_prompt(prompt_id), email)
    out.steps.append(Step("Prompt assembly", "ok", f"Template {prompt_id}, ~{len(prompt) // 4} tokens", ms(s)))

    # 3 + 4. Model call, validation and repair loop
    result, data = None, None
    attempt_prompt = prompt
    for attempt in range(max_repairs + 1):
        s = t()
        try:
            llm = complete(attempt_prompt, cfg, case_id=case.id, prompt_id=prompt_id)
        except LLMError as e:
            out.steps.append(Step("Model call", "fail", str(e), ms(s)))
            out.steps.append(Step("Route", "warn", "Model unavailable. Sent to human review.", 0))
            return out
        out.raw_outputs.append(llm.text)
        out.steps.append(Step(f"Model call (attempt {attempt + 1})", "ok",
                              f"{llm.provider} / {llm.model}, {llm.latency_s:.1f} s, out tokens: {llm.output_tokens}", ms(s)))
        s = t()
        data = parse_json(llm.text)
        result, err = validate(data) if data is not None else (None, "Output is not a JSON object")
        if result:
            out.steps.append(Step("Schema validation", "ok", "Output matches the TriageResult contract", ms(s)))
            break
        out.steps.append(Step("Schema validation", "fail", err or "Invalid", ms(s)))
        attempt_prompt = (prompt + "\n\nYour previous answer failed validation with this error:\n"
                          + (err or "invalid") + "\nReturn ONLY a corrected JSON object.")

    if result is None:
        out.steps.append(Step("Route", "warn", "Could not get valid output. Sent to human review.", 0))
        _log(case, prompt_id, out)
        return out

    # 5. Policy enforcement in code
    s = t()
    overrides = []
    if result.category in ALWAYS_ESCALATE and not result.escalate_to_human:
        result.escalate_to_human = True
        overrides.append(f"escalation forced for {result.category}")
    if injection_flag and not result.injection_detected:
        result.injection_detected = True
        result.escalate_to_human = True
        overrides.append("injection flagged by input guardrail")
    if result.category == "fraud_report" and result.priority != "P1":
        result.priority = "P1"
        overrides.append("fraud set to P1")
    out.steps.append(Step("Policy enforcement", "warn" if overrides else "ok",
                          "Overrides: " + "; ".join(overrides) if overrides else "Model output already complies with policy", ms(s)))

    # 6. Output guardrails
    s = t()
    problems = []
    if CARD_RE.search(result.reply_draft):
        problems.append("reply contains a card number")
    if APPROVED_RE.search(result.reply_draft):
        problems.append("reply promises an approval")
    if problems:
        out.steps.append(Step("Output guardrails", "fail", "Blocked: " + "; ".join(problems), ms(s)))
    else:
        out.reply_to_send = result.reply_draft
        out.steps.append(Step("Output guardrails", "ok", "Reply is safe to send as a draft", ms(s)))

    # 7. Route
    out.final = result
    if result.escalate_to_human or problems:
        out.queue = f"Human review, then {QUEUES[result.category]}"
    else:
        out.queue = QUEUES[result.category]
    out.steps.append(Step("Route", "ok", f"{out.queue} (priority {result.priority})", 0))
    _log(case, prompt_id, out)
    return out


def _log(case: Case, prompt_id: str, out: PipelineResult) -> None:
    log_run({"kind": "pipeline", "case_id": case.id, "prompt_id": prompt_id, "provider": "pipeline",
             "queue": out.queue, "steps": [s.__dict__ for s in out.steps]})
