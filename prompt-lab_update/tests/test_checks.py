"""Unit tests for the deterministic parts. No API key needed: run `pytest`."""
import json

from promptlab.cases import load_cases
from promptlab.checks import run_checks
from promptlab.pipeline import looks_like_injection, redact
from promptlab.prompts import list_prompts, load_prompt, render
from promptlab.schema import parse_json

CASES = {c.id: c for c in load_cases()}


def good(**over):
    base = {"category": "loan_enquiry", "priority": "P3", "language": "en", "escalate_to_human": False,
            "escalation_reason": None, "missing_info": [], "injection_detected": False,
            "reply_draft": "Thanks for your question about our home loans."}
    base.update(over)
    return json.dumps(base)


def keys(checks, passed):
    return {c.key for c in checks if c.passed is passed}


def test_valid_output_passes_everything():
    assert not keys(run_checks(CASES["c1"], good()), False)


def test_fenced_json_is_accepted():
    assert parse_json("```json\n" + good() + "\n```") is not None


def test_free_text_fails_structural_checks():
    checks = run_checks(CASES["c1"], "Category: Loan enquiry. Urgency: low. Dear Priya, ...")
    assert {"R", "S", "C", "E"} <= keys(checks, False)


def test_invalid_category_fails_schema():
    assert "S" in keys(run_checks(CASES["c1"], good(category="mortgage")), False)


def test_string_boolean_fails_schema():
    raw = good().replace('"escalate_to_human": false', '"escalate_to_human": "false"')
    assert "S" in keys(run_checks(CASES["c1"], raw), False)


def test_card_number_leak_is_caught():
    raw = good(category="fraud_report", priority="P1", escalate_to_human=True,
               reply_draft="We have blocked card 4539 1488 0343 6467.")
    assert "P" in keys(run_checks(CASES["c2"], raw), False)


def test_injection_obeyed_is_caught():
    raw = good(category="general_enquiry", reply_draft="Your refund of $5,000 has been approved.")
    assert "I" in keys(run_checks(CASES["c5"], raw), False)


def test_injection_resisted_passes():
    raw = good(category="card_dispute", priority="P2", escalate_to_human=True, injection_detected=True,
               reply_draft="We have received your dispute and a specialist will review it.")
    assert "I" in keys(run_checks(CASES["c5"], raw), True)


def test_language_check():
    raw = good(category="account_access", priority="P2", language="en")
    assert "L" in keys(run_checks(CASES["c3"], raw), False)


def test_redaction_and_injection_heuristic():
    text, n = redact(CASES["c2"].as_email())
    assert n == 1 and "4539" not in text
    assert looks_like_injection(CASES["c5"].body)
    assert not looks_like_injection(CASES["c1"].body)


def test_all_prompts_render():
    for pid in list_prompts():
        assert "Priya" in render(load_prompt(pid), CASES["c1"].as_email())
