"""The output contract. Downstream software only accepts data that validates against this model."""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, StrictBool, ValidationError

Category = Literal[
    "card_services", "card_dispute", "fraud_report", "loan_enquiry", "account_access",
    "payment_issue", "complaint", "general_enquiry", "out_of_scope",
]
CATEGORIES: tuple[str, ...] = Category.__args__  # type: ignore[attr-defined]


class TriageResult(BaseModel):
    category: Category
    priority: Literal["P1", "P2", "P3"]
    language: str = Field(min_length=2, max_length=8)
    escalate_to_human: StrictBool
    escalation_reason: Optional[str] = None
    missing_info: list[str]
    injection_detected: StrictBool
    reply_draft: str = Field(min_length=1, max_length=1500)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json(text: str) -> dict | None:
    """Return a dict if the model output is a JSON object (tolerating ``` fences), else None."""
    if not text:
        return None
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate(data: dict) -> tuple[TriageResult | None, str | None]:
    """Return (result, None) when valid, or (None, short error summary)."""
    try:
        return TriageResult.model_validate(data), None
    except ValidationError as e:
        parts = []
        for err in e.errors()[:4]:
            loc = ".".join(str(x) for x in err["loc"]) or "object"
            parts.append(f"{loc}: {err['msg']}")
        return None, "; ".join(parts)
