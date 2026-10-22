"""Load the golden test set: emails plus the answers we expect."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import GOLDEN_SET


@dataclass
class Expected:
    categories: list[str] = field(default_factory=list)
    escalate: bool | None = None
    language: str | None = None


@dataclass
class Case:
    id: str
    sender: str
    subject: str
    body: str
    tag: str = ""
    why: str = ""
    risk: bool = False
    injection: bool = False
    expected: Expected = field(default_factory=Expected)

    def as_email(self) -> str:
        """The exact text inserted into the prompt template."""
        return f"From: {self.sender}\nSubject: {self.subject}\n\n{self.body}"


def load_cases(path=GOLDEN_SET) -> list[Case]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for item in raw:
        exp = Expected(**item.pop("expected", {}))
        cases.append(Case(expected=exp, **item))
    return cases


def ad_hoc_case(body: str) -> Case:
    """An email pasted live by the audience: no expected answers, so only structural and safety checks run."""
    return Case(id="audience", sender="Customer", subject="(from Teams chat)", body=body, tag="Live from chat")
