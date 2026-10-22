"""Prompts are code: plain-text templates in /prompts, named <name>_v<version>.txt."""
from __future__ import annotations

import re

from .config import PROMPT_DIR

PLACEHOLDER = "{{email}}"
_NAME_RE = re.compile(r"^(?P<family>[a-z0-9_]+?)_v(?P<version>\d+)$")


def list_prompts() -> list[str]:
    """All prompt ids, e.g. ['engineered_v1', 'naive_v1'], newest version last within each family."""
    def key(name: str):
        m = _NAME_RE.match(name)
        return (m.group("family"), int(m.group("version"))) if m else (name, 0)
    return sorted((p.stem for p in PROMPT_DIR.glob("*.txt")), key=key)


def load_prompt(prompt_id: str) -> str:
    return (PROMPT_DIR / f"{prompt_id}.txt").read_text(encoding="utf-8")


def render(template: str, email: str) -> str:
    if PLACEHOLDER not in template:
        raise ValueError(f"Prompt template must contain {PLACEHOLDER}")
    return template.replace(PLACEHOLDER, email)


def next_version_id(prompt_id: str) -> str:
    m = _NAME_RE.match(prompt_id)
    family = m.group("family") if m else prompt_id
    versions = [int(_NAME_RE.match(p).group("version")) for p in list_prompts()
                if _NAME_RE.match(p) and _NAME_RE.match(p).group("family") == family]
    return f"{family}_v{max(versions, default=0) + 1}"


def save_new_version(prompt_id: str, text: str) -> str:
    """Save edited text as a new version instead of overwriting: prompt changes are releases."""
    if PLACEHOLDER not in text:
        raise ValueError(f"Prompt template must contain {PLACEHOLDER}")
    new_id = next_version_id(prompt_id)
    (PROMPT_DIR / f"{new_id}.txt").write_text(text, encoding="utf-8")
    return new_id
