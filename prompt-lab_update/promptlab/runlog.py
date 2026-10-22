"""Observability: every model call is appended to runs/log.jsonl. Replay mode reads it back."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from .config import RUN_LOG, RUNS_DIR

_lock = threading.Lock()


def log_run(record: dict) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with _lock, RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_runs() -> list[dict]:
    if not RUN_LOG.exists():
        return []
    rows = []
    for line in RUN_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_output(case_id: str, prompt_id: str) -> dict | None:
    """Most recent live (non-replay) run for this email and prompt version."""
    for row in reversed(read_runs()):
        if (row.get("kind") in ("triage", "ladder") and row.get("case_id") == case_id
                and row.get("prompt_id") == prompt_id and "output" in row):
            return row
    return None


def clear_runs() -> None:
    with _lock:
        if RUN_LOG.exists():
            RUN_LOG.unlink()
