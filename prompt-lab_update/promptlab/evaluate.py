"""Run the golden set against prompt versions. Used by the Evaluation tab and scripts/eval_gate.py."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd

from .cases import Case
from .llm import LLMConfig, LLMError
from .runner import RunOutcome, run_case


def run_eval(cases: list[Case], prompt_ids: list[str], cfg: LLMConfig, workers: int = 4,
             on_progress: Callable[[int, int], None] | None = None) -> tuple[list[RunOutcome], list[dict]]:
    jobs = [(c, p) for c in cases for p in prompt_ids]
    outcomes, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, c, p, cfg): (c, p) for c, p in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            c, p = futures[fut]
            try:
                outcomes.append(fut.result())
            except LLMError as e:
                errors.append({"case_id": c.id, "prompt_id": p, "error": str(e)})
            if on_progress:
                on_progress(i, len(jobs))
    return outcomes, errors


def to_frame(outcomes: list[RunOutcome]) -> pd.DataFrame:
    rows = []
    for o in outcomes:
        for ch in o.checks:
            rows.append({"case_id": o.case.id, "email": o.case.subject, "tests": o.case.tag,
                         "prompt_id": o.prompt_id, "check": ch.name, "key": ch.key,
                         "passed": ch.passed, "detail": ch.detail,
                         "latency_s": o.result.latency_s, "output_tokens": o.result.output_tokens})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    per_case = df.groupby(["prompt_id", "case_id"])["passed"].all().groupby("prompt_id").mean()
    out = df.groupby("prompt_id").agg(checks_passed=("passed", "sum"), checks_total=("passed", "size"),
                                      avg_latency_s=("latency_s", "mean"))
    out["pass_rate"] = out["checks_passed"] / out["checks_total"]
    out["emails_fully_correct"] = per_case
    return out.reset_index()
