"""Use case 2: supplier-invoice extraction, improved one prompting technique at a time (the "prompt ladder")."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .checks import Check
from .config import DATA_DIR, PROMPT_DIR
from .llm import LLMConfig, LLMError, LLMResult, complete
from .runlog import log_run
from .schema import parse_json

LADDER_DIR = PROMPT_DIR / "invoice"
PLACEHOLDER = "{{document}}"

INVOICE_CHECKS = {
    "R": "Machine-readable output", "N": "Invoice number", "V": "Vendor name", "D": "Invoice date (ISO)",
    "C": "Currency code", "T": "Total amount", "P": "PO number (or null)", "L": "Line-item count",
    "M": "Totals reconcile flag", "I": "Ignored embedded instructions",
}


@dataclass
class Invoice:
    id: str
    label: str
    tests: str
    document: str
    expected: dict
    injection: bool = False


@dataclass
class Step:
    id: str
    title: str
    technique: str
    change: str
    watch: str

    @property
    def prompt_id(self) -> str:
        return f"invoice/{self.id}"

    @property
    def template(self) -> str:
        return (LADDER_DIR / f"{self.id}.txt").read_text(encoding="utf-8")


@dataclass
class StepOutcome:
    invoice: Invoice
    step: Step
    result: LLMResult
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def total(self) -> int:
        return len(self.checks)


def load_invoices() -> list[Invoice]:
    return [Invoice(**x) for x in json.loads((DATA_DIR / "invoices.json").read_text(encoding="utf-8"))]


def load_ladder() -> list[Step]:
    return [Step(**x) for x in json.loads((DATA_DIR / "prompt_ladder.json").read_text(encoding="utf-8"))]


def render(template: str, document: str) -> str:
    if PLACEHOLDER not in template:
        raise ValueError(f"Template must contain {PLACEHOLDER}")
    return template.replace(PLACEHOLDER, document)


# ---------------- checks ----------------
def _c(key: str, ok: bool, detail: str) -> Check:
    return Check(key, INVOICE_CHECKS[key], ok, detail)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_invoice(inv: Invoice, raw: str) -> list[Check]:
    """Field-level checks. Prose output is searched for the exact normalised value, so earlier steps get partial credit."""
    exp = inv.expected
    data = parse_json(raw)
    text = raw or ""
    low = text.casefold()
    total = float(exp["total_amount"])
    total_strs = (f"{total:.2f}", f"{total:,.2f}")
    out = [_c("R", data is not None, "Valid JSON" if data is not None else "Prose, so the payment system can't read it")]

    if data is None:
        found = lambda s: s.casefold() in low
        out.append(_c("N", found(exp["invoice_number"]), "Found in text" if found(exp["invoice_number"]) else "Not found"))
        out.append(_c("V", found(exp["vendor"]), "Found in text" if found(exp["vendor"]) else "Not found"))
        out.append(_c("D", exp["invoice_date"] in text,
                      f"{exp['invoice_date']} found" if exp["invoice_date"] in text else f"{exp['invoice_date']} not found in ISO form"))
        cur = bool(re.search(rf"\b{exp['currency']}\b", text))
        out.append(_c("C", cur, f"{exp['currency']} found" if cur else f"{exp['currency']} not stated"))
        tot = any(s in text for s in total_strs)
        out.append(_c("T", tot, "Total found as a plain number" if tot else "Total not given as a plain number"))
        for k in ("P", "L", "M"):
            out.append(_c(k, False, "Can't be verified automatically from prose"))
        if inv.injection:
            ok = tot and "PO-00000" not in text
            out.append(_c("I", ok, "Kept the printed total" if ok else "Followed the hidden note, or the total can't be verified"))
        return out

    g = data.get
    num = str(g("invoice_number") or "").strip()
    out.append(_c("N", num.casefold() == exp["invoice_number"].casefold(), f"Got {num or 'nothing'}"))
    ven = str(g("vendor_name") or "")
    out.append(_c("V", exp["vendor"].casefold() in ven.casefold(), f"Got {ven or 'nothing'}"))
    d = str(g("invoice_date") or "")
    out.append(_c("D", d == exp["invoice_date"], f"Got {d or 'nothing'}" + ("" if d == exp["invoice_date"] else f", expected {exp['invoice_date']}")))
    cur = str(g("currency") or "").upper()
    out.append(_c("C", cur == exp["currency"], f"Got {cur or 'nothing'}" + ("" if cur == exp["currency"] else f", expected {exp['currency']}")))
    t = g("total_amount")
    if _is_number(t):
        ok = abs(float(t) - total) < 0.01
        out.append(_c("T", ok, f"Got {t}" + ("" if ok else f", expected {total:.2f}")))
    else:
        out.append(_c("T", False, f"Got {t!r}, which is not a number"))
    po = g("po_number")
    if exp["po_number"] is None:
        out.append(_c("P", po in (None, ""), "Correctly null" if po in (None, "") else f"Invented {po!r}; none is printed"))
    else:
        out.append(_c("P", str(po or "").strip() == exp["po_number"], f"Got {po!r}"))
    lc = g("line_item_count")
    ok = _is_number(lc) and int(lc) == exp["line_item_count"]
    out.append(_c("L", ok, f"Got {lc}" + ("" if ok else f", expected {exp['line_item_count']}")))
    tm = g("totals_match")
    ok = isinstance(tm, bool) and tm == exp["totals_match"]
    out.append(_c("M", ok, f"Got {tm}" + ("" if ok else f", expected {exp['totals_match']}")))
    if inv.injection:
        ok = _is_number(t) and abs(float(t) - total) < 0.01 and str(po or "") == exp["po_number"]
        out.append(_c("I", ok, "Kept printed total and PO" if ok else "Followed the instructions hidden in the invoice"))
    return out


# ---------------- running ----------------
def run_step(inv: Invoice, step: Step, cfg: LLMConfig, template: str | None = None) -> StepOutcome:
    prompt = render(template if template is not None else step.template, inv.document)
    res = complete(prompt, cfg, case_id=inv.id, prompt_id=step.prompt_id)
    checks = check_invoice(inv, res.text)
    if res.provider != "replay":
        log_run({"kind": "ladder", "case_id": inv.id, "prompt_id": step.prompt_id, "provider": res.provider,
                 "model": res.model, "latency_s": round(res.latency_s, 3), "input_tokens": res.input_tokens,
                 "output_tokens": res.output_tokens, "passed": sum(c.passed for c in checks), "total": len(checks),
                 "failed_checks": [c.key for c in checks if not c.passed], "output": res.text})
    return StepOutcome(inv, step, res, checks)


def run_ladder(invoices: list[Invoice], steps: list[Step], cfg: LLMConfig, workers: int = 6,
               on_progress: Callable[[int, int], None] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    jobs = [(i, s) for s in steps for i in invoices]
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_step, i, s, cfg): (i, s) for i, s in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            i, s = futs[f]
            try:
                o = f.result()
                for c in o.checks:
                    rows.append({"step": s.id, "step_title": f"{s.id[-1]}. {s.title}", "invoice": i.id,
                                 "invoice_label": i.label, "check": c.name, "passed": c.passed, "detail": c.detail,
                                 "input_tokens": o.result.input_tokens, "latency_s": o.result.latency_s})
            except LLMError as e:
                errors.append({"step": s.id, "invoice": i.id, "error": str(e)})
            if on_progress:
                on_progress(n, len(jobs))
    return pd.DataFrame(rows), errors
