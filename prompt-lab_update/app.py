"""Prompt Lab: from prompt engineering to production AI.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import difflib
import html
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from promptlab.cases import Case, ad_hoc_case, load_cases
from promptlab.checks import CHECK_NAMES
from promptlab.config import DEFAULT_MODELS, PROVIDER_LABELS, provider_ready
from promptlab.evaluate import run_eval, summary, to_frame
from promptlab.invoice import load_invoices, load_ladder, run_ladder, run_step
from promptlab.llm import LLMConfig, LLMError
from promptlab.pipeline import run_pipeline
from promptlab.prompts import list_prompts, load_prompt, next_version_id, save_new_version
from promptlab.runlog import clear_runs, read_runs
from promptlab.runner import RunOutcome, run_case

st.set_page_config(page_title="Prompt Lab: Meridian Bank", page_icon="🧪", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1500px;}
  .email {background: var(--secondary-background-color); border-radius: 12px; padding: 1rem 1.25rem; white-space: pre-wrap; font-size: 1.05rem;}
  .check {padding: .35rem 0; border-bottom: 1px solid rgba(128,128,128,.2);}
  .check small {opacity: .75;}
</style>
""", unsafe_allow_html=True)

CASES = load_cases()
CASE_BY_ID = {c.id: c for c in CASES}
INVOICES = {i.id: i for i in load_invoices()}
LADDER = load_ladder()
st.session_state.setdefault("live", {})
st.session_state.setdefault("ladder_live", {})


# ---------------- sidebar: model + prompt selection ----------------
with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", list(PROVIDER_LABELS), format_func=PROVIDER_LABELS.get)
    ready, msg = provider_ready(provider)
    (st.success if ready else st.warning)(msg)
    model = st.text_input("Model / deployment", value=DEFAULT_MODELS.get(provider, ""),
                          disabled=provider == "replay",
                          help="gpt-5.4-mini is fast and inexpensive for live demos. Change OPENAI_MODEL in .env to set the default.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                            help="0 keeps outputs as consistent as possible, which production systems prefer.")
    if model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
        st.caption("Reasoning models ignore temperature; reasoning effort is set by OPENAI_REASONING_EFFORT.")
    cfg = LLMConfig(provider=provider, model=model, temperature=temperature)

    st.header("Prompts to compare")
    prompt_ids = list_prompts()
    naive_default = next((p for p in prompt_ids if p.startswith("naive")), prompt_ids[0])
    eng_default = [p for p in prompt_ids if p.startswith("engineered")][-1] if any(
        p.startswith("engineered") for p in prompt_ids) else prompt_ids[-1]
    prompt_a = st.selectbox("Prompt A (baseline)", prompt_ids, index=prompt_ids.index(naive_default))
    prompt_b = st.selectbox("Prompt B (candidate)", prompt_ids, index=prompt_ids.index(eng_default))
    st.caption("All emails, names and card numbers are fictional test data.")


# ---------------- helpers ----------------
def show_outcome(o: RunOutcome | dict | None, label: str) -> None:
    if o is None:
        st.info(f"Run {label} to see its output.")
        return
    if isinstance(o, dict):
        st.error(o["error"])
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Checks passed", f"{o.passed} / {o.total}")
    c2.metric("Latency", f"{o.result.latency_s:.1f} s")
    c3.metric("Output tokens", o.result.output_tokens if o.result.output_tokens is not None else "n/a")
    lang = "json" if o.result.text.strip().startswith(("{", "```")) else "text"
    st.code(o.result.text, language=lang, wrap_lines=True)
    for ch in o.checks:
        icon = "✅" if ch.passed else "❌"
        st.markdown(f"<div class='check'>{icon} <b>{ch.name}</b><br><small>{html.escape(ch.detail)}</small></div>",
                    unsafe_allow_html=True)


def show_checks(o) -> None:
    for ch in o.checks:
        icon = "✅" if ch.passed else "❌"
        st.markdown(f"<div class='check'>{icon} <b>{ch.name}</b><br><small>{html.escape(ch.detail)}</small></div>",
                    unsafe_allow_html=True)


def safe_run(case: Case, prompt_id: str) -> RunOutcome | dict:
    try:
        return run_case(case, prompt_id, cfg)
    except (LLMError, ValueError) as e:
        return {"error": str(e)}


# ---------------- header ----------------
st.title("Prompt Lab: Meridian Bank inbox triage")
st.write("A global bank wants an LLM to read every customer email, route it to the right team and draft a reply. "
         "Same model, same emails, different prompts. Every output is scored by the checks production needs.")

tab_live, tab_ladder, tab_prompts, tab_pipe, tab_eval, tab_obs, tab_script = st.tabs(
    ["Use case 1: Email triage", "Use case 2: Step-by-step prompting", "Prompts", "Production pipeline",
     "Evaluation", "Observability", "Demo script"])

# ---------------- Live triage ----------------
with tab_live:
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        options = [c.id for c in CASES] + ["audience"]
        labels = {c.id: f"{c.sender}  ·  {c.tag}" for c in CASES} | {"audience": "Your own email  ·  from the chat"}
        choice = st.radio("Test inbox", options, format_func=labels.get)
    with right:
        if choice == "audience":
            body = st.text_area("Paste an email from the audience", height=180, placeholder="Paste the email here")
            case = ad_hoc_case(body) if body.strip() else None
            st.caption("No expected answers exist for this email, so only structural and safety checks run.")
        else:
            case = CASE_BY_ID[choice]
            st.markdown(f"**From:** {case.sender}  \n**Subject:** {case.subject}")
            st.markdown(f"<div class='email'>{html.escape(case.body)}</div>", unsafe_allow_html=True)
            st.caption(f"What this tests: {case.why}")

        b1, b2, b3 = st.columns(3)
        run_a = b1.button(f"Run {prompt_a}", width="stretch", disabled=case is None)
        run_b = b2.button(f"Run {prompt_b}", width="stretch", disabled=case is None)
        run_both = b3.button("Run both", type="primary", width="stretch", disabled=case is None)

        if case is not None and (run_a or run_b or run_both):
            todo = [prompt_a, prompt_b] if run_both else [prompt_a] if run_a else [prompt_b]
            with st.spinner("Calling the model..."):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda p: safe_run(case, p), todo))
            for p, r in zip(todo, results):
                st.session_state.live[(case.id, p, case.body)] = r

        if case is not None:
            ca, cb = st.columns(2, gap="large")
            with ca:
                st.subheader(f"A · {prompt_a}")
                show_outcome(st.session_state.live.get((case.id, prompt_a, case.body)), prompt_a)
            with cb:
                st.subheader(f"B · {prompt_b}")
                show_outcome(st.session_state.live.get((case.id, prompt_b, case.body)), prompt_b)

# ---------------- Step-by-step prompting (use case 2) ----------------
with tab_ladder:
    st.markdown("**Use case 2: global supplier-invoice extraction.** Accounts payable receives invoices from "
                "suppliers in many countries. Each step below adds **one** prompting technique to the previous "
                "prompt. Run a step, see which checks it fixes, then climb the whole ladder.")
    idx = st.radio("Prompt ladder", list(range(len(LADDER))), horizontal=True, key="ladder_step",
                   format_func=lambda i: f"{i + 1} · {LADDER[i].title}")
    step = LADDER[idx]
    lcol, rcol = st.columns([1.1, 1], gap="large")
    with lcol:
        st.subheader(f"Step {idx + 1}: {step.title}")
        st.caption(f"Technique: {step.technique}")
        st.markdown(f"**What changed:** {step.change}\n\n**What to watch for:** {step.watch}")
        view = st.radio("Show", ["Full prompt", "Only what changed from the previous step"], horizontal=True,
                        key="ladder_view", disabled=idx == 0)
        if idx > 0 and view.startswith("Only"):
            prev = LADDER[idx - 1]
            diff = difflib.unified_diff(prev.template.splitlines(), step.template.splitlines(),
                                        fromfile=f"step {idx}", tofile=f"step {idx + 1}", lineterm="", n=1)
            st.code("\n".join(diff), language="diff", height=460)
        else:
            st.code(step.template, language="text", wrap_lines=True, height=460)
        st.caption(f"Prompt file: prompts/invoice/{step.id}.txt · about {len(step.template) // 4} tokens before the invoice")
    with rcol:
        inv_id = st.selectbox("Invoice", list(INVOICES), key="ladder_inv",
                              format_func=lambda i: f"{INVOICES[i].label}  ·  tests: {INVOICES[i].tests}")
        inv = INVOICES[inv_id]
        with st.expander("Invoice document", expanded=False):
            st.code(inv.document, language="text")
        if st.button(f"Run step {idx + 1} on this invoice", type="primary", width="stretch"):
            with st.spinner("Calling the model..."):
                try:
                    st.session_state.ladder_live[(inv.id, step.id)] = run_step(inv, step, cfg)
                except (LLMError, ValueError) as e:
                    st.session_state.ladder_live[(inv.id, step.id)] = {"error": str(e)}
        o = st.session_state.ladder_live.get((inv.id, step.id))
        if isinstance(o, dict):
            st.error(o["error"])
        elif o is None:
            st.info("Run this step to see the output and its checks.")
        else:
            prev = st.session_state.ladder_live.get((inv.id, LADDER[idx - 1].id)) if idx else None
            delta = (o.passed - prev.passed) if prev is not None and not isinstance(prev, dict) else None
            m1, m2, m3 = st.columns(3)
            m1.metric("Checks passed", f"{o.passed} / {o.total}",
                      f"{delta:+d} vs step {idx}" if delta is not None else None)
            m2.metric("Latency", f"{o.result.latency_s:.1f} s")
            m3.metric("Prompt tokens", o.result.input_tokens if o.result.input_tokens is not None else "n/a")
            lang = "json" if o.result.text.strip().startswith(("{", "```")) else "text"
            st.code(o.result.text, language=lang, wrap_lines=True, height=260)
            show_checks(o)

    st.divider()
    st.subheader("Climb the whole ladder")
    st.write(f"Runs all {len(LADDER)} steps on all {len(INVOICES)} invoices ({len(LADDER) * len(INVOICES)} calls) "
             "and shows what each technique buys you, and what it costs in prompt size.")
    if st.button("Run all steps on all invoices", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        df_l, errs = run_ladder(list(INVOICES.values()), LADDER, cfg,
                                on_progress=lambda i, n: bar.progress(i / n, text=f"{i} of {n} runs"))
        bar.empty()
        st.session_state.ladder_df = (df_l, errs)
    if "ladder_df" in st.session_state:
        df_l, errs = st.session_state.ladder_df
        if errs:
            st.error(f"{len(errs)} run(s) failed")
            st.dataframe(pd.DataFrame(errs), width="stretch", hide_index=True)
        if not df_l.empty:
            order = [f"{s.id[-1]}. {s.title}" for s in LADDER]
            acc = df_l.groupby("step_title")["passed"].mean().reindex(order)
            per_call = df_l.drop_duplicates(["step", "invoice"]).groupby("step_title")["input_tokens"].mean().reindex(order)
            cols = st.columns(len(order))
            for k, (col, name) in enumerate(zip(cols, order)):
                if pd.isna(acc[name]):
                    continue
                d = None if k == 0 or pd.isna(acc.iloc[k - 1]) else f"{(acc[name] - acc.iloc[k - 1]) * 100:+.0f} pts"
                col.metric(name, f"{acc[name]:.0%}", d)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Share of checks passed, by step")
                st.line_chart(acc.rename("accuracy").reset_index(), x="step_title", y="accuracy")
            with c2:
                st.caption("Average prompt tokens per call, by step (the cost of each technique)")
                st.bar_chart(per_call.rename("prompt tokens").reset_index(), x="step_title", y="prompt tokens")
            st.caption("Pass rate for each check at each step: read across a row to see which technique fixed it")
            heat = df_l.pivot_table(index="check", columns="step_title", values="passed", aggfunc="mean").reindex(columns=order)
            st.dataframe(heat, width="stretch", column_config={
                c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="percent") for c in order})
            grid = (df_l.groupby(["invoice_label", "step_title"])["passed"]
                    .agg(lambda s: f"{int(s.sum())} / {len(s)}").unstack("step_title").reindex(columns=order))
            st.dataframe(grid, width="stretch")

# ---------------- Prompts ----------------
with tab_prompts:
    st.write("Prompts live in the `prompts/` folder as versioned files. Edits are saved as a **new version**, "
             "never overwritten, so every change can be reviewed, evaluated and rolled back.")
    pc1, pc2 = st.columns(2, gap="large")
    with pc1:
        edit_id = st.selectbox("Prompt to edit", prompt_ids, index=prompt_ids.index(prompt_b), key="edit_id")
        original = load_prompt(edit_id)
        edited = st.text_area("Template ({{email}} is replaced with the customer email)", original,
                              height=560, key=f"ta_{edit_id}")
        if st.button(f"Save as {next_version_id(edit_id)}", disabled=edited == original):
            try:
                new_id = save_new_version(edit_id, edited)
                st.success(f"Saved {new_id}. Select it in the sidebar to test it.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    with pc2:
        other = st.selectbox("Compare with", prompt_ids, index=prompt_ids.index(prompt_a), key="cmp_id")
        diff = difflib.unified_diff(load_prompt(other).splitlines(), edited.splitlines(),
                                    fromfile=other, tofile=edit_id + (" (edited)" if edited != original else ""),
                                    lineterm="")
        st.code("\n".join(diff) or "No differences.", language="diff")
        st.caption("Reviewing a prompt change should look exactly like reviewing a code change.")

# ---------------- Production pipeline ----------------
with tab_pipe:
    st.write("The prompt is one component. This runs the same email through the controls that surround it "
             "in production: redaction, injection screening, schema validation with repair, policy rules in "
             "code, output guardrails and routing.")
    pp1, pp2 = st.columns([1, 1])
    pipe_case_id = pp1.selectbox("Email", [c.id for c in CASES],
                                 format_func=lambda i: f"{CASE_BY_ID[i].sender} · {CASE_BY_ID[i].tag}")
    pipe_prompt = pp2.selectbox("Prompt version", prompt_ids, index=prompt_ids.index(prompt_b), key="pipe_prompt")
    if st.button("Run pipeline", type="primary"):
        with st.spinner("Running pipeline..."):
            st.session_state.pipe = run_pipeline(CASE_BY_ID[pipe_case_id], pipe_prompt, cfg)
            st.session_state.pipe_key = (pipe_case_id, pipe_prompt)
    res = st.session_state.get("pipe")
    if res and st.session_state.get("pipe_key") == (pipe_case_id, pipe_prompt):
        icons = {"ok": "✅", "warn": "⚠️", "fail": "❌"}
        s1, s2 = st.columns([1.2, 1], gap="large")
        with s1:
            st.subheader("Trace")
            for i, step in enumerate(res.steps, 1):
                st.markdown(f"{icons[step.status]} **{i}. {step.name}** ({step.ms} ms)  \n{step.detail}")
        with s2:
            st.subheader("Outcome")
            st.metric("Routed to", res.queue)
            if res.final:
                st.json(res.final.model_dump())
            if res.reply_to_send:
                st.text_area("Draft reply for the agent", res.reply_to_send, height=150, disabled=True)
            elif res.final:
                st.error("Reply blocked by output guardrails. An agent writes the reply.")

# ---------------- Evaluation ----------------
with tab_eval:
    st.write("Run every golden-set email through each prompt version and score them with the same checks. "
             "This is the release gate you would run in CI before any prompt change ships.")
    e1, e2 = st.columns([2, 1])
    eval_prompts = e1.multiselect("Prompt versions", prompt_ids, default=list(dict.fromkeys([prompt_a, prompt_b])))
    gate = e2.slider("Release gate: minimum pass rate", 0.5, 1.0, 0.9, 0.05, format="%.2f")
    if st.button("Run evaluation", type="primary", disabled=not eval_prompts):
        bar = st.progress(0.0, text="Starting...")
        outcomes, errors = run_eval(CASES, eval_prompts, cfg,
                                    on_progress=lambda i, n: bar.progress(i / n, text=f"{i} of {n} runs"))
        bar.empty()
        st.session_state.eval = (to_frame(outcomes), errors)

    if "eval" in st.session_state:
        df, errors = st.session_state.eval
        if errors:
            st.error(f"{len(errors)} run(s) failed")
            st.dataframe(pd.DataFrame(errors), width="stretch")
        if not df.empty:
            summ = summary(df)
            cols = st.columns(len(summ))
            for col, row in zip(cols, summ.itertuples()):
                passed_gate = row.pass_rate >= gate
                col.metric(row.prompt_id, f"{row.pass_rate:.0%}",
                           f"{row.emails_fully_correct:.0%} of emails fully correct", delta_color="off")
                col.markdown(("✅ **Passes release gate**" if passed_gate else "❌ **Blocked by release gate**")
                             + f"  \n{row.checks_passed} of {row.checks_total} checks · avg {row.avg_latency_s:.1f} s")

            st.subheader("Pass rate by check")
            by_check = df.pivot_table(index="check", columns="prompt_id", values="passed", aggfunc="mean")
            by_check = by_check.reindex([n for n in CHECK_NAMES.values() if n in by_check.index])
            st.bar_chart(by_check, horizontal=True, stack=False)

            st.subheader("Results by email")
            grid = (df.groupby(["case_id", "email", "tests", "prompt_id"])["passed"]
                    .agg(lambda s: f"{int(s.sum())} / {len(s)}").unstack("prompt_id").reset_index()
                    .drop(columns="case_id"))
            st.dataframe(grid, width="stretch", hide_index=True)
            with st.expander("Every check, with details"):
                st.dataframe(df[["case_id", "prompt_id", "check", "passed", "detail"]],
                             width="stretch", hide_index=True)

# ---------------- Observability ----------------
with tab_obs:
    st.write("Every model call is logged to `runs/log.jsonl`: prompt version, model, latency, tokens and check "
             "results. In production this feeds your tracing and monitoring stack.")
    runs = [r for r in read_runs() if r.get("kind") in ("triage", "ladder")]
    if not runs:
        st.info("No runs logged yet. Run an email on the Live triage tab.")
    else:
        log = pd.DataFrame(runs)
        log["pass_rate"] = log["passed"] / log["total"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model calls", len(log))
        m2.metric("Average latency", f"{log['latency_s'].mean():.1f} s")
        m3.metric("Output tokens", int(log["output_tokens"].fillna(0).sum()))
        m4.metric("Average pass rate", f"{log['pass_rate'].mean():.0%}")
        o1, o2 = st.columns(2)
        with o1:
            st.caption("Average pass rate by prompt version")
            st.bar_chart(log.groupby("prompt_id")["pass_rate"].mean())
        with o2:
            st.caption("Average latency (s) by prompt version")
            st.bar_chart(log.groupby("prompt_id")["latency_s"].mean())
        st.dataframe(log[["ts", "case_id", "prompt_id", "provider", "model", "latency_s",
                          "input_tokens", "output_tokens", "passed", "total", "failed_checks"]]
                     .sort_values("ts", ascending=False), width="stretch", hide_index=True)
        st.download_button("Download log (JSONL)", "\n".join(json.dumps(r, ensure_ascii=False) for r in runs),
                           file_name="prompt_lab_runs.jsonl")
        if st.button("Clear log"):
            clear_runs()
            st.rerun()

# ---------------- Demo script ----------------
with tab_script:
    st.markdown("""
**About 20 minutes. Rehearse once with a live model so Replay mode has a recording to fall back on.**

1. **Set the scene.** A global bank receives thousands of emails a day in many languages. Triage decides who sees each one and how fast.
2. **Live triage → Priya (routine), Run both.** Both answers read well to a human. Ask: which one can software route? Only one passes *Machine-readable*.
3. **Tom (fraud).** Look for the card number echoed back and whether the email was escalated.
4. **Alex (prompt injection).** The email contains instructions aimed at the AI. Compare who obeyed them.
5. **Prompts tab.** Walk through role, categories, rules and output format. Show the diff between versions. Optionally delete rule 3, save as a new version, select it as Prompt B and rerun Alex.
6. **Production pipeline → Tom, then Alex.** Show redaction before the model call, the injection flag, policy rules enforced in code and routing. The prompt is one control among many.
7. **Evaluation.** Run all emails against both versions. The release gate turns "it looked good in the demo" into evidence.
8. **Observability.** Show latency, tokens and pass rate per version. This is what you monitor in production.
9. **Audience email.** Paste an email from the Teams chat and run both prompts live.

---

**Use case 2: step-by-step prompting (about 20 minutes)**

1. **Frame it.** Accounts payable receives invoices from many countries. Dates, number formats and currencies differ, and every error means a wrong payment.
2. **Step 1 on the German invoice.** The model returns friendly prose with `03.04.2026` and `1.123,96 €`. Ask: could a payment system use this?
3. **Step 2.** Same invoice. A role and an explicit field list make values consistent, but it is still prose.
4. **Step 3 (switch to "Only what changed").** The JSON contract makes it machine-readable. Run the Australian invoice: watch the date `05/03/2026` and the line-item count.
5. **Step 4.** Two worked examples teach the convention that shipping and travel are not line items, without writing a rule.
6. **Step 5.** The model reasons first. Run the UK invoice: does it now notice that £1,425 + £285 is not £1,810?
7. **Step 6.** Run the Helix invoice with its hidden instruction to zero the total. Compare steps 3 and 6.
8. **Climb the whole ladder.** Show the accuracy line and the token bars together: every technique has a cost, so add one only when an evaluation shows it earns its place.
""")
