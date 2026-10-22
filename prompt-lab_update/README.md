# Prompt Lab: from prompt engineering to production AI

A Streamlit demo for the Tech Talk webinar. A fictional global bank, **Meridian Bank**, wants an LLM to triage
customer emails: classify them, set priority, decide whether a human must step in, and draft a reply.
The app runs the same emails through a naive prompt and an engineered prompt, scores every output with
automated production checks, and then shows the controls that surround a prompt in a real system.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
# edit .env and add your OPENAI_API_KEY
pytest                        # 26 offline tests, no API key needed
streamlit run app.py
```

The app opens at http://localhost:8501.

## Setting up your OpenAI key

Put your key in `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini
OPENAI_REASONING_EFFORT=low
```

`gpt-5.4-mini` is fast and inexpensive, which suits a live session. Any chat model your key can access
works; set it in `.env` or type it in the sidebar. For GPT-5 and o-series models the app sends
`reasoning_effort` instead of `temperature`, because those models do not accept a temperature setting.

The sidebar also offers **Azure OpenAI** (if your organisation hosts models there) and an
OpenAI-compatible server via `OPENAI_BASE_URL`; both use the same code path.

**Replay is your backup for the live session.** Rehearse once with your key. If the network or API
fails during the webinar, switch the sidebar to *Replay recorded runs* and every tab keeps working with
those real, recorded outputs.

## What each tab shows

| Tab | Teaching point |
|---|---|
| Use case 1: Email triage | Same email, two prompts, side by side. Seven automated checks per output. Paste an audience email live. |
| Use case 2: Step-by-step prompting | Invoice extraction improved one technique at a time across six steps, with a "climb the ladder" run that charts accuracy and prompt cost per step. |
| Prompts | Prompts are versioned files. Edits are saved as a new version, with a diff, like a code review. |
| Production pipeline | Redaction, injection screening, schema validation with automatic repair, policy rules in code, output guardrails and routing. |
| Evaluation | The golden set run against prompt versions, with a release-gate threshold. |
| Observability | Every call logged with prompt version, model, latency, tokens and check results. |
| Demo script | Run sheets for both use cases. |

## Use case 2: the prompt ladder

Accounts payable at a global company extracts data from supplier invoices from the USA, Germany, India,
the UK and Australia. Each step adds exactly one technique to the previous prompt:

| Step | Technique | What it fixes |
|---|---|---|
| 1 | Zero-shot request | Baseline: friendly prose, local date and number formats |
| 2 | Role + explicit field list | Consistent values (ISO dates, currency codes), still prose |
| 3 | Output contract (JSON) | Machine-readable output, nulls instead of guesses |
| 4 | Few-shot examples | Conventions shown rather than described (shipping is not a line item) |
| 5 | Reasoning first (`working` field) | Locale-dependent dates, total reconciliation, an audit trail |
| 6 | Guardrails | Delimited untrusted input, no invented or corrected values, injection reported as an anomaly |

The six invoices each test something specific: US MM/DD dates, German `1.234,56` amounts, Indian lakh
formatting, a UK invoice with no PO and a total that doesn't add up, a US invoice with a hidden instruction
to zero the total, and an Australian invoice where `$` means AUD and `05/03/2026` is 5 March.

Prompts are in `prompts/invoice/step1.txt` to `step6.txt`; step descriptions in `data/prompt_ladder.json`.
Exact results depend on the model: strong models may already handle some cases at step 3, which is itself
a lesson. Measure before adding complexity.

## The automated checks (use case 1)

| Key | Check | Why production needs it |
|---|---|---|
| R | Machine-readable output | Routing software can't act on prose |
| S | Schema and allowed values | Validated against a Pydantic contract (`promptlab/schema.py`) |
| C | Correct category | Wrong team, slower resolution |
| E | Correct escalation | Fraud and complaints must reach a person |
| L | Customer's language | Global customers expect replies in their language |
| P | No sensitive data echoed | Card numbers must never be repeated in a reply |
| I | Ignored embedded instructions | Customer text is data, never commands (prompt injection) |

## CI release gate

The same evaluation runs from the command line and returns a non-zero exit code when a prompt version
falls below the threshold, so it can block a merge in any CI system:

```bash
python scripts/eval_gate.py --prompt engineered_v1 --threshold 0.9   # uses OPENAI_API_KEY
python scripts/eval_gate.py --prompt naive_v1                       # expected: BLOCKED
```

## Project structure

```
prompt-lab/
├── app.py                    Streamlit UI (six tabs)
├── promptlab/
│   ├── config.py             Paths, providers, default models (.env)
│   ├── cases.py              Golden-set loader
│   ├── prompts.py            Versioned prompt templates
│   ├── schema.py             Pydantic output contract and JSON parsing
│   ├── checks.py             Automated production checks
│   ├── llm.py                OpenAI client (also Azure OpenAI, compatible servers, replay)
│   ├── runner.py             Run + score + log one email
│   ├── pipeline.py           Guardrails, repair loop, policy rules, routing
│   ├── evaluate.py           Parallel evaluation and summary
│   ├── invoice.py            Use case 2: invoice checks and the prompt ladder
│   └── runlog.py             JSONL run log (observability and replay)
├── prompts/                  naive_v1.txt, engineered_v1.txt, saved versions
│   └── invoice/              step1.txt … step6.txt (prompt ladder)
├── data/golden_set.json      8 test emails with expected answers
├── data/invoices.json        6 test invoices with expected values
├── data/prompt_ladder.json   Step titles, techniques and talking points
├── scripts/eval_gate.py      CI release gate
└── tests/                    Offline tests with a fake model
```

## Extending the demo

- **Add a test email:** append an object to `data/golden_set.json` with its expected category and escalation.
- **Add a prompt version:** edit and save in the Prompts tab, or drop a `<name>_v<n>.txt` file in `prompts/`.
- **Add a check:** add a function in `promptlab/checks.py`; it appears in every tab and in the CI gate.

## Notes

- Meridian Bank, every customer and every card number are fictional. The card number is a standard test number.
- Model output varies between runs, even at temperature 0. That is a teaching point: one good run proves
  nothing, which is why the Evaluation tab exists.
- `.env` and `runs/` are in `.gitignore`. Never commit API keys.
