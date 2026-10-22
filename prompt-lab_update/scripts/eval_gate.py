"""CI release gate: fail the build if a prompt version scores below the threshold.

Usage:
    python scripts/eval_gate.py --prompt engineered_v1 --threshold 0.9
    python scripts/eval_gate.py --prompt engineered_v2 --model gpt-5.4-mini
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from promptlab.cases import load_cases  # noqa: E402
from promptlab.config import DEFAULT_MODELS, provider_ready  # noqa: E402
from promptlab.evaluate import run_eval, summary, to_frame  # noqa: E402
from promptlab.llm import LLMConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="Prompt id, e.g. engineered_v1")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--provider", default="openai", choices=list(DEFAULT_MODELS))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    ready, msg = provider_ready(args.provider)
    if not ready:
        print(f"Provider not configured: {msg}")
        return 2
    cfg = LLMConfig(provider=args.provider, model=args.model or DEFAULT_MODELS[args.provider])
    outcomes, errors = run_eval(load_cases(), [args.prompt], cfg,
                                on_progress=lambda i, n: print(f"\r{i}/{n} runs", end="", flush=True))
    print()
    df = to_frame(outcomes)
    if df.empty:
        print("No successful runs.", errors)
        return 2
    row = summary(df).iloc[0]
    for _, r in df[~df["passed"]].iterrows():
        print(f"  FAIL {r.case_id:<4} {r.check:<32} {r.detail}")
    print(f"\n{args.prompt}: {row.pass_rate:.0%} of checks passed "
          f"({row.checks_passed}/{row.checks_total}), {row.emails_fully_correct:.0%} of emails fully correct")
    if errors:
        print(f"{len(errors)} run(s) errored")
    if row.pass_rate >= args.threshold and not errors:
        print(f"PASS: meets release gate of {args.threshold:.0%}")
        return 0
    print(f"BLOCKED: below release gate of {args.threshold:.0%}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
