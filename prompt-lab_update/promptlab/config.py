"""Central paths and settings. Everything reads from environment variables (.env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROMPT_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
GOLDEN_SET = DATA_DIR / "golden_set.json"
RUN_LOG = RUNS_DIR / "log.jsonl"

# Default models per provider. Override in .env as your organisation's access allows.
DEFAULT_MODELS = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
    "azure_openai": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
    "replay": "recorded",
}

PROVIDER_LABELS = {
    "openai": "OpenAI (API key)",
    "azure_openai": "Azure OpenAI",
    "replay": "Replay recorded runs (offline backup)",
}


def provider_ready(provider: str) -> tuple[bool, str]:
    """Return (ready, message) describing whether credentials are configured."""
    if provider == "openai":
        ok = bool(os.getenv("OPENAI_API_KEY"))
        return ok, "OPENAI_API_KEY found" if ok else "Set OPENAI_API_KEY in .env"
    if provider == "azure_openai":
        missing = [k for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT") if not os.getenv(k)]
        return (not missing), ("Azure settings found" if not missing else "Missing: " + ", ".join(missing))
    if provider == "replay":
        ok = RUN_LOG.exists()
        return ok, "Replaying runs/log.jsonl" if ok else "No recorded runs yet. Run live once to record."
    return False, "Unknown provider"
