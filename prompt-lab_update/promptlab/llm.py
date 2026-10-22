"""One function, several providers. The rest of the app never imports a vendor SDK directly."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .runlog import latest_output


@dataclass
class LLMConfig:
    provider: str = "openai"             # openai | azure_openai | replay
    model: str = ""
    temperature: float = 0.0             # production default: as deterministic as possible
    max_tokens: int = 2000             # reasoning models spend part of this on thinking


@dataclass
class LLMResult:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    provider: str
    model: str


class LLMError(RuntimeError):
    pass


def complete(prompt: str, cfg: LLMConfig, *, case_id: str = "", prompt_id: str = "") -> LLMResult:
    start = time.perf_counter()
    try:
        if cfg.provider in ("openai", "azure_openai"):
            text, tin, tout = _openai(prompt, cfg)
        elif cfg.provider == "replay":
            row = latest_output(case_id, prompt_id)
            if not row:
                raise LLMError(f"No recorded run for {case_id} / {prompt_id}. Run it live once first.")
            return LLMResult(row["output"], row.get("latency_s", 0.0), row.get("input_tokens"),
                             row.get("output_tokens"), "replay", row.get("model", "recorded"))
        else:
            raise LLMError(f"Unknown provider: {cfg.provider}")
    except LLMError:
        raise
    except Exception as e:  # surface vendor errors in one readable form
        raise LLMError(f"{type(e).__name__}: {e}") from e
    return LLMResult(text, time.perf_counter() - start, tin, tout, cfg.provider, cfg.model)


def _openai(prompt: str, cfg: LLMConfig):
    import openai

    if cfg.provider == "azure_openai":
        client = openai.AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    else:
        client = openai.OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)  # base_url enables Ollama, vLLM, etc.

    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": cfg.model, "messages": messages, "max_completion_tokens": cfg.max_tokens}
    if _is_reasoning_model(cfg.model):
        # GPT-5 and o-series models think before answering and ignore temperature.
        kwargs["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "low")
    else:
        kwargs["temperature"] = cfg.temperature

    try:
        resp = client.chat.completions.create(**kwargs)
    except openai.BadRequestError:
        # Older models or OpenAI-compatible servers may not accept these parameters.
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("temperature", None)
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        resp = client.chat.completions.create(**kwargs)

    choice = resp.choices[0]
    text = choice.message.content or ""
    if not text and choice.finish_reason == "length":
        raise LLMError("The model used its whole token budget before answering. "
                       "Raise max_tokens or set OPENAI_REASONING_EFFORT=minimal in .env.")
    usage = getattr(resp, "usage", None)
    return text, getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
