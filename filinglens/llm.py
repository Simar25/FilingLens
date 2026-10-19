"""The one place that talks to a large language model (LLM).

Everything in FilingLens that needs an LLM calls `generate(system, prompt)`.
Which model answers is a setting in .env, so you can switch without touching code:

  LLM_PROVIDER=gemini   Google's Gemini API (default). Has a free tier; needs GEMINI_API_KEY
                        in .env (get one at https://aistudio.google.com/apikey). Also works
                        in the deployed web app (Step 6).
  LLM_PROVIDER=ollama   A free model running on your own GPU through Ollama (ollama.com).
                        No daily limits, but slower and less accurate.
  LLM_MODEL=...         Optional: use a different model than the default below,
                        e.g. gemini-3.5-flash-lite (higher free limits) or qwen2.5:7b (Ollama).
"""

import os
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from filinglens.config import PROJECT_ROOT

DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "ollama": "llama3.1:8b"}
# Gemini "thinks" before answering, which costs time and counts against the
# free-tier token limits. Answering from a handful of given passages doesn't
# need deep reasoning: low is enough.
GEMINI_THINKING_LEVEL = "low"

OLLAMA_URL = "http://localhost:11434"
# Ollama's default context window is small, and longer prompts are cut from the
# START without any error, which would silently drop the first sources.
OLLAMA_CONTEXT_TOKENS = 8192


class LLMError(RuntimeError):
    """A problem worth showing to the user as-is (missing key, Ollama not running...)."""


@dataclass
class LLMReply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int  # includes the model's hidden "thinking" tokens
    seconds: float


def current_settings() -> tuple[str, str]:
    """(provider, model) from .env, with defaults."""
    load_dotenv(PROJECT_ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "").strip().lower() or "gemini"
    if provider not in DEFAULT_MODELS:
        raise LLMError(f'LLM_PROVIDER in .env must be "gemini" or "ollama", not "{provider}"')
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    return provider, model


def generate(system: str, prompt: str) -> LLMReply:
    provider, model = current_settings()
    start = time.perf_counter()
    if provider == "gemini":
        text, n_in, n_out = _generate_gemini(model, system, prompt)
    else:
        text, n_in, n_out = _generate_ollama(model, system, prompt)
    return LLMReply(text.strip(), model, n_in, n_out, time.perf_counter() - start)


def _generate_gemini(model: str, system: str, prompt: str) -> tuple[str, int, int]:
    from google import genai  # only needed when this provider is used
    # The Interactions API raises its own error classes, not google.genai.errors.APIError,
    # and google-genai (2.31) doesn't export them anywhere public yet.
    from google.genai._gaos.lib.compat_errors import APIConnectionError, APIError

    if not os.getenv("GEMINI_API_KEY"):
        raise LLMError("GEMINI_API_KEY is missing. Get a free key at https://aistudio.google.com/apikey "
                       'and add it to .env as  GEMINI_API_KEY="..."')
    try:
        # Keep the client in a `with` block: in a one-liner like genai.Client().interactions.create(...)
        # the client is thrown away, closing its connection, before the request is sent.
        with genai.Client() as client:
            interaction = client.interactions.create(
                model=model,
                system_instruction=system,
                input=prompt,
                generation_config={"thinking_level": GEMINI_THINKING_LEVEL},
            )
    except APIConnectionError:
        raise LLMError("Can't reach the Gemini API. Check your internet connection.") from None
    except APIError as exc:
        if exc.status_code == 429:
            raise LLMError("Gemini free-tier limit reached (requests per minute or per day). Wait a minute, "
                           "or set LLM_MODEL=gemini-3.5-flash-lite in .env, which has higher limits.") from None
        if exc.status_code in (400, 401, 403) and "key" in exc.message.lower():
            raise LLMError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.") from None
        raise LLMError(f"Gemini API error {exc.status_code}: {exc.message}") from None

    usage = interaction.usage
    output_tokens = (usage.total_output_tokens or 0) + (usage.total_thought_tokens or 0)
    return interaction.output_text or "", usage.total_input_tokens or 0, output_tokens


def _generate_ollama(model: str, system: str, prompt: str) -> tuple[str, int, int]:
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", timeout=600, json={
            "model": model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": {"num_ctx": OLLAMA_CONTEXT_TOKENS, "temperature": 0},
        })
    except requests.ConnectionError:
        raise LLMError("Can't reach Ollama. Install it from https://ollama.com, make sure it's running, "
                       f"then download the model once with:  ollama pull {model}") from None
    if response.status_code == 404:
        raise LLMError(f"Ollama doesn't have the model '{model}' yet. Download it with:  ollama pull {model}")
    response.raise_for_status()
    data = response.json()
    if data.get("prompt_eval_count", 0) >= OLLAMA_CONTEXT_TOKENS:
        raise LLMError("The prompt is longer than Ollama's context window; ask with fewer sources (-k).")
    return data["message"]["content"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)
