"""The one place that talks to a large language model (LLM).

Everything in FilingLens that needs an LLM calls `generate(system, prompt)`.
Which model answers is a setting in .env, so you can switch without touching code:

  LLM_PROVIDER=gemini   Google's Gemini API (default). Has a free tier; needs GEMINI_API_KEY
                        in .env (get one at https://aistudio.google.com/apikey). Also works
                        in the deployed web app (Step 6).
  LLM_PROVIDER=groq     Groq's API: open models (OpenAI's gpt-oss) on very fast hardware. Free
                        tier of about 1,000 requests and 200,000 tokens a day; needs GROQ_API_KEY
                        in .env (get one at https://console.groq.com/keys).
  LLM_PROVIDER=ollama   A free model running on your own GPU through Ollama (ollama.com).
                        No daily limits, but slower and less accurate.
  LLM_MODEL=...         Optional: use a different model than the default below, e.g.
                        gemini-3.5-flash-lite, openai/gpt-oss-20b (Groq) or qwen2.5:7b (Ollama).
"""

import os
import re
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from filinglens.config import PROJECT_ROOT

DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "groq": "openai/gpt-oss-120b", "ollama": "llama3.1:8b"}
# Gemini and gpt-oss "think" before answering, which costs time and counts against the
# free-tier token limits. Answering from a handful of given passages doesn't
# need deep reasoning: low is enough.
GEMINI_THINKING_LEVEL = "low"
# gpt-oss needs more: writing the SQL for one tricky question 4 times on 2026-10-19 was right 2/4 times at
# "low" (one silently wrong figure) and 3/4 at "medium" (its miss was an empty result, which text_to_sql
# now retries), for ~25% more tokens.
GROQ_REASONING_EFFORT = "medium"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"  # Groq copies OpenAI's API format

# What happens when an online LLM refuses or fails. All of this was learned the hard way on 2026-10-19:
#  - Free tiers have limits per minute AND per day. A refused request comes back as 429 ("too many
#    requests"), usually with a Retry-After header saying how long to wait: a few seconds for a
#    per-minute limit, many hours for a daily one ("retry in 13h42m").
#  - Google's client library obeys that header with NO upper limit, so a question silently waited
#    13 hours. It also had no time limit per request at all by default.
# So every request has a time limit, and this file decides about 429 and 5xx ("server problem")
# replies itself: a short wait is waited out and the request retried; a long one stops at once,
# with the provider's own message, so you can switch model or provider.
LLM_TIMEOUT_SECONDS = 60      # per try; normal answers take seconds, the slowest on a busy day 40 s
LLM_RETRIES = 2               # retries for 429 and 5xx replies
MAX_WAIT_SECONDS = 30         # a longer wait asked for by the provider isn't worth waiting for
BACKOFF_SECONDS = (2, 5)      # waits before retry 1 and 2 when the provider doesn't say how long
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
SWITCH_HINT = ("Options: wait until it resets; set LLM_MODEL to another model in .env (each has its own limit, "
               "e.g. gemini-3.5-flash-lite); set LLM_PROVIDER=groq in .env (needs GROQ_API_KEY); or "
               "LLM_PROVIDER=ollama (free, no limits, runs on your GPU).")

# Models differ in small ways that would break the rest of FilingLens, so replies are cleaned here.
# Found with gpt-oss on 2026-10-19:
#  - it cites like 【DB】 or 【2†L1-L3】 instead of [DB] or [2], so citations weren't recognised;
#  - it writes narrow non-breaking spaces and non-breaking hyphens, which show as '?' in some Windows
#    terminals, hide words like 'year-over-year' from searches, and would be a syntax error inside SQL.
MODEL_CITATION = re.compile("\u3010([^\u3011\u2020]+)(?:\u2020[^\u3011]*)?\u3011")  # 【2†L1-L3】 -> "2"
ODD_CHARACTERS = str.maketrans({"\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u2011": "-"})

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
        raise LLMError(f'LLM_PROVIDER in .env must be "gemini", "groq" or "ollama", not "{provider}"')
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    return provider, model


def generate(system: str, prompt: str) -> LLMReply:
    provider, model = current_settings()
    start = time.perf_counter()
    if provider == "gemini":
        text, n_in, n_out = _generate_gemini(model, system, prompt)
    elif provider == "groq":
        text, n_in, n_out = _generate_groq(model, system, prompt)
    else:
        text, n_in, n_out = _generate_ollama(model, system, prompt)
    return LLMReply(clean_text(text), model, n_in, n_out, time.perf_counter() - start)


def clean_text(text: str) -> str:
    """Iron out provider quirks (see MODEL_CITATION) so the rest of FilingLens sees plain text."""
    return MODEL_CITATION.sub(r"[\1]", text).translate(ODD_CHARACTERS).strip()


# ---------- deciding whether to retry (shared by Gemini and Groq) ----------

def _requested_wait(headers) -> float | None:
    """How many seconds the provider asks us to wait (Retry-After header), if it says."""
    try:
        if "retry-after-ms" in headers:
            return float(headers["retry-after-ms"]) / 1000
        if "retry-after" in headers:
            return float(headers["retry-after"])
    except ValueError:  # Retry-After may also be a date; then we use our own waits
        pass
    return None


def _wait_before_retry(status: int, headers, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None to give up now."""
    wait = _requested_wait(headers)
    if wait is None:
        wait = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
    if status not in RETRYABLE_STATUSES or attempt >= LLM_RETRIES or wait > MAX_WAIT_SECONDS:
        return None
    return wait


def _limit_or_server_error(name: str, model: str, status: int, headers, message: str) -> LLMError | None:
    """A clear message for 429 and 5xx replies (None for other errors)."""
    if status == 429:
        wait = _requested_wait(headers)
        if wait is not None and wait > MAX_WAIT_SECONDS:
            return LLMError(f"{name}'s free-tier limit for {model} is used up. {name} says: {message}\n{SWITCH_HINT}")
        return LLMError(f"{name} free-tier limit reached (requests or tokens per minute), still after {LLM_RETRIES} "
                        f"retries. Wait a minute and try again. {name} says: {message}")
    if status >= 500:
        return LLMError(f"{name} is overloaded or down right now (error {status}, still failing after "
                        f"{LLM_RETRIES} retries). Try again in a few minutes.\n{SWITCH_HINT}")
    return None


# ---------- Gemini ----------

def _generate_gemini(model: str, system: str, prompt: str) -> tuple[str, int, int]:
    from google import genai  # only needed when this provider is used
    from google.genai import types
    # The Interactions API raises its own error classes, not google.genai.errors.APIError,
    # and google-genai (2.31) doesn't export them anywhere public yet.
    from google.genai._gaos.lib.compat_errors import APIConnectionError, APIError, APITimeoutError

    if not os.getenv("GEMINI_API_KEY"):
        raise LLMError("GEMINI_API_KEY is missing. Get a free key at https://aistudio.google.com/apikey "
                       'and add it to .env as  GEMINI_API_KEY="..."')
    http_options = types.HttpOptions(
        timeout=LLM_TIMEOUT_SECONDS * 1000,  # in milliseconds
        # The library may only retry network errors (it insists on at least one retry) and 408 (request
        # timeout), which comes without a wait time. Every 429 and 5xx comes straight back here.
        retry_options=types.HttpRetryOptions(attempts=1, http_status_codes=[408]),
    )
    for attempt in range(LLM_RETRIES + 1):
        try:
            # Keep the client in a `with` block: in a one-liner like genai.Client().interactions.create(...)
            # the client is thrown away, closing its connection, before the request is sent.
            with genai.Client(http_options=http_options) as client:
                interaction = client.interactions.create(
                    model=model,
                    system_instruction=system,
                    input=prompt,
                    generation_config={"thinking_level": GEMINI_THINKING_LEVEL},
                )
            break
        except APITimeoutError:  # must come before APIConnectionError, its parent class
            raise LLMError(f"Gemini didn't answer within {LLM_TIMEOUT_SECONDS} s (2 tries). It's probably "
                           f"overloaded: try again in a few minutes.\n{SWITCH_HINT}") from None
        except APIConnectionError:
            raise LLMError("Can't reach the Gemini API. Check your internet connection.") from None
        except APIError as exc:
            response = getattr(exc, "response", None)
            headers = response.headers if response is not None else {}
            wait = _wait_before_retry(exc.status_code, headers, attempt)
            if wait is None:
                raise _gemini_error(exc, model, headers) from None
            time.sleep(wait)

    usage = interaction.usage
    output_tokens = (usage.total_output_tokens or 0) + (usage.total_thought_tokens or 0)
    return interaction.output_text or "", usage.total_input_tokens or 0, output_tokens


def _gemini_error(exc, model: str, headers) -> LLMError:
    status, message = exc.status_code, (exc.message or "").strip()
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict) and body["error"].get("message"):
        message = body["error"]["message"]  # Google's own sentence, without the "Error code: 429 - {...}" wrapper
    error = _limit_or_server_error("Gemini", model, status, headers, message)
    if error:
        return error
    if status in (400, 401, 403) and "key" in message.lower():
        return LLMError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.")
    return LLMError(f"Gemini API error {status}: {message}")


# ---------- Groq ----------

def _generate_groq(model: str, system: str, prompt: str) -> tuple[str, int, int]:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise LLMError("GROQ_API_KEY is missing. Get a free key at https://console.groq.com/keys "
                       'and add it to .env as  GROQ_API_KEY="..."')
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    if model.startswith("openai/gpt-oss"):
        # The thinking is still done (and counted in the token limits), just not sent back to us.
        body |= {"reasoning_effort": GROQ_REASONING_EFFORT, "include_reasoning": False}
    for attempt in range(LLM_RETRIES + 1):
        try:
            response = requests.post(GROQ_URL, json=body, timeout=LLM_TIMEOUT_SECONDS,
                                     headers={"Authorization": f"Bearer {key}"})
        except requests.Timeout:
            raise LLMError(f"Groq didn't answer within {LLM_TIMEOUT_SECONDS} s. Try again in a few minutes.") from None
        except requests.ConnectionError:
            raise LLMError("Can't reach the Groq API. Check your internet connection.") from None
        if response.ok:
            break
        wait = _wait_before_retry(response.status_code, response.headers, attempt)
        if wait is None:
            raise _groq_error(response, model) from None
        time.sleep(wait)

    data = response.json()
    usage = data.get("usage") or {}
    return (data["choices"][0]["message"].get("content") or "",
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def _groq_error(response: requests.Response, model: str) -> LLMError:
    status = response.status_code
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.text[:300]
    error = _limit_or_server_error("Groq", model, status, response.headers, message)
    if error:
        return error
    if status == 401:
        return LLMError("Groq rejected the API key. Check GROQ_API_KEY in .env.")
    if status == 413:  # one prompt bigger than the free tier's tokens-per-minute limit
        return LLMError(f"This prompt is too big for Groq's free tier ({message}). "
                        "Ask with fewer report passages, e.g. -k 4.")
    if status == 404:
        return LLMError(f"Groq doesn't offer the model '{model}' (any more). See https://console.groq.com/docs/models "
                        "and set LLM_MODEL in .env.")
    return LLMError(f"Groq API error {status}: {message}")


# ---------- Ollama ----------

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
