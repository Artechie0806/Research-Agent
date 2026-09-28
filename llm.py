"""LLM clients: any OpenAI-compatible chat-completions API, or the self-hosted
Qwen wrapper (FastAPI in front of vLLM).

Pick one with LLM_PROVIDER=openai|qwen. If unset, "openai" is used when
OPENAI_API_KEY or OPENAI_BASE_URL is set, else "qwen". `make_client()` returns
the right one; both expose `chat_text()` and `structured()`.

OpenAI-compatible (OpenAI, vLLM, Ollama, LM Studio, OpenRouter, Together, ...):

    POST {OPENAI_BASE_URL}/chat/completions   header: Authorization: Bearer ...
    body: {"model", "messages", "max_tokens", "temperature", "response_format"}

    OPENAI_BASE_URL          default https://api.openai.com/v1
    OPENAI_API_KEY           bearer token (may be blank for local servers)
    OPENAI_MODEL             model name (default gpt-4o-mini)
    OPENAI_MAX_TOKENS_PARAM  "max_tokens" (default) or "max_completion_tokens"
                             (needed by OpenAI reasoning models)
    OPENAI_JSON_MODE         "object" (default; also "1") sends
                             response_format=json_object; "schema" sends the
                             caller's JSON schema so the server enforces the
                             shape (LM Studio, vLLM, OpenAI); "0" sends neither
    OPENAI_REASONING_EFFORT  if set, sent as reasoning_effort (e.g. "none" stops
                             thinking models spending the whole budget thinking)
    LLM_TIMEOUT              per-request timeout seconds (default 300)

The Qwen wrapper exposes:

    POST /chat/text   header: api-key
    body: {"prompt": str, "max_tokens": int, "temperature": float}
    -> {"response": "<model text>"}

It uses a fixed system prompt and has no tool-use / structured-output mode, so
`structured()` folds the caller's system prompt + JSON schema into the single
`prompt` field, asks the model for JSON only, and parses it back out of the
reply (with one retry). Credentials/URL come from the environment (.env):

    QWEN_API_URL   base URL of the wrapper   (default http://127.0.0.1:8000)
    QWEN_API_KEY   the api-key the wrapper expects
    QWEN_TIMEOUT   per-request timeout seconds (default 300 — vision models are slow)
"""

from __future__ import annotations

import json
import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("QWEN_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("QWEN_API_KEY", "")
TIMEOUT = float(os.getenv("QWEN_TIMEOUT") or os.getenv("LLM_TIMEOUT") or "300")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS_PARAM = os.getenv("OPENAI_MAX_TOKENS_PARAM", "max_tokens")
_JSON_MODES = {"1": "object", "true": "object", "object": "object",
               "schema": "schema", "0": "", "false": "", "no": "", "": ""}
OPENAI_JSON_MODE = _JSON_MODES.get(os.getenv("OPENAI_JSON_MODE", "object").strip().lower(), "object")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip()

_ENDPOINT_SUFFIXES = ("/chat/text", "/chat/vision", "/chat/compare")


def _base_url(url: str) -> str:
    """Accept either the wrapper base URL or a full /chat/* endpoint URL, and
    return the base (so we can safely append /chat/text ourselves)."""
    url = url.strip().rstrip("/")
    for suffix in _ENDPOINT_SUFFIXES:
        if url.endswith(suffix):
            return url[: -len(suffix)].rstrip("/")
    return url

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a free-form model reply.

    Tolerates ```json fences and leading/trailing prose ("Here is the JSON: ...").
    """
    s = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON object in model reply")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError("unbalanced JSON in model reply")


class LLMClient:
    """Shared JSON-over-text logic; subclasses implement `_complete()`."""

    def _complete(self, system: str, user: str, max_tokens: int,
                  temperature: float, json_reply: bool = False,
                  schema: dict | None = None) -> str:
        """One completion. `json_reply=True` hints that the reply must be a JSON
        object; `schema`, when given, is the shape it must have."""
        raise NotImplementedError

    def chat_text(self, prompt: str, max_tokens: int,
                  temperature: float = 0.2) -> str:
        return self._complete("", prompt, max_tokens, temperature)

    def structured(self, system: str, user: str, schema: dict, max_tokens: int,
                   temperature: float = 0.2, retries: int = 1) -> tuple[dict, str]:
        """Get a schema-conforming JSON object back from a text-only endpoint.

        Returns (parsed_dict, raw_text). Raw text is handed back so the caller
        can estimate output tokens (the wrapper reports no usage)."""
        instructions = (
            "Respond with ONLY a single JSON object that conforms to this JSON "
            "schema. Do not include any explanation, preamble, or markdown code "
            f"fences.\n\nJSON schema:\n{json.dumps(schema)}"
        )
        nudge = ""
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            text = self._complete(system, f"{user}\n\n{instructions}{nudge}",
                                  max_tokens, temperature, json_reply=True,
                                  schema=schema)
            try:
                return _extract_json(text), text
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                nudge = ("\n\nYour previous reply was not valid JSON. Return "
                         "ONLY the JSON object, nothing else.")
        raise RuntimeError(f"model did not return valid JSON: {last_err}")


class QwenClient(LLMClient):
    """Thin sync client over the Qwen wrapper's /chat/text endpoint."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None):
        self.base_url = _base_url(base_url or API_URL)
        self.api_key = api_key if api_key is not None else API_KEY
        self.timeout = timeout or TIMEOUT

    def _complete(self, system: str, user: str, max_tokens: int,
                  temperature: float, json_reply: bool = False,
                  schema: dict | None = None) -> str:
        # The wrapper takes a single prompt with its own fixed system prompt,
        # so fold ours into the text.
        prompt = f"{system}\n\n{user}" if system else user
        r = httpx.post(
            f"{self.base_url}/chat/text",
            headers={"api-key": self.api_key},
            json={"prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["response"]


class OpenAIClient(LLMClient):
    """Sync client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float | None = None,
                 json_mode: str | None = None,
                 max_tokens_param: str | None = None):
        url = (base_url or OPENAI_BASE_URL).strip().rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]
        self.base_url = url
        self.api_key = api_key if api_key is not None else OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        self.timeout = timeout or TIMEOUT
        self.json_mode = OPENAI_JSON_MODE if json_mode is None else json_mode
        self.max_tokens_param = max_tokens_param or OPENAI_MAX_TOKENS_PARAM
        self.reasoning_effort = OPENAI_REASONING_EFFORT

    def _post(self, body: dict) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.post(f"{self.base_url}/chat/completions", headers=headers,
                          json=body, timeout=self.timeout)

    def _complete(self, system: str, user: str, max_tokens: int,
                  temperature: float, json_reply: bool = False,
                  schema: dict | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        body = {"model": self.model, "messages": messages,
                self.max_tokens_param: max_tokens, "temperature": temperature}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if json_reply and self.json_mode == "schema" and schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "reply", "strict": True, "schema": schema}}
        elif json_reply and self.json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self._post(body)
        if r.status_code == 400 and "response_format" in body:
            # Server doesn't support this JSON mode — stop asking for it.
            self.json_mode = ""
            body.pop("response_format")
            r = self._post(body)
        if r.is_error:
            # Surface the server's reason (context overflow, bad param, ...).
            raise httpx.HTTPStatusError(
                f"{r.status_code} from {r.url}: {r.text[:500]}",
                request=r.request, response=r)
        return r.json()["choices"][0]["message"]["content"] or ""


def provider() -> str:
    p = os.getenv("LLM_PROVIDER", "").strip().lower()
    if p:
        return p
    return "openai" if (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL")) else "qwen"


def missing_config() -> list[str]:
    """Env vars the selected provider needs but doesn't have."""
    if provider() == "openai":
        # Local OpenAI-compatible servers often need no key, only a base URL.
        return [] if (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL")) \
            else ["OPENAI_API_KEY"]
    return [] if os.getenv("QWEN_API_KEY") else ["QWEN_API_KEY"]


def make_client() -> LLMClient:
    p = provider()
    if p == "openai":
        return OpenAIClient()
    if p == "qwen":
        return QwenClient()
    raise ValueError(f"unknown LLM_PROVIDER {p!r} (expected 'openai' or 'qwen')")
