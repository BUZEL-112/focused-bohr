"""Zero-dependency OpenAI-compatible chat client driven entirely by environment config.

Resolution order (mirrors the hackathon's compute.env contract, see
``collusion_hackathon/COMPUTE.md``):

1. ``LLM_BASE_URL`` + ``LLM_API_KEY``  (custom OpenAI-compatible endpoint)
2. ``NEBIUS_API_KEY``                  (https://api.studio.nebius.ai/v1/)
3. ``OPENROUTER_API_KEY``              (https://openrouter.ai/api/v1)

``LLM_MODEL`` / ``MODEL_NAME`` override the default model id. Keys are also read from a
``.env`` / ``compute.env`` file when python-dotenv is unavailable (tiny manual parser).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/"
DEFAULT_MODEL = "openai/gpt-4.1-mini"

_ENV_FILE_CANDIDATES = (".env", "compute.env")


def load_env_file(search_from: Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from .env / compute.env without overwriting real env vars."""
    start = (search_from or Path.cwd()).resolve()
    for directory in [start, *start.parents]:
        for name in _ENV_FILE_CANDIDATES:
            candidate = directory / name
            if candidate.is_file():
                return _parse_env_file(candidate)
        if (directory / ".git").exists():
            break
    return {}


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        values.setdefault(key.strip(), value)
    return values


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    api_key: str
    model: str

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


def resolve_endpoint(explicit_model: str | None = None) -> EndpointConfig:
    env = load_env_file()
    def get(key: str) -> str:
        return os.getenv(key) or env.get(key) or ""

    base_url = get("LLM_BASE_URL")
    api_key = get("LLM_API_KEY")
    if base_url:
        api_key = api_key or get("OPENROUTER_API_KEY") or get("NEBIUS_API_KEY")
    if not base_url:
        if get("NEBIUS_API_KEY"):
            base_url, api_key = NEBIUS_BASE_URL, get("NEBIUS_API_KEY")
        elif get("OPENROUTER_API_KEY"):
            base_url, api_key = OPENROUTER_BASE_URL, get("OPENROUTER_API_KEY")
    model = explicit_model or get("LLM_MODEL") or get("MODEL_NAME") or DEFAULT_MODEL
    return EndpointConfig(base_url=base_url, api_key=api_key, model=model)


class LLMClientError(RuntimeError):
    """Raised when the endpoint is unconfigured or every retry fails."""


class LLMClient:
    """Minimal chat-completions client with retry/backoff; stdlib only."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 4,
        timeout_seconds: float = 120.0,
    ) -> None:
        config = resolve_endpoint(explicit_model=model)
        if not config.is_configured or not config.base_url:
            raise LLMClientError(
                "No LLM endpoint configured. Set OPENROUTER_API_KEY or NEBIUS_API_KEY "
                "(or LLM_BASE_URL + LLM_API_KEY) in the environment or compute.env."
            )
        self.base_url = config.base_url.rstrip("/") + "/"
        self.api_key = config.api_key
        self.model = config.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.base_url + "chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices") or []
                if not choices:
                    raise LLMClientError(f"no choices in response: {body}")
                content = choices[0].get("message", {}).get("content")
                if not content:
                    raise LLMClientError(f"empty completion: {body}")
                return str(content)
            except urllib.error.HTTPError as exc:
                last_error = exc
                retryable = exc.code in (429, 500, 502, 503, 504)
                if not retryable or attempt >= self.max_retries:
                    raise LLMClientError(f"HTTP {exc.code}: {exc.read()[:400]!r}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, LLMClientError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
            time.sleep(min(2**attempt, 15))
        raise LLMClientError(f"completion failed after {self.max_retries + 1} attempts: {last_error}")


def extract_number(text: str) -> float | None:
    """First float in the text (used for score/price parsing of judge replies)."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None
