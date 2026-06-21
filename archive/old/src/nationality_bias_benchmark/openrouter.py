"""Small, dependency-free OpenRouter client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    """An API or transport error that should be recorded, not scored."""


def complete_json(
    *,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str,
    site_url: str | None,
    app_name: str | None,
    timeout_seconds: float,
    retries: int,
) -> str:
    """Ask a model for a schema-constrained integer-answer JSON object."""
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        # Gemini 2.5 Flash Lite supports OpenRouter's reasoning control.  The
        # experiment measures direct answers only, not an extended thinking mode.
        "reasoning": {"effort": "none"},
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "integer_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    encoded = json.dumps(payload).encode("utf-8")
    last_error: str | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(API_URL, data=encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise OpenRouterError("response content was not text")
            return content
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {error.code}: {detail}"
            retryable = error.code in {408, 429, 500, 502, 503, 504}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, OpenRouterError) as error:
            last_error = f"{type(error).__name__}: {error}"
            retryable = True

        if not retryable or attempt == retries:
            break
        time.sleep(min(2**attempt, 8))

    raise OpenRouterError(last_error or "unknown OpenRouter error")
