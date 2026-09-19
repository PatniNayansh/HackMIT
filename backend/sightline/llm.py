"""The only module that talks to the Anthropic SDK.

Everything else depends on the `LLMClient` protocol, so tests substitute a fake and the
fix loop / intent prefill can share one client.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values

# Persona simulation is a latency-sensitive path (budget: 5 s per slide, three calls in
# parallel). Override with SIGHTLINE_MODEL if Sonnet 5 is too slow; measure before changing.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "low"


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"  # repo root; git-ignored


def load_env(path: Path = ENV_FILE) -> None:
    """Put non-empty values from the repo-root .env into os.environ. A variable already set
    in the real environment wins, and blank values (a copied .env.example) are ignored so
    they cannot shadow other credential sources."""
    for key, value in dotenv_values(path).items():
        if value and key not in os.environ:
            os.environ[key] = value


class LLMError(RuntimeError):
    """The model call completed but did not yield a usable structured answer."""


class LLMClient(Protocol):
    model: str

    async def complete_json(
        self,
        *,
        system: str,
        user_text: str,
        image_png: bytes | None,
        schema: dict[str, Any],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """One request, one JSON object conforming to `schema`. No conversation state."""
        ...


class AnthropicClient:
    def __init__(self, model: str | None = None, effort: str | None = None, client: Any = None):
        import anthropic

        load_env()
        self.model = model or os.environ.get("SIGHTLINE_MODEL", DEFAULT_MODEL)
        effort = effort if effort is not None else os.environ.get("SIGHTLINE_EFFORT", DEFAULT_EFFORT)
        # Haiku 4.5 rejects `effort`; "none" (or empty) omits it for any model.
        self.effort = None if effort in ("", "none") or "haiku" in self.model else effort
        self._client = client or anthropic.AsyncAnthropic()
        # Per-call token counts, for diagnosing latency. output_tokens includes any thinking.
        self.usage_log: list[dict[str, int]] = []

    async def complete_json(
        self,
        *,
        system: str,
        user_text: str,
        image_png: bytes | None,
        schema: dict[str, Any],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if image_png is not None:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(image_png).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": user_text})

        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config=output_config,
        )
        self.usage_log.append(
            {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"model refused the request: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("response contained no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"response was not valid JSON: {e}") from e
