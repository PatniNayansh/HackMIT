"""The only module that talks to an LLM SDK (OpenAI, through the Responses API).

Everything else depends on the `LLMClient` protocol, so tests substitute a fake and the
fix loop / intent prefill can share one client. Callers hand this module a JSON schema and get
a plain dict back; nothing in here leaks through that boundary.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Literal, Protocol, Union

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

# The one place to swap models or reasoning effort (at the booth, say). Both are set per role and
# passed on every call: OpenAI defaults to `medium` where the Claude setup ran at `low`, so leaving
# effort out would silently spend more and run slower. Valid efforts: none, low, medium, high,
# xhigh, max. If the gate fails on terra at `low`, raise the persona effort to `medium` before
# reaching for a bigger model.
#
#   persona      the three audience readers (also the profile inference and the recommendations,
#                which share their client)
#   structuring  field extraction and claim comparison over short text
#   helper       the figure describer (vision, at upload)
#   intent       the intent rephrasing model (kept in intent.py, unused by the pipeline)
CONFIG: dict[str, dict[str, str]] = {
    "models": {
        "persona": "gpt-5.6-terra",
        "structuring": "gpt-5.6-luna",
        "helper": "gpt-5.6-luna",
        "intent": "gpt-5.6-luna",
    },
    "effort": {
        "persona": "low",
        "structuring": "low",
        "helper": "low",
        "intent": "low",
    },
}
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
# Environment overrides, per role: (model variable, effort variable).
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "persona": ("SIGHTLINE_MODEL", "SIGHTLINE_EFFORT"),
    "structuring": ("SIGHTLINE_STRUCTURING_MODEL", "SIGHTLINE_STRUCTURING_EFFORT"),
    "helper": ("SIGHTLINE_HELPER_MODEL", "SIGHTLINE_HELPER_EFFORT"),
    "intent": ("SIGHTLINE_INTENT_MODEL", "SIGHTLINE_INTENT_EFFORT"),
}
KEY_VARIABLE = "OPENAI_API_KEY"

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


# ------------------------------------------------------------------ schemas, declared and validated
#
# Callers pass a JSON schema per output shape (the persona report, the structuring result, the
# proposition-coverage result, ...). It is sent as a strict structured-output format, and the reply is
# validated here by a Pydantic model built from that same schema, then handed back as a plain dict
# through `.model_dump()`. Types are checked, and NOTHING is repaired: a value the model got wrong is
# an error, never coerced or clamped (a confidence of 1.4 stays 1.4 all the way to the caller, which
# rejects it).

_UNSUPPORTED = ("default", "title", "examples")  # keywords strict mode does not take; harmless to drop


def strictify(schema: Any) -> Any:
    """The schema in the form strict structured outputs require: every object closed
    (`additionalProperties: false`) with every property listed as required. The schemas this project
    writes already are, and a test holds them to that, so this only guards a future omission."""
    if isinstance(schema, list):
        return [strictify(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: strictify(v) for k, v in schema.items() if k not in _UNSUPPORTED}
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"])
    return out


_SCALARS: dict[str, Any] = {"string": str, "number": float, "integer": int, "boolean": bool, "null": type(None)}


def _annotation(schema: dict[str, Any], name: str) -> Any:
    if "anyOf" in schema:
        return Union[tuple(_annotation(s, name) for s in schema["anyOf"])]  # noqa: UP007
    kind = schema.get("type")
    if isinstance(kind, list):
        return Union[tuple(_annotation({**schema, "type": k}, name) for k in kind)]  # noqa: UP007
    if "enum" in schema:
        return Literal[tuple(schema["enum"])]  # type: ignore[valid-type]
    if kind == "object":
        return _model(schema, name)
    if kind == "array":
        return list[_annotation(schema.get("items", {}), name)]  # type: ignore[misc]
    if kind in _SCALARS:
        return _SCALARS[kind]
    raise ValueError(f"unsupported schema at {name!r}: {schema!r}")


def _model(schema: dict[str, Any], name: str) -> type[BaseModel]:
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for i, (prop, sub) in enumerate(schema.get("properties", {}).items()):
        annotation = _annotation(sub, f"{name}.{prop}")
        # Internal names are positional so a property called `schema` or `copy` cannot shadow a
        # BaseModel attribute; the property name is the alias, and it is what goes back out.
        fields[f"f{i}"] = (annotation, Field(... if prop in required else None, alias=prop))
    return create_model(
        f"Shape_{abs(hash(name)) % 10**8}", __config__=ConfigDict(strict=True, extra="forbid", populate_by_name=False), **fields
    )


_MODELS: dict[str, type[BaseModel]] = {}


def validated(schema: dict[str, Any], raw: Any) -> Any:
    """`raw` checked against `schema` and returned as plain data. Raises pydantic.ValidationError."""
    key = json.dumps(schema, sort_keys=True)
    if key not in _MODELS:
        _MODELS[key] = _model(schema, "root")
    model = _MODELS[key]
    return model.model_validate(raw).model_dump(by_alias=True)


# ----------------------------------------------------------------------------- the client


def _redact(text: str) -> str:
    """An API error can quote part of the key it was sent. None of it may reach a log, a retry note
    or the UI."""
    key = os.environ.get(KEY_VARIABLE)
    if key:
        text = text.replace(key, "[redacted]")
    return re.sub(r"sk-[A-Za-z0-9_\-\*\.]{3,}", "sk-[redacted]", text)


class OpenAIClient:
    """`role` picks the model and effort from CONFIG (each overridable through the environment);
    `model` and `effort` set them explicitly."""

    def __init__(self, model: str | None = None, effort: str | None = None, client: Any = None, role: str = "persona"):
        import openai

        if role not in CONFIG["models"]:
            raise ValueError(f"unknown role {role!r}; expected one of {sorted(CONFIG['models'])}")
        load_env()
        model_var, effort_var = ENV_OVERRIDES[role]
        self.role = role
        self.model = model or os.environ.get(model_var) or CONFIG["models"][role]
        self.effort = effort or os.environ.get(effort_var) or CONFIG["effort"][role]
        if self.effort not in EFFORTS:
            raise ValueError(f"unknown reasoning effort {self.effort!r}; expected one of {EFFORTS}")
        if client is None:
            try:
                client = openai.AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
            except openai.OpenAIError as e:
                # Callers treat "no credential" as a TypeError (the app still starts, saved runs still
                # open). The message names the variable, never a value.
                raise TypeError(f"no credential: set {KEY_VARIABLE} in the environment or the repo-root .env") from e
        self._client = client
        # Per-call token counts, for diagnosing latency. output_tokens includes any reasoning.
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
        import openai

        content: list[dict[str, Any]] = []
        if image_png is not None:
            data = base64.standard_b64encode(image_png).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{data}"})
        content.append({"type": "input_text", "text": user_text})

        try:
            response = await self._client.responses.create(
                model=self.model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                text={"format": {"type": "json_schema", "name": "response", "schema": strictify(schema), "strict": True}},
                reasoning={"effort": self.effort},
                max_output_tokens=max_tokens,
                store=False,
            )
        except openai.OpenAIError as e:
            e.message = _redact(str(getattr(e, "message", e)))  # type: ignore[attr-defined]
            e.args = (e.message,)  # type: ignore[attr-defined]
            raise

        usage = response.usage
        if usage is not None:
            self.usage_log.append({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens})

        if response.status == "failed":
            err = response.error
            raise LLMError(f"model call failed: {_redact(f'{err.code}: {err.message}') if err else 'no detail'}")
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else None
            if reason == "content_filter":
                raise LLMError("response withheld by the content filter")
            raise LLMError("response truncated at max_tokens")
        if response.status != "completed":
            raise LLMError(f"response did not complete (status {response.status})")

        text = None
        for item in response.output:
            if item.type != "message":
                continue
            for part in item.content:
                if part.type == "refusal":
                    raise LLMError(f"model refused the request: {_redact(part.refusal)}")
                if part.type == "output_text" and text is None:
                    text = part.text
        if text is None:
            raise LLMError("response contained no text block")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"response was not valid JSON: {e}") from e
        try:
            return validated(schema, raw)
        except ValidationError as e:
            raise LLMError(f"response did not match the schema: {_redact(str(e))}") from e
