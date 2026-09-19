"""AnthropicClient against the REAL SDK with a mocked HTTP transport.

This is the closest to a live call available without credentials: it proves the SDK
accepts our kwargs, that the wire body has the image block / schema / effort, and that
refusal and truncation are surfaced. It cannot prove the model behaves.
"""

from __future__ import annotations

import base64
import json

import anthropic
import httpx2
import pytest

from sightline.audiences import RESPONSE_SCHEMA
from sightline.llm import AnthropicClient, LLMError

PAYLOAD = {
    "takeaway": "t", "confidence": 0.4, "unresolved_terms": ["x"], "questions": [], "inferred_claim": "c",
}


def sdk_client(reply: dict, seen: list) -> anthropic.AsyncAnthropic:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=reply)

    return anthropic.AsyncAnthropic(
        api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )


def message(text: str | None, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": [{"type": "text", "text": text}] if text is not None else [],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


async def test_request_body_and_parsed_response():
    seen: list = []
    llm = AnthropicClient(model="claude-opus-5", effort="low", client=sdk_client(message(json.dumps(PAYLOAD)), seen))
    out = await llm.complete_json(system="SYS", user_text="USER", image_png=b"\x89PNGdata", schema=RESPONSE_SCHEMA)

    assert out == PAYLOAD
    body = seen[0]
    assert body["model"] == "claude-opus-5" and body["system"] == "SYS"
    image, text = body["messages"][0]["content"]
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == b"\x89PNGdata"
    assert text == {"type": "text", "text": "USER"}
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": RESPONSE_SCHEMA}
    assert body["output_config"]["effort"] == "low"
    assert "temperature" not in body  # rejected by Opus 5
    assert llm.usage_log == [{"input_tokens": 1, "output_tokens": 1}]


async def test_no_image_block_when_slide_has_no_image():
    seen: list = []
    llm = AnthropicClient(model="claude-opus-5", client=sdk_client(message(json.dumps(PAYLOAD)), seen))
    await llm.complete_json(system="S", user_text="U", image_png=None, schema=RESPONSE_SCHEMA)
    assert [b["type"] for b in seen[0]["messages"][0]["content"]] == ["text"]


async def test_effort_is_omitted_for_haiku_and_when_disabled():
    for kwargs in ({"model": "claude-haiku-4-5", "effort": "low"}, {"model": "claude-opus-5", "effort": "none"}):
        seen: list = []
        llm = AnthropicClient(client=sdk_client(message(json.dumps(PAYLOAD)), seen), **kwargs)
        await llm.complete_json(system="S", user_text="U", image_png=None, schema=RESPONSE_SCHEMA)
        assert "effort" not in seen[0]["output_config"]


@pytest.mark.parametrize(
    "reply, match",
    [
        (message(json.dumps(PAYLOAD), "max_tokens"), "truncated"),
        (message("not json"), "not valid JSON"),
        (message(None), "no text block"),
    ],
)
async def test_unusable_responses_raise_llm_error(reply, match):
    llm = AnthropicClient(model="claude-opus-5", client=sdk_client(reply, []))
    with pytest.raises(LLMError, match=match):
        await llm.complete_json(system="S", user_text="U", image_png=None, schema=RESPONSE_SCHEMA)


# ------------------------------------------------------------------------ .env loading


def test_env_file_supplies_missing_values_but_never_overrides_the_shell(tmp_path, monkeypatch):
    from sightline.llm import load_env

    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-file\nSIGHTLINE_MODEL=from-file\nSIGHTLINE_EFFORT=\n")
    # setenv-then-delenv registers the variable with monkeypatch so teardown restores it
    for name in ("ANTHROPIC_API_KEY", "SIGHTLINE_EFFORT"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setenv("SIGHTLINE_MODEL", "from-shell")

    load_env(env)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"
    assert os.environ["SIGHTLINE_MODEL"] == "from-shell"  # shell wins
    assert "SIGHTLINE_EFFORT" not in os.environ  # blank value ignored


def test_missing_env_file_is_fine(tmp_path):
    from sightline.llm import load_env

    load_env(tmp_path / "nope.env")
