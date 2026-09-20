"""OpenAIClient against the REAL SDK with a mocked HTTP transport.

This is the closest to a live call available without credentials: it proves the SDK accepts our
kwargs, that the wire body has the image, the strict schema, the model and the reasoning effort,
and that refusal, truncation and malformed replies are surfaced as LLMError. It cannot prove the
model behaves (that is the live gate).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx2
import openai
import pytest
from slides import CLEAR_PROFILE, clear_slide

from sightline import compare, diagnose, ingest, intent, llm
from sightline.audiences import RESPONSE_SCHEMA, AudienceEngine, AudienceResponseError
from sightline.llm import CONFIG, LLMError, OpenAIClient

PAYLOAD = {
    "takeaway": "t", "confidence": 0.4, "unresolved_terms": ["x"], "questions": [], "inferred_claim": "c",
}
SECRET = "sk-proj-THISISNOTAREALKEY0123456789"


def sdk_client(reply: dict, seen: list, status: int = 200) -> openai.AsyncOpenAI:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append({"body": json.loads(request.content), "headers": dict(request.headers), "url": str(request.url)})
        return httpx2.Response(status, json=reply)

    return openai.AsyncOpenAI(
        api_key="test", max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )


def response(text: str | None = None, *, status: str = "completed", refusal: str | None = None,
             incomplete: str | None = None, error: dict | None = None, usage: dict | None = None) -> dict:
    content = []
    if text is not None:
        content.append({"type": "output_text", "text": text, "annotations": []})
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "status": status, "model": "gpt-5.6-terra",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            *([{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": content}] if content else []),
        ],
        "incomplete_details": {"reason": incomplete} if incomplete else None,
        "error": error,
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": usage or {
            "input_tokens": 11, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 3}, "total_tokens": 18,
        },
    }


def client_for(reply: dict, seen: list | None = None, **kwargs) -> OpenAIClient:
    return OpenAIClient(client=sdk_client(reply, [] if seen is None else seen), **kwargs)


ASK = dict(system="S", user_text="U", image_png=None, schema=RESPONSE_SCHEMA)

# ------------------------------------------------------------------ 1. a well-formed response


async def test_request_body_and_parsed_response():
    seen: list = []
    llm_ = OpenAIClient(client=sdk_client(response(json.dumps(PAYLOAD)), seen))
    out = await llm_.complete_json(system="SYS", user_text="USER", image_png=b"\x89PNGdata", schema=RESPONSE_SCHEMA)

    assert out == PAYLOAD and isinstance(out, dict)  # a plain dict, exactly the shape callers always got
    request = seen[0]
    assert request["url"].endswith("/responses")
    body = request["body"]
    assert body["model"] == CONFIG["models"]["persona"] and body["reasoning"] == {"effort": CONFIG["effort"]["persona"]}
    system, user = body["input"]
    assert system == {"role": "system", "content": "SYS"} and user["role"] == "user"
    image, text = user["content"]
    assert image["type"] == "input_image"
    assert base64.b64decode(image["image_url"].removeprefix("data:image/png;base64,")) == b"\x89PNGdata"
    assert text == {"type": "input_text", "text": "USER"}
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["schema"] == RESPONSE_SCHEMA
    assert "temperature" not in body  # reasoning models reject it


async def test_no_image_block_when_slide_has_no_image():
    seen: list = []
    await OpenAIClient(client=sdk_client(response(json.dumps(PAYLOAD)), seen)).complete_json(**ASK)
    assert [b["type"] for b in seen[0]["body"]["input"][1]["content"]] == ["input_text"]


async def test_the_reply_is_validated_but_never_repaired():
    """Types are checked; a well-typed value the model got wrong is handed on untouched for the caller to judge."""
    odd = {**PAYLOAD, "confidence": 1}  # an int where a number is declared is fine
    assert (await client_for(response(json.dumps(odd))).complete_json(**ASK))["confidence"] == 1
    for bad in ({**PAYLOAD, "confidence": "0.4"}, {**PAYLOAD, "questions": "none"}, {k: v for k, v in PAYLOAD.items() if k != "takeaway"}, {**PAYLOAD, "extra": 1}):
        with pytest.raises(LLMError, match="did not match the schema"):
            await client_for(response(json.dumps(bad))).complete_json(**ASK)


# --------------------------------------------------- 2. malformed, refused, truncated, failed


@pytest.mark.parametrize(
    "reply, match",
    [
        (response(json.dumps(PAYLOAD), status="incomplete", incomplete="max_output_tokens"), "truncated"),
        (response(None, status="incomplete", incomplete="content_filter"), "content filter"),
        (response("not json"), "not valid JSON"),
        (response(None), "no text block"),
        (response(None, refusal="I can't help with that."), "refused the request: I can't help"),
        (response(None, status="failed", error={"code": "server_error", "message": "boom"}), "server_error: boom"),
        (response(json.dumps(PAYLOAD), status="cancelled"), "did not complete"),
    ],
)
async def test_unusable_responses_raise_llm_error(reply, match):
    with pytest.raises(LLMError, match=match):
        await client_for(reply).complete_json(**ASK)


async def test_a_refusal_has_no_answer_to_return():
    """Even when a refusal arrives next to text, nothing is returned as if it were an answer."""
    reply = response(json.dumps(PAYLOAD), refusal="no")
    reply["output"][1]["content"].reverse()
    with pytest.raises(LLMError, match="refused"):
        await client_for(reply).complete_json(**ASK)


# ------------------------------------------ 3. out-of-range confidence is rejected, not clamped


async def test_the_client_hands_an_out_of_range_confidence_on_unchanged():
    out = await client_for(response(json.dumps({**PAYLOAD, "confidence": 1.4}))).complete_json(**ASK)
    assert out["confidence"] == 1.4  # not clamped to 1.0 here or anywhere on the way


async def test_the_engine_rejects_confidence_1_4_and_retries_and_never_clamps():
    calls: dict[str, int] = {}
    seen: list = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        persona = re.search(r"Your background: (\w+)\.", body["input"][0]["content"]).group(1)
        calls[persona] = calls.get(persona, 0) + 1  # each persona's first answer is out of range, its second is fine
        return httpx2.Response(200, json=response(json.dumps({**PAYLOAD, "confidence": 1.4 if calls[persona] == 1 else 0.6})))

    sdk = openai.AsyncOpenAI(api_key="test", max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    engine = AudienceEngine(OpenAIClient(client=sdk), None, offline=False)
    readings = await engine.read_slide(clear_slide(), CLEAR_PROFILE)
    assert {r.response.confidence for r in readings.values()} == {0.6}  # the second answer, never 1.4 clamped to 1.0
    assert all(r.attempts == 2 for r in readings.values())
    assert any("rejected" in json.dumps(b) for b in seen)  # the retry told the model why


async def test_the_engine_gives_up_on_a_confidence_that_stays_out_of_range():
    sdk = sdk_client(response(json.dumps({**PAYLOAD, "confidence": 1.4})), [])
    with pytest.raises(AudienceResponseError):
        await AudienceEngine(OpenAIClient(client=sdk), None, offline=False).read_slide(clear_slide(), CLEAR_PROFILE)


# --------------------------------------------------------------- 4. the key: environment only


def test_the_key_is_read_from_the_environment_and_reaches_the_wire_only_as_a_bearer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(llm, "load_env", lambda *a, **k: None)
    c = OpenAIClient()  # no key passed anywhere
    assert c._client.api_key == SECRET and SECRET not in repr(vars(c) | {"_client": None})


def test_no_key_means_a_type_error_that_names_the_variable_and_no_value(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.setattr(llm, "load_env", lambda *a, **k: None)
    with pytest.raises(TypeError, match="OPENAI_API_KEY"):
        OpenAIClient()  # the app's factories catch this and start with no client


def test_no_key_literal_is_in_the_code_and_nothing_is_written_to_disk():
    for path in (Path(llm.__file__).parent).glob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Anchored to a word boundary: a real key literal follows a quote or a space, where
        # the "task-performance" of a citation does not (research_lens.py cites one).
        assert not re.search(r"\bsk-[A-Za-z0-9_\-]{8,}", text), path.name
        assert not re.search(r"api_key\s*=\s*[\"']", text), path.name
    src = Path(llm.__file__).read_text(encoding="utf-8")
    assert "write_text" not in src and "open(" not in src  # llm.py reads .env through dotenv and writes nothing


async def test_an_api_error_never_carries_the_key():
    """OpenAI's 401 quotes a fragment of the key it was sent; that must not reach a log, a retry note or the UI."""
    err = {"error": {"message": f"Incorrect API key provided: {SECRET[:12]}****{SECRET[-4:]}. Find your key at ...", "type": "invalid_request_error", "code": "invalid_api_key"}}
    c = client_for(err)
    c._client = sdk_client(err, [], status=401)
    with pytest.raises(openai.AuthenticationError) as e:
        await c.complete_json(**ASK)
    assert SECRET[:12] not in str(e.value) and SECRET[-4:] not in str(e.value) and "sk-[redacted]" in str(e.value)


def test_redaction_masks_the_exact_key_value_too(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-shaped-like-a-key-12345")
    assert "not-shaped" not in llm._redact("bad credential not-shaped-like-a-key-12345 sent")


def test_env_file_supplies_missing_values_but_never_overrides_the_shell(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=from-file\nSIGHTLINE_MODEL=from-file\nSIGHTLINE_EFFORT=\n", encoding="utf-8")
    # setenv-then-delenv registers the variable with monkeypatch so teardown restores it
    for name in ("OPENAI_API_KEY", "SIGHTLINE_EFFORT"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setenv("SIGHTLINE_MODEL", "from-shell")

    llm.load_env(env)

    import os

    assert os.environ["OPENAI_API_KEY"] == "from-file"
    assert os.environ["SIGHTLINE_MODEL"] == "from-shell"  # shell wins
    assert "SIGHTLINE_EFFORT" not in os.environ  # blank value ignored


def test_missing_env_file_is_fine(tmp_path):
    llm.load_env(tmp_path / "nope.env")


# ---------------------------------------------------------------------- 5. token counts


async def test_token_counts_are_logged_from_the_responses_usage_fields():
    c = client_for(response(json.dumps(PAYLOAD)))
    await c.complete_json(**ASK)
    await c.complete_json(**ASK)
    assert c.usage_log == [{"input_tokens": 11, "output_tokens": 7}] * 2  # the same two fields as before; output includes reasoning


async def test_tokens_are_logged_even_when_the_call_is_unusable():
    c = client_for(response(json.dumps(PAYLOAD), status="incomplete", incomplete="max_output_tokens"))
    with pytest.raises(LLMError):
        await c.complete_json(**ASK)
    assert len(c.usage_log) == 1


# --------------------------------------- 6. model and effort come from the config, not the call site


async def _wire(client: OpenAIClient, seen: list) -> dict:
    await client.complete_json(**ASK)
    return seen[-1]["body"]


async def test_each_role_takes_its_model_and_effort_from_the_config_dict(monkeypatch):
    for var in ("SIGHTLINE_MODEL", "SIGHTLINE_EFFORT", "SIGHTLINE_STRUCTURING_MODEL", "SIGHTLINE_HELPER_MODEL", "SIGHTLINE_INTENT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "load_env", lambda *a, **k: None)
    assert CONFIG["models"] == {"persona": "gpt-5.6-terra", "structuring": "gpt-5.6-luna", "helper": "gpt-5.6-luna", "intent": "gpt-5.6-luna"}
    assert set(CONFIG["effort"].values()) == {"low"}  # OpenAI defaults to medium; the Claude setup ran at low
    for role in CONFIG["models"]:
        seen: list = []
        c = OpenAIClient(role=role, client=sdk_client(response(json.dumps(PAYLOAD)), seen))
        body = await _wire(c, seen)
        assert body["model"] == CONFIG["models"][role] == c.model
        assert body["reasoning"] == {"effort": CONFIG["effort"][role]} == {"effort": c.effort}


async def test_editing_the_config_dict_is_all_it_takes_to_swap_a_model_or_an_effort(monkeypatch):
    monkeypatch.setattr(llm, "load_env", lambda *a, **k: None)
    monkeypatch.delenv("SIGHTLINE_MODEL", raising=False)
    monkeypatch.delenv("SIGHTLINE_EFFORT", raising=False)
    monkeypatch.setitem(CONFIG["models"], "persona", "some-other-model")
    monkeypatch.setitem(CONFIG["effort"], "persona", "medium")
    seen: list = []
    body = await _wire(OpenAIClient(client=sdk_client(response(json.dumps(PAYLOAD)), seen)), seen)
    assert body["model"] == "some-other-model" and body["reasoning"] == {"effort": "medium"}


async def test_effort_is_sent_on_every_call_including_none(monkeypatch):
    monkeypatch.setattr(llm, "load_env", lambda *a, **k: None)
    for effort in llm.EFFORTS:
        seen: list = []
        body = await _wire(OpenAIClient(effort=effort, client=sdk_client(response(json.dumps(PAYLOAD)), seen)), seen)
        assert body["reasoning"] == {"effort": effort}  # never omitted, so the API default cannot apply silently


def test_an_unknown_effort_or_role_is_a_visible_error():
    with pytest.raises(ValueError, match="unknown reasoning effort"):
        OpenAIClient(effort="turbo", client=object())
    with pytest.raises(ValueError, match="unknown role"):
        OpenAIClient(role="nobody", client=object())


async def test_the_environment_can_override_a_role_without_touching_code(monkeypatch):
    monkeypatch.setattr(llm, "load_env", lambda *a, **k: None)
    monkeypatch.setenv("SIGHTLINE_STRUCTURING_MODEL", "gpt-5.6-terra")
    monkeypatch.setenv("SIGHTLINE_STRUCTURING_EFFORT", "medium")
    c = OpenAIClient(role="structuring", client=object())
    assert (c.model, c.effort) == ("gpt-5.6-terra", "medium")


def test_no_model_id_is_written_anywhere_but_the_config_dict():
    sightline = Path(llm.__file__).parent
    for path in sightline.glob("*.py"):
        lines = [(n, line) for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if re.search(r"gpt-\d|claude-", line)]
        if path.name == "llm.py":
            block = re.search(r"CONFIG: dict.*?\n}\n", path.read_text(encoding="utf-8"), re.S).group(0)
            assert all(line.strip() in block for _, line in lines if not line.lstrip().startswith("#")), (path.name, lines)
        else:
            assert not lines, (path.name, lines)


# --------------------------------------------------- the schemas callers hand in, unchanged


ALL_SCHEMAS = {
    "persona report": RESPONSE_SCHEMA,
    "structuring + coverage": compare.STRUCTURE_SCHEMA,
    "coverage re-ask": compare._COVERAGE_SCHEMA,
    "figure description": ingest.FIGURE_SCHEMA,
    "profile inference": ingest.PROFILE_SCHEMA,
    "recommendations": diagnose.RECS_SCHEMA,
    "intent": intent.INTENT_SCHEMA,
}


@pytest.mark.parametrize("name", ALL_SCHEMAS)
def test_every_schema_callers_pass_is_already_strict_so_the_shapes_do_not_change(name):
    assert llm.strictify(ALL_SCHEMAS[name]) == ALL_SCHEMAS[name]


def _example(schema):
    """The smallest value that satisfies a schema."""
    if "anyOf" in schema:
        return _example(schema["anyOf"][0])
    kind = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    return {
        "object": lambda: {k: _example(v) for k, v in schema.get("properties", {}).items()},
        "array": lambda: [_example(schema["items"])],
        "string": lambda: "s", "number": lambda: 0.5, "integer": lambda: 1, "boolean": lambda: True, "null": lambda: None,
    }[kind]()


@pytest.mark.parametrize("name", ALL_SCHEMAS)
def test_every_schema_round_trips_through_validation_as_the_same_plain_data(name):
    value = _example(ALL_SCHEMAS[name])
    assert llm.validated(ALL_SCHEMAS[name], value) == value
    with pytest.raises(Exception):  # noqa: B017 - a pydantic ValidationError
        llm.validated(ALL_SCHEMAS[name], {**value, "not_a_field": 1})


def test_strictify_closes_an_open_object_and_requires_every_property():
    open_ = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "object", "properties": {"c": {"type": "number"}}}}}
    out = llm.strictify(open_)
    assert out["additionalProperties"] is False and out["required"] == ["a", "b"]
    assert out["properties"]["b"]["additionalProperties"] is False and out["properties"]["b"]["required"] == ["c"]
