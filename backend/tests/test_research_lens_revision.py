"""research_lens.propose_adhd_friendly_revision: the one LLM call in the ADHD lens.

Kept separate from test_research_lens.py (which only covers the pure-arithmetic lens
logic) since this exercises an actual model call shape and needs its own fake client.
"""

from __future__ import annotations

import pytest

from sightline.audiences import SlideInput
from sightline.llm import LLMError
from sightline.research_lens import propose_adhd_friendly_revision

ORIGINAL_TEXT = "[title] Serving faster\n[body] Dense wall of acronyms, all at once, no structure"
REVISED_TEXT = "[title] Serving faster\n[body] 1. What changed\n[body] 2. Why it's faster"


class FakeReviseClient:
    model = "fake-model"

    def __init__(self, payload=None):
        self.payload = payload or {"revised_text": REVISED_TEXT, "rationale": "chunked into steps"}
        self.calls: list[dict] = []

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls.append({"system": system, "user_text": user_text, "schema": schema})
        return self.payload


@pytest.mark.asyncio
async def test_returns_the_revised_text():
    client = FakeReviseClient()
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    revised = await propose_adhd_friendly_revision(client, slide)
    assert revised == REVISED_TEXT


@pytest.mark.asyncio
async def test_prompt_includes_the_original_text_and_names_the_mechanism():
    client = FakeReviseClient()
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    await propose_adhd_friendly_revision(client, slide)
    assert len(client.calls) == 1
    assert ORIGINAL_TEXT in client.calls[0]["user_text"]
    assert "default-mode network" in client.calls[0]["system"]
    assert "not a proven fix" in client.calls[0]["system"]


@pytest.mark.asyncio
async def test_rejects_an_empty_revision():
    client = FakeReviseClient(payload={"revised_text": "   ", "rationale": "n/a"})
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    with pytest.raises(LLMError, match="empty"):
        await propose_adhd_friendly_revision(client, slide)
