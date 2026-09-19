from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from sightline.audiences import PERSONAS, Persona
from sightline.divergence import SentenceTransformerEmbedder


def persona_of(system: str) -> Persona:
    """Which persona a system prompt belongs to, via the label line in its WHO YOU ARE block."""
    found = [p for p in PERSONAS if f"Your background: {p.upper()}." in system]
    assert len(found) == 1, f"system prompt must identify exactly one persona, got {found}"
    return found[0]


def slide_no(user_text: str) -> int:
    return int(re.match(r"Slide (\d+)\.", user_text).group(1))


@dataclass
class Call:
    persona: Persona
    system: str
    user_text: str
    image_png: bytes | None
    schema: dict[str, Any]


def sentinel_payload(persona: Persona, user_text: str) -> dict[str, Any]:
    """A valid response whose strings are unique to (persona, slide), so any leak of one
    persona's output into another persona's prompt is detectable by substring."""
    n = slide_no(user_text)
    tag = f"SENTINEL-{persona.upper()}-{n}"
    return {
        "takeaway": f"{tag} takeaway",
        "confidence": 0.5,
        "unresolved_terms": [f"{tag} term"],
        "questions": [f"{tag} question"],
        "inferred_claim": f"{tag} claim",
    }


class FakeLLM:
    model = "fake-model"

    def __init__(
        self,
        responder: Callable[[Persona, int, str], dict[str, Any]] | None = None,
        delay: float = 0.0,
    ):
        # responder(persona, nth call by that persona, user_text) -> payload
        self.responder = responder or (lambda persona, n, user_text: sentinel_payload(persona, user_text))
        self.delay = delay
        self.calls: list[Call] = []
        self._in_flight = 0
        self.max_in_flight = 0

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        persona = persona_of(system)
        self.calls.append(Call(persona, system, user_text, image_png, schema))
        nth = sum(1 for c in self.calls if c.persona == persona)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.responder(persona, nth, user_text)
        finally:
            self._in_flight -= 1

    def calls_by(self, persona: Persona) -> list[Call]:
        return [c for c in self.calls if c.persona == persona]


@pytest.fixture(scope="session")
def real_embedder():
    """The real local model. If it cannot load, tests using it FAIL rather than skip:
    a skipped embedding test would silently hide a broken headline metric."""
    emb = SentenceTransformerEmbedder()
    try:
        emb.embed(["warm-up"])
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"sentence-transformers model unavailable: {e!r}")
    return emb
