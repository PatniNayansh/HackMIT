"""Targeted recommendations: the CONTRACT ONLY. There is no recommendation engine here.

`diagnose` has the signature the real module will implement, but today it returns a
checked-in fixture (`backend/fixtures/diagnose_sample.json`) for every slide, whatever the
slide says. `FIXTURE_BACKED` is what the server passes to the UI so it can badge the findings
as sample data; flip it to False only when `diagnose` is a real implementation, and delete the
fixture path with it.

Contract for a real implementation, from the step 2 brief:
  * `evidence` is a verbatim quote from a persona report or from the slide.
  * `trigger` says which metric or threshold fired, in plain words.
  * `confidence_note` says what the finding cannot tell you.
  * Nothing here may introduce a rating, grade or score: the personas return none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from .deck import DeckRollup, SlideResult
from .store import BACKEND

FIXTURE_BACKED = True
FIXTURE_PATH = BACKEND / "fixtures" / "diagnose_sample.json"


class Finding(TypedDict):
    id: str
    severity: Literal["high", "medium", "low"]
    trigger: str  # which metric/threshold fired, in plain words
    evidence: str  # verbatim quote from a persona report or the slide
    suggestion: str  # what to change
    confidence_note: str  # what this finding cannot tell you


_SEVERITIES = ("high", "medium", "low")


def load_fixture(path: Path = FIXTURE_PATH) -> list[Finding]:
    """The fixture, checked against the contract so a malformed edit fails loudly."""
    findings = json.loads(Path(path).read_text(encoding="utf-8"))
    for f in findings:
        missing = set(Finding.__annotations__) - set(f)
        if missing or f["severity"] not in _SEVERITIES:
            raise ValueError(f"fixture finding {f.get('id', '?')!r} breaks the Finding contract: {sorted(missing) or f['severity']}")
    return findings


def diagnose(slide_result: SlideResult, deck_context: DeckRollup) -> list[Finding]:
    """NOT IMPLEMENTED: returns the same sample findings for every slide."""
    return load_fixture()
