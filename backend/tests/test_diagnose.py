from __future__ import annotations

import json

import pytest

from sightline import diagnose
from sightline.deck import rollup

from builders import slide_result


def test_fixture_is_marked_as_fixture_and_matches_the_contract():
    assert diagnose.FIXTURE_BACKED is True
    findings = diagnose.diagnose(slide_result(1), rollup([slide_result(1)]))
    assert findings, "the UI is rendered from this fixture"
    for f in findings:
        assert set(f) == set(diagnose.Finding.__annotations__)
        assert f["severity"] in ("high", "medium", "low")
        assert all(isinstance(f[k], str) and f[k] for k in f)


def test_a_malformed_fixture_fails_loudly(tmp_path):
    bad = tmp_path / "f.json"
    bad.write_text(json.dumps([{"id": "x", "severity": "urgent"}]))
    with pytest.raises(ValueError, match="Finding contract"):
        diagnose.load_fixture(bad)
