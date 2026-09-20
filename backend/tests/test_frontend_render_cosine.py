"""Renders the real frontend views in Node (frontend/smoke/render.mjs, a tiny DOM stand-in)
against the bundled sample run's real payload, and checks what each screen says. It cannot judge
layout or colour; it does catch runtime errors and pins what text is, and is not, on screen."""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from sightline.audiences import FileCache
from sightline.server import FRONTEND_DIR, create_app
from sightline.store import BUNDLED_RUNS_DIR, RunStore

from builders import HashEmbedder

RUN = "cosine-demo"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to render the frontend")

TERMS = [["p99", "KV-cache"], ["goodput"], ["TTFT", "SLO", "goodput", "p99", "KV-cache", "block tables", "speculative decoding", "PagedAttention", "chunked prefill", "continuous batching"], [], ["alpha"], [], []]
ALIGN = [(0.45, 0.77), (0.44, 0.47), (0.31, 0.76), (0.54, 0.60), (0.70, 0.62), (0.60, 0.80), (0.77, 0.84)]


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    """A run made by the ORIGINAL cosine comparator, saved by hand from the cosine builder, and read
    back through the real server: the cosine UI must keep working when the flag is flipped."""
    from sightline.ingest import Slide
    from builders import slide_result

    tmp = tmp_path_factory.mktemp("render")
    store = RunStore(tmp / "history")
    slides = [Slide(i, f"[title] Slide {i}", b"\x89PNG") for i in range(1, 8)]
    meta = store.create_draft(title="Cosine demo", source_filename="d.pdf", slides=slides, inferred=None, inference_error=None, run_id=RUN)
    store.update_meta(RUN, status="complete", comparator="cosine", model="fake-model", started_at=meta["created_at"], finished_at=meta["created_at"],
                      intent="Our serving system delivers 2.4x higher goodput at p99 latency.",
                      profile={"domain": "LLM inference serving systems", "adjacent_field": "distributed systems", "confirmed": True, "edited": False})
    for i in range(1, 8):
        r = slide_result(i, align=ALIGN[i - 1], terms=(TERMS[i - 1], [], []), text=f"[title] Slide {i}\n[body] p99 goodput KV-cache")
        r["readings"]["expert"]["takeaway"] = r["metrics"]["takeaways"]["expert"] = r["metrics"]["intent"] = r["slide_intent"]["text"] = f"Expert takeaway for slide {i}, verbatim."
        for who in ("novice", "peer", "expert"):
            r["metrics"]["intent_alignment"][who]["inputs"]["intent"] = r["slide_intent"]["text"]
        r["metrics"]["intent_alignment"]["expert"]["inputs"]["expert"] = r["slide_intent"]["text"]
        store.save_result(RUN, r)
        term = TERMS[i - 1][0] if TERMS[i - 1] else "the setup"
        store.save_recs(RUN, i, {"novice": [{"audience": "novice", "bullet": f"Define {term} on first use", "evidence": f"cos={ALIGN[i-1][0]:.2f} takeaway of novice"}],
                                 "peer": [], "expert_flagged": [], "meta": {"model": "fake-sonnet", "dropped_without_evidence": 0}})
    app = create_app(store=store, cache=FileCache(tmp / "cache"), client_factory=lambda: None, embedder=HashEmbedder(), comparator="cosine")
    with TestClient(app) as c:
        run = c.get(f"/api/runs/{RUN}").json()
        recs = {n: c.get(f"/api/runs/{RUN}/slides/{n}/recommendations").json() for n in range(1, 8)}
    return {"run": run, "recs": recs}


def render(tmp_path, payload, slides="3"):
    f = tmp_path / "payload.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    # encoding is explicit: node writes UTF-8, and text=True would otherwise decode it with the
    # platform's locale encoding (cp1252 on Windows), which chokes on the typographic quotes.
    p = subprocess.run(["node", str(FRONTEND_DIR / "smoke" / "render.mjs"), str(f), RUN, slides],
                       capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout)


def everything(out):
    texts = {k: v for k, v in out.items() if isinstance(v, str)}
    for k, drawers in out.items():
        if k.endswith(":drawers"):
            for i, d in enumerate(drawers):
                texts[f"{k}[{i}]"] = d["body"]
    return texts


def test_saved_recommendations_replay_offline_for_every_slide(payload):
    assert all(r["available"] and r["cached"] for r in payload["recs"].values())  # no key, no calls


def test_the_screens_render_without_a_runtime_error(tmp_path, payload):
    out = render(tmp_path, payload, "1,3,6")
    assert out["errors"] == []
    assert out["overview_buttons"] > 0 and all(out[f"slide{n}"] for n in (1, 3, 6))


def test_the_overview_ranks_by_novice_difficulty_and_has_no_differ_most_section(tmp_path, payload):
    o = render(tmp_path, payload)["overview"]
    assert "Hardest slides for a newcomer" in o and "Narrative arc" in o
    assert "Where the audiences differ most" not in o and "Slides ranked by novice" not in o
    tiers = {p["tier"]["label"] for p in payload["run"]["rollup"]["per_slide"]}
    assert tiers and tiers <= {"Self-contained", "Background needed", "Expert-gated"}
    for label in tiers:  # whichever tiers this deck's slides fell in are shown as chips
        assert label in o
    # each row: takeaway, tier chip, novice unresolved-term count
    assert o.count("Novice takeaway") >= 5 and o.count("Novice unresolved terms") >= 5


def test_confidence_and_blind_spot_appear_nowhere_in_the_ui(tmp_path, payload):
    out = render(tmp_path, payload, "1,2,3,4,5,6,7")
    hits = {k: m.group(0) for k, t in everything(out).items() if (m := re.search(r"confiden|blind|diverg", t, re.I))}
    assert hits == {}


def test_every_slide_page_leads_with_a_tier_and_shows_the_expert_takeaway_once_as_the_intent(tmp_path, payload):
    out = render(tmp_path, payload, "1,2,3,4,5,6,7")
    declared = payload["run"]["meta"]["intent"]
    assert declared  # the sample has a declared intent, so its absence below means something
    for n in range(1, 8):
        page = out[f"slide{n}"]
        res = payload["run"]["results"][n - 1]
        takeaway = res["readings"]["expert"]["takeaway"]
        assert res["slide_intent"]["text"] == takeaway
        assert "What this slide demands of its reader" in page
        assert page.index("What this slide demands of its reader") < page.index("Takeaway, verbatim")  # summary above the cards
        # the block under the slide image: heading, then the green-dot label and the takeaway verbatim
        assert page.count("Intent of this slide") == 1
        assert ("Intent of this slide" + "Expert takeaway, verbatim" + takeaway) in page
        assert page.index("Intent of this slide") < page.index("What this slide demands of its reader")
        # ... and that text is on the page exactly once: the expert card does not repeat it
        assert page.count(takeaway) == 1
        assert page.count("Takeaway, verbatim") == 2 + page.count("Expert takeaway, verbatim") - 1  # novice and peer cards, plus the one block
        # the attribution line, the retired band and the declared intent are all gone from this page
        assert "Inferred from the expert reading" not in page and "derived from (verbatim)" not in page
        assert "What this slide is trying to establish" not in page
        assert declared not in page and "declared intent" not in page.lower()


def test_the_expert_card_points_to_the_intent_instead_of_repeating_it(tmp_path, payload):
    page = render(tmp_path, payload)["slide3"]
    expert = page[page.index("Works in LLM inference serving systems") :]
    assert "Its takeaway is the intent of this slide, shown under the slide image." in expert
    assert payload["run"]["results"][2]["readings"]["expert"]["takeaway"] not in expert
    assert "reference" in expert  # the rest of the card is unchanged


@pytest.mark.parametrize("takeaway", [
    "throughput rose 2.4x\u2014maybe more...",                     # ellipsis and dash, no final full stop, lower case
    "a" * 300 + " and then some, with no full stop",               # long: wraps, never truncated
    'She said "yes" & left \u2026 <b>not markup</b>',              # quotes, ampersand, angle brackets
])
def test_the_takeaway_is_rendered_exactly_as_returned(tmp_path, payload, takeaway):
    p = copy.deepcopy(payload)
    r = p["run"]["results"][2]
    r["readings"]["expert"]["takeaway"] = takeaway
    r["slide_intent"]["text"] = r["metrics"]["intent"] = r["metrics"]["takeaways"]["expert"] = takeaway
    r["metrics"]["intent_alignment"]["expert"]["inputs"]["expert"] = r["metrics"]["intent_alignment"]["expert"]["inputs"]["intent"] = takeaway
    for who in ("novice", "peer"):
        r["metrics"]["intent_alignment"][who]["inputs"]["intent"] = takeaway
    page = render(tmp_path, p)["slide3"]
    assert page.count(takeaway) == 1
    assert takeaway + "." not in page and takeaway + "\u2026" not in page  # nothing appended
    assert (takeaway + "Inferred") not in page


def test_no_provenance_panel_shows_the_expert_takeaway_twice(tmp_path, payload):
    out = render(tmp_path, payload, "3")
    takeaway = payload["run"]["results"][2]["readings"]["expert"]["takeaway"]
    for d in out["slide3:drawers"]:
        assert d["body"].count(takeaway) <= 1, d["opener"]  # the panel keeps its other content, minus the duplicate
    tier = next(d["body"] for d in out["slide3:drawers"] if d["opener"] in ("Self-contained", "Background needed", "Expert-gated"))
    assert takeaway in tier and "the expert takeaway, the reference" in tier
    assert "Intended reading, inferred" not in tier


def older_format(payload):
    """A run saved by the previous version: alignment was measured against a rephrased sentence."""
    old = copy.deepcopy(payload)
    for r in old["run"]["results"]:
        rephrased = f"A rephrased sentence for slide {r['index']}."
        r["slide_intent"] = {"text": rephrased, "source": "model", "model": "claude-haiku-4-5", "reason": None, "attempts": 1,
                             "latency_s": 1.0, "cached": False,
                             "derived_from": {"takeaway": r["readings"]["expert"]["takeaway"], "inferred_claim": r["readings"]["expert"]["inferred_claim"]}}
        r["metrics"]["intent"] = rephrased
        for who in ("novice", "peer", "expert"):
            r["metrics"]["intent_alignment"][who]["inputs"]["intent"] = rephrased
    return old


def test_a_run_measured_against_a_rephrased_sentence_still_shows_the_string_that_was_measured(tmp_path, payload):
    """The page must never show one string while the metrics used another. Runs saved before the
    expert takeaway became the intent keep showing their rephrased sentence, with its attribution."""
    out = render(tmp_path, older_format(payload), "3")
    page = out["slide3"]
    assert out["errors"] == []
    assert "A rephrased sentence for slide 3." in page and page.count("Inferred from the expert reading") == 1
    assert "Expert takeaway, verbatim" not in page  # that label would claim the takeaway is the intent
    assert page.count("Takeaway, verbatim") == 3  # the expert card keeps its takeaway: it is not what was measured against
    assert any("Inferred from the expert reading" in d["body"] or "rephrased" in d["body"] for d in out["slide3:drawers"])


def test_the_declared_intent_is_still_stored_and_still_shown_on_the_overview(tmp_path, payload):
    out = render(tmp_path, payload)
    assert payload["run"]["meta"]["intent"] in out["overview"]


def test_the_expert_card_says_reference_instead_of_a_score(tmp_path, payload):
    page = render(tmp_path, payload)["slide3"]
    expert = page[page.index("Expert") :]
    assert "Alignment to the intended reading" in expert and "reference" in expert
    assert "The intended reading is derived from this expert interpretation, so it defines the baseline rather than scoring against it." in expert
    assert not re.search(r"reference.{0,40}1\.00|1\.00", expert)  # no 1.00 dressed up as a score


def test_recommendations_quote_evidence_and_are_grouped_by_audience(tmp_path, payload):
    page = render(tmp_path, payload)["slide3"]
    recs = payload["recs"][3]["recommendations"]
    for audience in ("novice", "peer"):
        for r in recs[audience]:
            assert r["bullet"] in page and r["evidence"] in page
    assert page.count("What to change") == 2  # novice and peer, never the expert
    assert "Expert also flagged" not in page or recs["expert_flagged"]  # only when the expert flagged something
    assert "checked-in fixture" not in page


def test_what_the_expert_flagged_is_surfaced_above_the_summary(tmp_path, payload):
    p = copy.deepcopy(payload)
    expert = p["run"]["results"][2]["readings"]["expert"]
    p["recs"][3]["recommendations"]["expert_flagged"] = [
        {"note": "The expert could not resolve \u201cgamma\u201d.", "evidence": "gamma"},
        {"note": "The 2.4x has no stated baseline.", "evidence": expert["takeaway"][:20]},
    ]
    page = render(tmp_path, p)["slide3"]
    assert "Expert also flagged" in page and "The 2.4x has no stated baseline." in page
    assert page.index("Expert also flagged") < page.index("What this slide demands of its reader")


def test_every_number_opens_a_drawer_showing_real_text(tmp_path, payload):
    out = render(tmp_path, payload, "3")
    drawers = out["slide3:drawers"]
    assert len(drawers) >= 10
    r = payload["run"]["results"][2]
    for d in drawers:
        assert d["body"].strip(), d["opener"]
    by_opener = {d["opener"]: d["body"] for d in drawers}
    assert r["metrics"]["takeaways"]["novice"] in by_opener[f"{r['metrics']['intent_alignment']['novice']['value']:.2f}"]
    assert r["slide_intent"]["text"] in by_opener[f"{r['metrics']['intent_alignment']['peer']['value']:.2f}"]
    assert "definitional, not measured" in by_opener["reference"]
    tier = next(v for k, v in by_opener.items() if k in ("Self-contained", "Background needed", "Expert-gated"))
    assert "not met" in tier or "met" in tier
    assert r["metrics"]["takeaways"]["expert"] in tier  # the three takeaways drive it, and are shown


def test_a_saved_run_from_before_this_revision_opens_in_a_reduced_form(tmp_path, payload):
    legacy = copy.deepcopy(payload)
    legacy["run"]["meta"]["legacy"] = True
    out = render(tmp_path, legacy)
    o = out["overview"]
    assert out["errors"] == []
    assert "Saved in the earlier format" in o and "Hardest slides" not in o and "Narrative arc" not in o
    assert "Terms the novice could not resolve" in o  # what is still valid keeps showing
