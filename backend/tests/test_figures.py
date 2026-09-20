"""Ingest: deciding which slides carry a figure, describing it neutrally, and giving it to the personas."""

from __future__ import annotations

import io

import pymupdf
import pytest
from PIL import Image

from profe import ingest
from profe.ingest import FileFigureCache, Slide, describe_figure, describe_figures, figure_signals

from conftest import FakeLLM


def png(w=400, h=300, colour=(200, 60, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def make_pdf(path, pages) -> None:
    doc = pymupdf.open()
    for draw in pages:
        page = doc.new_page(width=960, height=540)
        draw(page)
    doc.save(path)


def text_only(page):
    page.insert_textbox(pymupdf.Rect(60, 40, 900, 150), "Supply and demand", fontsize=34)
    for i in range(4):
        page.insert_textbox(pymupdf.Rect(80, 170 + 60 * i, 900, 220 + 60 * i), "A full line of ordinary body text on this slide.", fontsize=20)


def big_image(page):
    page.insert_text((60, 60), "Results", fontsize=30)
    page.insert_image(pymupdf.Rect(80, 100, 880, 500), stream=png())


def chart_drawn_as_vectors(page):
    page.insert_text((60, 60), "Latency", fontsize=30)
    page.draw_line((100, 480), (860, 480))  # x axis
    page.draw_line((100, 480), (100, 120))  # y axis
    for i in range(1, 9):
        page.draw_line((100 + i * 90, 476), (100 + i * 90, 484))  # ticks
    page.draw_polyline([(100, 470), (400, 400), (860, 150)])  # a curve


def title_only(page):
    page.insert_textbox(pymupdf.Rect(60, 200, 900, 300), "Thank you", fontsize=40)


def small_image_no_text(page):
    page.insert_image(pymupdf.Rect(330, 220, 630, 320), stream=png(300, 100))  # ~6% of the page, and no text at all


def tiny_logo_with_text(page):
    text_only(page)
    page.insert_image(pymupdf.Rect(880, 490, 940, 530), stream=png(60, 40))  # a corner logo


# --------------------------------------------------------------------------------- the gate


@pytest.mark.parametrize("draw,carries", [
    (text_only, False), (title_only, False), (tiny_logo_with_text, False),
    (big_image, True), (chart_drawn_as_vectors, True),
])
def test_only_slides_with_non_text_content_are_flagged_for_description(tmp_path, draw, carries):
    make_pdf(tmp_path / "d.pdf", [draw])
    (slide,) = ingest.parse(tmp_path / "d.pdf")
    assert slide.carries_figure is carries, slide.figure_signals


def test_an_image_only_slide_is_flagged_even_when_the_image_is_below_the_area_threshold(tmp_path):
    make_pdf(tmp_path / "d.pdf", [small_image_no_text])
    (slide,) = ingest.parse(tmp_path / "d.pdf")
    sig = slide.figure_signals
    assert sig["text_chars"] < ingest.LOW_TEXT_CHARS and 0.02 <= sig["image_area_ratio"] < ingest.IMAGE_AREA_RATIO_MIN and slide.carries_figure


def test_the_gate_records_what_it_saw(tmp_path):
    make_pdf(tmp_path / "d.pdf", [text_only, big_image])
    a, b = ingest.parse(tmp_path / "d.pdf")
    assert set(a.figure_signals) == {"image_area_ratio", "drawing_paths", "text_chars", "carries_figure"}
    assert b.figure_signals["image_area_ratio"] > 0.2 and a.figure_signals["image_area_ratio"] == 0


# --------------------------------------------------------------------------- the description


class Describer:
    model = "fake-haiku"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls.append({"system": system, "user": user_text, "image": image_png})
        r = self.replies.pop(0) if self.replies else self.replies_default
        if isinstance(r, Exception):
            raise r
        return r

    replies_default = {"has_figure": True, "description": "Two lines cross on axes labelled Price and Quantity."}


def figure_slide(i=1, png_bytes=None):
    return Slide(i, "[title] x", png_bytes or png(), None, {"carries_figure": True})


async def test_text_only_slides_are_never_sent_anywhere():
    d = Describer()
    text = Slide(1, "[title] words", png(), None, {"carries_figure": False})
    out = await describe_figures(d, [text, Slide(2, "no signals at all", png())])
    assert d.calls == [] and [s.image_content for s in out] == [None, None]  # a text-only deck costs nothing extra


async def test_a_figure_slide_is_described_once_by_a_vision_call_on_its_own_image():
    d = Describer()
    s = figure_slide()
    (out,) = await describe_figures(d, [s])
    (call,) = d.calls
    assert call["image"] == s.image_png
    assert out.image_content["text"] == "Two lines cross on axes labelled Price and Quantity."
    assert out.image_content["source"] == "vision" and out.image_content["machine_generated"] is True and out.image_content["model"] == "fake-haiku"


async def test_the_prompt_is_descriptive_never_interpretive():
    d = Describer()
    await describe_figures(d, [figure_slide()])
    p = d.calls[0]["system"]
    for rule in ("describer, not an interpreter", "marks, labels, axes, values and spatial relationships", "name a principle",
                 "what the figure means", "expand or explain an acronym", "copy it as printed", "P*, Q*",
                 "what would happen", "describe only what IS drawn", "use the words show, showing"):
        assert rule in p, rule
    assert "showing market equilibrium and consumer surplus" in p  # the counter-example is in the prompt as the BAD case
    assert p.index("Good:") < p.index("Bad:")


@pytest.mark.parametrize("interpretive", [
    "A supply and demand diagram showing market equilibrium and consumer surplus.",
    "The chart demonstrates that latency rises with load.",
    "A curve that illustrates growth, therefore output rises.",
    "A dashed line meets the orange curve at a higher load than where it would meet the blue one.",
    "The blue curve is better than the orange one.",
])
async def test_a_description_that_interprets_is_retried_then_not_used(interpretive):
    bad = {"has_figure": True, "description": interpretive}
    d = Describer(bad, bad, bad)
    rec = await describe_figure(d, figure_slide())
    assert len(d.calls) == 3 and "interpreted the figure" in d.calls[1]["user"]
    assert rec["text"] is None and rec["source"] == "failed" and "interpreted the figure" in rec["error"]


async def test_a_retry_that_describes_plainly_is_accepted():
    d = Describer({"has_figure": True, "description": "A supply and demand diagram showing equilibrium."},
                  {"has_figure": True, "description": "Two lines on axes labelled Price and Quantity, crossing at a marked point."})
    rec = await describe_figure(d, figure_slide())
    assert rec["text"].startswith("Two lines") and rec["attempts"] == 2


async def test_a_third_attempt_rescues_a_description_the_first_two_got_wrong():
    bad = {"has_figure": True, "description": "A line showing steady growth."}
    d = Describer(bad, bad, {"has_figure": True, "description": "One orange line rises from left to right on axes labelled Hours and Volume."})
    assert (await describe_figure(d, figure_slide()))["attempts"] == 3


async def test_when_the_model_finds_no_figure_there_is_no_description():
    (out,) = await describe_figures(Describer({"has_figure": False, "description": None}), [figure_slide()])
    assert out.image_content is None


async def test_an_api_error_does_not_fail_the_deck_the_personas_still_have_the_image():
    rec = await describe_figure(Describer(ConnectionError("down"), ConnectionError("down"), ConnectionError("down")), figure_slide())
    assert rec["text"] is None and rec["source"] == "failed" and "ConnectionError" in rec["error"]


async def test_with_no_client_nothing_is_described():
    (out,) = await describe_figures(None, [figure_slide()])
    assert out.image_content is None


async def test_a_deck_is_described_once_the_cache_answers_the_second_time(tmp_path):
    cache = FileFigureCache(tmp_path)
    s = figure_slide()
    d1 = Describer()
    (a,) = await describe_figures(d1, [s], cache)
    d2 = Describer()
    (b,) = await describe_figures(d2, [s], cache)
    assert len(d1.calls) == 1 and d2.calls == [] and a.image_content == b.image_content


async def test_a_failed_description_is_not_cached_so_a_later_upload_can_succeed(tmp_path):
    cache = FileFigureCache(tmp_path)
    bad = {"has_figure": True, "description": "It shows the trend."}
    await describe_figures(Describer(bad, bad, bad), [figure_slide()], cache)
    d = Describer()
    (out,) = await describe_figures(d, [figure_slide()], cache)
    assert len(d.calls) == 1 and out.image_content["text"]


async def test_slides_are_described_in_parallel_and_returned_in_order():
    slides = [figure_slide(1, png(300, 200, (1, 2, 3))), Slide(2, "text", png(), None, {"carries_figure": False}), figure_slide(3, png(300, 200, (9, 8, 7)))]
    out = await describe_figures(Describer(), slides)
    assert [s.index for s in out] == [1, 2, 3] and [s.image_content is not None for s in out] == [True, False, True]


# ------------------------------------------------------------------- what the personas are given


def test_the_persona_input_carries_the_description_under_a_figure_marker():
    s = Slide(1, "[title] Latency\n[body] p99", png(), {"text": "A line rises from left to right.", "source": "vision"})
    assert s.to_input().text == "[title] Latency\n[body] p99\n\nFIGURE: [description of the image, not printed words] A line rises from left to right."
    assert Slide(1, "[title] Latency", png()).to_input().text == "[title] Latency"  # no figure, no marker
    assert Slide(1, "", png(), {"text": "A drawing."}).to_input().text.startswith("FIGURE:")  # an image-only slide
    assert Slide(1, "[title] x", png(), {"text": None, "source": "failed"}).to_input().text == "[title] x"  # a failed description is not shown


def test_a_figure_changes_what_the_personas_are_shown_so_it_cannot_be_served_from_a_stale_cache():
    from profe.audiences import slide_hash

    a, b = Slide(1, "[title] x", png()), Slide(1, "[title] x", png(), {"text": "A drawing."})
    prof = ingest.DeckProfile("d", "a")
    assert slide_hash(a.to_input(), prof) != slide_hash(b.to_input(), prof)


def test_the_description_is_stored_with_the_slide_and_restored(tmp_path):
    from profe.store import RunStore

    store = RunStore(tmp_path)
    s = Slide(1, "[title] x", png(), {"text": "Two lines.", "model": "fake-haiku", "source": "vision", "machine_generated": True}, {"carries_figure": True, "drawing_paths": 12})
    meta = store.create_draft(title="t", source_filename="t.pdf", slides=[s, Slide(2, "y", png())], inferred=None, inference_error=None)
    back = store.load_slides(meta["run_id"])
    assert back[0].image_content == s.image_content and back[0].carries_figure and back[1].image_content is None


def test_the_image_is_withheld_from_the_personas_only_when_asked_and_only_on_described_slides(monkeypatch):
    described = Slide(1, "[title] x", b"IMG", {"text": "One line rises."})
    plain = Slide(2, "[title] y", b"IMG2")
    monkeypatch.delenv("PROFE_FIGURE_INPUT", raising=False)
    assert described.to_input().image_png == b"IMG" and plain.to_input().image_png == b"IMG2"  # default: as it has always been
    monkeypatch.setenv("PROFE_FIGURE_INPUT", "description_only")
    assert described.to_input().image_png is None and "FIGURE:" in described.to_input().text
    assert plain.to_input().image_png == b"IMG2"  # no description, so nothing replaces the image
    monkeypatch.setenv("PROFE_FIGURE_INPUT", "nonsense")
    assert described.to_input().image_png == b"IMG"
