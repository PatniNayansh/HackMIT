from __future__ import annotations

import io

import pymupdf
import pytest
from PIL import Image

from profe import ingest
from profe.audiences import DeckProfile
from profe.ingest import IngestError, UnsupportedFormat, infer_profile, parse

from conftest import FakeLLM


def make_pdf(path, pages: list[list[tuple[str, float, float]]]) -> None:
    """pages -> [(text, fontsize, y)] per page. Left-aligned at x=60 on a 960x540 page."""
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page(width=960, height=540)
        for text, size, y in items:
            page.insert_text((60, y), text, fontsize=size)
    doc.save(path)
    doc.close()


def test_pdf_becomes_one_record_per_page_with_text_and_png(tmp_path):
    pdf = tmp_path / "deck.pdf"
    make_pdf(
        pdf,
        [
            [("Q3 sales grew 20%", 40, 90), ("Revenue was $6.0M", 18, 200), ("Renewals drove it", 18, 240)],
            [("Next quarter", 40, 90), ("Keep the plan", 18, 200)],
        ],
    )
    slides = parse(pdf)

    assert [s.index for s in slides] == [1, 2]  # 1-based
    assert slides[0].text.splitlines()[0] == "[title] Q3 sales grew 20%"
    assert "[body] Revenue was $6.0M" in slides[0].text
    img = Image.open(io.BytesIO(slides[0].image_png))
    assert img.format == "PNG" and img.width == ingest.RENDER_WIDTH_PX


def test_uniform_type_is_not_guessed_to_be_a_title(tmp_path):
    pdf = tmp_path / "flat.pdf"
    make_pdf(pdf, [[("first line", 18, 100), ("second line", 18, 140)]])
    text = parse(pdf)[0].text
    assert "[title]" not in text and text.count("[body]") == 2


def test_large_text_low_on_the_page_is_not_a_title(tmp_path):
    pdf = tmp_path / "footer.pdf"
    make_pdf(pdf, [[("small", 14, 100), ("small too", 14, 140), ("BIG FOOTER", 40, 480)]])
    assert "[title]" not in parse(pdf)[0].text


def test_image_only_page_has_empty_text_but_still_a_png(tmp_path):
    pdf = tmp_path / "blank.pdf"
    make_pdf(pdf, [[]])
    (slide,) = parse(pdf)
    assert slide.text == "" and slide.image_png.startswith(b"\x89PNG")


def test_slide_converts_to_the_audience_input_without_intent(tmp_path):
    pdf = tmp_path / "d.pdf"
    make_pdf(pdf, [[("hello", 30, 100)]])
    (slide,) = parse(pdf)
    si = slide.to_input()
    assert (si.index, si.text, si.image_png) == (1, slide.text, slide.image_png)


def test_unsupported_extension_is_a_clear_error(tmp_path):
    with pytest.raises(UnsupportedFormat, match=r"\.pptx.*supported: \.pdf"):
        parse(tmp_path / "deck.pptx")


def test_new_formats_plug_in_without_touching_callers(tmp_path):
    @ingest.register(".fake")
    def parse_fake(path):
        return [ingest.Slide(1, "[title] hi", b"png")]

    try:
        assert parse(tmp_path / "x.FAKE")[0].text == "[title] hi"
    finally:
        del ingest._PARSERS[".fake"]


def test_garbage_bytes_named_pdf_is_an_ingest_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(IngestError):
        parse(bad)


def test_page_limit_is_enforced(tmp_path, monkeypatch):
    pdf = tmp_path / "long.pdf"
    make_pdf(pdf, [[("x", 20, 100)]] * 3)
    monkeypatch.setattr(ingest, "MAX_SLIDES", 2)
    with pytest.raises(IngestError, match="limit is 2"):
        parse(pdf)


# ---------------------------------------------------------------- subfield inference


async def test_infer_profile_returns_a_deckprofile_and_sends_only_slide_text():
    llm = FakeLLM(lambda p, n, u: None)
    seen = {}

    async def complete_json(*, system, user_text, image_png, schema, max_tokens=4096):
        seen.update(user=user_text, image=image_png, schema=schema)
        return {"domain": " LLM inference serving systems ", "adjacent_field": "distributed systems"}

    llm.complete_json = complete_json  # type: ignore[method-assign]
    slides = [ingest.Slide(1, "[title] PagedAttention", b"png"), ingest.Slide(2, "[body] goodput", b"png")]

    profile = await infer_profile(llm, slides)

    assert profile == DeckProfile("LLM inference serving systems", "distributed systems")
    assert "PagedAttention" in seen["user"] and "goodput" in seen["user"]
    assert seen["image"] is None  # text was available, so no image is sent
    assert seen["schema"] is ingest.PROFILE_SCHEMA


async def test_infer_profile_falls_back_to_the_first_slide_image_when_there_is_no_text():
    got = {}

    class LLM:
        model = "m"

        async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
            got["image"] = image_png
            return {"domain": "d", "adjacent_field": "a"}

    await infer_profile(LLM(), [ingest.Slide(1, "", b"first"), ingest.Slide(2, "", b"second")])
    assert got["image"] == b"first"


async def test_infer_profile_rejects_an_empty_field():
    class LLM:
        model = "m"

        async def complete_json(self, **_):
            return {"domain": "  ", "adjacent_field": "a"}

    with pytest.raises(ingest.LLMError):
        await infer_profile(LLM(), [ingest.Slide(1, "text", b"png")])


def test_profile_prompt_samples_a_long_deck_evenly_and_truncates():
    slides = [ingest.Slide(i, f"[body] slide-{i} " + "x" * 2000, b"") for i in range(1, 41)]
    prompt = ingest._profile_prompt(slides)
    assert prompt.count("Slide ") <= ingest._PROFILE_MAX_SLIDES
    assert "slide-1 " in prompt and "x" * 601 not in prompt
