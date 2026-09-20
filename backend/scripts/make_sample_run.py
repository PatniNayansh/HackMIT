"""Build the bundled sample run: a made-up eight-slide talk read by the REAL audiences.

The deck is invented for demonstration (it is not anyone's research); the readings are genuine
model output. The result is written to backend/fixtures/runs/ and marked `sample`, which the UI
shows as "sample data" and the server never lets anyone re-run or modify. It exists so the demo
works with no API key and no network: open it from the history list.

    .venv/bin/python backend/scripts/make_sample_run.py        # needs OPENAI_API_KEY; ~50 model calls (three personas, one structuring call and one figure description where needed, per slide, plus recommendations)
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import pymupdf

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from profe import ingest  # noqa: E402
from profe.audiences import FileCache  # noqa: E402
from profe.compare import FileStructureCache  # noqa: E402
from profe.diagnose import recommend  # noqa: E402
from profe.divergence import default_embedder  # noqa: E402
from profe.llm import OpenAIClient  # noqa: E402
from profe.runner import TolerantEngine, run_deck  # noqa: E402
from profe.store import BUNDLED_RUNS_DIR, RunStore, data_dir  # noqa: E402

RUN_ID = "sample-llm-serving"
TITLE = "Sample: serving LLMs faster"
INTENT = (
    "Our serving system delivers 2.4x higher goodput at p99 latency by pairing "
    "PagedAttention-style KV-cache management with speculative decoding."
)
DOMAIN, ADJACENT = "LLM inference serving systems", "distributed systems and databases"

SLIDES: list[tuple[str, list[str]]] = [
    ("Serving LLMs faster: 2.4x goodput at p99", ["Alex Rivera", "Systems Group"]),
    ("Why serving is hard", [
        "Every request grows a KV-cache as it generates tokens",
        "Fragmented memory wastes GPU capacity",
        "Tail latency (p99) is what users feel",
    ]),
    ("PagedAttention + speculative decoding: 2.4x goodput at p99", [
        "KV-cache fragmentation: <4% waste via block tables",
        "Draft model acceptance alpha = 0.78 at gamma = 5",
        "Continuous batching; chunked prefill off to hold the TTFT SLO",
    ]),
    ("Block tables", [
        "KV-cache split into fixed-size blocks, mapped per sequence",
        "Copy-on-write shares blocks across parallel samples",
        "Prefix caching hit rate: 61% on the chat workload",
    ]),
    ("Speculative decoding", [
        "A small draft model proposes gamma tokens",
        "The target model verifies them in one forward pass",
        "Rejection sampling keeps the output distribution exact",
    ]),
    ("Result: 2.4x more requests per second", [
        "Same hardware: 8 GPUs, same latency target",
        "Median response time unchanged",
        "Measured on a real chat workload",
    ]),
    ("What to remember", [
        "Serve more users on the same GPUs",
        "No change to what the model says",
        "Open-sourced this quarter",
    ]),
    # A slide whose substance is a chart, drawn as vector graphics. Almost no text: what the readers
    # get from it depends on the figure description (and the image), which is exactly what it tests.
    ("Latency versus load", []),
]


def draw_chart(page: pymupdf.Page) -> None:
    ink, blue, orange = (0.1, 0.1, 0.1), (0.16, 0.47, 0.84), (0.85, 0.33, 0.12)
    page.draw_line((120, 470), (860, 470), color=ink, width=1.5)  # x axis
    page.draw_line((120, 470), (120, 150), color=ink, width=1.5)  # y axis
    for i in range(1, 8):
        page.draw_line((120 + i * 92, 466), (120 + i * 92, 474), color=ink)  # x ticks
    for i in range(1, 5):
        page.draw_line((116, 470 - i * 70), (124, 470 - i * 70), color=ink)  # y ticks
    page.draw_polyline([(120, 440), (350, 400), (560, 320), (700, 200), (780, 150)], color=orange, width=3)
    page.draw_polyline([(120, 445), (350, 425), (560, 395), (700, 350), (860, 300)], color=blue, width=3)
    page.draw_line((120, 250), (860, 250), color=ink, width=1, dashes="[4 4] 0")  # a horizontal reference line
    page.insert_text((380, 508), "Requests per second", fontsize=16, fontname="helv")
    page.insert_text((60, 320), "p99 latency (ms)", fontsize=16, fontname="helv", rotate=90)
    page.insert_text((790, 165), "baseline", fontsize=14, fontname="helv", color=orange)
    page.insert_text((770, 316), "ours", fontsize=14, fontname="helv", color=blue)
    page.insert_text((870, 254), "500 ms", fontsize=13, fontname="helv")


def build_pdf(path: Path) -> None:
    doc = pymupdf.open()
    for title, bullets in SLIDES:
        page = doc.new_page(width=960, height=540)
        page.draw_rect(pymupdf.Rect(0, 0, 14, 540), color=None, fill=(0.16, 0.47, 0.84))
        page.insert_textbox(pymupdf.Rect(60, 40, 900, 150), title, fontsize=34, fontname="helv")
        if not bullets:
            draw_chart(page)
        y = 170
        for b in bullets:
            page.draw_circle(pymupdf.Point(70, y + 14), 4, color=None, fill=(0.16, 0.47, 0.84))  # a shape, so no glyph is lost
            page.insert_textbox(pymupdf.Rect(90, y, 900, y + 60), b, fontsize=22, fontname="helv")
            y += 70
    doc.save(path)


async def main() -> None:
    client = OpenAIClient()
    store = RunStore(BUNDLED_RUNS_DIR)
    if (BUNDLED_RUNS_DIR / RUN_ID).exists():
        shutil.rmtree(BUNDLED_RUNS_DIR / RUN_ID)
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "sample-llm-serving.pdf"
        build_pdf(pdf)
        slides = ingest.parse(pdf)
    print("slides carrying a figure:", [s.index for s in slides if s.carries_figure])
    # Figure descriptions are made once, here (the helper model, with vision), exactly as the upload path does.
    slides = await ingest.describe_figures(OpenAIClient(role="helper"), slides, ingest.FileFigureCache(data_dir() / "cache" / "figures"))
    for s in slides:
        if s.image_content:
            print(f"figure description, slide {s.index}:", s.image_content["text"])
    inferred = await ingest.infer_profile(client, slides)
    print("model suggested:", inferred)
    meta = store.create_draft(
        title=TITLE, source_filename="sample-llm-serving.pdf", slides=slides,
        inferred=inferred, inference_error=None, run_id=RUN_ID,
    )
    store.update_meta(
        RUN_ID, intent=INTENT, sample=True,
        profile={"domain": DOMAIN, "adjacent_field": ADJACENT, "confirmed": True,
                 "edited": (inferred.domain, inferred.adjacent_field) != (DOMAIN, ADJACENT)},
    )
    engine = TolerantEngine(client, FileCache(data_dir() / "cache" / "audiences"))
    await run_deck(
        store, RUN_ID, engine, default_embedder(), comparator="fieldwise",
        structuring_client=OpenAIClient(role="structuring"), structure_cache=FileStructureCache(data_dir() / "cache" / "structured"),
    )
    final = store.load_meta(RUN_ID)
    print("status:", final["status"], "| model:", final["model"], "| error:", final["error"])
    results = store.load_results(RUN_ID)
    print("results:", len(results), "of", meta["slide_count"])

    # Recommendations are normally made the first time a slide is opened. The bundled sample is
    # read-only in the app, so they are made here, once, so the demo shows them with no key.
    for r in results:
        if r["metrics"]:
            store.save_recs(RUN_ID, r["index"], await recommend(r, r["slide_intent"], client=client))
    p = [r["timing"]["personas_s"] for r in results]
    st = [r["timing"].get("structuring_s", 0) for r in results]
    m = store.load_meta(RUN_ID)
    from datetime import datetime

    wall = (datetime.fromisoformat(m["finished_at"]) - datetime.fromisoformat(m["started_at"])).total_seconds()
    print(f"whole run {wall:.0f}s for {len(results)} slides = {wall / len(results):.1f}s per slide wall-clock (structuring overlaps the next slide's persona calls)")
    print(f"per slide: personas mean {sum(p) / len(p):.1f}s max {max(p):.1f}s | structuring mean {sum(st) / len(st):.1f}s max {max(st):.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
