"""Build the bundled sample run: a made-up seven-slide talk read by the REAL audiences.

The deck is invented for demonstration (it is not anyone's research); the readings are genuine
model output. The result is written to backend/fixtures/runs/ and marked `sample`, which the UI
shows as "sample data" and the server never lets anyone re-run or modify. It exists so the demo
works with no API key and no network: open it from the history list.

    .venv/bin/python backend/scripts/make_sample_run.py        # needs ANTHROPIC_API_KEY; ~22 model calls
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

from sightline import ingest  # noqa: E402
from sightline.audiences import FileCache  # noqa: E402
from sightline.divergence import default_embedder  # noqa: E402
from sightline.llm import AnthropicClient  # noqa: E402
from sightline.runner import TolerantEngine, run_deck  # noqa: E402
from sightline.store import BUNDLED_RUNS_DIR, RunStore, data_dir  # noqa: E402

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
]


def build_pdf(path: Path) -> None:
    doc = pymupdf.open()
    for title, bullets in SLIDES:
        page = doc.new_page(width=960, height=540)
        page.draw_rect(pymupdf.Rect(0, 0, 14, 540), color=None, fill=(0.16, 0.47, 0.84))
        page.insert_textbox(pymupdf.Rect(60, 40, 900, 150), title, fontsize=34, fontname="helv")
        y = 170
        for b in bullets:
            page.draw_circle(pymupdf.Point(70, y + 14), 4, color=None, fill=(0.16, 0.47, 0.84))  # a shape, so no glyph is lost
            page.insert_textbox(pymupdf.Rect(90, y, 900, y + 60), b, fontsize=22, fontname="helv")
            y += 70
    doc.save(path)


async def main() -> None:
    client = AnthropicClient()
    store = RunStore(BUNDLED_RUNS_DIR)
    if (BUNDLED_RUNS_DIR / RUN_ID).exists():
        shutil.rmtree(BUNDLED_RUNS_DIR / RUN_ID)
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "sample-llm-serving.pdf"
        build_pdf(pdf)
        slides = ingest.parse(pdf)
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
    await run_deck(store, RUN_ID, engine, default_embedder())
    final = store.load_meta(RUN_ID)
    print("status:", final["status"], "| model:", final["model"], "| error:", final["error"])
    print("results:", len(store.load_results(RUN_ID)), "of", meta["slide_count"])


if __name__ == "__main__":
    asyncio.run(main())
