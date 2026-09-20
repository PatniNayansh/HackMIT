"""Review a real PDF with the real audiences and save it like any other run.

    python backend/scripts/make_local_run.py <pdf> --run-id <id> --title "..."

Writes to data/history/, the same place the web app writes uploads, so the run shows up in
Saved runs and replays offline afterwards. Deliberately NOT backend/fixtures/runs/: that
directory is committed, and a deck someone else wrote is not ours to redistribute.

Needs OPENAI_API_KEY. Budget roughly six model calls a slide (three personas, one structuring
call, one figure description where a slide carries a figure, and one for recommendations).
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from profe import ingest  # noqa: E402
from profe.audiences import FileCache  # noqa: E402
from profe.compare import FileStructureCache  # noqa: E402
from profe.diagnose import recommend  # noqa: E402
from profe.divergence import default_embedder  # noqa: E402
from profe.llm import OpenAIClient, load_env  # noqa: E402
from profe.runner import TolerantEngine, run_deck  # noqa: E402
from profe.store import RunStore, data_dir  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--domain", help="the deck's subfield; inferred by the model when omitted")
    ap.add_argument("--adjacent", help="the field the peer comes from; inferred when omitted")
    args = ap.parse_args()

    load_env()
    client = OpenAIClient()
    store = RunStore(data_dir() / "history")
    out = data_dir() / "history" / args.run_id
    if out.exists():
        shutil.rmtree(out)

    slides = ingest.parse(args.pdf)
    print(f"{len(slides)} slides; carrying a figure: {[s.index for s in slides if s.carries_figure]}")
    slides = await ingest.describe_figures(
        OpenAIClient(role="helper"), slides, ingest.FileFigureCache(data_dir() / "cache" / "figures")
    )
    inferred = await ingest.infer_profile(client, slides)
    print("model suggested:", inferred)

    domain = args.domain or inferred.domain
    adjacent = args.adjacent or inferred.adjacent_field
    meta = store.create_draft(
        title=args.title, source_filename=args.pdf.name, slides=slides,
        inferred=inferred, inference_error=None, run_id=args.run_id,
    )
    store.update_meta(
        args.run_id,
        profile={"domain": domain, "adjacent_field": adjacent, "confirmed": True,
                 "edited": (inferred.domain, inferred.adjacent_field) != (domain, adjacent)},
    )

    engine = TolerantEngine(client, FileCache(data_dir() / "cache" / "audiences"))
    await run_deck(
        store, args.run_id, engine, default_embedder(), comparator="fieldwise",
        structuring_client=OpenAIClient(role="structuring"),
        structure_cache=FileStructureCache(data_dir() / "cache" / "structured"),
    )
    final = store.load_meta(args.run_id)
    print("status:", final["status"], "| model:", final["model"], "| error:", final["error"])

    results = store.load_results(args.run_id)
    print(f"results: {len(results)} of {meta['slide_count']}")
    # Made up front rather than on first open, so the saved run replays with no key.
    for r in results:
        if r["metrics"]:
            store.save_recs(args.run_id, r["index"], await recommend(r, r["slide_intent"], client=client))
    print("recommendations saved for", sum(1 for r in results if r["metrics"]), "slides")


if __name__ == "__main__":
    asyncio.run(main())
