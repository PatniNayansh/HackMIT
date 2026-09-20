"""Re-derive the bundled sample's field-wise metrics under the current comparator, keeping its readings.

The three readings per slide are genuine model output and are kept exactly as stored. What is redone is
the one Sonnet structuring call per slide (fields, the expert claim's propositions, and each reader's
coverage of them) and the recommendations built from it. Use this instead of make_sample_run.py when only
the comparison changed: it makes ~16 calls instead of ~50 and leaves the readings, figures and slides alone.

    .venv/bin/python backend/scripts/restructure_sample.py     # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from sightline.compare import build_fieldwise_metrics, structure_slide  # noqa: E402
from sightline.diagnose import recommend  # noqa: E402
from sightline.divergence import default_embedder  # noqa: E402
from sightline.llm import AnthropicClient  # noqa: E402
from sightline.store import BUNDLED_RUNS_DIR, RunStore  # noqa: E402

RUN_ID = "sample-llm-serving"


async def main() -> None:
    store = RunStore(BUNDLED_RUNS_DIR)
    client = AnthropicClient(model="claude-sonnet-5")
    recs_client = AnthropicClient()
    embedder = default_embedder()
    for r in store.load_results(RUN_ID):
        if not r["metrics"]:
            continue
        takeaways = {p: r["readings"][p]["takeaway"] for p in ("novice", "peer", "expert")}
        unresolved = {p: r["readings"][p]["unresolved_terms"] for p in ("novice", "peer", "expert")}
        structured = await structure_slide(client, takeaways, embedder=embedder)
        r["metrics"] = build_fieldwise_metrics(takeaways, structured, unresolved, (r.get("image_content") or {}).get("text"))
        r["scored_by"] = structured["meta"]["model"]
        r["timing"]["structuring_s"] = structured["meta"]["latency_s"]
        store.save_result(RUN_ID, r)
        store.save_recs(RUN_ID, r["index"], await recommend(r, r["slide_intent"], client=recs_client))
        c = r["metrics"]["comparisons"]
        print(r["index"], {a: (c[a]["claim"].get("outcome"), c[a]["claim"].get("coverage", {}).get("missed")) for a in ("novice", "peer")}, structured["meta"]["tripwire"])


if __name__ == "__main__":
    asyncio.run(main())
