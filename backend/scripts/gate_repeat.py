"""Run the live gate N times and report how stable it is.

One passing run is one sample from a non-deterministic system. This runs the real gate
(so the criteria live in exactly one place: tests/test_gate_live.py), each time with a
fresh cache, and shows per-run metrics, per-test pass rates, latency and output tokens.

    make gate-repeat          # 5 runs
    make gate-repeat N=10

Costs real API calls: 6 per run.
"""

from __future__ import annotations

import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPORT = BACKEND / ".cache" / "gate_last_run.json"


def run_once() -> tuple[dict[str, str], dict]:
    REPORT.unlink(missing_ok=True)  # never read a stale report
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "live", "-o", "addopts=", "-q", "--tb=no",
         "-rA", "-p", "no:cacheprovider"],
        cwd=BACKEND, capture_output=True, text=True,
    )
    outcomes = {
        m[2]: m[1]
        for line in proc.stdout.splitlines()
        if (m := re.match(r"(PASSED|FAILED|ERROR) tests/test_gate_live\.py::(\w+)", line))
    }
    if not REPORT.exists():
        sys.exit("gate could not run (no report written):\n" + "\n".join(proc.stdout.splitlines()[-25:]))
    return outcomes, json.loads(REPORT.read_text())


def row(report: dict) -> dict[str, float]:
    c, j = report["clear"]["scores"], report["jargon"]["scores"]
    lat = report["latency_s"]
    out_tokens = [u["output_tokens"] for u in report.get("usage", [])]
    conf = lambda slide, persona: report[slide]["responses"][persona]["confidence"]  # noqa: E731
    retries = sum(a - 1 for slide in report["attempts"].values() for a in slide.values())
    return {
        "clear_div": c["audience_divergence"]["value"],
        "jargon_div": j["audience_divergence"]["value"],
        "clear_blind": c["blind_spot_score"]["value"],
        "jargon_blind": j["blind_spot_score"]["value"],
        "jargon_novice_align": j["intent_alignment"]["novice"]["value"],
        "jargon_expert_align": j["intent_alignment"]["expert"]["value"],
        "clear_min_conf": min(conf("clear", p) for p in ("novice", "peer", "expert")),
        "jargon_conf_gap": conf("jargon", "expert") - conf("jargon", "novice"),
        "retries": retries,
        "clear_slowest_s": max(lat["clear"].values()),
        "jargon_slowest_s": max(lat["jargon"].values()),
        "max_output_tokens": max(out_tokens) if out_tokens else float("nan"),
        "median_output_tokens": st.median(out_tokens) if out_tokens else float("nan"),
    }


def main(n: int) -> None:
    rows, tally, model = [], {}, None
    for i in range(1, n + 1):
        outcomes, report = run_once()
        model = f'{report["model"]} (effort={report["effort"]})'
        rows.append(row(report))
        for name, outcome in outcomes.items():
            tally.setdefault(name, []).append(outcome == "PASSED")
        failed = [k for k, v in outcomes.items() if v != "PASSED"]
        print(f"run {i}/{n}: {'all pass' if not failed else 'FAILED ' + ', '.join(failed)}", flush=True)

    print(f"\nmodel: {model}\n")
    print(f"{'metric':22s} {'min':>8s} {'median':>8s} {'max':>8s}")
    for key in rows[0]:
        vals = [r[key] for r in rows]
        print(f"{key:22s} {min(vals):8.3f} {st.median(vals):8.3f} {max(vals):8.3f}")
    print("\npass rate per gate test:")
    for name, results in tally.items():
        print(f"  {sum(results)}/{len(results)}  {name}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
