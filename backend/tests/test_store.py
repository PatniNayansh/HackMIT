from __future__ import annotations

import json

import pytest

from sightline import store as store_mod
from sightline.audiences import DeckProfile
from sightline.ingest import Slide
from sightline.store import ReadOnlyRun, RunNotFound, RunStore

from builders import slide_result


def slides(n=3):
    return [Slide(i, f"[title] slide {i}", b"\x89PNG-" + str(i).encode()) for i in range(1, n + 1)]


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "history")


def draft(store, **kw):
    return store.create_draft(
        title="My talk", source_filename="talk.pdf", slides=slides(),
        inferred=DeckProfile("LLM serving", "distributed systems"), inference_error=None, **kw,
    )


def test_a_draft_keeps_the_slides_exactly_as_extracted(store):
    meta = draft(store)
    run_id = meta["run_id"]

    assert meta["status"] == "draft" and meta["slide_count"] == 3
    assert meta["profile"] is None  # nothing is confirmed until the presenter confirms it
    assert meta["profile_inferred"] == {"domain": "LLM serving", "adjacent_field": "distributed systems"}
    loaded = store.load_slides(run_id)
    assert [(s.index, s.text, s.image_png) for s in loaded] == [(s.index, s.text, s.image_png) for s in slides()]
    assert store.image_path(run_id, 2).read_bytes() == b"\x89PNG-2"


def test_drafts_are_not_history(store):
    draft(store)
    assert store.list_runs() == []
    assert len(store.list_runs(include_drafts=True)) == 1


def test_results_round_trip_and_come_back_in_slide_order(store):
    run_id = draft(store)["run_id"]
    for i in (3, 1, 2):
        store.save_result(run_id, slide_result(i))
    assert [r["index"] for r in store.load_results(run_id)] == [1, 2, 3]
    assert store.load_results(run_id)[0] == slide_result(1)  # nothing lost through JSON


def test_a_half_written_result_reads_as_pending_not_a_crash(store):
    run_id = draft(store)["run_id"]
    store.save_result(run_id, slide_result(1))
    (store.root / run_id / "results" / "002.json").write_text('{"index": 2, "rea', encoding="utf-8")
    assert [r["index"] for r in store.load_results(run_id)] == [1]


def test_update_meta_rejects_unknown_fields(store):
    run_id = draft(store)["run_id"]
    assert store.update_meta(run_id, status="running")["status"] == "running"
    with pytest.raises(KeyError):
        store.update_meta(run_id, bogus=1)


def test_history_lists_newest_first_with_what_the_list_view_needs(store):
    a = draft(store)["run_id"]
    b = draft(store)["run_id"]
    store.update_meta(a, status="complete", created_at="2026-01-01T00:00:00+00:00", model="m1")
    store.update_meta(b, status="complete", created_at="2026-02-01T00:00:00+00:00", model="m2")
    rows = store.list_runs()
    assert [r["run_id"] for r in rows] == [b, a]
    assert {"title", "created_at", "slide_count", "model"} <= set(rows[0])


def test_a_corrupt_run_does_not_take_the_list_down(store):
    good = draft(store)["run_id"]
    store.update_meta(good, status="complete")
    bad = store.root / "20260101-000000-bad"
    bad.mkdir()
    (bad / "run.json").write_text("{not json", encoding="utf-8")
    assert [r["run_id"] for r in store.list_runs()] == [good]


@pytest.mark.parametrize("run_id", ["../etc", "a/b", "..", "", ".hidden", "x" * 200])
def test_run_ids_cannot_escape_the_history_directory(store, run_id):
    with pytest.raises(RunNotFound):
        store.load_meta(run_id)


def test_unknown_run_is_not_found(store):
    with pytest.raises(RunNotFound):
        store.load_meta("20260101-000000-nope")


def test_bundled_runs_are_listed_and_readable_but_never_writable(tmp_path):
    bundled = tmp_path / "bundled"
    (bundled / "sample-1").mkdir(parents=True)
    (bundled / "sample-1" / "run.json").write_text(
        json.dumps({"run_id": "sample-1", "title": "Sample", "status": "complete", "sample": True,
                    "created_at": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    s = RunStore(tmp_path / "history", bundled=[bundled])
    assert [r["run_id"] for r in s.list_runs()] == ["sample-1"]
    assert s.load_meta("sample-1")["sample"] is True
    with pytest.raises(ReadOnlyRun):
        s.update_meta("sample-1", status="failed")
    with pytest.raises(ReadOnlyRun):
        s.save_result("sample-1", slide_result(1))


def test_data_dir_can_be_redirected(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGHTLINE_DATA_DIR", str(tmp_path / "elsewhere"))
    assert store_mod.data_dir() == tmp_path / "elsewhere"
