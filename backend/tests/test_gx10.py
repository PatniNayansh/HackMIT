"""The audio chunker, and the transport that carries chunks to the GX10.

Nothing here touches the real machine. The chunker runs the bundled ffmpeg on audio it
synthesizes, so it is real work on real files; the transport's ssh and scp calls are captured
so the exact command lines can be asserted -- which is the point, because every bug found in
this module so far was in a command line rather than in the logic around it.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from profe import gx10
from profe.audio import AudioError, Chunk, ffmpeg_exe, manifest, split, write_manifest


# ------------------------------------------------------------------------------ chunking


@pytest.fixture(scope="module")
def tone(tmp_path_factory) -> Path:
    """Five minutes of sine wave: long enough to cut into three chunks."""
    d = tmp_path_factory.mktemp("audio")
    src = d / "lecture.mp3"
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=300", str(src)],
        check=True, capture_output=True,
    )
    return src


def test_a_long_recording_is_cut_into_two_minute_pieces(tone, tmp_path):
    chunks = split(tone, tmp_path / "chunks")
    assert [c.name for c in chunks] == ["chunk_00", "chunk_01", "chunk_02"]
    assert [c.start_s for c in chunks] == [0.0, 120.0, 240.0]
    assert all(c.path.is_file() and c.path.stat().st_size > 0 for c in chunks)
    # `-c copy` keeps the source format rather than re-encoding
    assert all(c.path.suffix == tone.suffix for c in chunks)


def test_splitting_twice_does_not_leave_the_first_attempts_chunks_behind(tone, tmp_path):
    out = tmp_path / "chunks"
    split(tone, out)
    stale = out / f"chunk_09{tone.suffix}"
    stale.write_bytes(b"left over from a longer file")
    again = split(tone, out)
    assert not stale.exists()
    assert len(list(out.glob(f"chunk_*{tone.suffix}"))) == len(again)


def test_a_file_that_is_not_audio_is_refused_by_name_before_ffmpeg_sees_it(tmp_path):
    deck = tmp_path / "deck.pdf"
    deck.write_bytes(b"%PDF-1.4")
    with pytest.raises(AudioError, match="unsupported audio type"):
        split(deck, tmp_path / "chunks")


def test_a_missing_file_says_so_rather_than_producing_zero_chunks(tmp_path):
    with pytest.raises(AudioError, match="no audio file"):
        split(tmp_path / "gone.mp3", tmp_path / "chunks")


def test_the_manifest_describes_the_job_completely_enough_to_need_no_arguments(tone, tmp_path):
    chunks = split(tone, tmp_path / "chunks")
    m = manifest(chunks, source_name=tone.name)
    assert m["chunk_seconds"] == 120 and m["source_name"] == "lecture.mp3"
    assert [c["name"] for c in m["chunks"]] == ["chunk_00", "chunk_01", "chunk_02"]
    path = write_manifest(tmp_path / "chunks", m)
    assert json.loads(path.read_text(encoding="utf-8")) == m


# ----------------------------------------------------------------------------- transport


@pytest.fixture
def calls(monkeypatch):
    """Capture every ssh/scp invocation instead of running it."""
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        # launch() verifies liveness with a separate call and raises if nothing is running, so
        # a fake that always answers "" would make every launch test fail for the right reason
        # at the wrong moment. Answer pgrep as a live job would.
        remote = argv[-1] if argv else ""
        out = "4242 python run_lecture_audio.py --job-dir /home/asus/profe-jobs/job-1" if "pgrep" in remote else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(gx10.subprocess, "run", fake_run)
    monkeypatch.setattr(gx10, "_tool", lambda name: name)
    monkeypatch.setenv("GX10_HOST", "10.0.0.9")
    monkeypatch.setenv("GX10_SSH_USER", "asus")
    return seen


def last_remote_command(calls: list[list[str]]) -> str:
    return calls[-1][-1]


def test_every_remote_command_exports_the_path_a_login_shell_would_have(calls):
    """A non-interactive ssh shell does not source ~/.bashrc, so ~/.local/bin -- uv, uvx and the
    ffmpeg symlink -- is missing unless it is exported. Silent, and only visible as a command
    not found much later."""
    gx10.ssh("echo hello")
    assert last_remote_command(calls).startswith("export PATH=$HOME/.local/bin:$PATH;")


def test_the_launch_command_detaches_all_three_streams(calls):
    gx10.launch("job-1")
    launch_cmd = calls[0][-1]
    assert "nohup " in launch_cmd
    assert "> " in launch_cmd and "2>&1" in launch_cmd
    assert "< /dev/null" in launch_cmd  # without this the launching ssh itself can hang
    assert "disown" in launch_cmd


def test_the_launch_command_does_not_chain_cd_before_the_background_job(calls):
    """`cd X && nohup Y ... &` backgrounds the whole chain, so ssh returns before it has run and
    the closing connection takes the job with it. Measured on the real machine."""
    gx10.launch("job-1")
    launch_cmd = calls[0][-1]
    before_nohup = launch_cmd.split("nohup", 1)[0]
    assert "&&" not in before_nohup, launch_cmd


def test_the_launch_passes_path_through_to_the_background_process_too(calls):
    """The child inherits the environment as it was at launch; it does not re-source a shell."""
    gx10.launch("job-1")
    assert "nohup env PATH=$HOME/.local/bin:$PATH" in calls[0][-1]


def test_the_runner_is_told_it_is_not_the_server_so_the_tribe_guard_lets_it_run(calls):
    gx10.launch("job-1")
    assert "PROFE_PROCESS=precompute" in calls[0][-1]


def test_the_liveness_check_cannot_match_its_own_command_line(calls):
    """pgrep sees the shell running this very command, whose line contains the pattern. Without
    the bracket every job looks alive, including jobs that were never started."""
    gx10.running("job-1")
    cmd = last_remote_command(calls)
    assert "run_lecture_audio[.]py" in cmd


def test_the_liveness_check_does_not_depend_on_a_tilde_the_shell_will_expand(calls):
    """job_dir() starts with "~", which the remote shell expands to /home/<user> in the
    process's real argv -- so a pattern carrying the literal tilde never matches."""
    gx10.running("job-1")
    pattern = last_remote_command(calls).split("pgrep -af '", 1)[1].split("'", 1)[0]
    assert "~" not in pattern
    assert "job-1" in pattern


def test_ssh_refuses_to_sit_at_a_password_prompt(calls):
    gx10.ssh("true")
    assert "BatchMode=yes" in calls[-1]


def test_the_ssh_password_is_never_read_by_this_module():
    """It exists for sudo, which nothing here does. A password on a command line would be
    visible in the remote process table to anyone running ps."""
    src = Path(gx10.__file__).read_text(encoding="utf-8")
    for quote in ('"', "'"):
        assert f"environ.get({quote}GX10_SSH_PASSWORD" not in src
        assert f"environ[{quote}GX10_SSH_PASSWORD" not in src


def test_an_unreachable_machine_is_a_recoverable_condition_not_a_crash(monkeypatch):
    monkeypatch.setenv("GX10_HOST", "10.0.0.9")
    monkeypatch.setenv("GX10_SSH_USER", "asus")
    monkeypatch.setattr(gx10, "_tool", lambda name: name)

    def refuse(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: connect to host: Connection timed out")

    monkeypatch.setattr(gx10.subprocess, "run", refuse)
    with pytest.raises(gx10.Gx10Unavailable):
        gx10.ssh("true")
    assert gx10.reachable() is False


def test_missing_configuration_is_unavailable_rather_than_an_exception_in_a_request(monkeypatch):
    monkeypatch.delenv("GX10_HOST", raising=False)
    monkeypatch.delenv("GX10_SSH_USER", raising=False)
    with pytest.raises(gx10.Gx10Unavailable, match="GX10_HOST"):
        gx10.config()


def test_status_reads_missing_rather_than_failing_before_the_runner_writes_one(monkeypatch, calls):
    monkeypatch.setattr(gx10, "ssh", lambda *a, **k: "")
    assert gx10.status("job-1") == {"state": "missing"}
    monkeypatch.setattr(gx10, "ssh", lambda *a, **k: "{ half written")
    assert gx10.status("job-1") == {"state": "missing"}
    monkeypatch.setattr(gx10, "ssh", lambda *a, **k: json.dumps({"state": "running", "done": 2, "total": 4}))
    assert gx10.status("job-1")["done"] == 2


def test_send_ships_each_chunk_and_the_manifest(tmp_path, calls):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for i in range(3):
        (chunks / f"chunk_{i:02d}.mp3").write_bytes(b"audio")
    mp = chunks / "manifest.json"
    mp.write_text("{}", encoding="utf-8")

    gx10.send("job-1", chunks, mp)
    copied = [Path(c[-2]).name for c in calls if c[0] == "scp"]
    assert sorted(n for n in copied if n.startswith("chunk_")) == ["chunk_00.mp3", "chunk_01.mp3", "chunk_02.mp3"]
    assert "manifest.json" in copied
    assert "mkdir -p" in calls[0][-1]  # the job directory is made before anything is copied


# ------------------------------------------------------------------------- the endpoints
# The chunker runs for real on synthesized audio; the GX10 is replaced, because the point of
# these is the contract the page sees -- above all that the audio half can fail in every way
# it likes without touching the deck.

import asyncio  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from profe import audio_jobs  # noqa: E402
from profe.audiences import FileCache  # noqa: E402
from profe.server import create_app  # noqa: E402
from profe.store import BUNDLED_RUNS_DIR, RunStore  # noqa: E402

from builders import HashEmbedder  # noqa: E402

RUN = "sample-llm-serving"


@pytest.fixture
def app_with_fake_gx10(tmp_path, monkeypatch, tone):
    """A server whose GX10 is a dictionary. `remote` is what the machine would report."""
    monkeypatch.setenv("PROFE_DATA_DIR", str(tmp_path / "data"))
    # The GX10 reports the manifest it was sent, so this has to match what the real
    # chunker produces from the fixture: 300 s at 120 s a chunk.
    remote = {"state": "running", "done": 0, "total": 3}
    sent: list[str] = []

    monkeypatch.setattr(audio_jobs.gx10, "config", lambda: gx10.Gx10Config(host="h", user="u"))
    monkeypatch.setattr(audio_jobs.gx10, "ssh", lambda *a, **k: "")
    monkeypatch.setattr(audio_jobs.gx10, "scp_to", lambda local, rem, **k: sent.append(Path(local).name))
    monkeypatch.setattr(audio_jobs.gx10, "launch", lambda job_id, **k: None)
    monkeypatch.setattr(audio_jobs.gx10, "status", lambda job_id, **k: remote)
    monkeypatch.setattr(audio_jobs.gx10, "collect", lambda job_id, dest, **k: ["chunk_00", "chunk_01"])

    app = create_app(
        store=RunStore(tmp_path / "history", bundled=[BUNDLED_RUNS_DIR]),
        cache=FileCache(tmp_path / "cache"),
        client_factory=lambda: None,
        embedder=HashEmbedder(),
        frontend_dir=tmp_path / "none",
    )
    return app, remote, sent, tone


def post_audio(client, tone, run=RUN):
    return client.post(f"/api/runs/{run}/audio",
                       files={"file": ("lecture.mp3", tone.read_bytes(), "audio/mpeg")})


def settle(client, run=RUN, tries=60):
    """The job is prepared in a background task; wait for it to leave the local phases."""
    for _ in range(tries):
        body = client.get(f"/api/runs/{run}/audio").json()
        if body.get("state") not in ("chunking", "uploading"):
            return body
        time.sleep(0.2)
    return body


def test_audio_is_chunked_here_and_every_chunk_is_sent_separately(app_with_fake_gx10):
    app, _, sent, tone = app_with_fake_gx10
    with TestClient(app) as c:
        assert post_audio(c, tone).status_code == 202
        body = settle(c)
    assert body["chunks_total"] == 3            # 300s of audio at 120s a chunk
    assert body["chunks_sent"] == 3
    # three chunks and one manifest, not one large blob
    assert sorted(n for n in sent if n.startswith("chunk_")) == ["chunk_00.mp3", "chunk_01.mp3", "chunk_02.mp3"]
    assert "manifest.json" in sent


def test_a_deck_is_never_held_up_by_the_audio_half(app_with_fake_gx10, monkeypatch):
    """Acceptance: a failed audio upload must not block the slides-only path. The run's own
    status is untouched, and the failure is reported on the audio job alone."""
    app, _, _, tone = app_with_fake_gx10
    monkeypatch.setattr(audio_jobs.gx10, "launch",
                        lambda *a, **k: (_ for _ in ()).throw(gx10.Gx10Error("GPU on fire")))
    with TestClient(app) as c:
        before = c.get(f"/api/runs/{RUN}").json()["meta"]["status"]
        post_audio(c, tone)
        body = settle(c)
        after = c.get(f"/api/runs/{RUN}").json()["meta"]["status"]
    assert body["state"] == "failed" and "GPU on fire" in body["error"]
    assert before == after == "complete"        # the deck did not notice


def test_an_unreachable_machine_is_unavailable_rather_than_failed(app_with_fake_gx10, monkeypatch):
    """Nothing went wrong: the machine is asleep or on another network. The page says so
    differently because the presenter can act on it differently."""
    app, _, _, tone = app_with_fake_gx10
    monkeypatch.setattr(audio_jobs.gx10, "launch",
                        lambda *a, **k: (_ for _ in ()).throw(gx10.Gx10Unavailable("no route to host")))
    with TestClient(app) as c:
        post_audio(c, tone)
        body = settle(c)
    assert body["state"] == "unavailable"
    assert "deck is unaffected" in body["detail"]


def test_progress_follows_the_gx10_and_finishes_by_collecting_the_output(app_with_fake_gx10):
    app, remote, _, tone = app_with_fake_gx10
    with TestClient(app) as c:
        post_audio(c, tone)
        settle(c)

        remote.update(state="running", done=1, total=3)
        mid = c.get(f"/api/runs/{RUN}/audio").json()
        assert mid["state"] == "running" and mid["chunks_done"] == 1

        remote.update(state="complete", done=3, total=3)
        end = c.get(f"/api/runs/{RUN}/audio").json()
    assert end["state"] == "complete"
    assert end["chunks_collected"] == 2          # pulled back from the GX10


def test_a_run_that_failed_on_the_gx10_reports_why(app_with_fake_gx10):
    app, remote, _, tone = app_with_fake_gx10
    with TestClient(app) as c:
        post_audio(c, tone)
        settle(c)
        remote.update(state="failed", done=1, total=3, error="RequestException: timed out")
        body = c.get(f"/api/runs/{RUN}/audio").json()
    assert body["state"] == "failed" and "timed out" in body["error"]


def test_losing_contact_mid_run_is_not_reported_as_a_failed_run(app_with_fake_gx10, monkeypatch):
    """The work is on another machine and keeps going. A network blip must not be turned into
    a dead job, or a demo loses a run it still has."""
    app, _, _, tone = app_with_fake_gx10
    with TestClient(app) as c:
        post_audio(c, tone)
        settle(c)
        monkeypatch.setattr(audio_jobs.gx10, "status",
                            lambda *a, **k: (_ for _ in ()).throw(gx10.Gx10Unavailable("timed out")))
        body = c.get(f"/api/runs/{RUN}/audio").json()
    assert body["state"] == "running"
    assert "still polling" in body["detail"]


def test_a_file_that_is_not_audio_is_refused_at_the_door(app_with_fake_gx10):
    app, _, _, _ = app_with_fake_gx10
    with TestClient(app) as c:
        r = c.post(f"/api/runs/{RUN}/audio", files={"file": ("deck.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 415


def test_audio_for_an_unknown_run_is_a_404_not_a_stray_job(app_with_fake_gx10):
    app, _, _, tone = app_with_fake_gx10
    with TestClient(app) as c:
        assert post_audio(c, tone, run="no-such-run").status_code == 404


def test_a_run_with_no_audio_says_none_rather_than_failing(app_with_fake_gx10):
    app, _, _, _ = app_with_fake_gx10
    with TestClient(app) as c:
        assert c.get(f"/api/runs/{RUN}/audio").json() == {"state": "none"}


def test_removing_the_audio_forgets_it_locally(app_with_fake_gx10):
    app, _, _, tone = app_with_fake_gx10
    with TestClient(app) as c:
        post_audio(c, tone)
        settle(c)
        assert c.delete(f"/api/runs/{RUN}/audio").status_code == 204
        assert c.get(f"/api/runs/{RUN}/audio").json() == {"state": "none"}


def test_the_page_can_ask_whether_audio_is_possible_at_all(app_with_fake_gx10, monkeypatch):
    """So the page can say the audio half is off before someone picks a file, not after."""
    app, _, _, _ = app_with_fake_gx10
    monkeypatch.setattr(gx10, "reachable", lambda **k: True)
    with TestClient(app) as c:
        body = c.get("/api/gx10").json()
    assert body["configured"] is True and body["reachable"] is True

    # the machine is configured but asleep
    monkeypatch.setattr(gx10, "reachable", lambda **k: False)
    with TestClient(app) as c:
        body = c.get("/api/gx10").json()
    assert body["configured"] is True and body["reachable"] is False and body["reason"]

    # no credentials at all. The fixture patches config() globally -- audio_jobs.gx10 and
    # profe.gx10 are one module -- so being unconfigured is expressed as config() refusing.
    def unconfigured():
        raise gx10.Gx10Unavailable("GX10_HOST and GX10_SSH_USER are not set")

    monkeypatch.setattr(gx10, "config", unconfigured)
    with TestClient(app) as c:
        body = c.get("/api/gx10").json()
    assert body["configured"] is False and body["reachable"] is False and "GX10_HOST" in body["reason"]
