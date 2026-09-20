"""The audio chunker, and the transport that carries chunks to the GX10.

Nothing here touches the real machine. The chunker runs the bundled ffmpeg on audio it
synthesizes, so it is real work on real files; the transport's ssh and scp calls are captured
so the exact command lines can be asserted -- which is the point, because every bug found in
this module so far was in a command line rather than in the logic around it.
"""

from __future__ import annotations

import json
import subprocess
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
