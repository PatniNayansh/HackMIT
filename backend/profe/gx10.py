"""Talking to the GX10.

There is no HTTP endpoint on that box and there is not meant to be one. It is an ASUS DGX
Spark on the local network, reached over SSH with key auth already established, and neural
results have always arrived here as files written by a script over that link. So this module
is the "endpoint": it copies chunks across with scp, starts a detached run, polls a status
file, and copies the results back.

Everything here is deliberately thin. The model call, the atlas and the rendering all live on
the GX10 in scripts/precompute_neural/; this side knows only about files and process state, so
it needs none of TRIBE's dependencies and can run on a laptop.

Hard-won details this module encodes so nobody rediscovers them (see docs/GX10_OPERATIONS.md):

  * A non-interactive SSH shell does not source ~/.bashrc, so ~/.local/bin -- where uv, uvx and
    the ffmpeg symlink live -- is missing from PATH unless it is exported explicitly. The same
    PATH has to be passed through to the backgrounded process, which inherits the environment
    as it was at launch rather than re-sourcing a shell.
  * A background job must detach all three streams. `nohup cmd > log 2>&1 &` still leaves stdin
    on the SSH pty, which can hang the launching ssh invocation even though the job started
    fine. `< /dev/null` and `disown` are not optional.
  * For the same reason the launch command's exit status says nothing useful about whether the
    job is running. Confirm with a separate call.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REMOTE_ROOT = "~/profe-jobs"
REMOTE_REPO = "~/SightLine"
RUNNER = f"{REMOTE_REPO}/backend/scripts/precompute_neural/run_lecture_audio.py"
VENV_PYTHON = f"{REMOTE_REPO}/backend/scripts/precompute_neural/.venv/bin/python"

CONNECT_TIMEOUT_S = 8
SSH_OPTS = [
    "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}",
    "-o", "BatchMode=yes",          # never sit at a password prompt: key auth or fail fast
    "-o", "StrictHostKeyChecking=accept-new",
]


class Gx10Unavailable(RuntimeError):
    """The GX10 is not configured or not reachable. Always recoverable: the deck is analysed
    without it, and the presenter is told the audio half did not run."""


class Gx10Error(RuntimeError):
    """The GX10 answered, but the operation failed."""


@dataclass(frozen=True)
class Gx10Config:
    host: str
    user: str

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def config() -> Gx10Config:
    """Read from the environment (the repo-root .env is loaded by llm.load_env at startup).
    No password: this module only ever uses key auth. GX10_SSH_PASSWORD exists for sudo, which
    nothing here does, and is deliberately never read."""
    host, user = os.environ.get("GX10_HOST", "").strip(), os.environ.get("GX10_SSH_USER", "").strip()
    if not host or not user:
        raise Gx10Unavailable("GX10_HOST and GX10_SSH_USER are not set; see docs/GX10_OPERATIONS.md")
    return Gx10Config(host=host, user=user)


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise Gx10Unavailable(f"no {name} client on this machine; the GX10 is reached over SSH")
    return found


def ssh(command: str, *, timeout: int = 60, cfg: Gx10Config | None = None) -> str:
    """One remote command. PATH is exported first because a non-interactive shell will not."""
    cfg = cfg or config()
    argv = [_tool("ssh"), *SSH_OPTS, cfg.target, f"export PATH=$HOME/.local/bin:$PATH; {command}"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise Gx10Unavailable(f"the GX10 did not answer within {timeout}s") from e
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        if "Connection" in err or "Could not resolve" in err or "timed out" in err.lower():
            raise Gx10Unavailable(f"could not reach the GX10: {err[:200]}")
        raise Gx10Error(err[:300] or f"remote command failed ({proc.returncode})")
    return proc.stdout


def reachable(*, cfg: Gx10Config | None = None) -> bool:
    try:
        ssh("true", timeout=CONNECT_TIMEOUT_S + 4, cfg=cfg)
        return True
    except (Gx10Unavailable, Gx10Error):
        return False


def scp_to(local: Path, remote: str, *, timeout: int = 600, cfg: Gx10Config | None = None) -> None:
    cfg = cfg or config()
    argv = [_tool("scp"), *SSH_OPTS, "-r", str(local), f"{cfg.target}:{remote}"]
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise Gx10Error(f"could not copy {local.name} to the GX10: {(proc.stderr or '').strip()[:200]}")


def scp_from(remote: str, local: Path, *, timeout: int = 600, cfg: Gx10Config | None = None) -> None:
    cfg = cfg or config()
    local.mkdir(parents=True, exist_ok=True)
    argv = [_tool("scp"), *SSH_OPTS, "-r", f"{cfg.target}:{remote}", str(local)]
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise Gx10Error(f"could not copy results back: {(proc.stderr or '').strip()[:200]}")


# ------------------------------------------------------------------------------------ jobs


def job_dir(job_id: str) -> str:
    return f"{REMOTE_ROOT}/{job_id}"


def send(job_id: str, chunk_dir: Path, manifest_path: Path, *, cfg: Gx10Config | None = None) -> None:
    """Ship the already-cut chunks and the manifest. Chunking happens on this side so the
    upload is many small files rather than one large blob, and so progress is per chunk."""
    cfg = cfg or config()
    remote = job_dir(job_id)
    ssh(f"rm -rf {remote} && mkdir -p {remote}/chunks {remote}/out", cfg=cfg)
    for f in sorted(chunk_dir.glob("chunk_*")):
        scp_to(f, f"{remote}/chunks/", cfg=cfg)
    scp_to(manifest_path, f"{remote}/manifest.json", cfg=cfg)


def launch(job_id: str, *, cfg: Gx10Config | None = None) -> None:
    """Start the run detached, then confirm separately -- the launch call's own exit status
    cannot be trusted for a backgrounded remote process."""
    cfg = cfg or config()
    remote = job_dir(job_id)
    # No `cd ... &&` before the nohup. The trailing `&` backgrounds the WHOLE chain, so ssh
    # returns before the chain has run and the connection closing takes the job with it --
    # measured, not guessed. The cd is unnecessary anyway: python puts the script's own
    # directory on sys.path, which is what the runner's `from regions import ...` needs.
    cmd = (
        f"nohup env PATH=$HOME/.local/bin:$PATH PROFE_PROCESS=precompute "
        f"{VENV_PYTHON} {RUNNER} --job-dir {remote} "
        f"> {remote}/job.log 2>&1 < /dev/null & disown"
    )
    ssh(cmd, cfg=cfg)
    if not running(job_id, cfg=cfg) and status(job_id, cfg=cfg).get("state") in (None, "missing"):
        log = tail_log(job_id, cfg=cfg)
        raise Gx10Error(f"the run did not start on the GX10. Last output:\n{log[-400:]}")


def running(job_id: str, *, cfg: Gx10Config | None = None) -> bool:
    """Is the runner alive for this job?

    The bracket in `run_lecture_audio[.]py` is load-bearing. pgrep sees the whole process table
    including the shell this very command runs in, whose command line contains the pattern --
    so an unbracketed pattern always matches itself and every job looks alive. Bracketing makes
    the regex match "run_lecture_audio.py" while the searching shell's own line, which contains
    the literal brackets, does not match it."""
    # Matched on the job id, not on job_dir(): that starts with "~", which the remote shell
    # expands to /home/<user> in the process's real argv, so a pattern carrying the literal
    # tilde can never match a running job. The id is unique, which is all the pattern needs.
    out = ssh(f"pgrep -af 'run_lecture_audio[.]py .*{job_id}' || true", cfg=cfg)
    return bool(out.strip())


def status(job_id: str, *, cfg: Gx10Config | None = None) -> dict:
    """The remote status file, or {"state": "missing"} before the runner has written one."""
    out = ssh(f"cat {job_dir(job_id)}/status.json 2>/dev/null || true", cfg=cfg)
    if not out.strip():
        return {"state": "missing"}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"state": "missing"}


def tail_log(job_id: str, *, lines: int = 40, cfg: Gx10Config | None = None) -> str:
    return ssh(f"tail -n {lines} {job_dir(job_id)}/job.log 2>/dev/null || true", cfg=cfg)


def collect(job_id: str, dest: Path, *, cfg: Gx10Config | None = None) -> list[str]:
    """Pull the finished chunk outputs back. Returns the chunk names that arrived."""
    scp_from(f"{job_dir(job_id)}/out/*", dest, cfg=cfg)
    return sorted(p.name for p in dest.iterdir() if p.is_dir() and (p / "metrics.json").is_file())
