"""Sandboxed notebook execution.

Per submission notebook:
  1. copy the submission into a per-job scratch dir (everything writable lives here)
  2. create a venv with --system-site-packages (inherits the base image's
     packages; requirements.txt only adds/overrides)
  3. pip install requirements.txt   [sandboxed: Landlock + seccomp + rlimits]
  4. run the notebook via nbclient   [same sandbox], pinned to the venv Python
  5. read submission.csv from the scratch work dir

Confinement (see space-sandbox-capabilities): Landlock (RW scratch, RO system +
predownloaded inputs, everything else incl. /proc denied), seccomp blocklist
(ptrace/process_vm_readv/mount/...), rlimits, egress allowlist (seccomp
user-notif on connect()), no HF token, HF_HUB_OFFLINE.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from . import egress, landlock, seccomp
from ..config import RunnerConfig
from ..data import DataLayout

SUBMISSION_FILENAME = "submission.csv"

# Fixed resource caps. The configurable limits (CPU time, memory, wall clock)
# come from RunnerConfig.
# NOTE: RLIMIT_NPROC is intentionally NOT capped. It's a per-UID limit (shared
# across all concurrent runs + the parent), so a low cap made rayon/tokenizers
# fail to spawn their thread pool under load with EAGAIN ("Resource temporarily
# unavailable" -> "global thread pool has not been initialized"). Thread/process
# growth is bounded instead by RLIMIT_AS (memory) and the wall-clock timeout.
_RLIMIT_FSIZE = 2 * 1024**3
_RLIMIT_NOFILE = 4096

# Once a run's process tree is SIGKILLed, the write ends of the stdout pipe close
# and the reaping ``communicate()`` returns promptly. This only bounds the
# pathological tail (a descendant we couldn't reap, e.g. stuck in uninterruptible
# sleep) so a stuck run can never hold the (serial) submission slot forever.
_REAP_GRACE = 30


def _proc_ppids() -> dict[int, int]:
    """Map ``pid -> ppid`` for every live process by scanning ``/proc`` (Linux).

    Returns ``{}`` on platforms without ``/proc`` (e.g. a macOS ``--dry``), where
    the process-group kill is the only reaping mechanism."""
    ppids: dict[int, int] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return ppids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except OSError:
            continue
        # "pid (comm) state ppid ...": comm may contain spaces and parentheses, so
        # parse the fields after the final ')'.
        try:
            fields = stat[stat.rindex(b")") + 2:].split()
            ppids[int(entry.name)] = int(fields[1])
        except (ValueError, IndexError):
            continue
    return ppids


def _descendants(pid: int) -> list[int]:
    """Every descendant pid of ``pid`` (excluding ``pid`` itself), via a /proc walk."""
    children: dict[int, list[int]] = {}
    for child, parent in _proc_ppids().items():
        children.setdefault(parent, []).append(child)
    out: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(children.get(p, []))
    return out


def _kill_tree(pid: int) -> None:
    """SIGKILL a process, its process group, and its entire descendant tree.

    The sandboxed child is a session leader (``start_new_session=True``), so a
    ``killpg`` reaps its group — but nbclient launches the Jupyter kernel in its
    OWN session, which escapes that group. Left alive, the orphaned kernel keeps
    the inherited stdout pipe open (wedging the reaping ``communicate()``) and keeps
    tracing on NDIF. Snapshot the descendant tree from /proc FIRST — while the ppid
    links are still intact, since once the parent dies its children reparent to init
    and the chain is lost — then signal the group and every descendant."""
    descendants = _descendants(pid)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:                       # already gone / no such group
        pass
    for p in (pid, *descendants):
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:                  # already reaped / not ours
            pass


class Canceller:
    """Cooperative cancellation for a submission's sandboxed runs.

    A submission runs its subprocesses in sequence (pip install, then one child per
    ``(notebook, dataset)``); at most one is live at a time. ``register`` records the
    live child so ``cancel`` — called from another thread (e.g. an admin endpoint) —
    can SIGKILL its whole tree immediately, which makes the run's ``communicate``
    return. A cancel that arrives between runs is latched: the next ``register`` kills
    on arrival and the pipeline loop's ``cancelled`` check stops it starting more."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def register(self, proc: subprocess.Popen) -> None:
        with self._lock:
            if not self._cancelled:
                self._proc = proc
                return
        _kill_tree(proc.pid)             # cancelled before we started -> kill now

    def unregister(self, proc: subprocess.Popen) -> None:
        with self._lock:
            if self._proc is proc:
                self._proc = None

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            proc = self._proc
        if proc is not None:
            _kill_tree(proc.pid)

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


@dataclass
class SandboxResult:
    ok: bool
    submission_csv: Path | None = None
    error: str | None = None
    phase: str | None = None         # install | run | read


def _make_preexec(cpu_limit: int, mem_bytes: int | None,
                  sb: landlock.LandlockSandbox | None, csock: socket.socket | None,
                  enable_seccomp: bool, apply_rlimits: bool):
    """Build the closure that runs in the forked child before exec.

    With ``sb=None`` / ``csock=None`` / ``enable_seccomp=False`` / no rlimits, the
    confinement layers are skipped (used by ``--dry`` on platforms without
    Landlock/seccomp). The server always passes them on."""
    def apply():
        def lim(which, soft, hard=None):
            try:
                resource.setrlimit(which, (soft, hard if hard is not None else soft))
            except (ValueError, OSError):
                pass
        if apply_rlimits:
            lim(resource.RLIMIT_CORE, 0)
            # RLIMIT_CPU is a backstop tied to the wall-clock timeout (CPU time
            # can't exceed it); the wall-clock kill is the real, single timeout.
            if cpu_limit:
                lim(resource.RLIMIT_CPU, cpu_limit, cpu_limit + 10)
            lim(resource.RLIMIT_FSIZE, _RLIMIT_FSIZE)
            lim(resource.RLIMIT_NOFILE, _RLIMIT_NOFILE)
            if mem_bytes:
                lim(resource.RLIMIT_AS, mem_bytes)
        if sb is not None:
            sb.apply()                          # Landlock (irreversible)
        if csock is not None:                   # egress: hand listener fd to parent
            fd = egress.install_connect_notifier()
            egress.send_fd(csock, fd)
            os.close(fd)
            csock.close()
        if enable_seccomp:
            seccomp.install_blocklist()
    return apply


def _run(cmd, env, cwd, *, cpu_limit, mem_bytes, sb, allow_suffixes, timeout,
         enable_seccomp, apply_rlimits, mitm_suffixes=None, cancel=None) -> tuple[int, str]:
    """Run a sandboxed subprocess in its own session; kill the group on timeout.

    When ``allow_suffixes`` is given, egress is restricted by **hostname**: a
    loopback proxy (parent-side) allowlists those suffixes, the child is pointed at
    it via HTTP(S)_PROXY, and a seccomp ``connect()`` gate permits the child only
    loopback connects (so it can't bypass the proxy). NDIF is CONNECT-tunneled;
    hosts in ``mitm_suffixes`` are additionally TLS-terminated and restricted to read
    methods; the child gets a CA bundle (system CAs + the proxy's CA) so it trusts
    the minted leaf certs."""
    stop = threading.Event()
    psock = csock = None
    proxy = None
    run_env = dict(env)
    if allow_suffixes is not None:
        proxy = egress.AllowProxy(list(allow_suffixes),
                                  mitm_suffixes=list(mitm_suffixes or []))
        url = f"http://127.0.0.1:{proxy.start()}"
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                  "http_proxy", "https_proxy", "all_proxy"):
            run_env[k] = url
        run_env["NO_PROXY"] = run_env["no_proxy"] = "localhost,127.0.0.1"
        if proxy.ca_cert_pem:
            import certifi
            bundle = Path(cwd) / ".aletheia_ca_bundle.pem"
            bundle.write_bytes(Path(certifi.where()).read_bytes()
                               + b"\n" + proxy.ca_cert_pem)
            for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
                run_env[k] = str(bundle)
        psock, csock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)

        def supervise():
            try:
                fd = egress.recv_fd(psock)
            except OSError:
                return
            egress.serve(fd, egress._loopback_only, stop)
        threading.Thread(target=supervise, daemon=True).start()

    preexec = _make_preexec(cpu_limit, mem_bytes, sb, csock,
                            enable_seccomp, apply_rlimits)
    proc = subprocess.Popen(
        cmd, env=run_env, cwd=str(cwd), preexec_fn=preexec,
        start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    if csock is not None:
        csock.close()  # parent keeps only psock
    # Make the live child cancellable: an operator ``cancel`` from another thread
    # SIGKILLs its whole tree, which unblocks the ``communicate`` below.
    if cancel is not None:
        cancel.register(proc)
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
        if cancel is not None and cancel.cancelled:
            rc = 125
            out = "cancelled by operator" + (f"\n{out}" if out else "")
    except subprocess.TimeoutExpired:
        # Kill the whole tree (not just the group — the kernel escapes it) so the
        # reaping communicate() can't hang on an orphan holding the pipe, and bound
        # that reap so the slot is released even if a descendant survives the kill.
        _kill_tree(proc.pid)
        try:
            out, _ = proc.communicate(timeout=_REAP_GRACE)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            out = ""
        rc = 124
        out = f"timed out after {timeout}s" + (f"\n{out}" if out else "")
    finally:
        if cancel is not None:
            cancel.unregister(proc)
        stop.set()
        if psock is not None:
            psock.close()
        if proxy is not None:
            proxy.stop()
    return rc, out


def _child_env(scratch: Path, data_env: dict, venv: Path) -> dict:
    """Minimal, token-free base environment for the sandboxed child.

    Per-notebook bits (``DATASET_NAME``, the submitter's keys) are layered on at
    run time, not here — this env is built once per request."""
    tmp = scratch / "tmp"
    tmp.mkdir(exist_ok=True)
    env = {
        "PATH": f"{venv / 'bin'}:/usr/bin:/bin",
        "HOME": str(scratch),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(scratch / "cache"),
        "XDG_RUNTIME_DIR": str(tmp),
        "JUPYTER_RUNTIME_DIR": str(tmp / "jupyter-runtime"),
        "ALETHEIA_JUPYTER": str(scratch / "jupyter"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }
    env.update(data_env)      # HF_HUB_CACHE (RO), HF_DATASETS_CACHE (scratch), offline flags
    return env


@dataclass
class JobContext:
    """Per-request sandbox setup, reused across every (dataset, notebook) run.

    Built once by :func:`setup_job`: one submission copy, one venv, one
    requirements install, one dataset-cache copy. ``run_notebook`` then executes
    each notebook against this shared context."""
    scratch: Path
    work: Path
    venv: Path
    base_env: dict
    run_allow: list | None
    run_mitm: list | None
    common: dict


# Packages baked into the base image that must not be reinstalled from a
# submission's requirements.txt. nnsight ships as a git+github pin, which the
# install egress allowlist (PyPI-only) rejects — a pin would fail the whole run.
_PREINSTALLED_REQS = ("nnsight",)


def _strip_preinstalled_reqs(text: str) -> str:
    """Drop requirement lines referencing a pre-installed package (e.g. nnsight),
    keeping comments and every other dependency so PyPI installs still work."""
    kept = []
    for line in text.splitlines():
        s = line.strip().lower()
        if s and not s.startswith("#") and any(p in s for p in _PREINSTALLED_REQS):
            continue
        kept.append(line)
    out = "\n".join(kept)
    return out + "\n" if text.endswith("\n") and out else out


def setup_job(
    submission_root: Path,
    data_layout: DataLayout,
    scratch: Path,
    config: RunnerConfig,
    cancel: "Canceller | None" = None,
) -> tuple[JobContext | None, SandboxResult | None]:
    """Build the per-request sandbox: copy the submission, create the venv, copy
    the dataset cache, and install requirements — all once. Returns
    ``(context, None)`` on success or ``(None, failure)`` if venv/pip failed."""
    scratch = Path(scratch)
    work = scratch / "work"
    venv = scratch / "venv"
    for d in (scratch, scratch / "cache", scratch / "jupyter"):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Isolated copy of the submission (writable; submission.csv lands here).
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(submission_root, work)

    # 2. venv (created by the trusted parent; fast, no untrusted code).
    rc = subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                         str(venv)], capture_output=True, text=True)
    if rc.returncode != 0:
        return None, SandboxResult(False, error=f"venv create failed: {rc.stderr[-800:]}",
                                   phase="install")

    # 3. Base env incl. the per-request dataset-cache copy (one copytree, not one
    # per notebook). DATASET_NAME / the submitter's keys are added at run time.
    data_env = data_layout.child_env(scratch, offline=config.confine)
    env = _child_env(scratch, data_env, venv)
    env["PIP_NO_INPUT"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    # Landlock policy: RO system; RW scratch (incl. the dataset cache copy); deny
    # everything else. The aletheia_runner package path must be readable so the
    # child can run _run_entry (covers editable installs whose source is outside
    # sys.prefix).
    confine = config.confine
    if confine:
        pkg_path_entry = str(Path(__file__).resolve().parent.parent)
        ro = landlock.default_system_ro_paths() + [pkg_path_entry]
        rw = [str(scratch)] + landlock.essential_dev_paths()
        sb = landlock.LandlockSandbox(ro_paths=ro, rw_paths=rw)
        run_allow = config.egress_allowlist if config.enforce_egress else None
        run_mitm = config.egress_get_only_suffixes if config.enforce_egress else None
        install_allow = config.install_allowlist if config.enforce_egress else None
    else:
        sb = None                       # --dry on a host without Landlock/seccomp
        run_allow = install_allow = run_mitm = None

    mem_bytes = config.mem_mb * 1024 * 1024 if config.mem_mb else None
    # ``timeout`` is the single per-(notebook, dataset) wall-clock budget (kills the
    # process group); RLIMIT_CPU is a backstop derived from it.
    common = dict(cpu_limit=config.notebook_timeout, mem_bytes=mem_bytes, sb=sb,
                  timeout=config.notebook_timeout,
                  enable_seccomp=confine, apply_rlimits=confine)

    # 4. Install submission/requirements.txt once (sandboxed; egress limited to PyPI).
    req = work / "submission" / "requirements.txt"
    if req.exists() and req.read_text().strip():
        # nnsight (the hackathon build) is pre-installed in the base image. Strip any
        # participant pin of it: installs come from git+github, which the install
        # egress allowlist (PyPI-only) blocks — a pinned nnsight fails the whole
        # submission on every dataset. Everything else still installs from PyPI.
        req.write_text(_strip_preinstalled_reqs(req.read_text()))
    if req.exists() and req.read_text().strip():
        code, out = _run(
            [str(venv / "bin" / "python"), "-m", "pip", "install", "-r",
             "submission/requirements.txt"],
            env, work, allow_suffixes=install_allow, cancel=cancel, **common)
        if code != 0:
            return None, SandboxResult(False, error=f"pip install failed:\n{out[-1500:]}",
                                       phase="install")

    return JobContext(scratch=scratch, work=work, venv=venv, base_env=env,
                      run_allow=run_allow, run_mitm=run_mitm, common=common), None


def run_notebook(
    ctx: JobContext,
    notebook_rel: str,
    dataset_cfg,
    config: RunnerConfig,
    *,
    extra_env: dict[str, str] | None = None,
    timeout: int | None = None,
    cancel: "Canceller | None" = None,
) -> SandboxResult:
    """Run one notebook against one dataset inside an already-prepared job.

    The shared work dir is reused, so any stale ``submission.csv`` from a prior
    notebook/dataset is cleared first and the produced one snapshotted under a
    name unique to this (dataset, notebook)."""
    work = ctx.work
    stale = work / SUBMISSION_FILENAME
    if stale.exists():
        stale.unlink()

    # Run the entry as a standalone script so the heavy aletheia_runner package
    # __init__ (numpy/...) isn't imported here; only the kernel imports user libs.
    # The submitter's NDIF_API_KEY is injected only for this run phase (not install).
    entry = str(Path(__file__).resolve().parent / "_run_entry.py")
    run_env = {**ctx.base_env, **dataset_cfg.env(), **(extra_env or {})}
    if config.ndif_host:                      # point nnsight at the configured cluster
        run_env["NDIF_HOST"] = config.ndif_host
    # ``timeout`` clamps this run to the submission's remaining overall budget; the
    # CPU-time backstop tracks it. Fall back to the job's configured per-run budget.
    common = dict(ctx.common)
    if timeout is not None:
        common["timeout"] = timeout
        common["cpu_limit"] = timeout
    code, out = _run(
        [str(ctx.venv / "bin" / "python"), entry, notebook_rel],
        run_env, work, allow_suffixes=ctx.run_allow, mitm_suffixes=ctx.run_mitm,
        cancel=cancel, **common)
    if code != 0:
        # Keep a generous tail of the output: this becomes the organizer-only
        # ``error_detail`` (persisted to the bucket), so we want the full traceback,
        # not a 1.5 KB snippet — bounded only to keep results.jsonl from bloating.
        return SandboxResult(False, error=f"notebook failed (rc={code}):\n{out[-16000:]}",
                             phase="run")

    produced = work / SUBMISSION_FILENAME
    if not produced.exists():
        return SandboxResult(False, error="notebook did not write submission.csv",
                             phase="read")
    # Flatten slashes from BOTH the dataset key (HF ids are "org/name") and the
    # notebook path so the snapshot is a single file in the shared scratch dir.
    safe = f"{dataset_cfg.key}__{notebook_rel}".replace("/", "__")
    snap = ctx.scratch / f"{safe}.submission.csv"
    snap.write_bytes(produced.read_bytes())
    return SandboxResult(True, submission_csv=snap)
