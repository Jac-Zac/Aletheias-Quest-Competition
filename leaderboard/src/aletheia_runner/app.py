"""FastAPI app for the leaderboard Space.

Thin web layer over the runner core:
- ``POST /submit``  — accept a zip + team name, run + score it, persist, return scores
- ``GET  /api/leaderboard`` — JSON leaderboard (best score per team/dataset)
- ``GET  /`` — static leaderboard page that reads the JSON API

``create_app(config, store)`` is injectable for tests; the module-level ``app``
is built from the environment for the Space.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse

from . import ndif, pipeline
from .archive import SubmissionArchive
from .config import PRIMARY_METRIC, SECONDARY_METRIC, RunnerConfig, dataset_label
from .ratelimit import RateLimiter
from .registry import TeamRegistry
from .results import (BaseResultStore, is_bucket_uri, make_store, parse_bucket_uri,
                      summarize_submission)

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "250"))
# How many submissions may execute at once. Each run holds a model graph + a
# notebook kernel locally (~1-2 GB; the heavy compute is remote on NDIF), so this
# bounds RAM/CPU on a small Space. Extra submissions queue on the semaphore.
MAX_CONCURRENT_SUBMISSIONS = int(os.environ.get("MAX_CONCURRENT_SUBMISSIONS", "4"))
# The leaderboard page polls /api/leaderboard every 30s per open tab, and each
# read pulls the whole results object from the bucket. Cache the computed board
# in-process for this long so N viewers cost ~one read per window, not N; a
# submission invalidates the cache so a fresh score shows up immediately.
LEADERBOARD_CACHE_TTL = float(os.environ.get("LEADERBOARD_CACHE_TTL", "20"))


class _LeaderboardCache:
    """Thread-safe TTL cache over ``store.leaderboard()`` (called from threadpool
    workers). ``invalidate()`` forces the next read to recompute."""

    def __init__(self, store: BaseResultStore, ttl: float):
        self._store = store
        self._ttl = ttl
        self._lock = threading.Lock()
        self._at = 0.0
        self._val: list[dict] | None = None

    def get(self) -> list[dict]:
        with self._lock:
            if self._val is not None and time.monotonic() - self._at < self._ttl:
                return self._val
        val = self._store.leaderboard()        # network read; outside the lock
        with self._lock:
            self._val, self._at = val, time.monotonic()
        return val

    def invalidate(self) -> None:
        with self._lock:
            self._val = None


def _read_json_uri(uri: str, token: str | None):
    """Read JSON from a ``bucket://`` uri or a local path; None if missing/unset."""
    if not uri:
        return None
    if is_bucket_uri(uri):
        from huggingface_hub import download_bucket_files
        bucket_id, path = parse_bucket_uri(uri, "invalidations.json")
        with tempfile.TemporaryDirectory(prefix="aletheia-inv-") as tmp:
            local = Path(tmp) / "f.json"
            download_bucket_files(bucket_id, files=[(path, str(local))],
                                  raise_on_missing_files=False, token=token)
            return json.loads(local.read_text()) if local.exists() else None
    p = Path(uri)
    return json.loads(p.read_text()) if p.exists() else None


def _read_text_uri(uri: str, token: str | None):
    """Read raw text from a ``bucket://`` uri or local path; None if missing/unset."""
    if not uri:
        return None
    if is_bucket_uri(uri):
        from huggingface_hub import download_bucket_files
        bucket_id, path = parse_bucket_uri(uri, "max_concurrent.txt")
        with tempfile.TemporaryDirectory(prefix="aletheia-mc-") as tmp:
            local = Path(tmp) / "f.txt"
            download_bucket_files(bucket_id, files=[(path, str(local))],
                                  raise_on_missing_files=False, token=token)
            return local.read_text() if local.exists() else None
    p = Path(uri)
    return p.read_text() if p.exists() else None


class _DynamicSlots:
    """Async admission gate honoring a LIVE limit (read via ``limit_fn`` — must be
    fast/non-blocking). At most ``limit`` holders run at once. A lowered limit never
    preempts in-flight holders (it just stops admitting until running drops below it);
    a raised limit admits more within one poll interval. Same ``async with`` interface
    as asyncio.Semaphore, so call sites are unchanged."""
    def __init__(self, limit_fn, poll: float = 0.5):
        self._limit_fn = limit_fn
        self._poll = poll
        self._running = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        while True:
            async with self._lock:
                if self._running < max(1, int(self._limit_fn())):
                    self._running += 1
                    return self
            await asyncio.sleep(self._poll)

    async def __aexit__(self, *exc):
        async with self._lock:
            self._running -= 1

    @property
    def running(self) -> int:
        return self._running


class _Invalidations:
    """TTL-cached list of disqualified submissions from a read-only JSON file.

    Each entry is ``{"team", "submitted_at", "reason"[, "notebook"]}``; a leaderboard
    row matches on ``(team, submitted_at)``. Malformed/missing -> ``{}`` (a bad file
    must never break the board)."""

    def __init__(self, uri: str, token: str | None, ttl: float):
        self._uri, self._token, self._ttl = uri, token, ttl
        self._lock = threading.Lock()
        self._at = 0.0
        self._val: dict[tuple, str] = {}

    def get(self) -> dict[tuple, str]:
        """Map ``(team, submitted_at) -> reason`` for the current invalidations."""
        with self._lock:
            if self._at and time.monotonic() - self._at < self._ttl:
                return self._val
        try:
            data = _read_json_uri(self._uri, self._token)
        except Exception:  # noqa: BLE001 - never let a bad file break the leaderboard
            data = None
        val = {}
        if isinstance(data, list):
            for e in data:
                if isinstance(e, dict) and e.get("team") and e.get("submitted_at"):
                    val[(e["team"], e["submitted_at"])] = str(e.get("reason", ""))
        with self._lock:
            self._val, self._at = val, time.monotonic()
        return val


def _normalize_tag(raw: str | None) -> str | None:
    """Map a submitted method tag to the canonical "white" / "black" (or None).

    Accepts white / whitebox / white-box / wb and black / blackbox / black-box / bb
    (case-insensitive); anything else is ignored (treated as untagged)."""
    if not raw:
        return None
    t = raw.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if t in ("white", "whitebox", "wb"):
        return "white"
    if t in ("black", "blackbox", "bb"):
        return "black"
    return None


def _redact_dataset_names(text: str, config: RunnerConfig) -> str:
    """Replace any real dataset / labels-repo id with its public codename.

    The private split names must never reach the participant; the run path already
    reports failures by codename, but a data/labels load that fails *before* the run
    (e.g. a dataset not yet on the Hub/NDIF) surfaces HF's own message — which names
    the real dataset — through the 400. Scrub it here too, so the up-front rejection
    keeps the same guarantee as the run path."""
    for d in config.datasets:
        label = dataset_label(d.key)
        if d.name:
            text = text.replace(d.name, label)
        if getattr(d, "labels_uri", None):
            text = text.replace(d.labels_uri, label)
    return text


def _team_submissions(store: BaseResultStore, team: str, limit: int = 100) -> list[dict]:
    """A team's submission history (newest first), capped. One entry per submitted
    notebook: the mean of each metric across datasets, the per-dataset breakdown,
    and total runtime; a notebook with no successful dataset is ``ok=False``."""
    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)   # (notebook, stamp) -> records
    for r in store.all():
        if r.team == team:
            groups[(r.notebook, r.submitted_at)].append(r)

    entries = []
    for (notebook, stamp), recs in groups.items():
        summ = summarize_submission(recs)
        entries.append({"notebook": notebook.split("/")[-1], "submitted_at": stamp,
                        "ok": summ["ok"], "metrics": summ["metrics"],
                        "datasets": summ["datasets"],
                        "runtime_seconds": summ["runtime_seconds"],
                        "tag": summ.get("tag"),
                        "failed_dataset": summ["failed_dataset"], "error": summ["error"]})
    entries.sort(key=lambda e: e["submitted_at"] or "", reverse=True)
    return entries[:limit]


def _anonymize(entry: dict, label) -> dict:
    """Return a copy of a reported entry (a leaderboard row, /api/me submission, or
    /submit result) with every real dataset name replaced by its public label
    ("Dataset 1", ...). The real names are private — they must never leave the
    server. Copies rather than mutating, so the cached leaderboard rows stay intact.
    ``label`` maps a dataset key to its label (and passes ``None`` through)."""
    out = dict(entry)
    if out.get("datasets"):
        out["datasets"] = [{**d, "dataset": label(d.get("dataset"))}
                           for d in out["datasets"]]
    if "failed_dataset" in out:
        out["failed_dataset"] = label(out["failed_dataset"])
    if "dataset" in out:                 # /submit failure rows carry a bare key
        out["dataset"] = label(out["dataset"])
    return out


def create_app(config: RunnerConfig, store: BaseResultStore,
               registry: TeamRegistry,
               limiter: RateLimiter | None = None,
               archive: SubmissionArchive | None = None,
               backend=None) -> FastAPI:
    # ``backend`` runs the submission: production uses a FargateBackend (launches a
    # per-submission task). The Space never executes a submission in-process. Required
    # — with no backend configured, /submit returns 503. Team resolution, rate
    # limiting, and result persistence stay on the Space (bucket writes serialized by
    # ``bucket_lock``), unchanged. (Tests inject an in-process backend double.)
    app = FastAPI(title="Aletheia's Quest — Leaderboard")
    # No limiter passed (e.g. most tests) -> unlimited.
    limiter = limiter or RateLimiter("rate_limits.json", 0, 0)

    # Single worker (see Dockerfile), so in-process primitives suffice:
    #   - submit_slots bounds concurrent heavy runs (the submission queue);
    #   - bucket_lock serializes read-modify-write of the registry + results
    #     objects so concurrent submissions can't clobber each other.
    # Live-tunable concurrency: a background task refreshes `_conc["limit"]` from a
    # bucket text file (a single integer) every ~15s, so an operator can change how
    # many submissions run at once by editing that file — no redeploy. Falls back to
    # MAX_CONCURRENT_SUBMISSIONS if the file is missing/unreadable.
    _conc = {"limit": MAX_CONCURRENT_SUBMISSIONS}
    max_conc_uri = ""
    if is_bucket_uri(config.results_uri):
        _bid, _ = parse_bucket_uri(config.results_uri, "results.jsonl")
        max_conc_uri = f"bucket://{_bid}/max_concurrent.txt"
    submit_slots = _DynamicSlots(lambda: _conc["limit"])
    bucket_lock = asyncio.Lock()
    lb_cache = _LeaderboardCache(store, LEADERBOARD_CACHE_TTL)
    # Disqualified submissions (read-only JSON, edited by hand). Default: a sibling
    # of the results object in the same bucket.
    inv_uri = config.invalidations_uri
    if not inv_uri and is_bucket_uri(config.results_uri):
        bucket_id, _ = parse_bucket_uri(config.results_uri, "results.jsonl")
        inv_uri = f"bucket://{bucket_id}/invalidations.json"
    invalidations = _Invalidations(inv_uri, config.hf_token, LEADERBOARD_CACHE_TTL)
    # In-flight submissions per team (queued or running). Mutated only on the event
    # loop, so a plain dict is safe; read by /api/me.
    pending: dict[str, int] = {}
    # In-flight runs by id, for the operator-only admin endpoints. Each carries a
    # cancel handle (from the backend) whose .cancel() stops the live run from another
    # thread — StopTask on Fargate, SIGKILL in-process. Mutated only on the event
    # loop; the handle is thread-safe.
    runs: dict[int, dict] = {}
    run_seq = {"n": 0}

    def _require_admin(token: str | None) -> None:
        import hmac
        if not config.admin_token:
            raise HTTPException(404, "not found")   # feature disabled -> don't advertise it
        if not token or not hmac.compare_digest(token, config.admin_token):
            raise HTTPException(403, "forbidden")

    # Public dataset codenames ("Dataset <Greek deity>"). Real names are private and
    # must never appear in a response; ``label`` maps any key to its stable codename
    # (so even stale keys not in the current config get a real name, not a generic
    # placeholder). None passes through.
    def label(key: str | None) -> str | None:
        return dataset_label(key) if key else key

    _bg_tasks: list = []

    @app.on_event("startup")
    async def _start_bg():
        async def _refresh_concurrency():
            while True:
                try:
                    txt = await run_in_threadpool(_read_text_uri, max_conc_uri, config.hf_token)
                    _conc["limit"] = max(1, int(txt.strip())) if txt and txt.strip() else \
                        MAX_CONCURRENT_SUBMISSIONS
                except Exception:  # noqa: BLE001 - a bad file must never wedge the gate
                    pass
                await asyncio.sleep(15)
        _bg_tasks.append(asyncio.create_task(_refresh_concurrency()))

    @app.on_event("shutdown")
    async def _stop_bg():
        for t in _bg_tasks:
            t.cancel()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text()

    @app.get("/api/leaderboard")
    def api_leaderboard() -> dict:
        # Flag reference/baseline rows and disqualified submissions so the page can
        # mark them and leave them out of the competitive rank (both still show, in
        # score order — baselines as context, invalid ones in the spot they'd rank).
        baseline = set(config.baseline_teams)
        inval = invalidations.get()      # (team, submitted_at) -> reason

        def tag(r: dict) -> dict:
            row = _anonymize(r, label)
            row["baseline"] = r.get("team") in baseline
            reason = inval.get((r.get("team"), r.get("submitted_at")))
            if reason is not None:
                row["invalid"] = True
                row["invalid_reason"] = reason
            return row

        return {"primary": PRIMARY_METRIC, "secondary": SECONDARY_METRIC,
                "results": [tag(r) for r in lb_cache.get()]}

    @app.post("/api/me")
    async def me(ndif_api_key: str | None = Header(default=None,
                                                   alias="X-NDIF-API-Key")) -> dict:
        """A team's own standing: name, rate-limit status, in-flight + past
        submissions. Keyed by the NDIF key (hashed for lookup, never stored). All
        reads, so no bucket lock — the bucket objects are replaced atomically."""
        if not ndif_api_key:
            raise HTTPException(400, "provide your NDIF API key (X-NDIF-API-Key)")
        team = await run_in_threadpool(registry.lookup, ndif_api_key)
        if team is None:
            return {"registered": False, "team": None,
                    "rate_limit": limiter.status(""), "submissions": [], "pending": 0}
        rate_limit, submissions = await asyncio.gather(
            run_in_threadpool(limiter.status, team),
            run_in_threadpool(_team_submissions, store, team))
        active = next((r for r in runs.values()
                       if r["team"] == team and r.get("status") in ("queued", "running")), None)
        return {"registered": True, "team": team, "rate_limit": rate_limit,
                "submissions": [_anonymize(s, label) for s in submissions],
                "pending": pending.get(team, 0),
                "in_flight": ({"status": active["status"],
                               "progress": active.get("progress")} if active else None)}

    @app.get("/api/health")
    def health() -> dict:
        # Public codenames only — never the real dataset names. ``backend`` lets an
        # operator confirm whether execution runs on Fargate or in-process (useful at
        # cutover; it only reports the mode, exposes no config).
        return {"ok": True,
                "backend": "fargate" if backend is not None else "in-process",
                "datasets": list(config.dataset_label_map().values()),
                "max_concurrent": _conc["limit"],
                "running": submit_slots.running}

    @app.get("/admin/runs")
    def admin_runs(x_admin_token: str | None = Header(default=None,
                                                     alias="X-Admin-Token")) -> dict:
        """List in-flight runs (operator only). Requires ``X-Admin-Token`` matching
        the configured ``ADMIN_TOKEN``; disabled (404) when no token is set."""
        _require_admin(x_admin_token)
        now = datetime.datetime.now(datetime.timezone.utc)
        def _age(iso):
            try:
                return round((now - datetime.datetime.fromisoformat(iso)).total_seconds())
            except Exception:  # noqa: BLE001
                return None
        return {"runs": [{"id": r["id"], "team": r["team"],
                          "status": r.get("status"),
                          "cancelled": r["canceller"].cancelled,
                          "started_at": r["started_at"],
                          "dispatched_at": r.get("dispatched_at"),
                          "age_s": _age(r["started_at"]),
                          "progress": r.get("progress")}
                         for r in sorted(runs.values(), key=lambda r: r["id"])]}

    @app.post("/admin/cancel")
    def admin_cancel(run_id: int | None = None, cancel_all: bool = False,
                     x_admin_token: str | None = Header(default=None,
                                                       alias="X-Admin-Token")) -> dict:
        """Cancel an in-flight run by ``run_id`` (or every run with ``cancel_all=true``).

        SIGKILLs the run's live subprocess tree so a wedged run releases the shared
        (serial) submission slot immediately, without restarting the Space. The run
        is then recorded as a failed submission. Operator only (see ``/admin/runs``)."""
        _require_admin(x_admin_token)
        targets = (list(runs.values()) if cancel_all
                   else [r for r in runs.values() if r["id"] == run_id])
        for r in targets:
            r["canceller"].cancel()
        return {"cancelled": [r["id"] for r in targets]}

    @app.post("/admin/purge")
    def admin_purge(x_admin_token: str | None = Header(default=None,
                                                       alias="X-Admin-Token")) -> dict:
        """Drop tracking entries whose worker task has already finished (operator only).
        Teardown normally removes these automatically; this clears any that lingered."""
        _require_admin(x_admin_token)
        purged = [rid for rid, r in list(runs.items())
                  if r.get("task") is not None and r["task"].done()]
        for rid in purged:
            runs.pop(rid, None)
        return {"purged": purged}

    @app.get("/api/sandbox-probe")
    def sandbox_probe() -> dict:
        """Report which unprivileged sandboxing primitives work in this env."""
        proc = subprocess.run(
            [sys.executable, "-m", "aletheia_runner.sandbox.probe"],
            capture_output=True, text=True, timeout=30)
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise HTTPException(500, f"probe failed: {proc.stderr[:1000]}")

    @app.get("/api/landlock-selftest")
    def landlock_selftest() -> dict:
        """Apply Landlock in a child and report what it allows/denies here."""
        proc = subprocess.run(
            [sys.executable, "-m", "aletheia_runner.sandbox.landlock"],
            capture_output=True, text=True, timeout=30)
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            raise HTTPException(500, f"selftest failed: {proc.stderr[:1000]}")

    def _build_payload(records: list, team: str) -> dict:
        """Participant-facing result payload: per-notebook mean metrics + anonymized
        per-dataset breakdown. Shared by the JSON response and the stream's terminal
        ``result`` event (both give the submitter their own final scores)."""
        by_nb: dict[str, list] = {}
        for r in records:
            by_nb.setdefault(r.notebook, []).append(r)
        results_out, scores, failures = [], {}, []
        for nb, recs in by_nb.items():
            name = nb.split("/")[-1]
            summ = summarize_submission(recs)
            if summ["ok"]:
                scores[name] = summ["metrics"].get(PRIMARY_METRIC)
                results_out.append(_anonymize({
                    "notebook": name, "ok": True, "metrics": summ["metrics"],
                    "datasets": summ["datasets"],
                    "runtime_seconds": summ["runtime_seconds"]}, label))
            else:
                fail = _anonymize({"notebook": name, "dataset": summ["failed_dataset"],
                                   "error": summ["error"]}, label)
                failures.append(fail)
                results_out.append({**fail, "ok": False})
        return {"team": team, "primary": PRIMARY_METRIC, "results": results_out,
                "scores": scores, "failures": failures,
                "message": (f"scored {len(scores)} notebook(s)"
                            + (f", {len(failures)} failed" if failures else ""))}

    async def _finalize(records: list, method_tag: str | None) -> None:
        """Stamp records (submitted_at + tag), persist them, refresh the board cache."""
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tag = _normalize_tag(method_tag)
        for r in records:
            r.submitted_at = stamp
            r.tag = tag
        async with bucket_lock:
            await run_in_threadpool(store.append, records)
        lb_cache.invalidate()

    def _run_records(data, team, extra_env, on_progress, canceller, run_key):
        """Execute a submission on the configured backend (Fargate in production; a
        test double in the suite) and return its ResultRecords. Blocking; called in a
        threadpool worker."""
        return backend.run(config, run_key, data, team, extra_env,
                           on_progress, canceller)

    def _sse(event: str, **data) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    async def _stream_body(run_id, entry, q):
        """SSE progress for a streaming submission. The run itself is a detached
        worker task (spawned in the handler), so this generator only OBSERVES it —
        a disconnect here can't stop, unslot, or lose the run."""
        yield _sse("received", run_id=run_id, team=entry["team"],
                   total=len(config.datasets))
        yield _sse("queued", ahead=sum(
            1 for r in runs.values()
            if r["id"] < run_id and r.get("status") in ("queued", "running")))
        while True:
            kind, payload = await q.get()
            if kind == "done":
                break
            if kind == "running":
                yield _sse("running")
            elif kind == "dataset":
                yield _sse("dataset", **payload)
        try:
            records = await entry["task"]
        except (FileNotFoundError, ValueError) as e:
            yield _sse("error", message=_redact_dataset_names(
                f"invalid submission: {e}", config))
            return
        except BaseException:  # noqa: BLE001
            yield _sse("error", message="internal error while running your submission")
            return
        yield _sse("result", **_build_payload(records, entry["team"]))

    @app.post("/submit")
    async def submit(team: str = Form(default=""), file: UploadFile = File(...),
                     ndif_api_key: str | None = Header(default=None,
                                                       alias="X-NDIF-API-Key"),
                     hf_token: str | None = Header(default=None,
                                                   alias="X-HF-Token"),
                     row_limit: str | None = Header(default=None,
                                                    alias="X-Aletheia-Limit"),
                     method_tag: str | None = Header(default=None,
                                                     alias="X-Aletheia-Tag"),
                     accept: str | None = Header(default=None, alias="Accept")):
        """Run + score a submission. Returns one JSON body by default; a client that
        sends ``Accept: text/event-stream`` gets a live SSE progress stream instead
        (same run, redacted progress). Clients that don't send it are unaffected."""
        if not config.datasets:
            raise HTTPException(503, "runner has no datasets configured")
        if backend is None:
            raise HTTPException(503, "no execution backend configured")
        if not ndif_api_key:
            raise HTTPException(400, "an NDIF API key is required (X-NDIF-API-Key)")

        # A method category is mandatory: every entry must declare white-box or
        # black-box. Normalize once here and reject up front (before any NDIF call or
        # rate-limit charge) if it's missing or unrecognized.
        method_tag = _normalize_tag(method_tag)
        if method_tag is None:
            raise HTTPException(
                400, "a method category is required: submit with --tag white "
                     "(white-box) or --tag black (black-box).")

        # Validate the key against NDIF and decide which key drives the run, before
        # binding a team or charging a rate-limit attempt. whoami returns the
        # account (or null email for a key NDIF doesn't recognise), or None when
        # NDIF is unreachable. Only keys with the usable tier can actually run
        # traces, so:
        #   - recognised + tier_1   -> the submitter's OWN key
        #   - recognised, no tier_1 -> the shared leaderboard key (HF secret)
        #   - NOT recognised        -> reject immediately (definitive: null email)
        #   - unknown (NDIF down)   -> shared leaderboard key, so a blip doesn't
        #                              block valid users (fail-open on transient)
        info = await run_in_threadpool(ndif.whoami, ndif_api_key, config.ndif_host)
        if info is not None and not ndif.is_recognized(info):
            raise HTTPException(400, "NDIF does not recognize this API key")
        if ndif.has_usable_tier(info):
            run_ndif_key = ndif_api_key
        elif config.leaderboard_ndif_api_key:
            run_ndif_key = config.leaderboard_ndif_api_key
        else:
            # No shared key configured: keep the submitter's key (their run will
            # fail at the first trace if they lack the tier, but we don't block).
            run_ndif_key = ndif_api_key

        limit = MAX_UPLOAD_MB * 1_000_000
        # Reject by declared size before doing anything else (and before pulling the
        # body into memory); re-check the actual bytes after (size may be unset).
        if file.size is not None and file.size > limit:
            raise HTTPException(413, f"submission exceeds {MAX_UPLOAD_MB} MB")

        # The NDIF key identifies the team: bound to a name on first submission,
        # remembered after (the key alone suffices on later submissions). This is a
        # read-modify-write of the registry object, so serialize it under the bucket
        # lock (and run it off the event loop).
        async with bucket_lock:
            team, err = await run_in_threadpool(registry.resolve, ndif_api_key, team)
        if err:
            raise HTTPException(400, err)

        data = await file.read()
        if len(data) > limit:
            raise HTTPException(413, f"submission exceeds {MAX_UPLOAD_MB} MB")

        when = datetime.datetime.now(datetime.timezone.utc)

        # Unpack + validate structure (free — no rate-limit charge). tmp isn't needed
        # after this; the run works off `data` in memory, so drop it immediately.
        tmp = Path(tempfile.mkdtemp(prefix="aletheia-upload-"))
        try:
            zpath = tmp / "submission.zip"
            zpath.write_bytes(data)
            try:
                await run_in_threadpool(pipeline.unpack, zpath, tmp / "unpacked")
                await run_in_threadpool(pipeline.validate_submission, tmp / "unpacked")
            except (FileNotFoundError, ValueError, zipfile.BadZipFile) as e:
                # A truncated / non-zip upload raises BadZipFile (not ValueError) —
                # reject it cleanly as a 400 instead of leaking a 500.
                raise HTTPException(400, f"invalid submission: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        # One in-flight submission per team: reject a duplicate rather than queue it
        # (doesn't cost a rate-limit attempt). Checked against live runs only.
        if any(r["team"] == team and r.get("status") in ("queued", "running")
               for r in runs.values()):
            raise HTTPException(
                409, "you already have a submission queued or running — wait for it "
                     "to finish before submitting again.")

        # Charge a rate-limit attempt right before we commit to running it.
        async with bucket_lock:
            allowed, retry = await run_in_threadpool(limiter.check_and_consume, team)
        if not allowed:
            raise HTTPException(
                429, f"rate limit reached: {limiter.max} submission(s) per "
                     f"{limiter.window / 3600:g}h. Try again in ~{retry}s.",
                headers={"Retry-After": str(retry)})

        # Archive the raw zip (every submission we run) for later retrieval.
        if archive is not None:
            try:
                await run_in_threadpool(archive.save, team, data, when)
            except Exception as e:  # noqa: BLE001
                print(f"[archive] failed to store submission for {team!r}: {e}",
                      file=sys.stderr, flush=True)

        extra_env = {"NDIF_API_KEY": run_ndif_key}
        if hf_token:
            extra_env["HF_TOKEN"] = hf_token
        if row_limit is not None:
            try:
                if int(row_limit) > 0:
                    extra_env["ALETHEIA_LIMIT"] = str(int(row_limit))
            except ValueError:
                pass

        # Build the run and spawn its worker HERE — a detached task that owns the
        # slot, the run, and persistence, so it completes and scores regardless of the
        # client. Teardown is the task's done-callback, so runs/pending are cleared
        # exactly when the run finishes (no leaked entries on disconnect).
        run_seq["n"] += 1
        run_id = run_seq["n"]
        run_key = uuid.uuid4().hex          # unique S3 prefix (survives restarts)
        canceller = backend.new_canceller()
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        entry = {"id": run_id, "team": team, "status": "queued",
                 "started_at": when.isoformat(), "dispatched_at": None,
                 "progress": None, "canceller": canceller, "task": None}

        def _on_progress(ev: dict) -> None:      # runs on the worker thread
            out = {"phase": ev.get("phase"), "index": ev.get("index"),
                   "total": ev.get("total"), "dataset": label(ev.get("dataset"))}
            if ev.get("phase") == "done":
                out["ok"] = bool(ev.get("ok"))
            entry["progress"] = {"index": ev.get("index"), "total": ev.get("total")}
            loop.call_soon_threadsafe(q.put_nowait, ("dataset", out))

        async def _worker():
            try:
                async with submit_slots:                 # slot held only while running
                    entry["status"] = "running"
                    entry["dispatched_at"] = datetime.datetime.now(
                        datetime.timezone.utc).isoformat()
                    loop.call_soon_threadsafe(q.put_nowait, ("running", None))
                    records = await run_in_threadpool(
                        _run_records, data, team, extra_env, _on_progress,
                        canceller, run_key)
                await _finalize(records, method_tag)      # persist regardless of client
                entry["status"] = "cancelled" if canceller.cancelled else "done"
                return records
            except BaseException:  # noqa: BLE001
                entry["status"] = "cancelled" if canceller.cancelled else "failed"
                raise
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        def _teardown(_task):
            try:
                _task.exception()   # retrieve (avoids "exception never retrieved" warnings)
            except BaseException:   # noqa: BLE001
                pass
            runs.pop(run_id, None)
            pending[team] = pending.get(team, 1) - 1
            if pending[team] <= 0:
                pending.pop(team, None)

        pending[team] = pending.get(team, 0) + 1
        runs[run_id] = entry
        task = asyncio.create_task(_worker())
        entry["task"] = task
        task.add_done_callback(_teardown)

        # Stream clients get live progress; others wait for one JSON body. Either way
        # the run is already detached and will finish + persist on its own.
        if accept and "text/event-stream" in accept.lower():
            return StreamingResponse(
                _stream_body(run_id, entry, q),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            records = await task
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, _redact_dataset_names(f"invalid submission: {e}", config))
        return _build_payload(records, team)

    return app


def _app_from_env() -> FastAPI:
    cfg_path = os.environ.get("RUNNER_CONFIG", "runner.yaml")
    if Path(cfg_path).exists():
        config = RunnerConfig.from_yaml(cfg_path)
    else:  # allow the app to import/boot even before a config is mounted
        config = RunnerConfig(datasets=[]).with_env_overrides()
    results_uri = os.environ.get("RESULTS_URI", config.results_uri)
    store = make_store(results_uri, token=config.hf_token)
    registry = TeamRegistry(config.teams_uri, token=config.hf_token)
    # Developer exempt list: explicit uri, else a sibling of the rate-limits file
    # (same bucket) so it works with no extra config.
    exempt_uri = config.rate_limit_exempt_uri
    if not exempt_uri and is_bucket_uri(config.rate_limits_uri):
        bucket_id, _ = parse_bucket_uri(config.rate_limits_uri, "rate_limits.json")
        exempt_uri = f"bucket://{bucket_id}/rate_limit_exempt.json"
    limiter = RateLimiter(config.rate_limits_uri, config.rate_limit_max,
                          config.rate_limit_window_hours * 3600,
                          token=config.hf_token, exempt_uri=exempt_uri or None)
    archive = SubmissionArchive(config.submissions_uri, token=config.hf_token)
    # Fargate execution when configured (Space variables), else in-process.
    # The Space ONLY dispatches to Fargate — it never runs a submission in-process.
    # Unconfigured -> no backend -> /submit returns 503 (rather than silently running
    # heavy work on the Space). (`submit.py --dry` still rehearses locally on the
    # participant's own machine; that path doesn't touch the Space.)
    from .fargate_backend import FargateBackend, FargateConfig
    fargate_cfg = FargateConfig.from_env()
    backend = FargateBackend(fargate_cfg) if fargate_cfg is not None else None
    return create_app(config, store, registry, limiter, archive, backend)


app = _app_from_env()
