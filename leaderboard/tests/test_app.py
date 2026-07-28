import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aletheia_runner.app import create_app as _create_app
from aletheia_runner.config import DatasetConfig, RunnerConfig, dataset_label
from aletheia_runner.registry import TeamRegistry
from aletheia_runner.results import ResultStore


class _InProcessBackend:
    """Test double: runs the pipeline in the TEST process so the web-layer tests can
    exercise a full submission without AWS. Production ships no in-process path — the
    Space only dispatches to Fargate — so this deliberately lives in the tests, not
    the package. Same ``run`` / ``new_canceller`` interface as the FargateBackend."""

    def run(self, config, run_key, zip_bytes, team, extra_env, on_progress=None, cancel=None):
        import tempfile

        from aletheia_runner import pipeline   # module attr so tests can monkeypatch run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "s.zip"
            zp.write_bytes(zip_bytes)
            root = pipeline.unpack(zp, Path(tmp) / "u")
            return pipeline.run_pipeline(root, team, config, extra_env, None,
                                         on_progress, cancel)

    def new_canceller(self):
        from aletheia_runner.sandbox import Canceller
        return Canceller()


def create_app(*args, backend=None, **kwargs):
    """Tests exercise the web layer against an in-process backend (production uses the
    FargateBackend). Pass an explicit ``backend=`` to override."""
    return _create_app(*args, backend=backend or _InProcessBackend(), **kwargs)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _stub_ndif_whoami(monkeypatch):
    """Treat every submitted key as a valid tier_1 NDIF account by default, so the
    submit tests never reach the network. Tier-specific tests override this."""
    from aletheia_runner import app as app_module
    monkeypatch.setattr(app_module.ndif, "whoami",
                        lambda key, host=None, **kw: {"email": "t@x", "tiers": ["tier_1"]})


def _zip_with_fixture() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("submission/fixture.ipynb",
                    (FIXTURES / "fixture.ipynb").read_text())
    return buf.getvalue()


def _post(client, *, team=None, key="key-1", tag="white"):
    data = {}
    if team is not None:
        data["team"] = team
    headers = {"X-NDIF-API-Key": key} if key else {}
    if tag:                                   # a method tag is mandatory; default valid
        headers["X-Aletheia-Tag"] = tag
    return client.post("/submit", data=data,
                       files={"file": ("s.zip", _zip_with_fixture(), "application/zip")},
                       headers=headers)


@pytest.fixture
def client(tmp_path):
    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))])
    store = ResultStore(str(tmp_path / "results.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    return TestClient(create_app(cfg, store, registry))


def test_submit_scores_and_appears_on_leaderboard(client):
    res = _post(client, team="team-a", key="key-a")
    assert res.status_code == 200, res.text
    body = res.json()
    # `scores` is keyed by notebook = mean AUROC (the primary metric).
    assert body["scores"]["fixture.ipynb"] == 1.0 and not body["failures"]
    nb = body["results"][0]
    assert nb["metrics"] == {"balanced_accuracy": 1.0, "auroc": 1.0, "recall": 1.0, "fpr": 0.0}
    assert nb["runtime_seconds"] is not None
    board = client.get("/api/leaderboard").json()
    row = board["results"][0]
    assert row["team"] == "team-a" and row["balanced_accuracy"] == 1.0
    # Per-dataset breakdown surfaced, but the real dataset name ("dummy") is
    # private — the response carries only the public codename.
    assert {d["dataset"] for d in row["datasets"]} == {dataset_label("dummy")}


def _post_tagged(client, *, team, key, tag):
    return client.post(
        "/submit", data={"team": team},
        files={"file": ("s.zip", _zip_with_fixture(), "application/zip")},
        headers={"X-NDIF-API-Key": key, "X-Aletheia-Tag": tag})


def test_method_tag_recorded_normalized_and_on_desk(client):
    # A recognized tag (any accepted spelling) normalizes to white/black and rides
    # through to the leaderboard row + the entrant's desk.
    assert _post_tagged(client, team="wb", key="key-w", tag="white-box").status_code == 200
    rows = client.get("/api/leaderboard").json()["results"]
    assert next(r for r in rows if r["team"] == "wb")["tag"] == "white"
    me = client.post("/api/me", headers={"X-NDIF-API-Key": "key-w"}).json()
    assert me["submissions"][0]["tag"] == "white"
    # An unrecognized tag is rejected (a valid category is mandatory), not silently
    # recorded as untagged.
    assert _post_tagged(client, team="ut", key="key-x", tag="green").status_code == 400


def test_method_tag_is_mandatory(client):
    # No tag -> 400 up front, before charging a rate-limit attempt.
    r = _post(client, team="team-a", key="key-a", tag=None)
    assert r.status_code == 400
    assert "category" in r.json()["detail"].lower()
    # The rejected attempt didn't consume the team's rate budget: a valid submission
    # still succeeds.
    assert _post(client, team="team-a", key="key-a", tag="white").status_code == 200


def test_admin_endpoints_disabled_without_token(client):
    # No admin_token configured on the fixture -> endpoints are hidden (404), even
    # when a token is supplied, so their existence isn't advertised.
    assert client.get("/admin/runs", headers={"X-Admin-Token": "x"}).status_code == 404
    assert client.post("/admin/cancel", headers={"X-Admin-Token": "x"}).status_code == 404


def _admin_client(tmp_path):
    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        admin_token="s3cr3t")
    store = ResultStore(str(tmp_path / "results.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    return TestClient(create_app(cfg, store, registry))


def test_admin_requires_valid_token(tmp_path):
    client = _admin_client(tmp_path)
    assert client.get("/admin/runs").status_code == 403                       # none
    assert client.get("/admin/runs",
                      headers={"X-Admin-Token": "nope"}).status_code == 403   # wrong
    r = client.get("/admin/runs", headers={"X-Admin-Token": "s3cr3t"})
    assert r.status_code == 200 and r.json() == {"runs": []}


def test_admin_cancel_unknown_run_is_noop(tmp_path):
    client = _admin_client(tmp_path)
    r = client.post("/admin/cancel?run_id=999", headers={"X-Admin-Token": "s3cr3t"})
    assert r.status_code == 200 and r.json() == {"cancelled": []}


def _parse_sse(lines):
    """Collect (event, data-dict) frames from an SSE line iterator."""
    events, ev, data = [], None, []
    for raw in lines:
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        if line == "":
            if ev is not None:
                events.append((ev, json.loads("\n".join(data)) if data else {}))
            ev, data = None, []
        elif line.startswith("event:"):
            ev = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].strip())
    if ev is not None:
        events.append((ev, json.loads("\n".join(data)) if data else {}))
    return events


def test_submit_streams_progress_when_accept_sse(client):
    with client.stream(
            "POST", "/submit", data={"team": "team-s"},
            files={"file": ("s.zip", _zip_with_fixture(), "application/zip")},
            headers={"X-NDIF-API-Key": "key-s", "X-Aletheia-Tag": "white",
                     "Accept": "text/event-stream"}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = _parse_sse(r.iter_lines())

    kinds = [e for e, _ in events]
    assert kinds[0] == "received"                       # lifecycle order
    assert "queued" in kinds and "running" in kinds
    assert "dataset" in kinds
    assert kinds[-1] == "result"

    payload = json.dumps(events)
    assert "dummy" not in payload                       # real dataset name never leaks
    for ev, data in events:
        if ev == "dataset":
            # codename only, ok/fail only — no scores in progress
            assert data["dataset"] == dataset_label("dummy")
            assert "metrics" not in data
            assert "balanced_accuracy" not in json.dumps(data)

    result = next(d for e, d in events if e == "result")
    assert result["scores"]["fixture.ipynb"] == 1.0     # terminal event still has scores
    # And it still landed on the leaderboard (persisted like the JSON path).
    board = client.get("/api/leaderboard").json()["results"]
    assert any(row["team"] == "team-s" for row in board)


def test_submit_without_accept_header_is_still_json(client):
    """Back-compat: a client that doesn't ask for a stream gets the single JSON body."""
    r = _post(client, team="team-j", key="key-j")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["scores"]["fixture.ipynb"] == 1.0


def test_leaderboard_flags_baseline_teams(tmp_path):
    """Rows whose team is in ``baseline_teams`` are tagged ``baseline: true`` (so the
    page can mark + de-rank them); other rows are ``baseline: false``."""
    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        baseline_teams=["Baselines"])
    store = ResultStore(str(tmp_path / "results.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    client = TestClient(create_app(cfg, store, registry))
    assert _post(client, team="Baselines", key="key-b").status_code == 200
    assert _post(client, team="Contender", key="key-c").status_code == 200
    rows = {r["team"]: r for r in client.get("/api/leaderboard").json()["results"]}
    assert rows["Baselines"]["baseline"] is True
    assert rows["Contender"]["baseline"] is False


def test_invalidations_loader(tmp_path):
    from aletheia_runner.app import _Invalidations
    f = tmp_path / "i.json"
    f.write_text(json.dumps([
        {"team": "A", "submitted_at": "2026-01-01T00:00:00", "reason": "r1"},
        {"team": "B"},                                   # no submitted_at -> ignored
        {"team": "C", "submitted_at": "2026-01-02T00:00:00"},   # reason optional -> ""
    ]))
    m = _Invalidations(str(f), None, ttl=0).get()
    assert m == {("A", "2026-01-01T00:00:00"): "r1", ("C", "2026-01-02T00:00:00"): ""}
    # missing file / bad json -> empty, never raises
    assert _Invalidations(str(tmp_path / "nope.json"), None, ttl=0).get() == {}


def test_leaderboard_marks_invalid_submission(tmp_path, monkeypatch):
    """A submission listed in the invalidations file is flagged ``invalid`` + reason;
    others aren't. (TTL 0 so the read isn't cached across the two fetches.)"""
    from aletheia_runner import app as app_module
    monkeypatch.setattr(app_module, "LEADERBOARD_CACHE_TTL", 0.0)
    inval = tmp_path / "inval.json"
    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        invalidations_uri=str(inval))
    store = ResultStore(str(tmp_path / "results.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    client = TestClient(create_app(cfg, store, registry))
    assert _post(client, team="team-dq", key="key-dq").status_code == 200

    row = next(r for r in client.get("/api/leaderboard").json()["results"]
               if r["team"] == "team-dq")
    assert not row.get("invalid")
    inval.write_text(json.dumps([{"team": "team-dq", "submitted_at": row["submitted_at"],
                                  "reason": "used a disallowed external API"}]))
    row2 = next(r for r in client.get("/api/leaderboard").json()["results"]
                if r["team"] == "team-dq")
    assert row2["invalid"] is True
    assert row2["invalid_reason"] == "used a disallowed external API"


def test_ndif_key_required(client):
    assert _post(client, team="team-a", key=None).status_code == 400


def test_new_key_requires_team(client):
    assert _post(client, team="  ", key="brand-new-key").status_code == 400


def test_key_remembers_team_and_team_is_unique(client):
    # First submit binds key-a -> team-a.
    assert _post(client, team="team-a", key="key-a").status_code == 200
    # Same key, no team -> remembered.
    r = _post(client, team=None, key="key-a")
    assert r.status_code == 200 and r.json()["team"] == "team-a"
    # A different key can't claim the same team name.
    assert _post(client, team="team-a", key="key-b").status_code == 400


def test_index_and_health(client):
    assert "Aletheia" in client.get("/").text
    # Health reports anonymized codenames, never the real dataset names.
    assert client.get("/api/health").json()["datasets"] == [dataset_label("dummy")]


def test_me_endpoint_reports_team_history_and_limit(client):
    assert _post(client, team="team-a", key="key-a").status_code == 200
    r = client.post("/api/me", headers={"X-NDIF-API-Key": "key-a"})
    assert r.status_code == 200
    d = r.json()
    assert d["registered"] and d["team"] == "team-a" and d["pending"] == 0
    assert d["rate_limit"]["enabled"] is False           # no limiter on the fixture
    assert len(d["submissions"]) == 1
    sub = d["submissions"][0]
    assert sub["ok"] and sub["metrics"]["balanced_accuracy"] == 1.0
    assert {x["dataset"] for x in sub["datasets"]} == {dataset_label("dummy")}
    # An unknown key is reported as unregistered (and isn't registered as a side effect).
    u = client.post("/api/me", headers={"X-NDIF-API-Key": "never-seen"})
    assert u.status_code == 200 and u.json()["registered"] is False


def test_me_requires_a_key(client):
    assert client.post("/api/me").status_code == 400


def test_submit_rejects_more_than_one_notebook(client):
    """Only one notebook per submission — a zip with two is rejected with a 400."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        nb = (FIXTURES / "fixture.ipynb").read_text()
        zf.writestr("submission/a.ipynb", nb)
        zf.writestr("submission/b.ipynb", nb)
    r = client.post("/submit", data={"team": "team-a"},
                    files={"file": ("s.zip", buf.getvalue(), "application/zip")},
                    headers={"X-NDIF-API-Key": "key-a", "X-Aletheia-Tag": "white"})
    assert r.status_code == 400
    assert "one notebook" in r.json()["detail"]


def test_real_dataset_name_never_leaves_the_server(tmp_path):
    """The real dataset name is private: it must appear in NO participant-facing
    response (submit / leaderboard / me / health), yet still be persisted under its
    real key for the organizers."""
    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))])
    store = ResultStore(str(tmp_path / "results.jsonl"))
    client = TestClient(create_app(cfg, store, TeamRegistry(str(tmp_path / "teams.json"))))

    submit = _post(client, team="team-a", key="key-a")
    assert submit.status_code == 200, submit.text
    leaderboard = client.get("/api/leaderboard")
    me = client.post("/api/me", headers={"X-NDIF-API-Key": "key-a"})
    health = client.get("/api/health")
    for resp in (submit, leaderboard, me, health):
        assert "dummy" not in resp.text, resp.text          # real name never escapes
        assert dataset_label("dummy") in resp.text          # public codename is used

    # ...but the persisted record keeps the real key (organizer-only).
    assert {r.dataset_key for r in store.all()} == {"dummy"}


def test_anonymize_covers_failed_dataset_and_bare_key():
    from aletheia_runner.app import _anonymize
    label = {"dummy": "Dataset 1"}.get

    ok = _anonymize({"datasets": [{"dataset": "dummy", "auroc": 1.0}]}, label)
    assert ok["datasets"] == [{"dataset": "Dataset 1", "auroc": 1.0}]

    failed = _anonymize({"notebook": "n.ipynb", "dataset": "dummy",
                         "failed_dataset": "dummy"}, label)
    assert failed["dataset"] == "Dataset 1" and failed["failed_dataset"] == "Dataset 1"

    # A successful summary has failed_dataset=None — left untouched.
    assert _anonymize({"failed_dataset": None}, label)["failed_dataset"] is None


def test_submission_zip_is_archived(tmp_path):
    """Every uploaded zip is stored (under the team) for later retrieval."""
    import glob

    from aletheia_runner.archive import SubmissionArchive

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))])
    store = ResultStore(str(tmp_path / "r.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    archive = SubmissionArchive(str(tmp_path / "archive"))
    client = TestClient(create_app(cfg, store, registry, archive=archive))

    assert _post(client, team="team-a", key="key-a").status_code == 200
    zips = glob.glob(str(tmp_path / "archive" / "**" / "*.zip"), recursive=True)
    assert len(zips) == 1 and "team-a" in zips[0]


def test_submit_503_when_no_datasets(tmp_path):
    app = create_app(RunnerConfig(datasets=[]),
                     ResultStore(str(tmp_path / "r.jsonl")),
                     TeamRegistry(str(tmp_path / "teams.json")))
    assert _post(TestClient(app), team="x", key="k").status_code == 503


def test_rate_limit_returns_429_after_max(tmp_path):
    """A team gets `max` submissions per window; the next is 429'd with Retry-After.
    A different team has its own budget."""
    from aletheia_runner.ratelimit import RateLimiter

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))])
    store = ResultStore(str(tmp_path / "r.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    limiter = RateLimiter(str(tmp_path / "rl.json"), max_submissions=2,
                          window_seconds=3600)
    client = TestClient(create_app(cfg, store, registry, limiter))

    assert _post(client, team="team-a", key="key-a").status_code == 200
    assert _post(client, team="team-a", key="key-a").status_code == 200
    blocked = _post(client, team="team-a", key="key-a")
    assert blocked.status_code == 429
    assert blocked.headers.get("Retry-After")
    assert _post(client, team="team-b", key="key-b").status_code == 200  # own budget


def test_rejected_submission_does_not_consume_an_attempt(tmp_path):
    """A structurally-invalid submission (more than one notebook) is rejected before
    the rate limit is touched — so it doesn't burn one of the team's attempts."""
    from aletheia_runner.ratelimit import RateLimiter

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))])
    store = ResultStore(str(tmp_path / "r.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    limiter = RateLimiter(str(tmp_path / "rl.json"), max_submissions=1,
                          window_seconds=3600)
    client = TestClient(create_app(cfg, store, registry, limiter))

    # Send the invalid (two-notebook) submission first; with only one attempt in the
    # window, it must NOT cost that attempt — it's rejected before the limiter runs.
    two_nbs = io.BytesIO()
    with zipfile.ZipFile(two_nbs, "w") as zf:
        nb = (FIXTURES / "fixture.ipynb").read_text()
        zf.writestr("submission/a.ipynb", nb)
        zf.writestr("submission/b.ipynb", nb)
    rejected = client.post("/submit", data={"team": "team-a"},
                           files={"file": ("s.zip", two_nbs.getvalue(), "application/zip")},
                           headers={"X-NDIF-API-Key": "key-a", "X-Aletheia-Tag": "white"})
    assert rejected.status_code == 400

    # The team's one attempt is still available: a valid submission now succeeds.
    assert _post(client, team="team-a", key="key-a").status_code == 200


def test_concurrent_submissions_bounded_and_all_recorded(tmp_path, monkeypatch):
    """Submissions run off the event loop, capped by the semaphore, and the
    bucket lock keeps concurrent result writes from clobbering each other."""
    import asyncio
    import threading
    import time

    import httpx

    from aletheia_runner import app as app_module
    from aletheia_runner.results import ResultRecord

    monkeypatch.setattr(app_module, "MAX_CONCURRENT_SUBMISSIONS", 2)

    state = {"active": 0, "peak": 0}
    slock = threading.Lock()

    def fake_run_pipeline(root, team, config, extra_env=None, on_submission_csv=None,
                          on_progress=None, cancel=None):
        with slock:                       # runs in a threadpool thread
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.25)                  # hold the slot so runs actually overlap
        with slock:
            state["active"] -= 1
        return [ResultRecord(team=team, notebook="n.ipynb", dataset_key="dummy",
                             metrics={"balanced_accuracy": 1.0}, ok=True)]

    monkeypatch.setattr(app_module.pipeline, "run_pipeline", fake_run_pipeline)

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))])
    store = ResultStore(str(tmp_path / "results.jsonl"))
    app = create_app(cfg, store, TeamRegistry(str(tmp_path / "teams.json")))

    async def fire(n):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            async def one(i):
                return await ac.post(
                    "/submit", data={"team": f"team-{i}"},
                    files={"file": ("s.zip", _zip_with_fixture(), "application/zip")},
                    headers={"X-NDIF-API-Key": f"key-{i}", "X-Aletheia-Tag": "white"})
            return await asyncio.gather(*[one(i) for i in range(n)])

    resps = asyncio.run(fire(4))
    assert all(r.status_code == 200 for r in resps), [r.text for r in resps]
    assert state["peak"] == 2          # semaphore caps concurrency (and runs overlapped)
    assert len(store.all()) == 4       # bucket lock: all four writes landed, none lost


_VALID_TIER1 = {"email": "u@x", "tiers": ["tier_1"]}
_VALID_UNTIERED = {"email": "u@x", "tiers": []}
_UNRECOGNIZED = {"email": None, "tiers": []}   # NDIF's 200 for an unknown key


def _client_capturing_ndif_key(tmp_path, monkeypatch, *, leaderboard_key=None,
                               whoami_result=_VALID_TIER1):
    """A client whose run_pipeline records the NDIF key threaded into the run.

    ``whoami_result`` stubs ``ndif.whoami`` (a payload dict, or ``None`` for an
    unreachable NDIF). Returns ``(client, captured)`` where ``captured["key"]`` is
    set to the NDIF key the run was given (or stays None if the run never ran)."""
    from aletheia_runner import app as app_module
    from aletheia_runner.results import ResultRecord

    captured: dict[str, str | None] = {"key": None}

    def fake_run_pipeline(root, team, config, extra_env=None, on_submission_csv=None,
                          on_progress=None, cancel=None):
        captured["key"] = (extra_env or {}).get("NDIF_API_KEY")
        return [ResultRecord(team=team, notebook="n.ipynb", dataset_key="dummy",
                             metrics={"balanced_accuracy": 1.0}, ok=True)]

    monkeypatch.setattr(app_module.pipeline, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(app_module.ndif, "whoami",
                        lambda key, host=None, **kw: whoami_result)

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        leaderboard_ndif_api_key=leaderboard_key)
    store = ResultStore(str(tmp_path / "results.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    return TestClient(create_app(cfg, store, registry)), captured


def test_untiered_key_runs_under_shared_leaderboard_key(tmp_path, monkeypatch):
    """A recognized key without tier_1 runs under the shared leaderboard key."""
    client, captured = _client_capturing_ndif_key(
        tmp_path, monkeypatch, leaderboard_key="LB-KEY", whoami_result=_VALID_UNTIERED)
    assert _post(client, team="team-a", key="user-key").status_code == 200
    assert captured["key"] == "LB-KEY"


def test_tiered_key_runs_under_submitter_key(tmp_path, monkeypatch):
    """A key WITH tier_1 runs under the submitter's own key, not the shared one."""
    client, captured = _client_capturing_ndif_key(
        tmp_path, monkeypatch, leaderboard_key="LB-KEY", whoami_result=_VALID_TIER1)
    assert _post(client, team="team-a", key="user-key").status_code == 200
    assert captured["key"] == "user-key"


def test_unrecognized_key_is_rejected(tmp_path, monkeypatch):
    """A key NDIF doesn't recognize (200 with null email) is rejected up front,
    before the run — no team bound, no attempt charged."""
    client, captured = _client_capturing_ndif_key(
        tmp_path, monkeypatch, leaderboard_key="LB-KEY", whoami_result=_UNRECOGNIZED)
    res = _post(client, team="team-a", key="bogus-key")
    assert res.status_code == 400, res.text
    assert captured["key"] is None              # the run never started
    me = client.post("/api/me", headers={"X-NDIF-API-Key": "bogus-key"}).json()
    assert me["team"] is None                   # key was not registered to a team


def test_unreachable_ndif_falls_back_to_shared_key(tmp_path, monkeypatch):
    """When whoami is unreachable (None), a transient blip doesn't block the
    submission — it runs under the shared leaderboard key."""
    client, captured = _client_capturing_ndif_key(
        tmp_path, monkeypatch, leaderboard_key="LB-KEY", whoami_result=None)
    assert _post(client, team="team-a", key="user-key").status_code == 200
    assert captured["key"] == "LB-KEY"


def test_untiered_without_shared_key_uses_submitter_key(tmp_path, monkeypatch):
    """No shared key configured + recognized-but-untiered: keep the submitter's
    own key (the run may fail later, but we don't block here)."""
    client, captured = _client_capturing_ndif_key(
        tmp_path, monkeypatch, leaderboard_key=None, whoami_result=_VALID_UNTIERED)
    assert _post(client, team="team-a", key="user-key").status_code == 200
    assert captured["key"] == "user-key"
