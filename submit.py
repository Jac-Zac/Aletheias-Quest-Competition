#!/usr/bin/env python3
"""Submit your competition entry to the Aletheia's Quest leaderboard Space.

This compresses the repository (everything except git/cache cruft) into a zip
and POSTs it to the leaderboard Space, which runs every notebook in
``submission/`` against the private eval data and returns your score.

Usage:
    python submit.py --team "my-team-name"

Configuration (flags override environment):
    --space-url   Leaderboard Space base URL   (env: ALETHEIA_SPACE_URL)
    --team        Team / display name          (env: ALETHEIA_TEAM)
    --root        Repo root to package          (default: this file's directory)
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import itertools
import json
import os
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("submit.py needs `requests` — install it with: pip install requests")

# Files/dirs we never ship to the runner ("leaderboard" is the runner's own code).
EXCLUDE_DIRS = {".git", ".claude", "leaderboard", "__pycache__",
                ".ipynb_checkpoints", ".venv", "venv", "node_modules",
                ".mypy_cache", ".pytest_cache"}
EXCLUDE_GLOBS = ["*.pyc", "*.pyo", ".DS_Store", "submission.csv"]

MAX_ZIP_MB = 200  # guardrail; the Space may enforce its own limit.

# The competition leaderboard Space; used when --space-url / $ALETHEIA_SPACE_URL is unset.
DEFAULT_SPACE_URL = "https://ndif-aletheias-leaderboard.hf.space"

# The runner builds its per-job venv from THIS interpreter, so --dry needs Python >= 3.10.
MIN_PYTHON = (3, 10)


# ── Tiny ANSI styling + spinner (no deps; off when piped or NO_COLOR is set) ──
_STYLE = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _STYLE else s


def _bold(s): return _c("1", s)
def _dim(s): return _c("2", s)
def _ital(s): return _c("3", s)
def _ox(s): return _c("38;5;124", s)        # oxblood
def _grey(s): return _c("38;5;245", s)
def _green(s): return _c("38;5;71", s)
def _gold(s): return _c("38;5;179", s)


_RULE = _ox("─" * 44)


def _banner() -> None:
    print()
    print("  " + _gold("✦") + "  " + _bold("ALETHEIA’S QUEST")
          + _grey("   ·   a ledger of deception detection"))
    print("  " + _dim(_ital("ἀλήθεια — "
                            "disclosure; the state of not being hidden")))
    print("  " + _RULE)


def _info(msg): print("  " + _grey("▸ ") + msg)
def _ok(msg):   print("  " + _green("✓ ") + msg)
def _bad(msg):  print("  " + _ox("✗ ") + msg)


def _shorten(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


class Spinner:
    """Braille spinner + elapsed timer shown around a blocking call. When output
    isn't a TTY it just prints one static line, so logs stay clean."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, text: str):
        self.text = text
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.time()

    def __enter__(self):
        if _STYLE:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            print("  • " + self.text + " …")
        return self

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            el = int(time.time() - self._t0)
            clock = f"{el // 60}m {el % 60:02d}s" if el >= 60 else f"{el}s"
            line = f"  {_ox(frame)}  {self.text}" + (_dim(f"   {clock}") if el >= 1 else "")
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
            time.sleep(0.09)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        if _STYLE:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    return any(fnmatch.fnmatch(rel.name, pat) for pat in EXCLUDE_GLOBS)


def build_zip(root: Path) -> bytes:
    """Package ``root`` into an in-memory zip, skipping cruft."""
    # One notebook per submission — catch it here so we don't upload a package the
    # runner will reject anyway (it enforces the same rule and returns a 400).
    nbs = sorted((root / "submission").glob("*.ipynb"))
    if len(nbs) > 1:
        names = ", ".join(p.name for p in nbs)
        sys.exit(f"submission/ must contain exactly one notebook — found "
                 f"{len(nbs)} ({names}). Submit one notebook at a time.")

    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if _excluded(rel):
                continue
            zf.write(path, rel.as_posix())
            n += 1
    data = buf.getvalue()
    size_mb = len(data) / 1e6
    if size_mb > MAX_ZIP_MB:
        sys.exit(f"package is {size_mb:.0f} MB (> {MAX_ZIP_MB} MB limit). "
                 "Remove large files (e.g. heavy weights) before submitting.")
    _ok(f"packaged {_bold(str(n))} files  {_dim('·')}  {_bold(f'{size_mb:.1f} MB')}")
    return data


def submit(space_url: str, team: str, payload: bytes,
           ndif_api_key: str | None, hf_token: str | None,
           limit: int | None = None, tag: str | None = None) -> None:
    url = space_url.rstrip("/") + "/submit"
    data = {"team": team or ""}   # empty -> the runner uses this key's registered team
    files = {"file": ("submission.zip", payload, "application/zip")}
    # Your NDIF key is passed through to your sandboxed run so nnsight can
    # authenticate remote traces (it reads NDIF_API_KEY from the environment).
    headers = {"X-NDIF-API-Key": ndif_api_key} if ndif_api_key else {}
    if limit is not None:
        # Forwarded into your run as ALETHEIA_LIMIT so your notebook can score only
        # the first N rows of each dataset (a fast partial run; see example.ipynb).
        headers["X-Aletheia-Limit"] = str(limit)
    if tag:
        # Tags this submission's method category (white-box / black-box) on the
        # leaderboard (badge + filter); recorded with the result, not used by your run.
        headers["X-Aletheia-Tag"] = tag
    if hf_token:
        # Authorization lets an org member reach a private Space (harmless if
        # public); X-HF-Token is forwarded into your run as HF_TOKEN so your
        # notebook can load gated HF models you have access to.
        headers["Authorization"] = f"Bearer {hf_token}"
        headers["X-HF-Token"] = hf_token
    # Ask for a live progress stream. A Space that predates streaming just ignores
    # this and returns the single JSON body — handled transparently below.
    headers["Accept"] = "text/event-stream"

    _info("entering the lists as " + _bold(team or "(remembered team)")
          + (_dim("   ·   ") + _bold(f"{tag}-box") if tag else "")
          + _dim("   →   " + url))

    err = None
    body = None
    with Spinner("running your submission  "
                 + _dim("venv · deps · notebooks · NDIF traces")) as sp:
        try:
            resp = requests.post(url, data=data, files=files, headers=headers,
                                 stream=True, timeout=None)   # no client timeout: a
                                                 # full eval run over all datasets is long
            ctype = resp.headers.get("Content-Type", "")
            if resp.status_code == 200 and "text/event-stream" in ctype:
                body = _consume_stream(resp, sp)     # live progress -> spinner
            elif resp.status_code == 200:
                body = resp.json()                   # older Space: one JSON body
        except requests.RequestException as e:
            resp, err = None, e
    if resp is None:
        _bad("could not reach the Space  " + _dim("— " + _shorten(str(err), 160)))
        sys.exit(1)

    if resp.status_code != 200:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except ValueError:
            pass
        _bad(_bold(f"submission rejected  [{resp.status_code}]"))
        print("     " + _dim(_shorten(detail, 200)))
        sys.exit(1)

    if body is None:
        _bad("the stream ended before a result arrived  "
             + _dim("— check the Space, or retry"))
        sys.exit(1)
    if body.get("_error"):
        _bad(_bold("submission failed"))
        print("     " + _dim(_shorten(body["_error"], 200)))
        sys.exit(1)
    _render_results("the ledger answers", body.get("results") or [],
                    body.get("failures") or [], body.get("message"))


def _consume_stream(resp, sp) -> dict | None:
    """Read the SSE progress stream, updating the spinner as events arrive. Returns
    the terminal ``result`` payload (dict), ``{"_error": msg}`` on an ``error`` event,
    or ``None`` if the stream ended without either."""
    event, chunks, result = None, [], None
    for raw in resp.iter_lines():                    # bytes; SSE frames end on a blank line
        line = raw.decode("utf-8", "replace") if raw else ""
        if line == "":
            if event is not None:
                try:
                    payload = json.loads("\n".join(chunks)) if chunks else {}
                except ValueError:
                    payload = {}
                result = _on_stream_event(event, payload, sp) or result
            event, chunks = None, []
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            chunks.append(line[len("data:"):].lstrip())
    return result


def _on_stream_event(event: str, data: dict, sp) -> dict | None:
    """Translate one SSE event into a spinner update; return a terminal body if this
    is the final ``result`` / ``error`` event, else None."""
    if event == "received":
        sp.text = "received  " + _dim("· queued")
    elif event == "queued":
        ahead = data.get("ahead") or 0
        sp.text = "queued  " + _dim(f"· {ahead} ahead in line" if ahead else "· next up")
    elif event == "running":
        sp.text = "running  " + _dim("venv · deps · notebooks · NDIF traces")
    elif event == "dataset":
        i, n, ds = data.get("index"), data.get("total"), data.get("dataset", "?")
        mark = ("" if data.get("phase") != "done"
                else (_ok_mark() if data.get("ok") else _fail_mark()))
        sp.text = f"running  {_bold(f'{i}/{n}')}  {ds}{mark}"
    elif event == "result":
        return data
    elif event == "error":
        return {"_error": data.get("message", "submission failed")}
    return None


def _ok_mark() -> str:
    return _dim(" ✓")


def _fail_mark() -> str:
    return _dim(" ✗")


# Metric keys -> short labels, in display order (AUROC is the ranking metric,
# balanced accuracy breaks ties).
_METRICS = [("auroc", "AUROC"), ("balanced_accuracy", "Bal.Acc"),
            ("recall", "Recall"), ("fpr", "FPR")]


def _fmt_metric(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def _fmt_dur(s) -> str:
    if s is None:
        return "—"
    s = int(round(s)); m, ss = divmod(s, 60)
    return f"{m}m {ss}s" if m else f"{ss}s"


def _print_notebook(item: dict) -> None:
    """Print one notebook's mean metrics + per-dataset breakdown + runtime."""
    name = item.get("notebook") or "?"
    rt = item.get("runtime_seconds")
    _ok(f"{_bold(name)}" + (f"   {_dim('runtime ' + _fmt_dur(rt))}" if rt is not None else ""))
    m = item.get("metrics") or {}
    print("       " + "   ".join(
        f"{label} {_gold(_fmt_metric(m.get(key)))}" for key, label in _METRICS))
    for d in item.get("datasets") or []:
        ds = str(d.get("dataset", "")).split("/")[-1]
        cells = "  ".join(f"{label} {_fmt_metric(d.get(key))}" for key, label in _METRICS)
        print("         " + _dim(f"{ds:<26} {cells}"))


def _render_results(title: str, results: list, failures: list, message: str | None) -> None:
    print()
    print("  " + _ox(title))
    print("  " + _RULE)
    for item in results:
        if item.get("ok"):
            _print_notebook(item)
    seen = []
    for f in failures:
        ds = f.get("dataset")
        hdr = _bold(f.get("notebook") or "?")
        if ds:
            hdr += _dim("  — failed on " + str(ds).split("/")[-1])
        _bad(hdr)
        msg = (f.get("error") or "").strip()
        if msg and msg not in seen:
            seen.append(msg)
            print("       " + _dim(_shorten(msg, 180)))
    if not results and not failures:
        print("  " + _grey("no notebooks ran"))
    if message:
        print("  " + _RULE)
        print("  " + _grey(message))


def _resolve_hf_token(arg_token: str | None) -> str | None:
    """--hf-token / $HF_TOKEN, else the cached huggingface-cli login."""
    if arg_token:
        return arg_token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _clean_key(v: str | None) -> str | None:
    """A key value, or None if ``v`` is absent/blank or a stringified null.

    Guards the common footguns: ``$NDIF_API_KEY`` exported-but-empty, or set to
    the literal string ``None``/``null`` (e.g. from ``export NDIF_API_KEY=$(cmd
    that printed None)``). Sending those verbatim passes the server's "key
    present?" check but hangs its whoami lookup, so we drop them here."""
    if v is None:
        return None
    v = v.strip()
    if not v or v.lower() in ("none", "null", "nil"):
        return None
    return v


def _resolve_ndif_key(arg_key: str | None) -> str | None:
    """--ndif-api-key / $NDIF_API_KEY, else the key nnsight already has saved
    (CONFIG.API.APIKEY, e.g. from `CONFIG.set_default_api_key(...)`)."""
    key = _clean_key(arg_key)
    if key:
        return key
    try:
        from nnsight import CONFIG
        return _clean_key(CONFIG.API.APIKEY)
    except Exception:
        return None


def run_dry(root: Path, ndif_api_key: str | None, hf_token: str | None,
            limit: int | None = None) -> None:
    """Rehearse the submission locally via the leaderboard runner, no upload."""
    if sys.version_info[:2] < MIN_PYTHON:
        sys.exit(
            f"--dry needs Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} (the runner builds its "
            f"per-job venv from this interpreter); you're on "
            f"{sys.version_info.major}.{sys.version_info.minor}. Use a newer Python "
            f"(e.g. `./setup_dev.sh` creates a 3.10+ venv).")
    runner_src = root / "leaderboard" / "src"
    if not runner_src.is_dir():
        sys.exit("--dry needs the bundled runner at leaderboard/ (run from the repo root).")
    sys.path.insert(0, str(runner_src))
    try:
        from aletheia_runner.dryrun import dry_run
    except ImportError as e:
        sys.exit(f"--dry needs the runner's deps (datasets, nbclient, ipykernel, "
                 f"scikit-learn, pyyaml): {e}")

    _info("rehearsing locally  "
          + _dim("venv · requirements.txt · datasets from dry.yaml"))
    print("  " + _dim("  server-side Landlock/seccomp/egress is skipped here; "
                      "it runs on the Space"))

    # Live per-dataset progress: a spinner (with elapsed timer) while each dataset
    # traces on NDIF, then a one-line result as it finishes — so a slow run reads as
    # progress, not a hang. (--dry runs locally on public data, so showing the real
    # dataset name here is fine.)
    state = {"spin": None}

    def _progress(ev):
        ds = str(ev.get("dataset", "")).split("/")[-1]
        tag = _dim(f"[{ev.get('index')}/{ev.get('total')}]")
        if ev["phase"] == "start":
            sp = Spinner(f"{tag} {ds}  " + _dim("running"))
            sp.__enter__()
            state["spin"] = sp
            return
        if state["spin"]:
            state["spin"].__exit__()
            state["spin"] = None
        if ev.get("ok"):
            m = ev.get("metrics") or {}
            cells = "   ".join(f"{lbl} {_gold(_fmt_metric(m.get(k)))}" for k, lbl in _METRICS)
            _ok(f"{tag} {_bold(ds)}   {cells}")
        else:
            _bad(f"{tag} {_bold(ds)}   " + _dim(_shorten(ev.get("error") or "failed", 90)))

    try:
        records = dry_run(root, ndif_api_key, hf_token, limit=limit, on_progress=_progress)
    except (FileNotFoundError, ValueError) as e:
        # e.g. no/too-many notebooks, or a bad dry.yaml — a clear message beats
        # a traceback. (The Space reports the same rejection as a 400.)
        if state["spin"]:
            state["spin"].__exit__()
        records, err = None, e
    if records is None:
        _bad(_shorten(str(err), 200))
        sys.exit(1)

    # Group the per-(notebook, dataset) records into one summary per notebook.
    from collections import defaultdict
    from aletheia_runner.results import summarize_submission
    by_nb: dict[str, list] = defaultdict(list)
    for r in records:
        by_nb[r.notebook].append(r)

    print()
    print("  " + _ox("dry-run results"))
    print("  " + _RULE)
    ok = 0
    for nb, recs in by_nb.items():
        name = nb.split("/")[-1]
        summ = summarize_submission(recs)
        if summ["ok"]:
            ok += 1
            _print_notebook({"notebook": name, **summ})
        else:
            ds = summ.get("failed_dataset")
            _bad(_bold(name) + (_dim("  — failed on " + str(ds).split("/")[-1]) if ds else ""))
            # --dry shows the real error (public data, local) — print it in full.
            for ln in (summ.get("error") or "").strip().splitlines() or ["(no error captured)"]:
                print("       " + _dim(ln))
    print("  " + _RULE)
    if not ok:
        _bad("no notebook produced a valid submission.csv — fix the errors above")
        sys.exit(1)
    _ok(f"{_bold(f'{ok}/{len(by_nb)}')} run(s) OK  "
        + _dim("· submit for real by dropping --dry"))


def main(argv: list[str] | None = None) -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space-url",
                   default=os.environ.get("ALETHEIA_SPACE_URL", DEFAULT_SPACE_URL),
                   help=f"leaderboard Space base URL (default: $ALETHEIA_SPACE_URL "
                        f"or {DEFAULT_SPACE_URL})")
    p.add_argument("--team", default=os.environ.get("ALETHEIA_TEAM"),
                   help="team name — required only on your NDIF key's FIRST "
                        "submission, then remembered (default: $ALETHEIA_TEAM)")
    p.add_argument("--root", default=str(here), type=Path)
    p.add_argument("--ndif-api-key", default=os.environ.get("NDIF_API_KEY"),
                   help="your NDIF API key — required (default: $NDIF_API_KEY, else "
                        "the key nnsight has saved via CONFIG.set_default_api_key)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HF token to reach a private Space (default: $HF_TOKEN "
                        "or your cached huggingface-cli login)")
    p.add_argument("--dry", action="store_true",
                   help="run the full pipeline locally against a public dataset to "
                        "verify your submission works — no upload")
    p.add_argument("--tag", choices=["white", "black"], default=os.environ.get("ALETHEIA_TAG"),
                   help="tag your method's category on the leaderboard: 'white' (white-box "
                        "— uses activations/weights) or 'black' (black-box — query-only). "
                        "Shows a badge and is filterable; default: $ALETHEIA_TAG, else untagged.")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N rows of each dataset (forwarded to "
                        "your notebook as $ALETHEIA_LIMIT; see example.ipynb). Handy "
                        "for a fast partial --dry; omit to score every row.")
    args = p.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be a positive integer")
    if not args.dry and not args.tag:
        p.error("--tag is required for a real submission — declare your method "
                "category: --tag white (white-box) or --tag black (black-box)")

    args.ndif_api_key = _resolve_ndif_key(args.ndif_api_key)
    if not args.ndif_api_key:
        p.error("--ndif-api-key (or $NDIF_API_KEY, or a key saved in nnsight's "
                "CONFIG) is required")
    # An NDIF key is a UUID. Reject anything else here rather than sending it: a
    # malformed key sails past the Space's "key present?" check and then stalls
    # its whoami lookup, which — with no client timeout — looks like a hung run.
    try:
        uuid.UUID(str(args.ndif_api_key))
    except (ValueError, AttributeError, TypeError):
        p.error(f"NDIF API key {args.ndif_api_key!r} is not a valid key "
                "(expected a UUID like 1234abcd-56ef-...). Check --ndif-api-key "
                "/ $NDIF_API_KEY / the key saved in nnsight's CONFIG.")

    _banner()
    hf_token = _resolve_hf_token(args.hf_token)

    if args.dry:
        run_dry(Path(args.root).resolve(), args.ndif_api_key, hf_token,
                limit=args.limit)
        return

    if not args.space_url:
        p.error("--space-url (or ALETHEIA_SPACE_URL) is required")
    # --team is required only on this key's first submission; the runner enforces it.

    payload = build_zip(Path(args.root).resolve())
    submit(args.space_url, args.team, payload, args.ndif_api_key, hf_token,
           limit=args.limit, tag=args.tag)


if __name__ == "__main__":
    main()
