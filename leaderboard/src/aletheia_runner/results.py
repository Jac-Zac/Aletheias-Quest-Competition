"""Result records and persistence.

Two backends behind a common interface:
- ``LocalResultStore`` — append-only JSONL on disk (default; used by tests).
- ``BucketResultStore`` — read-modify-write a JSONL object in a HF bucket.

``make_store(uri, token)`` picks one: a ``bucket://org/name/path`` uri selects
the bucket backend, anything else is a local path. The leaderboard logic lives
on the base class, so it's identical for both.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import (METRIC_KEYS, PRIMARY_METRIC, SECONDARY_METRIC,
                     dataset_model_lora, dataset_split)


def is_bucket_uri(uri: str) -> bool:
    return uri.startswith("bucket://")


def parse_bucket_uri(uri: str, default_path: str) -> tuple[str, str]:
    """Split ``bucket://org/name[/path]`` into ``(org/name, path)``.

    ``path`` falls back to ``default_path`` when the uri names only the bucket.
    Raises ``ValueError`` if the org/name pair is missing.
    """
    parts = uri[len("bucket://"):].split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"bucket uri must be bucket://org/name[/path], got {uri!r}")
    return "/".join(parts[:2]), "/".join(parts[2:]) or default_path


@dataclass
class ResultRecord:
    team: str
    notebook: str
    dataset_key: str
    # Per-dataset metrics (``METRIC_KEYS`` -> value, or None when undefined). Empty
    # on a failed run.
    metrics: dict[str, float | None] = field(default_factory=dict)
    ok: bool = False
    # Participant-facing message: generic for sandboxed execution failures (the raw
    # error can echo the private inputs), the real message for format/scoring errors
    # (which describe the participant's own submission.csv).
    error: str | None = None
    # Organizer-only: the FULL real error/traceback for a redacted execution failure.
    # Persisted to the bucket (so failures can be diagnosed from S3) but never returned
    # to participants. None when ``error`` already holds the real error (no redaction).
    error_detail: str | None = None
    submitted_at: str | None = None  # ISO timestamp, stamped by the caller
    # Wall-clock seconds for the whole submission (same value across its records).
    runtime_seconds: float | None = None
    # Method category the participant tagged this submission with: "white" (uses
    # activations/weights) or "black" (query-only), or None if untagged. Same value
    # across a submission's records; surfaced on the leaderboard for the badge/filter.
    tag: str | None = None


def _parse_jsonl(text: str) -> list[ResultRecord]:
    """Parse a results.jsonl. Lines that don't match the current schema (e.g. an
    older format) are skipped rather than crashing the whole read."""
    fields = set(ResultRecord.__dataclass_fields__)
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            out.append(ResultRecord(**{k: v for k, v in data.items() if k in fields}))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def _to_jsonl(records: list[ResultRecord]) -> str:
    return "".join(json.dumps(asdict(r)) + "\n" for r in records)


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rank_key(row: dict) -> tuple[float, float]:
    """Sort key for a leaderboard row: primary metric, secondary as the tie-break.

    An undefined metric (``None`` — AUROC on a single-class dataset) sorts last in
    a descending sort rather than raising on the comparison."""
    def val(key: str) -> float:
        v = row.get(key)
        return float("-inf") if v is None else v

    return (val(PRIMARY_METRIC), val(SECONDARY_METRIC))


def summarize_submission(recs: list["ResultRecord"]) -> dict:
    """Summarize one submission's per-dataset records (all sharing a team/notebook/
    submitted_at). **All-or-nothing**: the submission counts only if *every* dataset
    scored (the runner fails fast on the first failure). When ``ok``, reports the
    headline mean of each metric plus the per-dataset breakdown; otherwise the
    metrics are ``None`` and ``failed_dataset``/``error`` describe the failure.
    Runtime is always reported.

    **The headline mean (what the leaderboard shows and ranks on) averages only the
    VALIDATION datasets.** Every dataset is still scored and surfaced in the
    breakdown (each row carries ``counted``: whether it feeds the mean), but
    ``dev-test`` rows don't move the headline number. If a submission has no
    validation datasets at all (e.g. a local ``--dry`` on dev data), the mean falls
    back to averaging every dataset so the rehearsal still reports a score."""
    runtime = next((r.runtime_seconds for r in recs
                    if r.runtime_seconds is not None), None)
    tag = next((r.tag for r in recs if getattr(r, "tag", None)), None)
    failed = next((r for r in recs if not r.ok), None)
    ok = bool(recs) and failed is None
    if not ok:
        return {"ok": False, "metrics": {k: None for k in METRIC_KEYS},
                "datasets": [], "runtime_seconds": runtime, "tag": tag,
                "failed_dataset": failed.dataset_key if failed else None,
                "error": failed.error if failed else None}
    # The headline mean averages validation datasets only; fall back to ALL when a
    # submission has no validation datasets (dev-only --dry rehearsals, tests).
    has_validation = any(dataset_split(r.dataset_key) == "validation" for r in recs)

    def _counts(r) -> bool:
        return dataset_split(r.dataset_key) == "validation" if has_validation else True

    datasets = []
    for r in sorted(recs, key=lambda r: r.dataset_key):
        model_id, lora_id = dataset_model_lora(r.dataset_key)
        datasets.append({"dataset": r.dataset_key, "model_id": model_id, "lora_id": lora_id,
                         "counted": _counts(r),
                         **{k: r.metrics.get(k) for k in METRIC_KEYS}})
    counted_recs = [r for r in recs if _counts(r)]
    metrics = {k: _mean([r.metrics.get(k) for r in counted_recs]) for k in METRIC_KEYS}
    return {"ok": True, "metrics": metrics, "datasets": datasets,
            "runtime_seconds": runtime, "tag": tag,
            "failed_dataset": None, "error": None}


class BaseResultStore:
    def all(self) -> list[ResultRecord]:  # pragma: no cover - interface
        raise NotImplementedError

    def append(self, records: list[ResultRecord]) -> None:  # pragma: no cover
        raise NotImplementedError

    def leaderboard(self) -> list[dict]:
        """Best submission per (team, notebook), ordered by mean AUROC.

        Each notebook is scored against every dataset; a submission's headline
        numbers are the **mean across datasets** of each metric. The per-dataset
        breakdown (now surfaced, with dataset names) and total runtime ride along
        for the click-to-expand view. Resubmitting updates a row only if the new
        submission's primary metric (AUROC) is better, with balanced accuracy
        breaking ties; ``submitted_at`` reflects when that best was achieved.

        Unlike balanced accuracy, AUROC is **undefined** on a single-class dataset
        (``None``), so a scored submission can land here without one: it keeps its
        row and sorts last rather than dropping off the board.
        """
        from collections import defaultdict

        # A submission's runs share one ``submitted_at`` stamp, so (team, notebook,
        # stamp) identifies it.
        subs: dict[tuple, list[ResultRecord]] = defaultdict(list)
        for r in self.all():
            subs[(r.team, r.notebook, r.submitted_at)].append(r)

        best: dict[tuple, tuple[tuple, dict]] = {}     # (team, notebook) -> (order, row)
        for (team, notebook, stamp), recs in subs.items():
            summ = summarize_submission(recs)
            if not summ["ok"]:                         # not every dataset scored
                continue
            row = {"team": team, "notebook": notebook, "submitted_at": stamp,
                   PRIMARY_METRIC: summ["metrics"].get(PRIMARY_METRIC),
                   SECONDARY_METRIC: summ["metrics"].get(SECONDARY_METRIC),
                   "metrics": summ["metrics"], "datasets": summ["datasets"],
                   "runtime_seconds": summ["runtime_seconds"], "tag": summ.get("tag")}
            tn = (team, notebook)
            order = _rank_key(row)
            if tn not in best or order > best[tn][0]:
                best[tn] = (order, row)

        rows = [row for _, row in best.values()]
        rows.sort(key=_rank_key, reverse=True)
        return rows


class LocalResultStore(BaseResultStore):
    """Append-only JSONL on the local filesystem."""

    def __init__(self, uri: str):
        self.uri = uri

    def append(self, records: list[ResultRecord]) -> None:
        path = Path(self.uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(_to_jsonl(records))

    def all(self) -> list[ResultRecord]:
        path = Path(self.uri)
        return _parse_jsonl(path.read_text()) if path.exists() else []


class BucketResultStore(BaseResultStore):
    """JSONL object inside a HuggingFace bucket (read-modify-write).

    Suitable for low submission volume — each append downloads the current
    object, appends, and re-uploads. ``bucket_id`` is ``org/bucket-name``.
    """

    def __init__(self, bucket_id: str, remote_path: str = "results.jsonl",
                 token: str | None = None):
        self.bucket_id = bucket_id
        self.remote_path = remote_path
        self.token = token

    def _download_text(self) -> str:
        from huggingface_hub import download_bucket_files

        with tempfile.TemporaryDirectory(prefix="aletheia-bucket-") as tmp:
            local = Path(tmp) / "results.jsonl"
            download_bucket_files(
                self.bucket_id, files=[(self.remote_path, str(local))],
                raise_on_missing_files=False, token=self.token)
            return local.read_text() if local.exists() else ""

    def all(self) -> list[ResultRecord]:
        return _parse_jsonl(self._download_text())

    def append(self, records: list[ResultRecord]) -> None:
        from huggingface_hub import batch_bucket_files

        existing = self._download_text()
        payload = (existing + _to_jsonl(records)).encode("utf-8")
        batch_bucket_files(
            self.bucket_id, add=[(payload, self.remote_path)], token=self.token)


def make_store(uri: str, token: str | None = None) -> BaseResultStore:
    """Build a store from a uri. ``bucket://org/name/path`` → bucket; else local."""
    if is_bucket_uri(uri):
        bucket_id, remote_path = parse_bucket_uri(uri, "results.jsonl")
        return BucketResultStore(bucket_id, remote_path, token=token)
    return LocalResultStore(uri)


# Backwards-compatible alias (the local store was the original name).
ResultStore = LocalResultStore
