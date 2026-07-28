"""Runner configuration: which datasets to score on, where labels/results live.

Loadable from a YAML file (production) or constructed directly (tests). Kept free
of HuggingFace/Space concerns so the core can be exercised locally.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


# Every dataset (inputs and labels) is a single-split repo standardized to this
# split — no per-dataset config subsets, no split selection.
SPLIT = "test"

# Public, private-safe codenames for datasets — lesser-known Greek deities. The
# real dataset names are private and must never appear in a response; instead a
# dataset shows as "Dataset <codename>". A key is mapped by a stable hash, so the
# SAME dataset always gets the SAME codename — consistent across submissions and
# even for keys not in the current runner config. (The pool is large enough that
# the competition's datasets don't collide; verified against the live key set.)
_DATASET_CODENAMES = (
    "Asteria", "Eurynome", "Menoetius", "Crius", "Coeus", "Pallas", "Astraeus",
    "Perses", "Ophion", "Eos", "Selene", "Metis", "Tethys", "Phoebe", "Theia",
    "Iapetus", "Hyperion", "Mnemosyne", "Themis", "Dione", "Leto", "Nyx", "Erebus",
    "Aether", "Hemera", "Thalassa", "Pontus", "Ananke", "Hecate", "Nemesis", "Eris",
    "Iris", "Hebe", "Enyo", "Deimos", "Phobos", "Tyche", "Nike", "Bia", "Kratos",
    "Zelus", "Styx", "Triton", "Nereus", "Proteus", "Glaucus", "Phorcys", "Ceto",
    "Palaemon", "Amphitrite", "Doris", "Electra", "Maia", "Taygete", "Alcyone",
    "Celaeno", "Sterope", "Hesperus", "Notus", "Boreas", "Eurus", "Zephyrus",
    "Aeolus", "Aristaeus", "Carpo", "Thallo", "Auxo", "Eunomia", "Dike", "Eirene",
    "Clotho", "Lachesis", "Atropos", "Aglaea", "Euphrosyne", "Hypnos", "Thanatos",
    "Momus", "Geras", "Oizys", "Apate", "Dolos", "Moros", "Ker", "Eosphorus",
    "Triptolemus", "Asterope", "Harmonia", "Peitho",
)


# Base models behind the datasets, as the token appears in a dataset id -> full HF id.
# Used to split a dataset name into <split>-<task>-<base>-<lora> (see ``_parse_name``).
_BASE_MODELS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}
_SPLIT_PREFIXES = ("dev-test-", "validation-", "dev-")   # longest first


def _parse_name(name: str) -> tuple[str, str, str | None, str | None]:
    """Split a dataset id into ``(split, task, model_id, lora_id)``.

    Datasets are named ``<org>/<split>-<task>-<base>[-<lora>]`` (e.g.
    ``…/validation-soft-trigger-gemma-3-27b-it-gemma-3-27b-it-lora-greeting``). We
    strip the split prefix, then cut at the (earliest) known base-model token: what's
    before it is the **task** (scenario), the token gives the **model id**, and what's
    after is the **lora id** (``None`` if absent). Falls back to ``(split, rest, None,
    None)`` for ids that don't match the convention (e.g. test fixtures)."""
    s = str(name).split("/")[-1]
    split = ""
    for p in _SPLIT_PREFIXES:
        if s.startswith(p):
            split, s = p.rstrip("-"), s[len(p):]
            break
    best = None
    for tok, full in _BASE_MODELS.items():
        idx = s.find(tok)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, tok, full)
    if best is None:
        return split, (s.strip("-") or s), None, None
    idx, tok, full = best
    task = s[:idx].strip("-")
    rest = s[idx + len(tok):].strip("-")
    lora = rest if rest and rest.lower() != "none" else None
    return split, (task or s), full, lora


def dataset_split(name: str) -> str:
    """The split a dataset belongs to (``"dev-test"``, ``"validation"``, ``"dev"``,
    or ``""`` if unrecognized) — used to decide which datasets count toward the
    leaderboard's headline mean / ranking (validation only)."""
    return _parse_name(name)[0]


def dataset_task(name: str) -> str:
    """The task/scenario a dataset belongs to (e.g. ``soft-trigger``), used to GROUP
    datasets under one codename — multiple datasets for the same task (different
    models/loras) share a name."""
    return _parse_name(name)[1]


def dataset_model_lora(name: str) -> tuple[str | None, str | None]:
    """``(model_id, lora_id)`` a dataset's conversations were generated with, parsed
    from its id. These are public (shown on the leaderboard breakdown); ``lora_id`` is
    ``None`` for a plain base model."""
    _, _, model_id, lora_id = _parse_name(name)
    return model_id, lora_id


def dataset_label(key: str) -> str:
    """Public, private-safe codename for a dataset — ``"Dataset <Deity>"``.

    Grouped **by task**: the codename is a stable hash of the dataset's *task*
    (scenario), so every dataset for the same task (across models/loras) shows the
    same name; the model/lora columns in the breakdown disambiguate them. The real
    name never leaves the server."""
    digest = int(hashlib.sha256(dataset_task(key).encode("utf-8")).hexdigest(), 16)
    return f"Dataset {_DATASET_CODENAMES[digest % len(_DATASET_CODENAMES)]}"

# Metrics computed per dataset, in display order. ``auroc`` is the primary metric
# (leaderboard ordering); ``balanced_accuracy`` is shown secondarily and breaks
# ties. Defined here (dependency-light) so scoring, the result store, and the API
# all share them.
METRIC_KEYS = ("auroc", "balanced_accuracy", "recall", "fpr")
PRIMARY_METRIC = "auroc"
SECONDARY_METRIC = "balanced_accuracy"


@dataclass(frozen=True)
class DatasetConfig:
    """One entry in the runner's inner loop: a dataset to predict on plus where
    its (private, held-out) labels live.

    One repo == one dataset (single ``test`` split, no config subsets).
    ``labels_uri`` may be an HF dataset id or a local ``.csv`` path; the labels
    table must have an id column and a binary label column.
    """

    name: str
    labels_uri: str
    id_column: str = "id"
    label_column: str = "deceptive"

    @property
    def key(self) -> str:
        """Stable identifier used in result records / the leaderboard."""
        return self.name

    def env(self) -> dict[str, str]:
        """Environment the notebook reads to know what to predict on."""
        return {"DATASET_NAME": self.name}


@dataclass(frozen=True)
class RunnerConfig:
    datasets: list[DatasetConfig]
    results_uri: str = "results.jsonl"
    teams_uri: str = "teams.json"   # NDIF-key -> team registry (bucket:// or local)
    submissions_uri: str = "submissions"   # archive of every uploaded zip (bucket:// or local dir)
    # Per-team submission rate limit (fixed window). max <= 0 disables it.
    rate_limits_uri: str = "rate_limits.json"
    rate_limit_max: int = 0
    rate_limit_window_hours: float = 0.0
    # Teams exempt from the rate limit (developers): a JSON list of team names in a
    # separate, read-only bucket file the app never writes, so it can be edited by
    # hand at any time. Empty -> derived as ``rate_limit_exempt.json`` next to
    # ``rate_limits_uri`` (so it lives in the same bucket by default).
    rate_limit_exempt_uri: str = ""
    # Disqualified submissions: a JSON list the app only READS (edited by hand),
    # each entry ``{"team", "submitted_at", "reason"[, "notebook"]}``. A matching
    # leaderboard row is shown muted, left out of the ranking, and carries the
    # reason. Empty -> derived as ``invalidations.json`` next to ``results_uri``.
    invalidations_uri: str = ""
    # The single timeout: wall-clock seconds allowed per (notebook, dataset) run.
    # Each run gets a fresh budget (e.g. 1800 = 30 min each for two datasets). The
    # sandbox kills the run's process group when it's exceeded; there is no separate
    # per-cell timeout.
    notebook_timeout: int = 1800
    # Overall wall-clock budget (seconds) for a whole submission — a backstop ABOVE
    # the per-(notebook, dataset) ``notebook_timeout``. The submission (serial) run
    # slot is shared across all teams, so a run that the per-run kill can't reclaim,
    # or one with many datasets, must still release the slot in bounded time: the
    # pipeline clamps each run to the remaining budget and stops once it's spent.
    # 0 disables it (unbounded, the historical behavior).
    submission_timeout: int = 0
    # Sandboxed execution (Landlock + seccomp + rlimits + predownloaded RO data).
    sandbox: bool = False
    confine: bool = True            # apply Landlock/seccomp/egress/rlimits (False for --dry)
    # NDIF endpoint nnsight traces hit (injected as NDIF_HOST). None -> nnsight default.
    ndif_host: str | None = None
    # Replace a notebook's real error with a generic message in returned records
    # (the raw error can echo the private inputs). Off for --dry: local rehearsal on
    # the PUBLIC dataset, where the participant should see their actual error.
    redact_errors: bool = True
    # Score only the rows the submission actually predicted (inner-join preds↔labels
    # on index) instead of requiring one prediction per label. Off on the real
    # leaderboard (a submission MUST cover every row); on for --dry, so a partial
    # rehearsal (e.g. `--limit N`) still reports a score on the N rows it produced.
    score_partial: bool = False
    cache_dir: str = "data/cache"   # where predownloaded inputs live
    mem_mb: int | None = None       # RLIMIT_AS (off by default; breaks some ML libs)
    # Egress is allowlisted by HOSTNAME SUFFIX via a loopback CONNECT proxy
    # (CDN-proof; the child is forced through it by a loopback-only seccomp gate).
    enforce_egress: bool = True
    # Run-phase hosts the loopback proxy permits. NDIF is CONNECT-tunneled (HTTPS,
    # unfiltered); HF is CONNECT/MITM (GET-only). api.ndif.us covers the hackathon
    # subdomain (aletheias.api.ndif.us) and any central auth call. The S3 host is
    # NDIF's trace-results bucket: a completed remote trace returns a presigned S3
    # URL nnsight streams the result from, so it must be reachable or every
    # completed trace 403s on the download (blind-tunneled; read-only, not team-writable).
    egress_allowlist: list[str] = field(
        default_factory=lambda: ["api.ndif.us", "huggingface.co", "hf.co",
                                 "ndif-hackathon-results.s3.amazonaws.com"])
    # Hosts within egress_allowlist that are MITM-terminated and restricted to
    # read methods (GET/HEAD). HF is reachable for model configs but can't be used
    # to push the private eval inputs out. NDIF is left blind-tunneled (unfiltered).
    egress_get_only_suffixes: list[str] = field(
        default_factory=lambda: ["huggingface.co", "hf.co"])
    install_allowlist: list[str] = field(          # install phase: PyPI
        default_factory=lambda: ["pypi.org", "pythonhosted.org"])
    hf_token: str | None = None
    # Shared NDIF key threaded into a run when the submitter's own key lacks the
    # usable tier (ndif.USABLE_TIER). Sourced from the LEADERBOARD_NDIF_API_KEY
    # env var (an HF Space secret); never set in the committed config.
    leaderboard_ndif_api_key: str | None = None
    # Bearer token for the operator-only admin endpoints (list / cancel in-flight
    # runs). Sourced from the ADMIN_TOKEN env var (an HF Space secret); never set in
    # the committed config. Unset -> the admin endpoints are disabled (404).
    admin_token: str | None = None
    # Team names shown as non-competing REFERENCE entries ("baselines"): the
    # leaderboard marks their rows and leaves them out of the competitive ranking
    # (they still appear, in score order, as context to beat). Multiple names can
    # map to the same reference set (e.g. a renamed account's stragglers).
    baseline_teams: list[str] = field(default_factory=list)

    def dataset_label_map(self) -> dict[str, str]:
        """Map each configured dataset key to its public codename (``dataset_label``).

        The real dataset names are private: participants must never see them on the
        leaderboard or in a submission's reported scores (they may still flow into
        the sandboxed notebook env, which can't exfiltrate them)."""
        return {d.key: dataset_label(d.key) for d in self.datasets}

    def base_env(self) -> dict[str, str]:
        """Environment shared across every notebook run."""
        e: dict[str, str] = {}
        if self.hf_token:
            e["HF_TOKEN"] = self.hf_token
        return e

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunnerConfig":
        datasets = [DatasetConfig(**d) for d in data.get("datasets", [])]
        known = {f for f in cls.__dataclass_fields__ if f != "datasets"}
        rest = {k: v for k, v in data.items() if k in known}
        return cls(datasets=datasets, **rest)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunnerConfig":
        import yaml  # local import so tests that build configs directly need no dep

        data = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls.from_dict(data)
        return cfg.with_env_overrides()

    def with_env_overrides(self) -> "RunnerConfig":
        """Pull secrets/overrides from the process environment."""
        return replace(
            self,
            hf_token=self.hf_token or os.environ.get("HF_TOKEN"),
            leaderboard_ndif_api_key=(
                self.leaderboard_ndif_api_key
                or os.environ.get("LEADERBOARD_NDIF_API_KEY")),
            admin_token=self.admin_token or os.environ.get("ADMIN_TOKEN"),
        )
