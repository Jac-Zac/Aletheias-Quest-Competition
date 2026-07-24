#!/usr/bin/env python3
"""Post-deploy gate: confirm every competition model traces on NDIF from THIS image.

Run this from inside the built runner image against the live NDIF *before* opening
submissions. It exercises the exact path a participant hits — plain
``LanguageModel(model_id)`` + one remote ``.trace`` — so a client/worker version skew
(the kind that yields ``Unknown persistent id: Module:...`` or ``Module
nnsight.intervention.backends.remote is not whitelisted``) fails here, loudly, instead
of silently zeroing live submissions.

Usage:
    NDIF_API_KEY=<key> python fargate/verify_models.py
    # optional: NDIF_HOST=https://aletheias.api.ndif.us  (default)

Exit code 0 iff ALL models trace successfully; non-zero otherwise (usable as a CI/deploy gate).
"""
from __future__ import annotations

import os
import sys

# The five models a submission may trace against. Plain LanguageModel on purpose:
# it must match however NDIF has each model deployed (see the Dockerfile pin note).
MODELS = [
    "google/gemma-3-27b-it",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-9B",
    "trohrbaugh/Qwen3.5-9B-heretic-v2",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
]
PROMPT = "The Eiffel Tower is located in the city of"
DEFAULT_HOST = "https://aletheias.api.ndif.us"


def _last_lines(text: str, n: int = 6) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n      ".join(lines[-n:])


def main() -> int:
    key = os.environ.get("NDIF_API_KEY")
    if not key:
        print("ERROR: set NDIF_API_KEY", file=sys.stderr)
        return 2
    host = os.environ.get("NDIF_HOST", DEFAULT_HOST)

    from nnsight import CONFIG, LanguageModel
    CONFIG.API.HOST = host
    CONFIG.API.APIKEY = key

    # Version alignment report (nnsight/torch skew shows as CRITICAL here).
    try:
        from nnsight import ndif
        print(f"== ndif.compare() against {host} ==")
        ndif.compare()
    except Exception as e:  # noqa: BLE001 — informational only
        print(f"(ndif.compare() unavailable: {type(e).__name__}: {e})")

    print(f"\n== tracing {len(MODELS)} models via plain LanguageModel ==")
    failures = []
    for mid in MODELS:
        try:
            model = LanguageModel(mid)
            with model.trace(PROMPT, remote=True):
                out = model.output.save()
            print(f"  [PASS] {mid}  -> {type(out).__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(mid)
            print(f"  [FAIL] {mid}\n      {_last_lines(str(e))}")

    n_ok = len(MODELS) - len(failures)
    print(f"\n== {n_ok}/{len(MODELS)} models OK ==")
    if failures:
        print("FAILED: " + ", ".join(failures))
        print("Do NOT open submissions until these trace — a failure here fail-fasts every "
              "submission that touches the model.")
        return 1
    print("All models trace cleanly. Safe to open submissions against this NDIF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
