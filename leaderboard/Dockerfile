# HuggingFace Docker Space image for the Aletheia's Quest leaderboard.
# CPU-only: heavy model compute runs remotely on NDIF.
# Python 3.12+ is required by the NDIF client (the server rejects older clients).
FROM python:3.12-slim

# Non-root user (HF Spaces convention) with a writable home.
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

WORKDIR /app
COPY . /app

# Install the runner + web extras, then register a "python3" Jupyter kernel
# so nbclient can launch notebooks.
RUN pip install --no-cache-dir ".[space]" \
    && python -m ipykernel install --name python3 --display-name python3

# Pre-install the competition's required heavy deps (CPU torch + nnsight) so
# participant notebooks inherit them via the per-job venv (--system-site-packages)
# instead of reinstalling ~1GB per job. Heavy compute still runs remotely on NDIF.
# transformers/nnsight/torch are pinned to EXACTLY the versions NDIF is deployed on:
# nnsight resolves each model's architecture locally (on meta) and serializes the
# intervention graph, so a client/worker skew makes the worker reject the trace
# (e.g. `Unknown persistent id: Module:model.model.visual` / `...embed_tokens`, or
# `Module nnsight.intervention.backends.remote is not whitelisted`). Pinned to the
# deploy of 2026-07-22 (verified: all 5 competition models trace via plain
# `LanguageModel(...)` — see fargate/verify_models.py): transformers 5.15.0.dev0
# @c7f9c88 (registers Qwen3.5 as its multimodal class — the release 5.12.x loaded it
# as CausalLM and no longer matches the deploy), nnsight hackathon/peft @d901da3e
# ("load multimodal repos as their registered class"), torch 2.13.0. When you
# redeploy NDIF, re-pin all three here to match and re-run verify_models.py from the
# built image before opening submissions.
# NOTE: the soft-trigger LoRA-on-sharded-120B failure was NOT a
# transformers issue and NOT fixed by transformers main (main's WeightConverter also
# lacks the kwarg) — it's peft: keep the NDIF *workers* on peft<0.19 (peft 0.19.x
# calls transformers' WeightConverter with a distributed_operation kwarg no
# transformers release or main accepts). The WeightConverter fix itself lives on the
# workers (the client only resolves on meta; the workers do the remote weight loading),
# but we pin the client to peft<0.19 too so client and workers stay on the same peft line.
# nnsight is pinned to the competition's hackathon branch (PEFT/LoRA remote support
# + the hackathon NDIF cluster). Pinned to an explicit commit SHA (the HEAD of
# hackathon/peft as of this deploy) rather than the bare branch name: it guarantees
# the newest commit is installed AND changes this line so the Docker layer cache is
# busted on rebuild (a bare @hackathon/peft could otherwise be served from a stale
# cached layer). Re-pin to the branch's new HEAD to pull later updates.
# git + a C toolchain: the hackathon/peft branch builds a C extension
# (nnsight._c.py_mount) from source, so it needs gcc + headers (build-essential).
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0 \
    && pip install --no-cache-dir \
       "transformers @ git+https://github.com/huggingface/transformers.git@c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d" \
       "peft<0.19" \
       "nnsight @ git+https://github.com/ndif-team/nnsight.git@d901da3ed772c815d0d146136de6f7e35f913221"

# RESULTS_URI is left unset so runner.yaml's bucket:// results store is used.
# (Set RESULTS_URI=/data/results.jsonl + attach a volume to use local storage instead.)
ENV RUNNER_CONFIG=/app/runner.yaml \
    PORT=7860
# /data holds the predownloaded input cache (runner.yaml cache_dir). On this Space
# /data is a PERSISTENT HF bucket mount, so the cache survives rebuilds/restarts —
# to refresh it after re-pushing a dataset, clear the bucket's cache/ prefix
# (see leaderboard-dev-storage), not a rebuild.
RUN mkdir -p /data && chown -R user /app /data
USER user

EXPOSE 7860
CMD ["uvicorn", "aletheia_runner.app:app", "--host", "0.0.0.0", "--port", "7860"]
