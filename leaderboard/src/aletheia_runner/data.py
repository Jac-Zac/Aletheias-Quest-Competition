"""Predownload dataset INPUTS (and LoRA adapter CONFIGS) so sandboxed notebooks
load them offline / from cache.

Datasets: the trusted parent builds each dataset's Arrow cache under
``cache_dir/datasets`` via ``load_dataset``; per job the child gets a writable
copy plus ``HF_DATASETS_OFFLINE`` and loads it with no token — that's what keeps
the private eval set readable without exposing it (labels are scored separately).

LoRA adapters: the parent predownloads each adapter referenced by the datasets'
``lora`` column into ``cache_dir/hf_hub`` — but only the **config/tokenizer**
files, never the weights. nnsight loads the model on ``meta`` and the LoRA is
applied remotely on NDIF, so the sandbox never needs the (multi-GB) adapter
safetensors; downloading them would only risk tripping the child's RLIMIT_FSIZE.
The configs are tiny, so per job we just copy the cache into the child's scratch.
The HF hub stays reachable so notebooks can fetch model configs live.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import SPLIT, RunnerConfig, dataset_model_lora

# Weight files we never predownload — nnsight loads on meta and the LoRA is applied
# remotely on NDIF, so the sandbox only needs the adapter config, not the weights.
_WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf",
                    "*.h5", "*.msgpack", "*.onnx"]

# Bump this whenever a dataset's CONTENTS change under an unchanged name (e.g. a
# re-pushed parquet fixing a bad model id). The Arrow cache dir is namespaced by
# this value, so a bump gives a guaranteed-CLEAN directory — the prepared-cache
# marker is otherwise keyed only on dataset *names* and a same-name content change
# would NOT invalidate it, leaving the stale Arrow served. Old epoch dirs are left
# behind (rebuilt inputs, small); prune them manually if /data ever gets tight.
_CACHE_EPOCH = "2026-07-02.2-heal-qwen-validation-cache"


@dataclass
class DataLayout:
    """Predownloaded inputs: the dataset Arrow cache and the adapter-config hub
    cache, both copied into the child's scratch per job."""

    datasets_cache: Path
    hub_cache: Path | None = None

    def child_env(self, job_scratch: Path, offline: bool = True) -> dict[str, str]:
        """Env for a sandboxed child to load the inputs. ``offline`` forces the
        datasets library offline (private data from the copied cache, no token);
        the HF *hub* stays online so notebooks can fetch model configs live."""
        scratch = Path(job_scratch)
        ds_copy = scratch / "hf_datasets_cache"
        if ds_copy.exists():
            shutil.rmtree(ds_copy)
        shutil.copytree(self.datasets_cache, ds_copy)  # datasets needs a writable lock

        # Copy the predownloaded adapter configs into the child's writable hub cache
        # (tiny — configs only). The child then loads the adapter config as a cache
        # hit; live fetches for anything else still write here.
        hub_copy = scratch / "hf_hub_cache"
        if hub_copy.exists():
            shutil.rmtree(hub_copy)
        if self.hub_cache and Path(self.hub_cache).is_dir():
            shutil.copytree(self.hub_cache, hub_copy)
        else:
            hub_copy.mkdir(parents=True, exist_ok=True)

        env = {
            "HF_DATASETS_CACHE": str(ds_copy),
            "HF_HUB_CACHE": str(hub_copy),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Classic HTTP download path (hf_xet doesn't route through the proxy).
            "HF_HUB_DISABLE_XET": "1",
            # Downloads go through the MITM egress proxy — give generous budgets.
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "HF_HUB_ETAG_TIMEOUT": "60",
        }
        if offline:
            env["HF_DATASETS_OFFLINE"] = "1"
        return env


def prepare_inputs(config: RunnerConfig) -> DataLayout:
    """Build the dataset Arrow cache and predownload LoRA adapter configs (in the
    trusted parent). Idempotent via a marker."""
    cache = Path(config.cache_dir)
    # Namespace the Arrow cache by epoch: a bump points at a fresh, empty directory,
    # so a re-pushed dataset is rebuilt cleanly with no in-place wipe (an in-place
    # rmtree risked leaving a half-deleted, corrupt cache dir). The marker lives
    # inside the epoch dir, so it's inherently per-epoch.
    datasets_cache = cache / "datasets" / _CACHE_EPOCH
    hub_cache = cache / "hf_hub"
    marker = datasets_cache / ".prepared"
    layout = DataLayout(datasets_cache=datasets_cache, hub_cache=hub_cache)

    key = json.dumps(sorted(d.name for d in config.datasets), sort_keys=True)
    if marker.exists() and marker.read_text() == key:
        return layout

    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    # Load each eval dataset once (builds the Arrow cache) and, in the same pass,
    # collect the distinct LoRA adapter repos it references. The labels dataset is
    # never loaded here.
    loras: set[str] = set()
    for cfg in config.datasets:
        ds = load_dataset(cfg.name, split=SPLIT, cache_dir=str(datasets_cache),
                          token=config.hf_token)
        if "lora" in ds.column_names:
            for value in ds.unique("lora"):
                if isinstance(value, str) and "/" in value:
                    loras.add(value)

    # Predownload each adapter's CONFIG (never its weights).
    from huggingface_hub import snapshot_download
    for repo in sorted(loras):
        snapshot_download(repo, cache_dir=str(hub_cache), token=config.hf_token,
                          ignore_patterns=_WEIGHT_PATTERNS)

    # Predownload each BASE MODEL's config + tokenizer (never its weights), so the
    # sandbox builds the nnsight meta-model from this baked cache instead of a live
    # Hub fetch. Without this, a notebook loads the base config at run time using the
    # *submitter's* HF token — which fails for GATED repos (e.g. google/gemma-3-27b-it)
    # whose license the submitter hasn't accepted. Weights are never needed locally
    # (they live on the NDIF cluster); only the small config/tokenizer files.
    bases = {m for cfg in config.datasets
             if (m := dataset_model_lora(cfg.name)[0])}
    for repo in sorted(bases):
        snapshot_download(repo, cache_dir=str(hub_cache), token=config.hf_token,
                          ignore_patterns=_WEIGHT_PATTERNS)

    marker.write_text(key)
    return layout
