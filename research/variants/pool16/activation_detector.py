"""Single-strategy trusted-model activation detector used by example.ipynb."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActivationProbe:
    model_id: str
    layers: tuple[int, ...]
    pool_width: int
    threshold: float
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float


def load_probe(path: str | Path) -> ActivationProbe:
    with np.load(path, allow_pickle=False) as data:
        schema_version = int(data["schema_version"])
        if schema_version not in (2, 3):
            raise ValueError("unsupported activation-probe schema")
        probe = ActivationProbe(
            model_id=str(data["model_id"]),
            layers=tuple(int(value) for value in data["layers"]),
            pool_width=(
                int(data["pool_width"]) if schema_version >= 3 else 1
            ),
            threshold=float(data["threshold"]),
            mean=data["mean"].astype(np.float32),
            scale=data["scale"].astype(np.float32),
            coefficient=data["coefficient"].astype(np.float32),
            intercept=float(data["intercept"]),
        )
    if not (
        probe.layers
        and all(layer >= 0 for layer in probe.layers)
        and probe.pool_width > 0
        and probe.mean.shape == probe.scale.shape == probe.coefficient.shape
        and np.isfinite(probe.mean).all()
        and np.isfinite(probe.scale).all()
        and np.isfinite(probe.coefficient).all()
        and 0 < probe.threshold < 1
    ):
        raise ValueError("invalid activation-probe artifact")
    return probe


def _split_final_answer(example: Mapping):
    messages = list(example.get("messages") or [])
    positions = [
        i for i, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not positions:
        return None
    position = positions[-1]
    context = [dict(message) for message in messages[:position]]
    answer = "\n".join(
        str(message.get("content", ""))
        for message in messages[position:]
        if message.get("role") == "assistant"
    )
    return context, answer


def _clip_tokens(tokenizer, text: str, budget: int) -> str:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    if len(ids) <= budget:
        return str(text)
    left = budget // 2
    return (
        tokenizer.decode(ids[:left])
        + "\n...[middle omitted]...\n"
        + tokenizer.decode(ids[-(budget - left) :])
    )


def compact_evidence(tokenizer, example: Mapping) -> str:
    """User-visible evidence only; hidden instructions are a brittle label shortcut."""
    split = _split_final_answer(example)
    if split is None:
        raise ValueError("conversation has no assistant response")
    context, answer = split
    visible = [
        message
        for message in context
        if message.get("role") not in {"system", "developer"}
    ]
    recent = "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in visible[-4:]
    )
    return (
        "<visible_context>\n"
        + _clip_tokens(tokenizer, recent, 350)
        + "\n</visible_context>\n<final_response>\n"
        + _clip_tokens(tokenizer, answer, 300)
        + "\n</final_response>"
    )


def _render_prompt(tokenizer, system_prompt: str, evidence: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": evidence},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _length_batches(
    order: Sequence[int],
    encoded: Sequence[Sequence[int]],
    *,
    max_rows: int,
    max_tokens: int,
) -> list[list[int]]:
    """Pack length-sorted rows without exceeding a padded-token budget."""
    batches: list[list[int]] = []
    current: list[int] = []
    for position in order:
        padded_tokens = (len(current) + 1) * len(encoded[position])
        if current and (len(current) >= max_rows or padded_tokens > max_tokens):
            batches.append(current)
            current = []
        current.append(position)
    if current:
        batches.append(current)
    return batches


def score_examples(
    examples: Sequence[Mapping],
    *,
    probe: ActivationProbe,
    system_prompt: str,
    batch_size: int = 32,
    max_batch_tokens: int = 12_000,
    attempts: int = 5,
) -> np.ndarray:
    """Score a dataset in one queued NDIF session, with length-batched traces."""
    import torch
    from nnsight import LanguageModel
    from util import decoder_layers

    model = LanguageModel(probe.model_id)
    tokenizer = model.tokenizer
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    layers = decoder_layers(model)
    encoded = [
        tokenizer.encode(
            _render_prompt(
                tokenizer, system_prompt, compact_evidence(tokenizer, example)
            ),
            add_special_tokens=False,
        )
        for example in examples
    ]
    if not encoded:
        return np.empty(0, dtype=np.float32)
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
    batches = _length_batches(
        order, encoded, max_rows=batch_size, max_tokens=max_batch_tokens
    )
    transient = (
        "bad gateway",
        "502",
        "503",
        "504",
        "timed out",
        "timeout",
        "temporarily",
        "unavailable",
        "connection",
        "overloaded",
        "try again",
        "submitting request",
        "remote end closed",
        "reset by peer",
        "websocket",
        "broken pipe",
    )
    cutoff = float(np.log(probe.threshold / (1.0 - probe.threshold)))
    layer_ids = tuple(probe.layers)
    pool_width_tokens = int(probe.pool_width)
    mean = torch.tensor(probe.mean)
    scale = torch.tensor(probe.scale)
    coefficient = torch.tensor(probe.coefficient)
    intercept = float(probe.intercept)
    for attempt in range(attempts):
        try:
            with model.session(remote=True):
                pieces = []
                for positions in batches:
                    chunk = [encoded[index] for index in positions]
                    width = max(len(ids) for ids in chunk)
                    input_ids = torch.full(
                        (len(chunk), width), pad, dtype=torch.long
                    )
                    attention = torch.zeros(
                        (len(chunk), width), dtype=torch.long
                    )
                    for row, ids in enumerate(chunk):
                        input_ids[row, : len(ids)] = torch.tensor(ids)
                        attention[row, : len(ids)] = 1

                    with model.trace(
                        {"input_ids": input_ids, "attention_mask": attention}
                    ) as tracer:
                        per_layer = []
                        for layer in layer_ids:
                            hidden = layers[layer].output
                            if isinstance(hidden, tuple):
                                hidden = hidden[0]
                            per_layer.append(
                                torch.stack(
                                    [
                                        torch.stack(
                                            [
                                                hidden[
                                                    row,
                                                    max(
                                                        0,
                                                        len(ids) - pool_width,
                                                    ) : len(ids),
                                                ]
                                                .float()
                                                .mean(0)
                                                for row, ids in enumerate(chunk)
                                            ]
                                        )
                                        for pool_width in (pool_width_tokens,)
                                    ],
                                    dim=1,
                                )
                            )
                        # Match the float16 feature cache used to fit the probe,
                        # then reduce to one scalar per row before downloading.
                        features = torch.cat(per_layer, dim=-1).half().float()
                        device = features.device
                        normalized = (
                            features - mean.to(device)
                        ) / scale.to(device)
                        raw = (
                            normalized @ coefficient.to(device) + intercept
                        )
                        pieces.append(torch.sigmoid(raw - cutoff).float())
                        # The deepest requested residual is already available;
                        # skip the trusted model's remaining decoder blocks and
                        # output head.
                        tracer.stop()
                saved_scores = torch.cat(pieces).cpu().save()
            values = saved_scores.float().numpy().reshape(-1)
            break
        except Exception as error:
            message = f"{type(error).__name__}: {error}".lower()
            if attempt == attempts - 1 or not any(
                word in message for word in transient
            ):
                raise
            time.sleep(min(2**attempt, 8))
    else:
        raise RuntimeError("unreachable")
    outputs = np.empty(len(values), dtype=np.float32)
    outputs[np.asarray(order)] = values
    return np.clip(outputs, 0, 1)
