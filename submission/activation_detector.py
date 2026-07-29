"""Efficient trusted-model low-rank activation detector."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRANSIENT_ERROR_TEXT = (
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
    "remote end closed",
    "reset by peer",
    "websocket",
    "broken pipe",
)


@dataclass(frozen=True)
class ActivationProbe:
    model_id: str
    layer: int
    pool_widths: tuple[int, ...]
    threshold: float
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float


def load_probe(path: str | Path) -> ActivationProbe:
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != 4:
            raise ValueError("unsupported activation-probe schema")
        layers = tuple(int(value) for value in data["layers"])
        if len(layers) != 1 or str(data["feature_mode"]) != "outer_contrast":
            raise ValueError("unsupported activation-probe feature layout")
        probe = ActivationProbe(
            model_id=str(data["model_id"]),
            layer=layers[0],
            pool_widths=tuple(int(value) for value in data["pool_widths"]),
            threshold=float(data["threshold"]),
            mean=data["mean"].astype(np.float32),
            scale=data["scale"].astype(np.float32),
            coefficient=data["coefficient"].astype(np.float32),
            intercept=float(data["intercept"]),
        )
    if not (
        probe.layer >= 0
        and len(probe.pool_widths) == 2
        and all(width > 0 for width in probe.pool_widths)
        and probe.mean.shape == probe.scale.shape == probe.coefficient.shape
        and np.isfinite(probe.mean).all()
        and np.isfinite(probe.scale).all()
        and np.isfinite(probe.coefficient).all()
        and 0 < probe.threshold < 1
    ):
        raise ValueError("invalid activation-probe artifact")
    return probe


def is_transient_error(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return any(text in message for text in TRANSIENT_ERROR_TEXT)


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


def encode_examples(
    tokenizer,
    examples: Sequence[Mapping],
    system_prompt: str,
) -> list[list[int]]:
    return [
        tokenizer.encode(
            _render_prompt(
                tokenizer,
                system_prompt,
                compact_evidence(tokenizer, example),
            ),
            add_special_tokens=False,
        )
        for example in examples
    ]


def length_batches(
    encoded: Sequence[Sequence[int]],
    *,
    max_rows: int,
    max_tokens: int,
) -> tuple[list[int], list[list[int]]]:
    """Pack length-sorted rows without exceeding a padded-token budget."""
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
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
    return order, batches


def score_examples(
    examples: Sequence[Mapping],
    *,
    probe: ActivationProbe,
    system_prompt: str,
    batch_size: int = 32,
    max_batch_tokens: int = 12_000,
    attempts: int = 3,
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
    encoded = encode_examples(tokenizer, examples, system_prompt)
    if not encoded:
        return np.empty(0, dtype=np.float32)
    order, batches = length_batches(
        encoded,
        max_rows=batch_size,
        max_tokens=max_batch_tokens,
    )
    cutoff = float(np.log(probe.threshold / (1.0 - probe.threshold)))
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
                        hidden = layers[probe.layer].output
                        if isinstance(hidden, tuple):
                            hidden = hidden[0]
                        pooled = torch.stack(
                            [
                                torch.stack(
                                    [
                                        hidden[
                                            row,
                                            max(0, len(ids) - pool_width) : len(ids),
                                        ]
                                        .float()
                                        .mean(0)
                                        for row, ids in enumerate(chunk)
                                    ]
                                )
                                for pool_width in probe.pool_widths
                            ],
                            dim=1,
                        )
                        features = torch.cat(
                            [pooled[:, 0], pooled[:, 0] - pooled[:, 1]],
                            dim=-1,
                        )
                        # Match the float16 feature cache used to fit the probe,
                        # then reduce to one scalar per row before downloading.
                        features = features.half().float()
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
            if attempt == attempts - 1 or not is_transient_error(error):
                raise
            time.sleep(min(2**attempt, 8))
    outputs = np.empty(len(values), dtype=np.float32)
    outputs[np.asarray(order)] = values
    return outputs
