"""Rebuild the expanded low-rank trusted activation probe.

Training combines the public Hugging Face development rows listed in
``training_datasets.yaml`` with public ``liars-bench-expanded`` metadata. It
uses only conversation text and labels from that metadata; source-model
activation files are never read.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from activation_detector import run_feature_batches
from liars_bench_data import load_liars_bench_rows

# nnsight does not read NDIF_API_KEY / NDIF_HOST from the environment on its
# own -- wire it explicitly, matching train_twoway.py.
from nnsight import CONFIG  # noqa: E402
if os.environ.get("NDIF_HOST"):
    CONFIG.API.HOST = os.environ["NDIF_HOST"]
if os.environ.get("NDIF_API_KEY"):
    CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])


MODEL_ID = "Qwen/Qwen3.5-9B"
SEED = 20260723
LAYERS = (23,)
POOL_WIDTHS = (16, 24)
FEATURE_MODE = "outer_contrast"
LOGISTIC_C = 0.001


def _source_group(name: str) -> str:
    """Preserve the public source's group name used by the frozen artifact."""
    return name.rsplit("/", 1)[-1]


def _hf_scenario_rows(config_path=None):
    """Load unsampled Hugging Face rows grouped by scenario and source group."""
    if config_path is None:
        config_path = Path(__file__).with_name("training_datasets.yaml")
    config = yaml.safe_load(Path(config_path).read_text())
    items = {item["name"].split("/", 1)[1]: item for item in config["datasets"]}
    instructed = {
        name.removeprefix("dev-instructed-deception-"): item
        for name, item in items.items()
        if name.startswith("dev-instructed-deception-Qwen3.5-27B-")
    }
    varied = {
        name.removeprefix("dev-varied-deception-"): item
        for name, item in items.items()
        if name.startswith("dev-varied-deception-Qwen3.5-27B-")
    }
    out = {"instructed": {}, "varied": {}}
    for organism in sorted(set(instructed) & set(varied)):
        for scenario, item in (("instructed", instructed[organism]), ("varied", varied[organism])):
            inputs = load_dataset(item["name"], split="test")
            label_ds = load_dataset(item["labels_uri"], split="test")
            id_col = item.get("id_column", "index")
            label_col = item.get("label_column", "deceptive")
            by_id = dict(zip(label_ds[id_col], label_ds[label_col]))
            ex, y = [], []
            for i in range(len(inputs)):
                row_id = inputs[i][id_col]
                if row_id not in by_id:
                    continue
                ex.append(inputs[i])
                y.append(int(bool(by_id[row_id])))
            out[scenario][_source_group(organism)] = (ex, np.asarray(y))
    return out


def _s3_scenario_rows():
    """Load public metadata rows grouped by scenario and source group."""
    out = {}
    datasets = (
        ("instructed", "instructed-deception"),
        ("varied", "varied-deception"),
    )
    for scenario, dataset_name in datasets:
        ex, y, org = load_liars_bench_rows(dataset_name, split="dev")
        org = np.asarray([_source_group(o) for o in org])
        bucket = {}
        for organism in np.unique(org):
            sel = org == organism
            bucket[organism] = ([e for e, s in zip(ex, sel) if s], y[sel])
        out[scenario] = bucket

    organisms_both = set(out["instructed"]) & set(out["varied"])
    return {scenario: {o: rows for o, rows in buckets.items() if o in organisms_both}
            for scenario, buckets in out.items()}


PER_LABEL = 32


def load_all_rows(seed: int = SEED):
    """Sample 32 rows per label, scenario, and source-specific group."""
    hf = _hf_scenario_rows()
    s3 = _s3_scenario_rows()

    merged = {"instructed": {}, "varied": {}}
    for scenario in merged:
        organisms = set(hf[scenario]) | set(s3.get(scenario, {}))
        for organism in organisms:
            ex_parts, y_parts = [], []
            if organism in hf[scenario]:
                ex_h, y_h = hf[scenario][organism]
                ex_parts.append(ex_h)
                y_parts.append(y_h)
            if organism in s3.get(scenario, {}):
                ex_s, y_s = s3[scenario][organism]
                ex_parts.append(ex_s)
                y_parts.append(y_s)
            ex_all = [e for part in ex_parts for e in part]
            y_all = np.concatenate(y_parts)
            merged[scenario][organism] = (ex_all, y_all)

    valid_organisms = [
        organism for organism in set(merged["instructed"]) & set(merged["varied"])
        if all(
            (merged[scenario][organism][1] == 0).sum() >= PER_LABEL
            and (merged[scenario][organism][1] == 1).sum() >= PER_LABEL
            for scenario in ("instructed", "varied")
        )
    ]

    rng = np.random.default_rng(seed)
    examples, labels, organisms = [], [], []
    for organism in sorted(valid_organisms):
        for scenario in ("instructed", "varied"):
            ex_all, y_all = merged[scenario][organism]
            chosen = np.concatenate(
                [
                    rng.choice(
                        np.flatnonzero(y_all == label),
                        PER_LABEL,
                        replace=False,
                    )
                    for label in (0, 1)
                ]
            )
            rng.shuffle(chosen)
            examples.extend(ex_all[int(i)] for i in chosen)
            labels.extend(int(y_all[i]) for i in chosen)
            organisms.extend([organism] * len(chosen))

    print(f"  PER_LABEL={PER_LABEL} across {len(valid_organisms)} source groups")
    return examples, np.asarray(labels), np.asarray(organisms)


def extract_features(
    examples,
    batch_size=32,
    max_batch_tokens=12_000,
    attempts=3,
):
    """Extract all training features in one length-batched NDIF session."""
    if not examples:
        raise ValueError("training set is empty")
    return run_feature_batches(
        examples,
        model_id=MODEL_ID,
        layer=LAYERS[0],
        pool_widths=POOL_WIDTHS,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        attempts=attempts,
    )

def fit_low_rank_probe(
    features,
    labels,
    organisms,
    logistic_c=LOGISTIC_C,
    random_state=SEED,
):
    scaler = StandardScaler().fit(features)
    normalized = scaler.transform(features)
    organism_directions = []
    for organism in np.unique(organisms):
        selected = organisms == organism
        values = normalized[selected]
        selected_labels = labels[selected]
        direction = values[selected_labels == 1].mean(0) - values[selected_labels == 0].mean(0)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        organism_directions.append(direction)
    basis = np.stack(organism_directions)
    low_rank = normalized @ basis.T
    classifier = LogisticRegression(
        C=logistic_c,
        solver="liblinear",
        max_iter=2_000,
        random_state=random_state,
    ).fit(low_rank, labels)
    direction = classifier.coef_[0] @ basis
    intercept = float(classifier.intercept_[0])
    return scaler, basis, direction, intercept


def _score(features, scaler, direction, intercept):
    normalized = (features - scaler.mean_) / scaler.scale_
    return normalized @ direction + intercept


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def _pick_threshold(logits, labels):
    probs = _sigmoid(logits)
    best = (-np.inf, 0.5)
    for threshold in np.unique(np.r_[0.0, probs, 1.0]):
        candidate = balanced_accuracy_score(labels, probs >= threshold)
        if (candidate, -abs(threshold - 0.5)) > (best[0], -abs(best[1] - 0.5)):
            best = (candidate, float(threshold))
    return best


def main():
    if os.environ.get("NDIF_API_KEY") is None:
        raise SystemExit("set NDIF_API_KEY before rebuilding the probe")

    examples, labels, organisms = load_all_rows()
    print(
        f"Loaded {len(labels)} balanced rows across "
        f"{len(np.unique(organisms))} source groups"
    )

    print("Extracting layer-23 features in one NDIF session...")
    features = extract_features(examples).astype(np.float32)

    print("Fitting the final low-rank readout...")
    scaler, basis, direction, intercept = fit_low_rank_probe(features, labels, organisms)
    final_logits = _score(features, scaler, direction, intercept)
    _, threshold = _pick_threshold(final_logits, labels)
    print(f"  chosen threshold={threshold:.6f}")

    output = Path(__file__).with_name("trusted_activation_probe.npz")
    np.savez_compressed(
        output,
        schema_version=np.asarray(4, dtype=np.int16),
        model_id=np.asarray(MODEL_ID),
        layers=np.asarray(LAYERS, dtype=np.int16),
        pool_widths=np.asarray(POOL_WIDTHS, dtype=np.int16),
        feature_mode=np.asarray(FEATURE_MODE),
        threshold=np.asarray(threshold),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coefficient=direction.astype(np.float32),
        intercept=np.asarray(intercept),
        training_rows=np.asarray(len(labels), dtype=np.int32),
        training_organisms=np.asarray(len(np.unique(organisms)), dtype=np.int16),
        low_rank_dimensions=np.asarray(len(basis), dtype=np.int16),
        logistic_c=np.asarray(LOGISTIC_C),
    )
    print(f"saved {output}: {len(labels)} rows across {len(np.unique(organisms))} organisms")


if __name__ == "__main__":
    main()
