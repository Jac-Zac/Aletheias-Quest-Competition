"""Offline checks for the trusted low-rank submission.

This script deliberately avoids dataset downloads and NDIF calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission"
EXPECTED_ARTIFACT_SHA256 = (
    "54743cf37a875c3842417d363e1a244667ae1947b6e761906a7d1c44c86caf1f"
)

sys.path.insert(0, str(SUBMISSION))
from activation_detector import length_batches, load_probe  # noqa: E402
from train_trusted_activation_probe import (  # noqa: E402
    SEED,
    fit_low_rank_probe,
)


def check_structure() -> None:
    notebooks = sorted(SUBMISSION.glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "trusted_efficient_lowrank.ipynb"
    ]
    notebook = json.loads(notebooks[0].read_text())
    report = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    for phrase in (
        "## Method",
        "2,176",
        "each of the 17 groups",
        "train_trusted_activation_probe.py",
    ):
        assert phrase in report, phrase


def check_artifact() -> None:
    artifact = SUBMISSION / "trusted_activation_probe.npz"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == EXPECTED_ARTIFACT_SHA256
    probe = load_probe(artifact)
    assert probe.layer == 23
    assert probe.pool_widths == (16, 24)
    assert probe.mean.shape == probe.scale.shape == probe.coefficient.shape
    assert probe.mean.shape == (8192,)


def check_training_config() -> None:
    config = yaml.safe_load(
        (SUBMISSION / "training_datasets.yaml").read_text()
    )
    datasets = config["datasets"]
    assert len(datasets) == 19
    pairs = {(item["name"], item["labels_uri"]) for item in datasets}
    assert len(pairs) == len(datasets)


def check_batching() -> None:
    lengths = [20, 5, 12, 8, 30, 7]
    encoded = [list(range(length)) for length in lengths]
    order, batches = length_batches(
        encoded,
        max_rows=2,
        max_tokens=40,
    )
    assert order == [1, 5, 3, 2, 0, 4]
    assert [index for batch in batches for index in batch] == order
    for batch in batches:
        assert len(batch) <= 2
        assert len(batch) * max(lengths[index] for index in batch) <= 40


def check_deterministic_fit() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(96, 24)).astype(np.float32)
    labels = np.tile([0, 1], 48)
    groups = np.repeat(np.array(["a", "b", "c"]), 32)

    np.random.seed(1)
    first = fit_low_rank_probe(features, labels, groups)
    np.random.seed(999)
    second = fit_low_rank_probe(features, labels, groups)

    for left, right in zip(
        (
            first[0].mean_,
            first[0].scale_,
            first[1],
            first[2],
            np.asarray(first[3]),
        ),
        (
            second[0].mean_,
            second[0].scale_,
            second[1],
            second[2],
            np.asarray(second[3]),
        ),
    ):
        np.testing.assert_array_equal(left, right)


def main() -> None:
    check_structure()
    check_artifact()
    check_training_config()
    check_batching()
    check_deterministic_fit()
    print(
        "offline validation passed:",
        f"seed={SEED}, layer=23, pools=(16, 24)",
    )


if __name__ == "__main__":
    main()
