"""Train the visible-reasoning probe from public datasets.

Example:
    python submission/train_reasoning_probe.py \
        --output submission/reasoning_probe.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from reasoning_detector import extract_reasoning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

RANDOM_STATE = 20260715
SCHEMA_VERSION = 1


def load_training_data(config_path):
    """Load labeled public reasoning traces listed in the run configuration."""
    config = yaml.safe_load(Path(config_path).read_text())
    records = []
    for item in config["datasets"]:
        if "varied-deception" not in item["name"]:
            continue
        inputs = load_dataset(item["name"], split="test").to_pandas()
        labels = load_dataset(item["labels_uri"], split="test").to_pandas()
        id_column = item.get("id_column", "index")
        label_column = item.get("label_column", "deceptive")
        merged = inputs.merge(
            labels[[id_column, label_column]], on=id_column, how="inner"
        )
        for row in merged.to_dict("records"):
            records.append(
                {
                    "dataset": item["name"],
                    "index": row[id_column],
                    "text": extract_reasoning(row),
                    "label": int(bool(row[label_column])),
                }
            )
    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("no varied-deception training rows found")
    if (data["text"].str.strip() == "").any():
        raise ValueError("a training row has no visible reasoning")
    return data


def new_model():
    """Construct the fixed vectorizer and logistic classifier."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    return vectorizer, classifier


def fit_predict(train_text, train_label, test_text):
    """Fit one fold and return deception probabilities for its held-out rows."""
    vectorizer, classifier = new_model()
    classifier.fit(vectorizer.fit_transform(train_text), train_label)
    return classifier.predict_proba(vectorizer.transform(test_text))[:, 1]


def macro_threshold(labels, scores, groups):
    """Choose the threshold maximizing mean balanced accuracy across groups."""
    best = (-np.inf, 0.5)
    for threshold in np.unique(np.r_[0.0, scores, 1.0]):
        values = [
            balanced_accuracy_score(
                labels[groups == group], scores[groups == group] >= threshold
            )
            for group in np.unique(groups)
        ]
        candidate = (float(np.mean(values)), float(threshold))
        if (candidate[0], -abs(candidate[1] - 0.5)) > (
            best[0],
            -abs(best[1] - 0.5),
        ):
            best = candidate
    return best[1]


def cross_validate(data):
    """Hold out each Qwen variant and choose one pooled deployment threshold."""
    texts = data["text"].to_numpy()
    labels = data["label"].to_numpy(dtype=int)
    groups = data["dataset"].to_numpy()
    scores = np.full(len(data), np.nan)

    for group in np.unique(groups):
        held = groups == group
        scores[held] = fit_predict(texts[~held], labels[~held], texts[held])

    threshold = macro_threshold(labels, scores, groups)
    predictions = scores >= threshold

    rows = []
    for group in np.unique(groups):
        held = groups == group
        rows.append(
            {
                "dataset": group,
                "n": int(held.sum()),
                "auroc": roc_auc_score(labels[held], scores[held]),
                "balanced_accuracy": balanced_accuracy_score(
                    labels[held], predictions[held]
                ),
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows), threshold


def fit_and_save(data, threshold, output):
    """Fit all public rows and save the parameters required for inference."""
    texts = data["text"].to_numpy()
    labels = data["label"].to_numpy(dtype=int)
    vectorizer, classifier = new_model()
    classifier.fit(vectorizer.fit_transform(texts), labels)
    np.savez_compressed(
        output,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int16),
        vocabulary=vectorizer.get_feature_names_out().astype(str),
        idf=vectorizer.idf_.astype(np.float32),
        coefficient=classifier.coef_[0].astype(np.float32),
        intercept=np.asarray(classifier.intercept_[0]),
        threshold=np.asarray(threshold),
    )


def main():
    """Run public evaluation and write a fully fitted probe artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dry.yaml")
    parser.add_argument("--output", required=True, help="path for the trained .npz")
    args = parser.parse_args()
    data = load_training_data(args.config)
    print(f"loaded {len(data):,} rows from {data['dataset'].nunique()} Qwen variants")
    metrics, threshold = cross_validate(data)
    print(metrics.to_string(index=False))
    print(f"macro held-out AUROC: {metrics['auroc'].mean():.3f}")
    print(f"macro balanced accuracy: {metrics['balanced_accuracy'].mean():.3f}")
    print(f"deployment threshold: {threshold:.6f}")
    fit_and_save(data, threshold, args.output)
    print(f"saved {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
