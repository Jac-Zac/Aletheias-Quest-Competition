"""Load public conversation metadata used to reproduce the trusted probe.

Only the metadata parquet files are read. Source-model activation files in the
same bucket are not accessed.
"""

from __future__ import annotations

import io
import json
from urllib.request import urlopen

import numpy as np
import pandas as pd


BUCKET_URL = "https://liars-bench-expanded.s3.amazonaws.com"


def fetch_records(dataset: str, split: str = "dev") -> pd.DataFrame:
    """Download one public metadata parquet file."""
    key = f"metadata/{dataset}/{split}/records.parquet"
    with urlopen(f"{BUCKET_URL}/{key}") as response:
        blob = response.read()
    return pd.read_parquet(io.BytesIO(blob))


def _valid_messages(raw: str):
    """Parse a conversation that contains at least one assistant turn."""
    try:
        messages = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(messages, list) or not messages:
        return None
    if not any(m.get("role") == "assistant" for m in messages):
        return None
    return messages


def load_liars_bench_rows(dataset: str, split: str = "dev"):
    """Return usable examples, labels, and captured source-group names."""
    df = fetch_records(dataset, split=split)

    keep = df["error"].isna() & (df["n_tokens"] > 0)
    dropped = int((~keep).sum())
    if dropped:
        print(f"  [{dataset}/{split}] dropping {dropped}/{len(df)} rows "
              f"(error set or n_tokens==0 -- failed capture)")
    df = df[keep]

    examples, labels, organisms = [], [], []
    n_unparseable = 0
    for _, row in df.iterrows():
        messages = _valid_messages(row["messages"])
        if messages is None:
            n_unparseable += 1
            continue
        examples.append({"messages": messages, "index": int(row["index"])})
        labels.append(bool(row["deceptive"]))
        organisms.append(str(row["captured"]))

    if n_unparseable:
        print(
            f"  [{dataset}/{split}] skipped {n_unparseable} "
            "unparseable conversations"
        )
    print(
        f"  [{dataset}/{split}] loaded {len(examples)} usable rows across "
        f"{len(set(organisms))} source groups"
    )
    return examples, np.asarray(labels), np.asarray(organisms)
