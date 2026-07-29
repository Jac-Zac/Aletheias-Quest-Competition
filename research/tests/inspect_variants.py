"""Print comparable metadata for the current and archived probe artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = {
    "current": ROOT / "submission/trusted_activation_probe.npz",
    "pool16": ROOT / "research/variants/pool16/trusted_activation_probe.npz",
    "lowrank9": ROOT / "research/variants/lowrank9/trusted_activation_probe.npz",
}


def scalar(data, key, default="—"):
    return data[key].item() if key in data.files else default


def main() -> None:
    header = (
        "variant",
        "schema",
        "rows",
        "groups",
        "rank",
        "layers",
        "pools",
        "threshold",
    )
    print("\t".join(header))
    for name, path in ARTIFACTS.items():
        with np.load(path, allow_pickle=False) as data:
            pools = (
                tuple(int(value) for value in data["pool_widths"])
                if "pool_widths" in data.files
                else (int(data["pool_width"]),)
            )
            row = (
                name,
                str(scalar(data, "schema_version")),
                str(scalar(data, "training_rows")),
                str(scalar(data, "training_organisms")),
                str(scalar(data, "low_rank_dimensions")),
                str(tuple(int(value) for value in data["layers"])),
                str(pools),
                f"{float(data['threshold']):.6f}",
            )
        print("\t".join(row))


if __name__ == "__main__":
    main()
