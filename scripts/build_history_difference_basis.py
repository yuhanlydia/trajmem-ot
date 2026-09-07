#!/usr/bin/env python3
"""Build coherent memory directions from matched RoboMME history pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajmem_ot.memory_basis import history_difference_basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="JSON list of [a,b] or {a,b} dataset indices")
    parser.add_argument("--history-config", default="perceptual-framesamp-modul.yaml")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _pair_indices(row) -> tuple[int, int]:
    if isinstance(row, list) and len(row) == 2:
        return int(row[0]), int(row[1])
    if isinstance(row, dict):
        for left, right in (("a", "b"), ("left_index", "right_index"), ("first", "second")):
            if left in row and right in row:
                return int(row[left]), int(row[right])
    raise ValueError(f"invalid pair entry: {row!r}")


def main() -> None:
    args = parse_args()
    try:
        from mme_vla_suite.models.config.utils import get_history_config
        from mme_vla_suite.training.dataset import RoboMMEDataset
    except ImportError as exc:
        raise RuntimeError("Run this command inside the bootstrapped RoboMME uv environment") from exc

    rows = json.loads(args.pairs.read_text())
    pairs = [_pair_indices(row) for row in rows]
    if not pairs:
        raise ValueError("pairs file is empty")
    history_config = get_history_config(args.history_config)
    dataset = RoboMMEDataset(
        str(args.data), None, history_config, action_horizon=20, compute_norm_stats=True
    )
    differences = []
    for first, second in pairs:
        left = np.asarray(dataset[first]["static_image_emb"], dtype=np.float32)
        right = np.asarray(dataset[second]["static_image_emb"], dtype=np.float32)
        if left.shape != right.shape:
            raise ValueError(f"pair {(first, second)} has mismatched memory shapes")
        differences.append(left - right)
    difference_array = np.stack(differences)
    flat = difference_array.reshape(difference_array.shape[0], -1).astype(np.float64)
    centered = flat - flat.mean(axis=0, keepdims=True) if flat.shape[0] > 1 else flat
    singular_values = np.linalg.svd(centered, compute_uv=False)
    numerical_rank = int(np.sum(singular_values > 1e-12))
    if numerical_rank == 0:
        raise ValueError("paired histories produce no nonzero memory-difference direction")
    count = min(args.count, numerical_rank)
    basis = history_difference_basis(difference_array, count=count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        basis=basis,
        singular_values=singular_values,
        pairs=np.asarray(pairs, dtype=np.int64),
        history_config=np.asarray(args.history_config),
    )
    metadata = {
        "pairs": len(pairs),
        "basis_count": int(basis.shape[0]),
        "memory_shape": list(basis.shape[1:]),
        "history_config": args.history_config,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
