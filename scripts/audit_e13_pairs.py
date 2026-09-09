#!/usr/bin/env python3
"""Audit candidate E13-B history pairs before expensive action sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trajmem_ot.robomme_jax import extract_observation_history
from trajmem_ot.robomme_runtime import load_runtime_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--history-config")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def _normalized_image_mae(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        return float("inf")
    scale = 255.0 if max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) > 1.5 else 1.0
    return float(np.mean(np.abs(a - b)) / scale)


def _relative_l2(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = max(0.5 * (np.linalg.norm(a) + np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b) / denominator)


def _mask_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def main() -> None:
    args = parse_args()
    pairs = json.loads(args.pairs.read_text())
    if not pairs:
        raise ValueError("pairs file is empty")
    unique_indices = sorted({int(row[key]) for row in pairs for key in ("a", "b")})
    states = load_runtime_states(
        checkpoint=args.checkpoint,
        data=args.data,
        indices=unique_indices,
        seed=args.seed,
        train_config_name=args.config,
        history_config_name=args.history_config,
    )
    by_index = {state.index: state for state in states}

    rows = []
    for pair_index, pair in enumerate(pairs):
        a = by_index[int(pair["a"])]
        b = by_index[int(pair["b"])]
        bundle_a = extract_observation_history(a.observation, representation="static")
        bundle_b = extract_observation_history(b.observation, representation="static")
        prompt_a = _to_string(a.item.get("prompt"))
        prompt_b = _to_string(b.item.get("prompt"))
        rows.append(
            {
                "pair_index": pair_index,
                "a": a.index,
                "b": b.index,
                "prompt_match": prompt_a == prompt_b,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "front_image_mae": _normalized_image_mae(
                    a.item.get("image"), b.item.get("image")
                ),
                "wrist_image_mae": _normalized_image_mae(
                    a.item.get("wrist_image"), b.item.get("wrist_image")
                ),
                "current_state_l2": float(
                    np.linalg.norm(
                        np.asarray(a.item.get("state"), dtype=np.float64)
                        - np.asarray(b.item.get("state"), dtype=np.float64)
                    )
                ),
                "history_image_relative_l2": _relative_l2(
                    bundle_a["static_image_emb"], bundle_b["static_image_emb"]
                ),
                "history_position_relative_l2": _relative_l2(
                    bundle_a["static_pos_emb"], bundle_b["static_pos_emb"]
                ),
                "history_state_relative_l2": _relative_l2(
                    bundle_a["static_state_emb"], bundle_b["static_state_emb"]
                ),
                "history_mask_iou": _mask_iou(
                    bundle_a["static_mask"], bundle_b["static_mask"]
                ),
                "valid_history_tokens_a": int(
                    np.asarray(bundle_a["static_mask"], dtype=bool).sum()
                ),
                "valid_history_tokens_b": int(
                    np.asarray(bundle_b["static_mask"], dtype=bool).sum()
                ),
                "metadata": pair,
            }
        )

    report = {
        "experiment": "E13B_pair_audit",
        "n_pairs": len(rows),
        "pairs": rows,
        "selection_rule": (
            "retain pairs based only on prompt/task compatibility, current-context similarity, "
            "and a task-semantic history contrast; do not select pairs using E13-B outcome"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
