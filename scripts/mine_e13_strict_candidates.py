#!/usr/bin/env python3
"""Mine pre-outcome, episode-disjoint E13 pair candidates from execution starts."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from trajmem_ot.pair_mining import ContextRecord, mine_disjoint_context_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--image-stride", type=int, default=16)
    parser.add_argument("--max-front-mae", type=float, default=0.04)
    parser.add_argument("--max-wrist-mae", type=float, default=0.08)
    parser.add_argument("--max-state-l2", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.image_stride <= 0:
        raise ValueError("limit and image-stride must be positive")
    starts: dict[tuple[str, str, int], tuple[tuple[int, int], ContextRecord]] = {}
    for path in (args.data / "data").glob("*.pkl"):
        with path.open("rb") as handle:
            item = pickle.load(handle)
        episode = int(np.asarray(item["epis_idx"]).reshape(-1)[0])
        step = int(np.asarray(item["step_idx"]).reshape(-1)[0])
        execution_start = int(np.asarray(item["exec_start_idx"]).reshape(-1)[0])
        prompt = str(item["prompt"])
        subgoal = str(item["simple_subgoal"])
        record = ContextRecord(
            index=int(path.stem),
            episode=episode,
            prompt=prompt,
            subgoal=subgoal,
            front=np.asarray(item["image"])[:: args.image_stride, :: args.image_stride],
            wrist=np.asarray(item["wrist_image"])[:: args.image_stride, :: args.image_stride],
            state=np.asarray(item["state"], dtype=np.float32),
        )
        key = (prompt, subgoal, episode)
        rank = (abs(step - execution_start), step)
        if key not in starts or rank < starts[key][0]:
            starts[key] = (rank, record)

    pairs = mine_disjoint_context_pairs(
        [row[1] for row in starts.values()],
        max_front_mae=args.max_front_mae,
        max_wrist_mae=args.max_wrist_mae,
        max_state_l2=args.max_state_l2,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pairs, indent=2) + "\n")
    print(json.dumps({"execution_start_records": len(starts), "selected_pairs": len(pairs), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
