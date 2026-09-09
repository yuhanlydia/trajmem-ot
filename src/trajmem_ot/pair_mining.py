from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ContextRecord:
    index: int
    episode: int
    prompt: str
    subgoal: str
    front: np.ndarray
    wrist: np.ndarray
    state: np.ndarray


def _image_mae(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        return float("inf")
    scale = 255.0 if max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) > 1.5 else 1.0
    return float(np.mean(np.abs(a - b)) / scale)


def mine_disjoint_context_pairs(
    records: Sequence[ContextRecord],
    *,
    max_front_mae: float = 0.04,
    max_wrist_mae: float = 0.08,
    max_state_l2: float = 0.75,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Greedily select closest cross-episode pairs without using outcomes."""

    candidates: list[tuple[float, ContextRecord, ContextRecord, float, float, float]] = []
    for offset, left in enumerate(records):
        for right in records[offset + 1 :]:
            if left.episode == right.episode:
                continue
            if (left.prompt, left.subgoal) != (right.prompt, right.subgoal):
                continue
            front = _image_mae(left.front, right.front)
            wrist = _image_mae(left.wrist, right.wrist)
            state = float(np.linalg.norm(np.asarray(left.state) - np.asarray(right.state)))
            if front > max_front_mae or wrist > max_wrist_mae or state > max_state_l2:
                continue
            score = front + wrist + 0.05 * state
            candidates.append((score, left, right, front, wrist, state))

    selected: list[dict[str, Any]] = []
    used_episodes: set[int] = set()
    for score, left, right, front, wrist, state in sorted(
        candidates, key=lambda row: (row[0], row[1].index, row[2].index)
    ):
        if left.episode in used_episodes or right.episode in used_episodes:
            continue
        selected.append(
            {
                "a": left.index,
                "b": right.index,
                "prompt": left.prompt,
                "subgoal": left.subgoal,
                "episodes": [left.episode, right.episode],
                "mining_diagnostics": {
                    "score": score,
                    "front_image_mae_downsampled": front,
                    "wrist_image_mae_downsampled": wrist,
                    "current_state_l2": state,
                },
            }
        )
        used_episodes.update((left.episode, right.episode))
        if limit is not None and len(selected) >= limit:
            break
    return selected
