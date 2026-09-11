from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class BranchOutcome:
    branch_id: str
    return_sum: float
    success: bool
    status: str
    steps: int
    start_fingerprint: str
    final_fingerprint: str
    start_progress: dict | None = None
    final_progress: dict | None = None


def task_progress_snapshot(env: object) -> dict:
    """Count execution subgoals without demonstrations or unrecorded setup steps."""
    raw = getattr(env, "unwrapped", env)
    tasks = getattr(raw, "task_list", None)
    index = getattr(raw, "current_task_index", None)
    cursor = getattr(raw, "timestep", index)
    if tasks is None or cursor is None or any(not isinstance(task, dict) for task in tasks):
        return {"task_index": None, "completed_subgoals": None,
                "completion_cursor": None, "total_subgoals": None, "progress_fraction": None}
    index = None if index is None else int(index)
    cursor = int(cursor)
    eligible = [i for i, task in enumerate(tasks)
                if not task.get("demonstration", False)
                and task.get("subgoal_segment", task.get("name")) != "NO RECORD"]
    # Upstream increments timestep immediately on completion but may leave the
    # display index stale until the next evaluation call.
    completed = sum(i < cursor for i in eligible)
    return {"task_index": index, "completion_cursor": cursor, "completed_subgoals": completed,
            "total_subgoals": len(eligible),
            "progress_fraction": completed / len(eligible) if eligible else None}


def observation_fingerprint(observation: dict) -> str:
    digest = hashlib.sha256()
    for key in ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list"):
        values = observation.get(key, [])
        if values:
            array = np.asarray(values[-1])
            digest.update(key.encode())
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _update_fingerprint(digest: "hashlib._Hash", value: object) -> None:
    if isinstance(value, dict):
        entries = []
        for key, child in value.items():
            stable_key = re.sub(r"(?<=_)\d{6,}(?=_|$)", "<runtime-id>", str(key))
            child_digest = hashlib.sha256()
            _update_fingerprint(child_digest, child)
            entries.append((stable_key, child_digest.digest()))
        for stable_key, child_hash in sorted(entries):
            digest.update(stable_key.encode())
            digest.update(child_hash)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _update_fingerprint(digest, item)
        return
    array = np.asarray(value)
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def branch_fingerprint(env: object, observation: dict) -> str:
    """Fingerprint physics state when available, otherwise rendered observations."""

    unwrapped = getattr(env, "unwrapped", env)
    get_state_dict = getattr(unwrapped, "get_state_dict", None)
    if callable(get_state_dict):
        digest = hashlib.sha256()
        _update_fingerprint(digest, get_state_dict())
        return digest.hexdigest()
    return observation_fingerprint(observation)


class ReplayBranchRunner:
    """Exact paired branching by deterministic episode reconstruction.

    Replaying the common prefix restores both simulator physics and RoboMME's
    Python-side task counters, which are absent from ManiSkill's state_dict.
    """

    def __init__(self, env_factory: Callable[[], object]):
        self.env_factory = env_factory

    def _restore_prefix(self, prefix: Sequence[np.ndarray]) -> tuple[object, dict, dict]:
        env = self.env_factory()
        observation, info = env.reset()
        for action in prefix:
            observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            if bool(terminated) or bool(truncated):
                env.close()
                raise RuntimeError("common action prefix terminates before branch point")
        return env, observation, info

    def assert_deterministic(self, prefix: Sequence[np.ndarray]) -> str:
        first, obs_a, _ = self._restore_prefix(prefix)
        fp_a = branch_fingerprint(first, obs_a)
        first.close()
        second, obs_b, _ = self._restore_prefix(prefix)
        fp_b = branch_fingerprint(second, obs_b)
        second.close()
        if fp_a != fp_b:
            raise RuntimeError(f"RoboMME replay is not deterministic: {fp_a} != {fp_b}")
        return fp_a

    def rollout(
        self, branch_id: str, prefix: Sequence[np.ndarray], future_actions: Sequence[np.ndarray]
    ) -> BranchOutcome:
        env, observation, info = self._restore_prefix(prefix)
        start = branch_fingerprint(env, observation)
        start_progress = task_progress_snapshot(env)
        total_return = 0.0
        status = str(info.get("status", "unknown"))
        steps = 0
        for action in future_actions:
            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            total_return += float(np.asarray(reward).reshape(-1)[-1])
            status = str(info.get("status", "unknown"))
            steps += 1
            if bool(terminated) or bool(truncated):
                break
        final = branch_fingerprint(env, observation)
        final_progress = task_progress_snapshot(env)
        env.close()
        return BranchOutcome(
            branch_id=branch_id,
            return_sum=total_return,
            success=status == "success",
            status=status,
            steps=steps,
            start_fingerprint=start,
            final_fingerprint=final,
            start_progress=start_progress,
            final_progress=final_progress,
        )
