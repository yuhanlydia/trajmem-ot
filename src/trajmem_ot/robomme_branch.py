from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
        first.close()
        second, obs_b, _ = self._restore_prefix(prefix)
        second.close()
        fp_a, fp_b = observation_fingerprint(obs_a), observation_fingerprint(obs_b)
        if fp_a != fp_b:
            raise RuntimeError(f"RoboMME replay is not deterministic: {fp_a} != {fp_b}")
        return fp_a

    def rollout(
        self, branch_id: str, prefix: Sequence[np.ndarray], future_actions: Sequence[np.ndarray]
    ) -> BranchOutcome:
        env, observation, info = self._restore_prefix(prefix)
        start = observation_fingerprint(observation)
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
        final = observation_fingerprint(observation)
        env.close()
        return BranchOutcome(
            branch_id=branch_id,
            return_sum=total_return,
            success=status == "success",
            status=status,
            steps=steps,
            start_fingerprint=start,
            final_fingerprint=final,
        )

