#!/usr/bin/env python3
import argparse
from dataclasses import asdict
import json
import random

import numpy as np

from trajmem_ot.robomme_branch import ReplayBranchRunner
from trajmem_ot.headless_physics import patch_sapien_for_physics_only, physics_only_gym_make


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickXtimes")
    parser.add_argument("--dataset", choices=("train", "test"), default="test")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--prefix-steps", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--physics-only", action="store_true")
    args = parser.parse_args()

    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    import robomme.robomme_env  # noqa: F401

    if args.physics_only:
        import gymnasium as gym
        import sapien
        from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper

        patch_sapien_for_physics_only(sapien)
        gym.make = physics_only_gym_make(gym.make)
        DemonstrationWrapper._augment_obs_and_info = lambda self, obs, info, action: (obs, info)

    def factory():
        builder = BenchmarkEnvBuilder(
            args.task, dataset=args.dataset, action_space="joint_angle", gui_render=False, max_steps=100
        )
        if args.physics_only:
            import torch

            episode_seed, _ = builder.resolve_episode(args.episode)
            replay_seed = args.episode if episode_seed is None else episode_seed
            random.seed(replay_seed)
            np.random.seed(replay_seed)
            torch.manual_seed(replay_seed)
        return builder.make_env_for_episode(args.episode)

    runner = ReplayBranchRunner(factory)
    base = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4, 1], dtype=np.float32)
    prefix = [base.copy() for _ in range(args.prefix_steps)]
    start = runner.assert_deterministic(prefix)
    branches = {}
    for branch_id, offset in {"negative": -0.03, "original": 0.0, "positive": 0.03}.items():
        actions = []
        for _ in range(args.future_steps):
            action = base.copy()
            action[0] += offset
            actions.append(action)
        branches[branch_id] = asdict(runner.rollout(branch_id, prefix, actions))
    starts = {row["start_fingerprint"] for row in branches.values()}
    if starts != {start}:
        raise RuntimeError(f"branches did not start from the same state: {starts}")
    print(json.dumps({"deterministic_start": start, "branches": branches}, indent=2))


if __name__ == "__main__":
    main()
