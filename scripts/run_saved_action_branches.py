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
    parser.add_argument("--actions", required=True)
    parser.add_argument("--task", default="PickXtimes")
    parser.add_argument("--dataset", choices=("train", "test"), default="test")
    parser.add_argument("--episode", type=int, default=0)
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
    archive = np.load(args.actions)
    prefix = []
    start = runner.assert_deterministic(prefix)
    outcomes = {
        name: asdict(runner.rollout(name, prefix, archive[name]))
        for name in ("original", "positive", "negative", "random")
    }
    if {row["start_fingerprint"] for row in outcomes.values()} != {start}:
        raise RuntimeError("action branches did not begin from the identical state")
    print(json.dumps({"start": start, "outcomes": outcomes}, indent=2))


if __name__ == "__main__":
    main()
