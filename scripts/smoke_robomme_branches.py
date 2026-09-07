#!/usr/bin/env python3
import argparse
from dataclasses import asdict
import json

import numpy as np

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env import *  # noqa: F403
from trajmem_ot.robomme_branch import ReplayBranchRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickXtimes")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--prefix-steps", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=8)
    args = parser.parse_args()

    def factory():
        builder = BenchmarkEnvBuilder(
            args.task, dataset="test", action_space="joint_angle", gui_render=False, max_steps=100
        )
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
