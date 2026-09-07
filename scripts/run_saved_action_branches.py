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
    parser.add_argument("--actions", required=True)
    parser.add_argument("--task", default="PickXtimes")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    def factory():
        return BenchmarkEnvBuilder(
            args.task, dataset="test", action_space="joint_angle", gui_render=False, max_steps=100
        ).make_env_for_episode(args.episode)

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
