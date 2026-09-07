#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

import numpy as np

from mme_vla_suite.training.dataset import SampleDataset


def classify(prompt: str, episode: int) -> str | None:
    prompt = prompt.lower()
    if "repeating this action" in prompt and "place it on the target" in prompt:
        return "PickXtimes"
    if 20 <= episode <= 24 and "watch the video" in prompt and "container hiding" in prompt:
        return "VideoUnmask"
    if "target right after the button" in prompt or "target right before the button" in prompt:
        return "VideoPlaceButton"
    if "retrace the same pattern" in prompt:
        return "PatternLock"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--states-per-task", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = SampleDataset(args.data)
    rows = []
    by_episode = defaultdict(list)
    for index in range(len(dataset)):
        item = dataset[index]
        episode = int(item["epis_idx"].item())
        task = classify(str(item["prompt"]), episode)
        if task is None:
            continue
        by_episode[(task, episode)].append((index, int(item["step_idx"].item()), int(item["exec_start_idx"].item())))

    for task in ("PickXtimes", "VideoUnmask", "VideoPlaceButton", "PatternLock"):
        eligible = []
        for (candidate_task, episode), samples in sorted(by_episode.items()):
            if candidate_task != task:
                continue
            max_step = max(step for _, step, _ in samples)
            exec_start = samples[0][2]
            threshold = exec_start + 0.3 * (max_step - exec_start)
            eligible.extend((index, episode, step, exec_start, max_step) for index, step, _ in samples if step > threshold)
        if len(eligible) < args.states_per_task:
            raise RuntimeError(f"only {len(eligible)} eligible states for {task}")
        positions = np.linspace(0, len(eligible) - 1, args.states_per_task).round().astype(int)
        for slot, position in enumerate(positions):
            index, episode, step, exec_start, max_step = eligible[position]
            rows.append({"task": task, "slot": slot, "index": index, "episode": episode, "step": step,
                         "exec_start": exec_start, "episode_max_step": max_step})
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(json.dumps({"states": len(rows), "per_task": args.states_per_task, "output": args.output}))


if __name__ == "__main__":
    main()
