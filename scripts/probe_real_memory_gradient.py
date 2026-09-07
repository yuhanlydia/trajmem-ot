#!/usr/bin/env python3
"""Frozen MME-VLA memory-gradient probe using demonstration action error as Q.

This is an integration gate, not a RoboMME return result. It estimates a
black-box directional gradient in a low-rank memory subspace while holding the
model, observation, instruction and flow noise exactly fixed.
"""

import argparse
import json

import numpy as np

from mme_vla_suite.training.dataset import SampleDataset
from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--probe-radius", type=float, default=5e-4)
    parser.add_argument("--trust-radius", type=float, default=1e-3)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--actions-output")
    args = parser.parse_args()

    rng = np.random.default_rng(2026)
    item = SampleDataset(args.data)[args.index]
    client = MMEVLAWebsocketClientPolicy(args.host, args.port)
    client.reset()
    client.add_buffer(
        {
            "images": item["image"][None, None].astype(np.uint8),
            "state": item["state"][None].astype(np.float32),
            "add_buffer": True,
            "exec_start_idx": 0,
        }
    )
    memory = client.get_history()["static_image_emb"].astype(np.float32)
    memory_norm = np.linalg.norm(memory)
    noise = rng.normal(size=(20, 32)).astype(np.float32)
    observation = {
        "observation/image": item["image"],
        "observation/wrist_image": item["wrist_image"],
        "observation/state": item["state"],
        "prompt": item["prompt"],
        "_trajmem_noise": noise,
    }
    target = np.asarray(item["actions"][:20], dtype=np.float32)

    def score(delta: np.ndarray | None) -> float:
        if delta is None:
            client.clear_history_delta()
        else:
            client.set_history_delta(delta.astype(np.float32))
        action = client.infer(observation)["actions"]
        return -float(np.mean(np.square(action - target)))

    base_q = score(None)
    directions = []
    derivatives = []
    probe_norm = args.probe_radius * memory_norm
    for _ in range(args.rank):
        temporal = rng.normal(size=(memory.shape[0], 1)).astype(np.float32)
        feature = rng.normal(size=(1, memory.shape[1])).astype(np.float32)
        direction = temporal @ feature
        direction *= probe_norm / np.linalg.norm(direction)
        q_plus = score(direction)
        q_minus = score(-direction)
        directions.append(direction / probe_norm)
        derivatives.append((q_plus - q_minus) / (2 * probe_norm))

    gradient = sum(coef * direction for coef, direction in zip(derivatives, directions))
    delta = gradient * (args.trust_radius * memory_norm / max(np.linalg.norm(gradient), 1e-12))
    random_scores = []
    for _ in range(args.random_controls):
        random_delta = rng.normal(size=memory.shape).astype(np.float32)
        random_delta *= np.linalg.norm(delta) / np.linalg.norm(random_delta)
        random_scores.append(score(random_delta))
    positive_q = score(delta)
    result = {
        "metric": "negative demonstration action MSE (oracle integration probe, not environment return)",
        "memory_shape": list(memory.shape),
        "rank": args.rank,
        "relative_delta_norm": float(np.linalg.norm(delta) / memory_norm),
        "q_original": base_q,
        "q_positive": positive_q,
        "q_negative": score(-delta),
        "q_random_mean": float(np.mean(random_scores)),
        "q_random_std": float(np.std(random_scores)),
        "q_random_all": random_scores,
        "positive_percentile_vs_random": float(np.mean(positive_q > np.asarray(random_scores))),
        "directional_derivatives": derivatives,
    }
    client.clear_history_delta()
    if args.actions_output:
        def action_for(edit):
            if edit is None:
                client.clear_history_delta()
            else:
                client.set_history_delta(edit.astype(np.float32))
            return client.infer(observation)["actions"]

        np.savez(
            args.actions_output,
            original=action_for(None),
            positive=action_for(delta),
            negative=action_for(-delta),
            random=action_for(random_delta),
            delta=delta,
            noise=noise,
        )
        client.clear_history_delta()
        result["actions_output"] = args.actions_output
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
