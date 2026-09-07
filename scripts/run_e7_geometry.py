#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.dataset import RoboMMEDataset
from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy


RADII = np.asarray([0.0, 0.00003125, 0.0000625, 0.000125, 0.00025, 0.0005, 0.001], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--probe-radius", type=float, default=0.00025)
    parser.add_argument("--random-controls", type=int, default=64)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    manifest = [row for i, row in enumerate(manifest) if i % args.num_shards == args.shard]
    if args.limit is not None:
        manifest = manifest[: args.limit]
    output = Path(args.output)
    completed = set()
    if output.exists():
        completed = {json.loads(line)["index"] for line in output.read_text().splitlines() if line.strip()}

    history_config = get_history_config("perceptual-framesamp-modul.yaml")
    dataset = RoboMMEDataset(args.data, None, history_config, action_horizon=20, compute_norm_stats=True)
    client = MMEVLAWebsocketClientPolicy(args.host, args.port)

    for row in manifest:
        if row["index"] in completed:
            continue
        item = dataset[row["index"]]
        history = {
            "static_image_emb": item["static_image_emb"].astype(np.float32),
            "static_pos_emb": item["static_pos_emb"].astype(np.float32),
            "static_state_emb": item["static_state_emb"].astype(np.float32),
            "static_mask": item["static_mask"],
        }
        memory = history["static_image_emb"]
        memory_norm = float(np.linalg.norm(memory))
        rng = np.random.default_rng(100000 + row["index"])
        noise = rng.normal(size=(20, 32)).astype(np.float32)
        observation = {
            "observation/image": item["image"], "observation/wrist_image": item["wrist_image"],
            "observation/state": item["state"], "prompt": item["prompt"], "_trajmem_noise": noise,
        }
        target = np.asarray(item["actions"], dtype=np.float32)
        client.set_history_override(history)

        def score(delta=None):
            if delta is None:
                client.clear_history_delta()
            else:
                client.set_history_delta(delta.astype(np.float32))
            action = client.infer(observation)["actions"]
            return -float(np.mean(np.square(action - target)))

        q0 = score()
        probe_norm = args.probe_radius * memory_norm
        directions, derivatives = [], []
        for _ in range(args.rank):
            temporal = rng.normal(size=(memory.shape[0], 1)).astype(np.float32)
            feature = rng.normal(size=(1, memory.shape[1])).astype(np.float32)
            direction = temporal @ feature
            direction /= np.linalg.norm(direction)
            derivative = (score(probe_norm * direction) - score(-probe_norm * direction)) / (2 * probe_norm)
            directions.append(direction)
            derivatives.append(derivative)
        gram = np.asarray(
            [[float(np.vdot(a, b)) for b in directions] for a in directions],
            dtype=np.float64,
        )
        coefficients = np.linalg.solve(gram + 1e-6 * np.eye(args.rank), np.asarray(derivatives))
        gradient = sum(coef * direction for coef, direction in zip(coefficients, directions))
        gradient /= max(np.linalg.norm(gradient), 1e-12)

        sweep = []
        for rho in RADII:
            delta = rho * memory_norm * gradient
            q_plus, q_minus = score(delta), score(-delta)
            random_q = []
            if rho > 0:
                for _ in range(args.random_controls):
                    random_direction = rng.normal(size=memory.shape).astype(np.float32)
                    random_direction *= rho * memory_norm / np.linalg.norm(random_direction)
                    random_q.append(score(random_direction))
            sweep.append({"rho": float(rho), "q_plus": q_plus, "q_minus": q_minus,
                          "q_random": random_q})
        result = {**row, "q0": q0, "rank": args.rank,
                  "probe_radius": args.probe_radius,
                  "directional_derivatives": derivatives, "sweep": sweep}
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        client.clear_history_delta()
        client.clear_history_override()
        print(json.dumps({"completed": row["index"], "task": row["task"]}))


if __name__ == "__main__":
    main()
