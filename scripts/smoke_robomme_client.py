#!/usr/bin/env python3
"""Send one official RoboMME sample through a running MME-VLA server."""

import argparse
import json
import time

import numpy as np

from mme_vla_suite.training.dataset import SampleDataset
from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    item = SampleDataset(args.data)[args.index]
    client = MMEVLAWebsocketClientPolicy(host=args.host, port=args.port)
    client.reset()
    buffer_response = client.add_buffer(
        {
            "images": item["image"][None, None].astype(np.uint8),
            "state": item["state"][None].astype(np.float32),
            "add_buffer": True,
            "exec_start_idx": 0,
        }
    )
    observation = {
        "observation/image": item["image"],
        "observation/wrist_image": item["wrist_image"],
        "observation/state": item["state"],
        "prompt": item["prompt"],
    }
    start = time.monotonic()
    output = client.infer(observation)
    actions = output["actions"]
    print(
        json.dumps(
            {
                "buffer": buffer_response,
                "actions_shape": list(actions.shape),
                "finite": bool(np.isfinite(actions).all()),
                "wall_latency_s": round(time.monotonic() - start, 3),
                "server_infer_ms": round(float(output["infer_time_ms"]), 2),
                "first_action": actions[0].tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
