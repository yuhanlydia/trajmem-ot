#!/usr/bin/env python3
"""Compact, auditable frozen-policy memory sensitivity pilot.

This intentionally does not claim return-gradient optimization: online return
VJP is not implemented yet. It measures paired closed-loop sensitivity to a
fixed, norm-matched low-rank history edit on RoboMME train episodes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "third_party/robomme_policy_learning/third_party/robomme_benchmark/src"))
sys.path.insert(0, str(_ROOT / "third_party/robomme_policy_learning/src"))
from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
from robomme.env_record_wrapper import BenchmarkEnvBuilder


TASKS = ["PickXtimes", "VideoUnmask", "VideoPlaceButton", "PatternLock"]
CONDITIONS = ["M", "probe_plus", "probe_minus", "random"]


def obs_pack(obs, goal):
    state = np.concatenate([obs["joint_state_list"][-1], obs["gripper_state_list"][-1][:1]]).astype(np.float32)
    return {
        "observation/image": np.asarray(obs["front_rgb_list"][-1]),
        "observation/wrist_image": np.asarray(obs["wrist_rgb_list"][-1]),
        "observation/state": state,
        "prompt": goal,
    }, state


def run_one(task, episode, condition, port, max_steps, seed):
    builder = BenchmarkEnvBuilder(task, dataset="train", action_space="joint_angle", gui_render=False, max_steps=max_steps)
    env = builder.make_env_for_episode(episode)
    obs, info = env.reset()
    goal = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]
    client = MMEVLAWebsocketClientPolicy("127.0.0.1", port)
    client.reset()
    packed, state = obs_pack(obs, goal)
    client.add_buffer({"images": packed["observation/image"][None, None], "state": state[None], "add_buffer": True, "exec_start_idx": 0})
    history = client.get_history()
    memory = history["static_image_emb"].astype(np.float32)
    norm = float(np.linalg.norm(memory))
    rng = np.random.default_rng(seed)
    temporal = rng.normal(size=(memory.shape[0], 1)).astype(np.float32)
    feature = rng.normal(size=(1, memory.shape[1])).astype(np.float32)
    direction = temporal @ feature
    direction *= 0.00025 * norm / np.linalg.norm(direction)
    if condition == "probe_plus":
        client.set_history_delta(direction)
    elif condition == "probe_minus":
        client.set_history_delta(-direction)
    elif condition == "random":
        random_delta = rng.normal(size=memory.shape).astype(np.float32)
        random_delta *= np.linalg.norm(direction) / np.linalg.norm(random_delta)
        client.set_history_delta(random_delta)
    total_return = 0.0
    status = "unknown"
    steps = 0
    max_chunk = 20
    while steps < max_steps:
        packed, state = obs_pack(obs, goal)
        packed["_trajmem_noise"] = np.random.default_rng(seed + steps).normal(size=(20, 32)).astype(np.float32)
        actions = client.infer(packed)["actions"]
        for action in actions[:max_chunk]:
            obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            total_return += float(np.asarray(reward).reshape(-1)[-1])
            steps += 1
            status = str(info.get("status", "unknown"))
            if terminated or truncated or steps >= max_steps:
                break
        _, state = obs_pack(obs, goal)
        client.add_buffer({"images": np.asarray(obs["front_rgb_list"][-1])[None, None], "state": state[None], "add_buffer": True, "exec_start_idx": 0})
        if status in {"success", "fail", "timeout", "error"}:
            break
    env.close()
    return {"task": task, "episode": episode, "condition": condition, "steps": steps,
            "return": total_return, "status": status, "success": status == "success"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--episodes-per-task", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    jobs = [(task, ep, cond) for task in TASKS for ep in range(args.episodes_per_task) for cond in CONDITIONS]
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard]
    out = Path(args.output)
    done = {json.loads(line)["key"] for line in out.read_text().splitlines()} if out.exists() else set()
    for task, ep, cond in jobs:
        key = f"{task}:{ep}:{cond}"
        if key in done:
            continue
        try:
            row = run_one(task, ep, cond, args.port, args.max_steps, 20260822 + ep)
        except Exception as exc:
            row = {"task": task, "episode": ep, "condition": cond, "error": repr(exc)}
        row["key"] = key
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
