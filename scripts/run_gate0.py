#!/usr/bin/env python3
import argparse
import json

from trajmem_ot.video_gate0 import run_gate0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [{"seed": seed, **vars(run_gate0(seed, args.device))} for seed in args.seeds]
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
