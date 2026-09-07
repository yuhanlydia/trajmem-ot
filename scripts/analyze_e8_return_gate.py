#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from trajmem_ot.evaluation import summarize_paired_return


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze strictly paired full-horizon E8 branches")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for filename in args.inputs:
        rows.extend(json.loads(line) for line in Path(filename).read_text().splitlines() if line.strip())
    report = summarize_paired_return(rows)
    plus = report["paired_vs_baseline"].get("M_plus")
    report["gate_pass"] = bool(
        plus
        and report["by_condition"]["M_plus"]["success_rate"] > report["by_condition"]["M"]["success_rate"]
        and plus["success_gains"] > plus["success_losses"]
    )
    report["claim_scope"] = "paired closed-loop environment return and task success"
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
