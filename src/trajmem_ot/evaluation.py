from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import Iterable, Mapping

import numpy as np


def summarize_memory_line_search(rows: list[Mapping]) -> list[dict]:
    """Summarize a symmetric E7 sweep without treating it as environment return."""
    if not rows:
        raise ValueError("at least one E7 row is required")
    summary = []
    for sweep_index in range(1, len(rows[0]["sweep"])):
        values = [row["sweep"][sweep_index] for row in rows]
        rho = float(values[0]["rho"])
        if rho <= 0 or any(float(value["rho"]) != rho for value in values):
            raise ValueError("E7 rows must share the same positive radii")
        q0 = np.asarray([row["q0"] for row in rows], dtype=np.float64)
        plus = np.asarray([value["q_plus"] for value in values], dtype=np.float64)
        minus = np.asarray([value["q_minus"] for value in values], dtype=np.float64)
        random_wins = [plus[i] > score for i, value in enumerate(values) for score in value["q_random"]]
        # rho is relative memory norm, so this is dQ / d(relative edit) at zero.
        central_slope = (plus - minus) / (2.0 * rho)
        summary.append(
            {
                "rho": rho,
                "n_states": len(rows),
                "p_plus_gt_base": float(np.mean(plus > q0)),
                "p_plus_gt_minus": float(np.mean(plus > minus)),
                "p_plus_gt_random": float(np.mean(random_wins)),
                "p_positive_central_slope": float(np.mean(central_slope > 0)),
                "mean_central_slope": float(np.mean(central_slope)),
                "mean_delta_q": float(np.mean(plus - q0)),
                "median_delta_q": float(np.median(plus - q0)),
            }
        )
    return summary


def select_trust_radius(
    summaries: Iterable[Mapping], *, probability_floor: float = 0.65
) -> float | None:
    """Choose the smallest radius passing the preregistered E7 checks.

    Returning None is deliberate: a failed geometry gate must not silently
    promote a merely descriptive best radius into a simulator experiment.
    Random controls and plus-vs-base probabilities remain required report
    fields, but they are not used to tune the radius.
    """
    passing = [
        row
        for row in summaries
        if row["p_positive_central_slope"] >= probability_floor
        and row["median_delta_q"] > 0
    ]
    return min((float(row["rho"]) for row in passing), default=None)


def _two_sided_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize_paired_return(rows: Iterable[Mapping], *, baseline: str = "M") -> dict:
    """Summarize E8 outcomes paired by branch state and flow-noise seed."""
    groups: dict[tuple, dict[str, Mapping]] = defaultdict(dict)
    for row in rows:
        if row.get("error"):
            continue
        pair_key = (row["task"], row["episode"], row["branch_step"], row["noise_seed"])
        condition = str(row["condition"])
        if condition in groups[pair_key]:
            raise ValueError(f"duplicate condition {condition!r} for pair {pair_key}")
        groups[pair_key][condition] = row

    conditions = sorted({condition for group in groups.values() for condition in group})
    complete = [group for group in groups.values() if all(condition in group for condition in conditions)]
    if baseline not in conditions or not complete:
        raise ValueError("no complete paired E8 groups with the requested baseline")

    report = {"n_pairs": len(complete), "baseline": baseline, "by_condition": {}, "paired_vs_baseline": {}}
    for condition in conditions:
        part = [group[condition] for group in complete]
        report["by_condition"][condition] = {
            "success_rate": float(np.mean([bool(row["success"]) for row in part])),
            "mean_return": float(np.mean([float(row["return"]) for row in part])),
            "mean_steps": float(np.mean([int(row["steps"]) for row in part])),
        }
        if condition == baseline:
            continue
        wins = losses = ties = success_gains = success_losses = 0
        for group in complete:
            candidate, control = group[condition], group[baseline]
            delta = float(candidate["return"]) - float(control["return"])
            wins += delta > 0
            losses += delta < 0
            ties += delta == 0
            success_gains += bool(candidate["success"]) and not bool(control["success"])
            success_losses += bool(control["success"]) and not bool(candidate["success"])
        report["paired_vs_baseline"][condition] = {
            "return_wins": wins,
            "return_losses": losses,
            "return_ties": ties,
            "return_sign_test_p": _two_sided_sign_test(wins, losses),
            "success_gains": success_gains,
            "success_losses": success_losses,
            "success_sign_test_p": _two_sided_sign_test(success_gains, success_losses),
        }
    return report
