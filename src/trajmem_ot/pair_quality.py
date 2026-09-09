from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PairQualityCriteria:
    require_prompt_match: bool = True
    max_front_image_mae: float = 0.06
    max_wrist_image_mae: float = 0.15
    max_current_state_l2: float = 1.5
    min_history_mask_iou: float = 0.90
    min_history_image_relative_l2: float = 0.03


@dataclass(frozen=True)
class PairQualityAssessment:
    eligible: bool
    failed_reasons: tuple[str, ...]


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pair audit row is missing a finite {key!r}") from exc
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"pair audit row contains a non-finite {key!r}")
    return value


def assess_pair_quality(
    row: Mapping[str, Any], criteria: PairQualityCriteria
) -> PairQualityAssessment:
    """Apply pre-outcome context/contrast criteria to one audited pair."""

    failures: list[str] = []
    if criteria.require_prompt_match and not bool(row.get("prompt_match", False)):
        failures.append("prompt_mismatch")
    if _finite_float(row, "front_image_mae") > criteria.max_front_image_mae:
        failures.append("front_image_mae")
    if _finite_float(row, "wrist_image_mae") > criteria.max_wrist_image_mae:
        failures.append("wrist_image_mae")
    if _finite_float(row, "current_state_l2") > criteria.max_current_state_l2:
        failures.append("current_state_l2")
    if _finite_float(row, "history_mask_iou") < criteria.min_history_mask_iou:
        failures.append("history_mask_iou")
    if (
        _finite_float(row, "history_image_relative_l2")
        < criteria.min_history_image_relative_l2
    ):
        failures.append("insufficient_history_contrast")
    return PairQualityAssessment(eligible=not failures, failed_reasons=tuple(failures))


def load_pair_quality_tiers(path: str | Path) -> dict[str, PairQualityCriteria]:
    """Load named pair-quality tiers from YAML without outcome-dependent fields."""

    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("pair-quality config must be a non-empty mapping")
    tiers: dict[str, PairQualityCriteria] = {}
    allowed = set(PairQualityCriteria.__dataclass_fields__)
    for name, values in payload.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("pair-quality tier names must be non-empty strings")
        if not isinstance(values, Mapping):
            raise ValueError(f"pair-quality tier {name!r} must be a mapping")
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown pair-quality fields for {name!r}: {unknown}")
        tiers[name] = PairQualityCriteria(**dict(values))
    return tiers
