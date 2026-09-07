from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .robomme_jax import prepare_policy_observation


@dataclass(frozen=True)
class RoboMMERuntimeState:
    policy: Any
    item: dict[str, Any]
    observation: Any
    history_config_name: str


def resolve_history_config_name(checkpoint: str | Path, explicit: str | None) -> str:
    if explicit is not None:
        value = explicit.strip()
        if not value:
            raise ValueError("explicit history config cannot be empty")
        return value
    checkpoint_path = Path(checkpoint)
    metadata_path = checkpoint_path.parent / "history_config.txt"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"history_config metadata not found at {metadata_path}; pass --history-config explicitly"
        )
    value = metadata_path.read_text().strip()
    if not value:
        raise ValueError(f"history_config metadata is empty: {metadata_path}")
    return value


def load_runtime_state(
    *,
    checkpoint: str | Path,
    data: str | Path,
    index: int,
    seed: int,
    train_config_name: str = "mme_vla_suite",
    history_config_name: str | None = None,
) -> RoboMMERuntimeState:
    """Load one released RoboMME checkpoint state in the policy process.

    Imports are deliberately local so ordinary package imports and CPU unit
    tests do not require the large upstream RoboMME/OpenPI environment.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    try:
        from mme_vla_suite.models.config.utils import get_history_config
        from mme_vla_suite.policies import policy_config as _policy_config
        from mme_vla_suite.training import config as _config
        from mme_vla_suite.training.dataset import RoboMMEDataset
    except ImportError as exc:  # pragma: no cover - only available in real runtime
        raise RuntimeError(
            "RoboMME/OpenPI is not importable. Run scripts/bootstrap_robomme.sh "
            "and execute this command from the upstream uv environment."
        ) from exc

    checkpoint_path = Path(checkpoint)
    config_name = resolve_history_config_name(checkpoint_path, history_config_name)
    policy = _policy_config.create_trained_policy(
        _config.get_config(train_config_name), checkpoint_path, seed=seed
    )
    history_config = get_history_config(config_name)
    dataset = RoboMMEDataset(
        str(data),
        None,
        history_config,
        action_horizon=int(policy._model.action_horizon),
        compute_norm_stats=True,
    )
    if index >= len(dataset):
        raise IndexError(f"index {index} is outside dataset length {len(dataset)}")
    item = dataset[index]
    observation = prepare_policy_observation(policy, item)
    return RoboMMERuntimeState(
        policy=policy,
        item=item,
        observation=observation,
        history_config_name=config_name,
    )
