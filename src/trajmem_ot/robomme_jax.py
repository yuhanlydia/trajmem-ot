from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace as dataclass_replace
from typing import Any, Callable, Mapping

import numpy as np

_ALLOWED_HISTORY_FIELDS = frozenset({"static_image_emb", "recur_image_emb"})
_HISTORY_INPUT_FIELDS = (
    "static_image_emb",
    "static_mask",
    "static_pos_emb",
    "static_state_emb",
    "recur_image_emb",
    "recur_mask",
    "recur_pos_emb",
    "recur_state_emb",
)


@dataclass(frozen=True)
class FixedNoiseActionProblem:
    """One frozen VLA state exposed as ``memory -> unbatched actions``."""

    memory: Any
    action_fn: Callable[[Any], Any]
    memory_field: str
    noise: Any
    observation: Any


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("JAX is required for the RoboMME in-process adapter") from exc
    return jax, jnp


def _get_field(observation: Any, field: str) -> Any:
    if isinstance(observation, Mapping):
        if field not in observation:
            raise KeyError(field)
        return observation[field]
    if not hasattr(observation, field):
        raise AttributeError(f"observation has no field {field!r}")
    return getattr(observation, field)


def replace_observation_memory(observation: Any, replacement: Any, *, field: str) -> Any:
    """Replace only a recognized historical memory tensor.

    Current images, current state, prompt/instruction fields, masks, and model
    parameters remain untouched by construction.
    """
    if field not in _ALLOWED_HISTORY_FIELDS:
        choices = ", ".join(sorted(_ALLOWED_HISTORY_FIELDS))
        raise ValueError(f"{field!r} is not a history memory field; choose one of: {choices}")
    current = _get_field(observation, field)
    if tuple(np.shape(replacement)) != tuple(np.shape(current)):
        raise ValueError(
            f"replacement shape {tuple(np.shape(replacement))} != {field} shape {tuple(np.shape(current))}"
        )
    if isinstance(observation, Mapping):
        result = dict(observation)
        result[field] = replacement
        return result
    replace_method = getattr(observation, "replace", None)
    if callable(replace_method):
        return replace_method(**{field: replacement})
    if is_dataclass(observation):
        return dataclass_replace(observation, **{field: replacement})
    raise TypeError("observation must be a mapping or immutable dataclass-like object with replace()")


def build_fixed_noise_action_problem(
    policy: Any,
    observation: Any,
    *,
    noise: Any,
    rng: Any,
    memory_field: str = "static_image_emb",
    num_steps: int = 10,
) -> FixedNoiseActionProblem:
    """Expose an upstream MME-VLA policy as a pure fixed-context action function.

    The upstream policy's compiled ``_sample_actions`` function is used in the
    same process as JAX autodiff. The public function accepts an *unbatched*
    history memory and returns an unbatched action chunk. This avoids crossing
    the WebSocket/NumPy boundary that broke the original E7 gradient path.
    """
    if memory_field not in _ALLOWED_HISTORY_FIELDS:
        raise ValueError(f"{memory_field!r} is not a history memory field")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    sample_actions = getattr(policy, "_sample_actions", None)
    if not callable(sample_actions):
        raise TypeError("policy must expose a callable _sample_actions")

    _, jnp = _jax_modules()
    batched_memory = _get_field(observation, memory_field)
    if batched_memory is None:
        raise ValueError(f"observation field {memory_field!r} is None")
    if len(np.shape(batched_memory)) < 2 or int(np.shape(batched_memory)[0]) != 1:
        raise ValueError(f"{memory_field} must have batch size 1")
    memory = batched_memory[0]

    noise_shape = tuple(np.shape(noise))
    if len(noise_shape) == 2:
        fixed_noise = jnp.asarray(noise)[None, ...]
    elif len(noise_shape) == 3 and noise_shape[0] == 1:
        fixed_noise = noise
    else:
        raise ValueError("noise must have shape [H,A] or [1,H,A]")

    def action_fn(unbatched_memory):
        if tuple(np.shape(unbatched_memory)) != tuple(np.shape(memory)):
            raise ValueError(
                f"memory shape {tuple(np.shape(unbatched_memory))} != base shape {tuple(np.shape(memory))}"
            )
        edited_observation = replace_observation_memory(
            observation, unbatched_memory[None, ...], field=memory_field
        )
        actions = sample_actions(
            rng,
            edited_observation,
            num_steps=num_steps,
            noise=fixed_noise,
        )
        if int(np.shape(actions)[0]) != 1:
            raise ValueError("sample_actions must return batch size 1")
        return actions[0]

    return FixedNoiseActionProblem(
        memory=memory,
        action_fn=action_fn,
        memory_field=memory_field,
        noise=fixed_noise,
        observation=observation,
    )


def _batch_tree_value(value: Any, jnp: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _batch_tree_value(item, jnp) for key, item in value.items()}
    if isinstance(value, (str, bytes)) or value is None:
        return value
    return jnp.asarray(value)[None, ...]


def prepare_policy_observation(
    policy: Any,
    item: Mapping[str, Any],
    *,
    observation_cls: type | None = None,
) -> Any:
    """Prepare one offline RoboMME dataset item for in-process policy calls."""
    _, jnp = _jax_modules()
    transform = getattr(policy, "_input_transform", None)
    if not callable(transform):
        raise TypeError("policy must expose a callable _input_transform")
    required = ("image", "wrist_image", "state", "prompt")
    missing = [key for key in required if key not in item]
    if missing:
        raise KeyError(f"dataset item is missing required fields: {missing}")

    inputs: dict[str, Any] = {
        "observation/image": item["image"],
        "observation/wrist_image": item["wrist_image"],
        "observation/state": item["state"],
        "prompt": item["prompt"],
    }
    for field in _HISTORY_INPUT_FIELDS:
        if field in item and item[field] is not None:
            inputs[field] = item[field]
    transformed = transform(inputs)
    batched = {key: _batch_tree_value(value, jnp) for key, value in transformed.items()}

    if observation_cls is None:
        try:
            from mme_vla_suite.models.integration.history_observation import HistAugObservation
        except ImportError as exc:  # pragma: no cover - only available in RoboMME runtime
            raise RuntimeError(
                "RoboMME is not installed; pass observation_cls for tests or run scripts/bootstrap_robomme.sh"
            ) from exc
        observation_cls = HistAugObservation
    return observation_cls.from_dict(batched)
