from dataclasses import dataclass, replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from trajmem_ot.robomme_jax import (
    build_fixed_noise_action_problem,
    extract_observation_history,
    prepare_policy_observation,
    replace_observation_history,
    replace_observation_memory,
)


@dataclass(frozen=True)
class DummyObservation:
    static_image_emb: object
    static_mask: object
    state: object
    images: object
    prompt: object = None

    def replace(self, **changes):
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data):
        return cls(
            static_image_emb=data["static_image_emb"],
            static_mask=data["static_mask"],
            state=data["observation/state"],
            images={"base": data["observation/image"]},
            prompt=data.get("prompt"),
        )


@dataclass(frozen=True)
class FullDummyObservation:
    static_image_emb: object
    static_pos_emb: object
    static_state_emb: object
    static_mask: object
    state: object
    images: object
    prompt: object = None

    def replace(self, **changes):
        return replace(self, **changes)


class DummyPolicy:
    def __init__(self):
        self._input_transform = lambda inputs: inputs
        self.seen_states = []
        self.seen_noises = []

    def _sample_actions(self, rng, observation, *, num_steps, noise):
        del rng, num_steps
        self.seen_states.append(observation.state)
        self.seen_noises.append(noise)
        memory_signal = observation.static_image_emb.mean(axis=(-2, -1))[:, None, None]
        return noise + memory_signal


def test_replace_observation_memory_changes_only_history_field():
    observation = DummyObservation(
        static_image_emb=jnp.zeros((1, 3, 2)),
        static_mask=jnp.ones((1, 3), dtype=bool),
        state=jnp.asarray([[1.0, 2.0]]),
        images={"base": jnp.ones((1, 2, 2, 3))},
    )
    replacement = jnp.ones((1, 3, 2))
    changed = replace_observation_memory(
        observation, replacement, field="static_image_emb"
    )
    np.testing.assert_array_equal(
        np.asarray(changed.static_image_emb), np.ones((1, 3, 2))
    )
    assert changed.state is observation.state
    assert changed.images is observation.images
    with pytest.raises(ValueError, match="editable history memory"):
        replace_observation_memory(observation, observation.state, field="state")


def test_history_bundle_transplant_preserves_current_inputs():
    base = FullDummyObservation(
        static_image_emb=jnp.zeros((1, 3, 2)),
        static_pos_emb=jnp.zeros((1, 3, 1)),
        static_state_emb=jnp.zeros((1, 3, 2)),
        static_mask=jnp.asarray([[True, True, False]]),
        state=jnp.asarray([[1.0, 2.0]]),
        images={"base": jnp.ones((1, 2, 2, 3))},
        prompt="task",
    )
    donor = FullDummyObservation(
        static_image_emb=jnp.ones((1, 3, 2)),
        static_pos_emb=jnp.ones((1, 3, 1)),
        static_state_emb=jnp.ones((1, 3, 2)),
        static_mask=jnp.asarray([[False, True, True]]),
        state=jnp.asarray([[9.0, 9.0]]),
        images={"base": jnp.zeros((1, 2, 2, 3))},
        prompt="other",
    )
    changed = replace_observation_history(
        base, extract_observation_history(donor, representation="static")
    )
    np.testing.assert_array_equal(changed.static_image_emb, donor.static_image_emb)
    np.testing.assert_array_equal(changed.static_pos_emb, donor.static_pos_emb)
    np.testing.assert_array_equal(changed.static_state_emb, donor.static_state_emb)
    np.testing.assert_array_equal(changed.static_mask, donor.static_mask)
    assert changed.state is base.state
    assert changed.images is base.images
    assert changed.prompt == base.prompt
    with pytest.raises(ValueError, match="non-history"):
        replace_observation_history(base, {"state": donor.state})


def test_fixed_noise_action_problem_exposes_unbatched_memory_and_reuses_noise():
    policy = DummyPolicy()
    observation = DummyObservation(
        static_image_emb=jnp.zeros((1, 3, 2)),
        static_mask=jnp.ones((1, 3), dtype=bool),
        state=jnp.asarray([[1.0, 2.0]]),
        images={"base": jnp.ones((1, 2, 2, 3))},
    )
    noise = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4)
    problem = build_fixed_noise_action_problem(
        policy,
        observation,
        noise=noise,
        rng=jax.random.key(0),
        memory_field="static_image_emb",
        num_steps=5,
    )
    assert problem.memory.shape == (3, 2)
    first = problem.action_fn(problem.memory)
    second = problem.action_fn(problem.memory)
    changed = problem.action_fn(problem.memory + 1.0)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(changed))
    assert all(seen is noise for seen in policy.seen_noises)
    assert all(seen is observation.state for seen in policy.seen_states)


def test_fixed_noise_action_problem_accepts_explicit_sampler():
    policy = DummyPolicy()
    observation = DummyObservation(
        static_image_emb=jnp.zeros((1, 3, 2)),
        static_mask=jnp.ones((1, 3), dtype=bool),
        state=jnp.asarray([[1.0, 2.0]]),
        images={"base": jnp.ones((1, 2, 2, 3))},
    )
    noise = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    calls = []

    def explicit_sampler(rng, obs, *, num_steps, noise):
        calls.append((rng, obs, num_steps, noise))
        return policy._sample_actions(rng, obs, num_steps=num_steps, noise=noise)

    problem = build_fixed_noise_action_problem(
        policy,
        observation,
        noise=noise,
        rng=jax.random.key(0),
        sample_actions_fn=explicit_sampler,
    )
    problem.action_fn(problem.memory)
    assert len(calls) == 1


def test_prepare_policy_observation_batches_current_inputs_and_history():
    policy = DummyPolicy()
    item = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
        "prompt": "pick",
        "static_image_emb": np.zeros((3, 2), dtype=np.float32),
        "static_mask": np.ones((3,), dtype=bool),
        "static_pos_emb": np.zeros((3, 1), dtype=np.float32),
        "static_state_emb": np.zeros((3, 2), dtype=np.float32),
    }
    observation = prepare_policy_observation(
        policy, item, observation_cls=DummyObservation
    )
    assert observation.static_image_emb.shape == (1, 3, 2)
    assert observation.state.shape == (1, 2)
    assert observation.images["base"].shape == (1, 4, 4, 3)
