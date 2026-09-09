from types import SimpleNamespace

import numpy as np

from trajmem_ot.headless_physics import (
    PhysicsOnlyTaskEnv,
    patch_sapien_for_physics_only,
    physics_only_gym_make,
)


def test_physics_only_gym_make_overrides_rendering_arguments():
    calls = []

    def make(env_id, **kwargs):
        calls.append((env_id, kwargs))
        return "env"

    wrapped = physics_only_gym_make(make)
    assert wrapped("Task", obs_mode="rgb", render_mode="rgb_array") == "env"
    assert calls == [
        (
            "Task",
            {"obs_mode": "none", "render_mode": None, "render_backend": "none"},
        )
    ]


def test_sapien_patch_drops_visuals_but_restores_urdf_model():
    class ActorBuilder:
        def add_box_visual(self, *args, **kwargs):
            raise AssertionError("visual builder should be disabled")

        def add_box_collision(self, *args, **kwargs):
            return "collision"

    class LinkBuilder(ActorBuilder):
        pass

    class URDFLoader:
        def _build_link(self, link, builder):
            assert link.visuals == []
            return list(link.collisions)

    fake_sapien = SimpleNamespace(render=SimpleNamespace(RenderMaterial=object))
    patch_sapien_for_physics_only(
        fake_sapien,
        actor_builder_cls=ActorBuilder,
        link_builder_cls=LinkBuilder,
        urdf_loader_cls=URDFLoader,
    )

    actor = ActorBuilder()
    assert actor.add_box_visual(material="ignored") is actor
    assert actor.add_box_collision() == "collision"
    material = fake_sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
    material.set_base_color([0, 1, 0, 1])
    material.set_metallic(0.0)
    assert material is not None

    link = SimpleNamespace(visuals=["mesh"], collisions=["shape"])
    assert URDFLoader()._build_link(link, LinkBuilder()) == ["shape"]
    assert link.visuals == ["mesh"]


def test_physics_only_task_env_normalizes_stick_actions_and_reports_success():
    class Environment:
        unwrapped = None

        def __init__(self):
            self.unwrapped = self
            self.action = None

        def reset(self, **kwargs):
            return {}, {"success": np.array([False])}

        def step(self, action):
            self.action = action
            return {}, 1.0, True, False, {"success": np.array([True])}

        def close(self):
            pass

    raw = Environment()
    env = PhysicsOnlyTaskEnv(raw, "PatternLock")
    _, reward, terminated, truncated, info = env.step(np.arange(8))
    np.testing.assert_array_equal(raw.action, np.arange(7, dtype=np.float32))
    assert (reward, terminated, truncated, info["status"]) == (1.0, True, False, "success")
