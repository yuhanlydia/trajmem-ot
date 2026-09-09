from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class _DummyRenderMaterial:
    """Value sink for task code that constructs a material before adding a visual."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("set_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def physics_only_gym_make(original_make: Callable[..., Any]) -> Callable[..., Any]:
    """Force a Gym environment onto ManiSkill's renderer-free physics path."""

    def make(env_id: str, **kwargs: Any) -> Any:
        kwargs.update(obs_mode="none", render_mode=None, render_backend="none")
        return original_make(env_id, **kwargs)

    return make


class PhysicsOnlyTaskEnv:
    """Thin action/status adapter around a renderer-free RoboMME task."""

    _STICK_TASKS = {"PatternLock", "RouteStick"}

    def __init__(self, env: Any, task: str) -> None:
        self.env = env
        self.task = task

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, **kwargs: Any) -> tuple[dict, dict]:
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict]:
        width = 7 if self.task in self._STICK_TASKS else 8
        normalized = np.asarray(action, dtype=np.float32).reshape(-1)[:width]
        observation, reward, terminated, truncated, info = self.env.step(normalized)
        success = bool(np.asarray(info.get("success", False)).reshape(-1)[-1])
        if success:
            status = "success"
        elif bool(np.asarray(terminated).reshape(-1)[-1]):
            status = "fail"
        elif bool(np.asarray(truncated).reshape(-1)[-1]):
            status = "timeout"
        else:
            status = "ongoing"
        return observation, reward, terminated, truncated, {**info, "status": status}

    def close(self) -> None:
        self.env.close()


def patch_sapien_for_physics_only(
    sapien_module: Any,
    *,
    actor_builder_cls: type | None = None,
    link_builder_cls: type | None = None,
    urdf_loader_cls: type | None = None,
) -> None:
    """Disable visual asset creation while retaining collisions and articulations.

    SAPIEN's URDF and RoboMME primitive builders instantiate render materials even
    when ManiSkill uses ``render_backend='none'``. This process-local patch makes
    action-only simulator evaluation possible without a Vulkan device.
    """

    if actor_builder_cls is None or link_builder_cls is None or urdf_loader_cls is None:
        from sapien.wrapper.actor_builder import ActorBuilder
        from sapien.wrapper.articulation_builder import LinkBuilder
        from sapien.wrapper.urdf_loader import URDFLoader

        actor_builder_cls = actor_builder_cls or ActorBuilder
        link_builder_cls = link_builder_cls or LinkBuilder
        urdf_loader_cls = urdf_loader_cls or URDFLoader

    sapien_module.render.RenderMaterial = _DummyRenderMaterial

    def no_visual(builder: Any, *args: Any, **kwargs: Any) -> Any:
        return builder

    for builder_cls in (actor_builder_cls, link_builder_cls):
        for name in dir(builder_cls):
            if name.startswith("add_") and "visual" in name:
                setattr(builder_cls, name, no_visual)

    original_build_link = urdf_loader_cls._build_link

    def build_link_without_visuals(loader: Any, link: Any, builder: Any) -> Any:
        visuals = link.visuals
        link.visuals = []
        try:
            return original_build_link(loader, link, builder)
        finally:
            link.visuals = visuals

    urdf_loader_cls._build_link = build_link_without_visuals
