from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputePreset:
    """Memory-safe experiment sizing for one accelerator.

    Batching is intentionally spent on JVP directions and diffusion particles
    within one environment state. This maximizes reuse of the frozen model and
    current observation while keeping peak activation memory predictable.
    """

    name: str
    basis_directions: int
    jvp_chunk_size: int
    memory_views: int
    noise_samples_per_view: int
    response_rank: int
    xla_memory_fraction: float

    def __post_init__(self) -> None:
        integer_fields = (
            self.basis_directions,
            self.jvp_chunk_size,
            self.memory_views,
            self.noise_samples_per_view,
            self.response_rank,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("all compute dimensions must be positive")
        if self.jvp_chunk_size > self.basis_directions:
            raise ValueError("jvp_chunk_size cannot exceed basis_directions")
        if self.response_rank > self.basis_directions:
            raise ValueError("response_rank cannot exceed basis_directions")
        if not 0.0 < self.xla_memory_fraction <= 1.0:
            raise ValueError("xla_memory_fraction must be in (0, 1]")

    @property
    def total_trajectories(self) -> int:
        return self.memory_views * self.noise_samples_per_view


_PRESETS = {
    "16gb": ComputePreset(
        name="16gb",
        basis_directions=8,
        jvp_chunk_size=2,
        memory_views=4,
        noise_samples_per_view=2,
        response_rank=4,
        xla_memory_fraction=0.75,
    ),
    "24gb": ComputePreset(
        name="24gb",
        basis_directions=16,
        jvp_chunk_size=4,
        memory_views=8,
        noise_samples_per_view=4,
        response_rank=8,
        xla_memory_fraction=0.88,
    ),
}


def get_compute_preset(name: str) -> ComputePreset:
    key = name.strip().lower()
    try:
        return _PRESETS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_PRESETS))
        raise ValueError(f"unknown compute preset {name!r}; choose one of: {choices}") from exc
