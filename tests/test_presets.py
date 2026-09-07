import pytest

from trajmem_ot.presets import ComputePreset, get_compute_preset


def test_16gb_preset_matches_research_budget():
    preset = get_compute_preset("16gb")
    assert preset == ComputePreset(
        name="16gb",
        basis_directions=8,
        jvp_chunk_size=2,
        memory_views=4,
        noise_samples_per_view=2,
        response_rank=4,
        xla_memory_fraction=0.75,
    )
    assert preset.total_trajectories == 8


def test_24gb_preset_matches_research_budget():
    preset = get_compute_preset("24gb")
    assert preset.basis_directions == 16
    assert preset.jvp_chunk_size == 4
    assert preset.memory_views == 8
    assert preset.noise_samples_per_view == 4
    assert preset.response_rank == 8
    assert preset.xla_memory_fraction == 0.88
    assert preset.total_trajectories == 32


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown compute preset"):
        get_compute_preset("48gb")


def test_invalid_preset_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        ComputePreset(
            name="bad",
            basis_directions=0,
            jvp_chunk_size=1,
            memory_views=1,
            noise_samples_per_view=1,
            response_rank=1,
            xla_memory_fraction=0.8,
        )
