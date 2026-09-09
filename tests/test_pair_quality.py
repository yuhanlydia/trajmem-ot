from trajmem_ot.pair_quality import PairQualityCriteria, assess_pair_quality


def _row(**updates):
    row = {
        "prompt_match": True,
        "front_image_mae": 0.01,
        "wrist_image_mae": 0.02,
        "current_state_l2": 0.1,
        "history_mask_iou": 1.0,
        "history_image_relative_l2": 0.1,
    }
    row.update(updates)
    return row


def test_pair_quality_accepts_close_context_with_distinct_history():
    criteria = PairQualityCriteria(
        max_front_image_mae=0.03,
        max_wrist_image_mae=0.08,
        max_current_state_l2=0.75,
        min_history_mask_iou=0.95,
        min_history_image_relative_l2=0.03,
    )
    result = assess_pair_quality(_row(), criteria)
    assert result.eligible is True
    assert result.failed_reasons == ()


def test_pair_quality_rejects_context_mismatch_and_trivial_history():
    criteria = PairQualityCriteria()
    result = assess_pair_quality(
        _row(
            prompt_match=False,
            current_state_l2=3.0,
            history_image_relative_l2=0.0,
        ),
        criteria,
    )
    assert result.eligible is False
    assert "prompt_mismatch" in result.failed_reasons
    assert "current_state_l2" in result.failed_reasons
    assert "insufficient_history_contrast" in result.failed_reasons


def test_load_pair_quality_tiers_rejects_outcome_dependent_fields(tmp_path):
    from trajmem_ot.pair_quality import load_pair_quality_tiers

    path = tmp_path / "quality.yaml"
    path.write_text("strict:\n  max_front_image_mae: 0.03\n  coverage_gain: 1.0\n")
    import pytest
    with pytest.raises(ValueError, match="unknown pair-quality fields"):
        load_pair_quality_tiers(path)
