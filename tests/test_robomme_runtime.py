from pathlib import Path

import pytest

from trajmem_ot.robomme_runtime import resolve_history_config_name


def test_resolve_history_config_prefers_explicit_name(tmp_path: Path):
    checkpoint = tmp_path / "model" / "79999"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "history_config.txt").write_text("wrong.yaml\n")
    assert resolve_history_config_name(checkpoint, "chosen.yaml") == "chosen.yaml"


def test_resolve_history_config_reads_checkpoint_metadata(tmp_path: Path):
    checkpoint = tmp_path / "model" / "79999"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "history_config.txt").write_text("perceptual-framesamp-modul.yaml\n")
    assert resolve_history_config_name(checkpoint, None) == "perceptual-framesamp-modul.yaml"


def test_resolve_history_config_requires_metadata_or_explicit(tmp_path: Path):
    checkpoint = tmp_path / "model" / "79999"
    checkpoint.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="history_config"):
        resolve_history_config_name(checkpoint, None)
