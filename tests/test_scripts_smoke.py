import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("jax")


@pytest.mark.parametrize("preset", ["16gb", "24gb"])
def test_e11_smoke_script_runs_and_writes_operator_metrics(tmp_path: Path, preset: str):
    output = tmp_path / f"e11_{preset}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    subprocess.run(
        [
            sys.executable,
            "scripts/run_e11_jvp_smoke.py",
            "--preset",
            preset,
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["preset"] == preset
    assert report["basis_directions"] in {8, 16}
    assert report["jvp_secant_min_cosine"] > 0.999
    assert report["jvp_secant_max_relative_error"] < 0.01
    assert report["pullback_residual_after"] < report["pullback_residual_before"]
    assert report["effective_rank_90"] >= 1


@pytest.mark.parametrize(
    "script",
    [
        "scripts/run_e11_robomme_jvp.py",
        "scripts/run_e11_stage_localization.py",
        "scripts/run_e12_operator_basis.py",
        "scripts/build_history_difference_basis.py",
        "scripts/run_e13_memory_view_branching.py",
        "scripts/analyze_e11.py",
    ],
)
def test_experiment_scripts_expose_help_without_robomme_install(script: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
