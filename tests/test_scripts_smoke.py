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
        "scripts/run_e12_secant_recovery.py",
        "scripts/build_history_difference_basis.py",
        "scripts/run_e13_memory_view_branching.py",
        "scripts/run_e13_oracle_transplant.py",
        "scripts/run_e13_readout_mask_branching.py",
        "scripts/audit_e13_pairs.py",
        "scripts/analyze_e11.py",
        "scripts/analyze_e12_secant.py",
        "scripts/analyze_e13_oracle.py",
        "scripts/analyze_e13_readout.py",
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


def test_e13_oracle_analyzer_ignores_pair_audit_report(tmp_path: Path):
    pair = tmp_path / "pair_0.json"
    audit = tmp_path / "pair_audit.json"
    output = tmp_path / "summary.json"
    pair.write_text(
        json.dumps(
            {
                "experiment": "E13B_oracle_history_transplant",
                "pair_index": 0,
                "directions": [
                    {
                        "coverage": {
                            "coverage_gain": 0.25,
                            "mean_distance_improvement": 0.1,
                        },
                        "paired_memory_effect_mean": 0.2,
                    }
                ],
                "context_diagnostics": {"prompt_match": True},
            }
        )
    )
    audit.write_text(json.dumps({"experiment": "E13B_pair_audit", "rows": []}))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_e13_oracle.py",
            str(pair),
            str(audit),
            "--output",
            str(output),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["n_input_reports"] == 1
    assert report["subsets"]["all"]["n_pairs"] == 1
    assert report["subsets"]["all"]["mean_pair_coverage_gain_ci"]["estimate"] == pytest.approx(0.25)
