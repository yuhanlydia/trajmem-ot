import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("jax")


@pytest.mark.parametrize(
    "script",
    [
        "scripts/run_e12_secant_recovery.py",
        "scripts/analyze_e12_secant.py",
        "scripts/audit_e13_pairs.py",
        "scripts/mine_e13_strict_candidates.py",
        "scripts/run_e13_oracle_transplant.py",
        "scripts/analyze_e13_oracle.py",
        "scripts/run_e13_readout_mask_branching.py",
        "scripts/analyze_e13_readout.py",
        "scripts/run_e13_pair_readout_recovery.py",
        "scripts/analyze_e13_pair_readout.py",
        "scripts/run_e14_ot_pullback.py",
        "scripts/analyze_e14.py",
        "scripts/smoke_robomme_branches.py",
        "scripts/run_saved_action_branches.py",
    ],
)
def test_rigorous_experiment_scripts_expose_help(script: str):
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
