import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_dry_run_needs_no_checkpoint_or_gpu_and_counts_actual_trajectories():
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    run = subprocess.run([sys.executable, str(ROOT/'scripts/run_frmd_smoke.py'), '--dry-run'],
                         capture_output=True, text=True, check=True, env=env)
    report = json.loads(run.stdout)
    assert report['query_budget']['finite_response_probes'] == 256
    assert report['query_budget']['repair_total'] == 340
    assert report['continuation_policy'] == 'exploratory_no_fixed_metric_gate'


def test_existing_archive_is_rejected_before_checkpoint_loading(tmp_path):
    archive = tmp_path/'result.npz'
    archive.write_bytes(b'preserve-me')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    run = subprocess.run([sys.executable, str(ROOT/'scripts/run_frmd_smoke.py'),
                          '--checkpoint', '/missing', '--data', '/missing',
                          '--output', str(tmp_path/'result.json')],
                         capture_output=True, text=True, env=env)
    assert 'artifact already exists' in run.stderr
    assert archive.read_bytes() == b'preserve-me'


def test_output_extension_must_keep_json_distinct_from_npz(tmp_path):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    run = subprocess.run([sys.executable, str(ROOT/'scripts/run_frmd_smoke.py'),
                          '--checkpoint', '/missing', '--data', '/missing',
                          '--output', str(tmp_path/'result.npz')],
                         capture_output=True, text=True, env=env)
    assert 'output must use .json' in run.stderr
