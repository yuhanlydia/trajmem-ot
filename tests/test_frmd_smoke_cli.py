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


def test_native_pilot_dry_run_reports_each_case_without_loading_gpu(tmp_path):
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'case_id':'A-0-k3','task':'A','episode':0,
                                              'interference_level':3,'persistence_queries':{'1':{},'3':{}}}]}))
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
    run=subprocess.run([sys.executable,str(ROOT/'scripts/run_e14r_pilot.py'),
                        '--manifest',str(manifest),'--dry-run'],capture_output=True,text=True,env=env,check=True)
    report=json.loads(run.stdout)
    assert report['case_count']==1
    assert report['cases'][0]['finite_response_probes']==256
    assert report['cases'][0]['total_policy_queries']==438
    assert report['continuation_policy']=='exploratory_no_fixed_metric_gate'
