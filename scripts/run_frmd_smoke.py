#!/usr/bin/env python3
"""FRMD deployment smoke on synthetic row corruption; not native interference."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import yaml
from trajmem_ot.finite_distillation import NoisePartition, RepairConfig, repair_memory
from trajmem_ot.memory_basis import random_rank_one_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--corruption', choices=['random_rows', 'contiguous_rows'], default='random_rows')
    parser.add_argument('--drop-fraction', type=float, default=.2)
    parser.add_argument('--target', choices=['paired', 'centroid', 'ot'])
    parser.add_argument('--history-basis', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text()) if args.config else {}
    settings = settings or {}
    repair_settings = dict(settings.get('repair', {}))
    if args.target:
        repair_settings['target'] = args.target
    if args.iterations is not None:
        repair_settings['iterations'] = args.iterations
    config = RepairConfig(**repair_settings)
    k = int(settings.get('basis_directions', 8))
    nopt, nval, neval = (int(settings.get(name, default)) for name, default in
                         [('optimize_noises', 8), ('validation_noises', 4), ('evaluation_noises', 8)])
    if min(k, nopt, nval, neval) <= 0:
        parser.error('basis and noise counts must be positive')
    noises = NoisePartition(tuple(range(1000+args.seed, 1000+args.seed+nopt)),
                            tuple(range(10000+args.seed, 10000+args.seed+nval)),
                            tuple(range(100000+args.seed, 100000+args.seed+neval)))
    probes = 2*k*nopt*config.iterations
    total = probes + config.iterations*(nopt+nval+len(config.alphas)*nval) + nopt+nval+3*neval
    report = {'kind': 'synthetic_row_corruption_frmd_smoke', 'config': asdict(config),
              'basis_directions': k, 'noises': asdict(noises),
              'query_budget': {'finite_response_probes': probes, 'repair_total': total,
                               'controls_additional': 5*neval, 'total': total+5*neval},
              'continuation_policy': 'exploratory_no_fixed_metric_gate',
              'limitation': 'Synthetic corruption only; no native interference, persistence, or physical outcome claim.'}
    if args.dry_run:
        print(json.dumps(report, indent=2)); return
    if args.checkpoint is None or args.data is None or args.output is None:
        parser.error('--checkpoint, --data, and --output are required outside dry-run')
    if not 0 < args.drop_fraction < 1:
        parser.error('--drop-fraction must lie in (0, 1)')
    if args.output.suffix != '.json':
        parser.error('output must use .json to keep report and archive distinct')
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        parser.error('artifact already exists; choose a new path to preserve results')
    import jax
    import jax.numpy as jnp
    from trajmem_ot.robomme_runtime import load_runtime_state
    from trajmem_ot.robomme_jax import build_fixed_noise_action_problem
    from trajmem_ot.finite_response import quantized_norm_matched_memory
    from trajmem_ot import finite_distillation, self_teacher, repair_acceptance
    started = time.time()
    source_paths = [Path(__file__), *(Path(module.__file__) for module in
                                    (finite_distillation, self_teacher, repair_acceptance))]
    report['implementation_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    runtime = load_runtime_state(checkpoint=args.checkpoint, data=args.data, index=args.index, seed=args.seed)
    memory = np.asarray(runtime.observation.static_image_emb[0], dtype=np.float32)
    quantize = lambda m: np.asarray(jnp.asarray(m).astype(jnp.bfloat16), dtype=np.float32)
    teacher_memory = quantize(memory)
    valid = np.flatnonzero(np.asarray(runtime.observation.static_mask[0]))
    if not valid.size:
        raise ValueError('no valid memory rows')
    rng = np.random.default_rng(args.seed)
    count = max(1, int(round(len(valid)*args.drop_fraction)))
    if args.corruption == 'random_rows':
        dropped = np.sort(rng.choice(valid, count, replace=False))
    else:
        start = int(rng.integers(0, len(valid)-count+1)); dropped = valid[start:start+count]
    student_memory = teacher_memory.copy(); student_memory[dropped] = 0.
    basis_source = {'type': 'independent_random_rank_one', 'seed': args.seed+211}
    if args.history_basis:
        with np.load(args.history_basis, allow_pickle=False) as archive:
            basis = np.asarray(archive['basis'], dtype=np.float32)
        if len(basis) != k:
            raise ValueError('history basis direction count differs from declared budget')
        basis_source = {'type': 'explicit_history_basis', 'path': str(args.history_basis),
                        'sha256': hashlib.sha256(args.history_basis.read_bytes()).hexdigest()}
    else:
        basis = random_rank_one_basis(memory.shape, count=k, seed=args.seed+211)
    problems = {}
    latest_eval_actions = {}
    actual_calls = 0
    def action(m, seed):
        nonlocal actual_calls
        if seed not in problems:
            model = runtime.policy._model
            noise = jax.random.normal(jax.random.key(seed), (1, int(model.action_horizon), int(model.action_dim)))
            problems[seed] = build_fixed_noise_action_problem(runtime.policy, runtime.observation,
                                                             noise=noise, rng=jax.random.key(args.seed), num_steps=10)
        output = problems[seed].action_fn(jnp.asarray(m, dtype=jnp.bfloat16))
        actual_calls += 1
        if actual_calls % 32 == 0:
            print(f'policy queries: {actual_calls}; elapsed: {time.time()-started:.1f}s', file=sys.stderr, flush=True)
        observed = np.asarray(output.block_until_ready(), dtype=np.float32)[:, :8]
        if seed in noises.evaluation:
            latest_eval_actions[seed] = observed.copy()
        return observed
    result = repair_memory(student_memory, basis, action, lambda seed: action(teacher_memory, seed),
                           quantize, noises, config)
    repaired_actions = np.stack([latest_eval_actions[seed] for seed in noises.evaluation])
    delta = result.memory-student_memory
    applied_norm = float(np.linalg.norm(delta))
    control_memories, matching = {}, {}
    for name, direction in [('negative', -delta), ('random', rng.normal(size=memory.shape)),
                            ('interpolation', teacher_memory-student_memory)]:
        if applied_norm == 0:
            control_memories[name] = student_memory.copy()
            matching[name] = {'applied_norm': 0., 'relative_error': 0., 'no_edit': True}
        else:
            matched = quantized_norm_matched_memory(jnp.asarray(student_memory, dtype=jnp.bfloat16),
                                                   direction, target_norm=applied_norm, relative_tolerance=.03)
            control_memories[name] = np.asarray(matched.quantized_memory, dtype=np.float32)
            matching[name] = {'applied_norm': float(matched.applied_norm),
                              'relative_error': float(matched.relative_error),
                              'matched_within_3_percent': bool(matched.relative_error <= .03)}
    teacher_actions = np.stack([action(teacher_memory, seed) for seed in noises.evaluation])
    baseline_actions = np.stack([action(student_memory, seed) for seed in noises.evaluation])
    baseline_distance = np.linalg.norm((baseline_actions-teacher_actions).reshape(neval, -1), axis=1)
    controls = {}
    saved_actions = {'teacher': teacher_actions, 'interfered': baseline_actions, 'repaired': repaired_actions}
    for name, candidate in control_memories.items():
        observed = np.stack([action(candidate, seed) for seed in noises.evaluation])
        saved_actions[name] = observed
        distances = np.linalg.norm((observed-teacher_actions).reshape(neval, -1), axis=1)
        controls[name] = {'recovery_mean': float(np.mean((baseline_distance-distances)/(baseline_distance+1e-12))),
                          'loss_mean': float(np.mean((observed-teacher_actions)**2)), **matching[name]}
    report.update({'checkpoint': str(args.checkpoint.resolve()), 'data': str(args.data.resolve()),
                   'history_config': runtime.history_config_name, 'index': args.index, 'seed': args.seed,
                   'corruption': args.corruption, 'dropped_rows': dropped.tolist(), 'basis_source': basis_source,
                   'memory_shape': list(memory.shape), 'memory_dtype': 'bfloat16',
                   'trace': result.trace, 'evaluation': result.evaluation, 'controls': controls,
                   'query_counts': result.query_counts, 'actual_policy_queries_including_controls': actual_calls,
                   'applied_norm': applied_norm, 'seconds': time.time()-started})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two concurrent runs from overwriting evidence.
    with args.output.with_suffix('.npz').open('xb') as archive:
        np.savez_compressed(archive, teacher_memory=teacher_memory,
                            student_memory=student_memory, repaired_memory=result.memory, **saved_actions)
    with args.output.open('x') as report_file:
        report_file.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
