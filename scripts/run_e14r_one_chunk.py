#!/usr/bin/env python3
"""Execute saved physical action chunks from identical renderer-free source states."""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import random
import time
import numpy as np
from trajmem_ot.robomme_branch import ReplayBranchRunner
from trajmem_ot.headless_physics import patch_sapien_for_physics_only, physics_only_gym_make


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--noise-index',type=int,default=0)
    p.add_argument('--all-noises',action='store_true')
    args=p.parse_args()
    if args.output.exists(): p.error('output exists; preserve previous results')
    report=json.loads(args.report.read_text()); case=report['case']
    if case['query_step']!=case['execution_start']:
        p.error('this entrypoint currently restores first-execution queries only')
    if report.get('kind')!='native_interference_development_pilot':
        p.error('a native pilot report with source alignment is required')
    labels=['interfered','repaired','teacher','interpolation','negative','random']
    with np.load(args.report.with_suffix('.npz'),allow_pickle=False) as archive:
        actions={label:archive['physical_'+label].copy() for label in labels}
    seeds=report['noises']['evaluation']
    if any(value.ndim!=3 or value.shape[0]!=len(seeds) or not np.isfinite(value).all() for value in actions.values()):
        p.error('action archive must contain finite [noises,horizon,robot_dims] arrays')
    indices=list(range(len(seeds))) if args.all_noises else [args.noise_index]
    if any(index<0 or index>=len(seeds) for index in indices): p.error('noise index out of range')
    import torch
    import h5py
    import gymnasium as gym
    import sapien
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
    import robomme.robomme_env
    patch_sapien_for_physics_only(sapien); gym.make=physics_only_gym_make(gym.make)
    DemonstrationWrapper._augment_obs_and_info=lambda self,obs,info,action:(obs,info)
    task=case['task']; width=7 if task in ('PatternLock','RouteStick') else 8
    builder=BenchmarkEnvBuilder(task,dataset=case['split'],action_space='joint_angle',gui_render=False,max_steps=1000)
    seed,_=builder.resolve_episode(case['raw_episode'])
    if seed!=case['seed']: raise ValueError('source and simulator metadata seeds differ')
    with h5py.File(case['raw_file'],'r') as f:
        expected=np.asarray(f[f"episode_{case['raw_episode']}/timestep_{case['query_step']}/obs/joint_state"][()])
    alignment=[]
    class AuditedEnvironment:
        def __init__(self,env): self.env=env
        @property
        def unwrapped(self): return self.env.unwrapped
        def reset(self):
            observation,info=self.env.reset()
            observed=np.asarray(self.unwrapped.agent.robot.get_qpos()).reshape(-1)[:7]
            alignment.append(float(np.max(np.abs(observed-expected))))
            return observation,info
        def step(self,action): return self.env.step(action)
        def close(self): self.env.close()
    def factory():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        return AuditedEnvironment(builder.make_env_for_episode(case['raw_episode']))
    started=time.time(); runner=ReplayBranchRunner(factory); fingerprint=runner.assert_deterministic([])
    outcomes=[]
    for index in indices:
        for label in labels:
            outcome=runner.rollout(label,[],actions[label][index,:,:width])
            if outcome.start_fingerprint!=fingerprint:
                raise RuntimeError('paired branches did not start at the same physical fingerprint')
            row={**asdict(outcome),'noise_seed':seeds[index],'noise_index':index}
            a=outcome.start_progress['completed_subgoals']; b=outcome.final_progress['completed_subgoals']
            row['subgoal_advancement']=None if a is None or b is None else b-a
            outcomes.append(row)
            print(label,seeds[index],outcome.status,'advance',row['subgoal_advancement'],flush=True)
    result={'kind':'headless_native_one_chunk','case_id':case['case_id'],'task':task,'episode':case['episode'],
            'raw_episode':case['raw_episode'],'split':case['split'],'interference_level':case['interference_level'],
            'source_report':str(args.report.resolve()),
            'source_report_sha256':hashlib.sha256(args.report.read_bytes()).hexdigest(),
            'start_fingerprint':fingerprint,
            'source_joint_alignment_max_abs_errors':alignment,'outcomes':outcomes,'seconds':time.time()-started,
            'continuation_policy':'exploratory_no_fixed_metric_gate',
            'limitation':'Same source seed with measured joint-state reconstruction error; identical branch fingerprints, not an exact raw-recording physics snapshot. One chunk, not closed loop.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
