#!/usr/bin/env python3
"""Paired native FRMD rollouts with real RGB feedback and surviving-token edits.

Run in the benchmark environment. A persistent child uses the policy environment.
Set a working process-local Vulkan ICD; CPU readback does not remove rendering.
"""
import argparse,hashlib,json,os,random,select,subprocess,time
from pathlib import Path
import numpy as np
from trajmem_ot.live_history import collect_step_frames
from trajmem_ot.robomme_branch import branch_fingerprint,observation_fingerprint,task_progress_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--policy-python',type=Path,required=True)
    p.add_argument('--policy-cwd',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--queries',type=int,default=10)
    p.add_argument('--noise-seeds',type=int,nargs='+',default=[200007])
    p.add_argument('--conditions',nargs='+',default=['interfered','repaired','teacher','interpolation','negative','random'],
                   choices=['interfered','repaired','teacher','interpolation','negative','random'])
    args=p.parse_args()
    if args.queries<1 or len(set(args.noise_seeds))!=len(args.noise_seeds):p.error('positive queries and unique seeds required')
    if args.output.suffix!='.json':p.error('output must use .json')
    artifacts=args.output.with_suffix('.artifacts')
    if args.output.exists() or artifacts.exists():p.error('preserve existing report/artifacts; choose a fresh output')
    report=json.loads(args.report.read_text());case=report['case']
    if report.get('kind')!='native_interference_development_pilot' or case['query_step']!=case['execution_start']:
        p.error('native first-execution repair report required')
    used={n for group in report['noises'].values() for n in group}
    schedule={seed+1000*q for seed in args.noise_seeds for q in range(args.queries)}
    if used & schedule or len(schedule)!=len(args.noise_seeds)*args.queries:
        p.error('closed-loop query noises must be distinct and disjoint from prior noise partitions')
    artifacts.mkdir(parents=True)
    import gymnasium as gym,torch,h5py
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    import robomme.robomme_env
    original_make=gym.make
    def cpu_readback_make(*a,**kw):
        kw.update(sim_backend='cpu',render_backend='cpu')
        return original_make(*a,**kw)
    gym.make=cpu_readback_make
    builder=BenchmarkEnvBuilder(case['task'],dataset=case['split'],action_space='joint_angle',
                               gui_render=False,max_steps=args.queries*20)
    seed,_=builder.resolve_episode(case['raw_episode'])
    if seed!=case['seed']:raise ValueError('source and simulator seeds differ')
    with h5py.File(case['raw_file'],'r') as f:
        source_joints=f[f"episode_{case['raw_episode']}/timestep_{case['query_step']}/obs/joint_state"][()]
    root=Path(__file__).resolve().parents[1]
    command=[str(args.policy_python.resolve()),'-u',str(root/'scripts/e14r_live_policy_worker.py'),
             '--report',str(args.report.resolve()),'--data',str(args.data.resolve())]
    worker_log=(artifacts/'worker.log').open('x')
    worker=subprocess.Popen(command,cwd=args.policy_cwd,env={**os.environ,'PYTHONPATH':str(root/'src'),
                              'XLA_PYTHON_CLIENT_PREALLOCATE':'false'},stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,stderr=worker_log,text=True,bufsize=1)
    def receive():
        ready,_,_=select.select([worker.stdout],[],[],900)
        if not ready:raise TimeoutError('policy worker response exceeded 900 seconds')
        line=worker.stdout.readline()
        if not line.startswith('FRMD_RPC '):raise RuntimeError('policy worker exited or sent invalid response; inspect worker.log')
        response=json.loads(line[len('FRMD_RPC '):])
        if 'error' in response:raise RuntimeError(response['error'])
        return response
    def rpc(request):
        worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();return receive()
    rows=[];expected_fingerprint=None;expected_observation=None;started=time.time()
    try:
        receive()
        for noise_seed in args.noise_seeds:
            for condition in args.conditions:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                env=builder.make_env_for_episode(case['raw_episode'])
                branch_id=f'{case["case_id"]}/{condition}/{noise_seed}'
                prefix=condition+'-'+str(noise_seed)
                try:
                    obs,info=env.reset();initial_fingerprint=branch_fingerprint(env,obs)
                    if expected_fingerprint is None:expected_fingerprint=initial_fingerprint
                    if initial_fingerprint!=expected_fingerprint:raise RuntimeError('paired initial physical fingerprints differ')
                    initial_observation=observation_fingerprint(obs)
                    if expected_observation is None:expected_observation=initial_observation
                    if initial_observation!=expected_observation:raise RuntimeError('paired initial RGB/state observations differ')
                    initial_progress=task_progress_snapshot(env)
                    joints=np.asarray(obs['joint_state_list'][-1]).reshape(-1)
                    alignment_error=float(np.max(np.abs(joints-source_joints)))
                    rpc({'op':'reset','condition':condition,'branch_identity':branch_id})
                    pending_images=[];pending_states=[];trace=[];steps=0;total_return=0.;done=False
                    status=str(info.get('status','unknown'))
                    for query in range(args.queries):
                        pack=collect_step_frames(obs)
                        images=np.concatenate(pending_images) if pending_images else np.empty((0,*pack['images'].shape[1:]),np.uint8)
                        states=np.concatenate(pending_states) if pending_states else np.empty((0,8),np.float32)
                        goal=info['task_goal'];goal=goal[0] if isinstance(goal,list) else goal
                        observation_path=artifacts/f'{prefix}-q{query}-observation.npz'
                        action_path=artifacts/f'{prefix}-q{query}-action.npz'
                        with observation_path.open('xb') as f:
                            np.savez_compressed(f,image=np.asarray(obs['front_rgb_list'][-1]),
                                wrist_image=np.asarray(obs['wrist_rgb_list'][-1]),state=pack['states'][-1],
                                prompt=str(goal).lower(),update_images=images,update_states=states)
                        before=task_progress_snapshot(env)
                        response=rpc({'op':'act','observation':str(observation_path.resolve()),
                                      'output':str(action_path.resolve()),'noise_seed':noise_seed+1000*query})
                        pending_images=[];pending_states=[]
                        with np.load(action_path,allow_pickle=False) as saved:actions=saved['actions'].copy()
                        if actions.ndim!=2 or not np.isfinite(actions).all():raise ValueError('invalid policy actions')
                        width=7 if case['task'] in ('PatternLock','RouteStick') else 8
                        for action in actions[:20,:width]:
                            obs,reward,terminated,truncated,info=env.step(np.asarray(action,dtype=np.float32))
                            steps+=1;total_return+=float(np.asarray(reward).reshape(-1)[-1])
                            status=str(info.get('status','unknown'));done=bool(terminated) or bool(truncated)
                            if not done:
                                updates=collect_step_frames(obs);pending_images.append(updates['images']);pending_states.append(updates['states'])
                            if done:break
                        trace.append({'query':query+1,'noise_seed':noise_seed+1000*query,
                                      'before_progress':before,'after_progress':task_progress_snapshot(env),
                                      'observation_sha256':hashlib.sha256(observation_path.read_bytes()).hexdigest(),**response})
                        print('QUERY',branch_id,query+1,'steps',steps,'status',status,flush=True)
                        if done:break
                    final_progress=task_progress_snapshot(env)
                    row={'condition':condition,'noise_seed':noise_seed,'steps':steps,'queries':len(trace),
                         'return_sum':total_return,'status':status,'success':status=='success',
                         'terminated_or_truncated':done,'start_fingerprint':initial_fingerprint,
                         'initial_observation_fingerprint':initial_observation,'final_fingerprint':branch_fingerprint(env,obs),
                         'source_joint_alignment_max_abs_error':alignment_error,
                         'start_progress':initial_progress,'final_progress':final_progress,
                         'subgoal_advancement':final_progress['completed_subgoals']-initial_progress['completed_subgoals'],
                         'query_trace':trace}
                    rows.append(row)
                    (artifacts/'progress.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
                finally:env.close()
        result={'kind':'native_rgb_closed_loop_development','case_id':case['case_id'],'case':case,
                'source_report':str(args.report.resolve()),'source_report_sha256':hashlib.sha256(args.report.read_bytes()).hexdigest(),
                'source_archive_sha256':hashlib.sha256(args.report.with_suffix('.npz').read_bytes()).hexdigest(),
                'checkpoint':report['checkpoint'],'history_config':report['history_config'],
                'query_limit':args.queries,'noise_seeds':args.noise_seeds,'conditions':args.conditions,
                'renderer':'NVIDIA Vulkan with CPU tensor readback and original visual geometry',
                'vulkan_icd':os.environ.get('VK_ICD_FILENAMES'),'outcomes':rows,'seconds':time.time()-started,
                'persistence_protocol':'Each subsequent RGB frame encoded by frozen policy; original edit transferred only to surviving source tokens; no teacher retrieval/refit in repaired branch',
                'continuation_policy':'exploratory_no_fixed_metric_gate',
                'limitation':'Development/train source, bounded multi-query rollout; no held-out benchmark claim. Initial repair used recorded query; deployment uses reconstructed real RGB state.'}
        with args.output.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    finally:
        if worker.poll() is None:
            try:
                worker.stdin.write('{"op":"close"}\n');worker.stdin.flush();worker.wait(timeout=30)
            except (BrokenPipeError,subprocess.TimeoutExpired):worker.terminate();worker.wait(timeout=30)
        worker_log.close()


if __name__=='__main__':main()
