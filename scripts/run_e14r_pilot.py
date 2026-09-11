#!/usr/bin/env python3
"""Native history-interference development pilot with offline persistence scoring."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import yaml
from trajmem_ot.finite_distillation import NoisePartition, RepairConfig, repair_memory
from trajmem_ot.memory_basis import normalize_basis, random_rank_one_basis
from trajmem_ot.native_history import encode_native_history, encode_native_frame_sampling, transfer_memory_delta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--case-id',action='append')
    p.add_argument('--target',choices=['paired','centroid','ot'])
    p.add_argument('--iterations',type=int)
    p.add_argument('--basis',choices=['random','history_difference'],default='history_difference')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true',help='skip complete outputs only when source and run settings match')
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text()); cases=manifest['cases']
    if args.case_id:
        requested=set(args.case_id); cases=[case for case in cases if case['case_id'] in requested]
        if {case['case_id'] for case in cases}!=requested:
            p.error('requested case ID is absent from manifest')
    if not cases or len({c['case_id'] for c in cases})!=len(cases):
        p.error('manifest selection must have nonempty unique case IDs')
    settings=yaml.safe_load(args.config.read_text()) if args.config else {}
    settings=settings or {}; options=dict(settings.get('repair',{}))
    if args.target: options['target']=args.target
    if args.iterations is not None: options['iterations']=args.iterations
    config=RepairConfig(**options)
    k=int(settings.get('basis_directions',8))
    nopt,nval,neval=(int(settings.get(key,value)) for key,value in
                    [('optimize_noises',8),('validation_noises',4),('evaluation_noises',8)])
    noises=NoisePartition(tuple(range(1000+args.seed,1000+args.seed+nopt)),
                          tuple(range(10000+args.seed,10000+args.seed+nval)),
                          tuple(range(100000+args.seed,100000+args.seed+neval)))
    probes=2*k*nopt*config.iterations
    repair_total=probes+config.iterations*(nopt+nval+len(config.alphas)*nval)+nopt+nval+3*neval
    budget=[{'case_id':case['case_id'],'finite_response_probes':probes,
             'repeat_calibration_queries':2,
             'total_policy_queries':2+repair_total+6*neval+3*neval*len(case['persistence_queries'])}
            for case in cases]
    summary={'case_count':len(cases),'cases':budget,'basis':args.basis,'config':asdict(config),
             'continuation_policy':'exploratory_no_fixed_metric_gate'}
    if args.dry_run:
        print(json.dumps(summary,indent=2)); return
    if args.checkpoint is None or args.output_dir is None:
        p.error('--checkpoint and --output-dir required outside dry-run')
    if not manifest.get('summary',{}).get('source_files_verified'):
        p.error('manifest source files have not been verified')
    if k < 1: p.error('basis count must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    pending=[]; skipped=[]; pending_budget=[]
    for case,planned in zip(cases,budget):
        output=args.output_dir/f"{case['case_id']}.json"
        archive=output.with_suffix('.npz')
        if output.exists():
            if not args.resume or not archive.is_file():
                p.error(f'existing or incomplete evidence: {output}')
            previous=json.loads(output.read_text())
            requested_basis=previous.get('requested_basis',
                'history_difference' if previous.get('basis_source','').startswith('retrieved_teacher') else 'random')
            previous_basis_count=previous.get('basis_directions')
            if previous_basis_count is None:
                previous_basis_count=(previous.get('query_budget',{}).get('finite_response_probes',0)
                                      // (2*len(previous.get('noises',{}).get('optimize',[0]))
                                          * previous.get('config',{}).get('iterations',1)))
            compatible=(previous.get('case')==case and previous.get('manifest_sha256')==manifest_hash
                        and previous.get('checkpoint')==str(args.checkpoint.resolve())
                        and json.dumps(previous.get('config'),sort_keys=True)==json.dumps(asdict(config),sort_keys=True)
                        and previous.get('noises')==json.loads(json.dumps(asdict(noises)))
                        and previous_basis_count==k and requested_basis==args.basis)
            if not compatible:p.error(f'cannot resume incompatible run settings: {output}')
            skipped.append({'case_id':case['case_id'],'status':'previously_complete'})
        elif archive.exists():
            p.error(f'orphan archive preserved; choose another output directory: {archive}')
        else:
            pending.append(case); pending_budget.append(planned)
    cases,budget=pending,pending_budget
    if not cases:
        print(json.dumps({'pending_cases':0,'completed_cases':len(skipped)},indent=2));return
    data=Path(manifest['data'])
    import h5py
    import jax
    import jax.numpy as jnp
    from trajmem_ot.robomme_runtime import load_runtime_state
    from trajmem_ot.robomme_jax import prepare_policy_observation, build_fixed_noise_action_problem
    from trajmem_ot.finite_response import quantized_norm_matched_memory
    from trajmem_ot import native_history,finite_distillation,self_teacher,repair_acceptance
    implementation_hashes={Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                           for module in (native_history,finite_distillation,self_teacher,repair_acceptance)}
    implementation_hashes[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runtime=load_runtime_state(checkpoint=args.checkpoint,data=data,index=cases[0]['query_index'],seed=args.seed)
    policy=runtime.policy
    if runtime.history_config_name not in ('perceptual-tokendrop-modul.yaml','perceptual-framesamp-modul.yaml'):
        raise ValueError('native encoders support the released tokendrop/framesamp modulation configurations')
    quantize=lambda value:np.asarray(jnp.asarray(value,dtype=jnp.bfloat16),dtype=np.float32)
    handles={}
    @lru_cache(maxsize=256)
    def feature(episode,step):
        return np.load(data/'features'/f'episode_{episode}'/f'token_emb_{step}.npy',allow_pickle=True).item()
    def pixels(ref):
        path=ref['raw_file']
        if path not in handles: handles[path]=h5py.File(path,'r')
        return handles[path][f"episode_{ref['raw_episode']}/timestep_{ref['step']}/obs/front_rgb"][()]
    def encode(segments):
        if runtime.history_config_name=='perceptual-framesamp-modul.yaml':
            return encode_native_frame_sampling(segments,
                feature_loader=lambda ref:feature(ref['episode'],ref['step']),
                token_budget=512,token_per_image=16)
        return encode_native_history(segments,feature_loader=lambda ref:feature(ref['episode'],ref['step']),
                                     pixel_loader=pixels,token_budget=512)
    def observation(index,encoded):
        with (data/'data'/f'{index}.pkl').open('rb') as f: item=pickle.load(f)
        item.update(encoded.history)
        return prepare_policy_observation(policy,item)
    def problem(obs,seed):
        noise=jax.random.normal(jax.random.key(seed),(1,int(policy._model.action_horizon),int(policy._model.action_dim)))
        return build_fixed_noise_action_problem(policy,obs,noise=noise,rng=jax.random.key(args.seed),num_steps=10)
    all_status=list(skipped)
    try:
        for case,planned in zip(cases,budget):
            case_id=case['case_id']; output=args.output_dir/f'{case_id}.json'
            archive=output.with_suffix('.npz')
            if output.exists() or archive.exists():
                raise FileExistsError(f'preserve existing evidence: {output}')
            started=time.time(); counts={'actual':0}
            print('START',case_id,flush=True)
            student=encode(case['student_history']); teacher=encode(case['teacher_history'])
            student_obs=observation(case['query_index'],student)
            teacher_obs=observation(case['query_index'],teacher)
            initial=quantize(student.history['static_image_emb']); teacher_memory=quantize(teacher.history['static_image_emb'])
            cache={}
            def full_action(memory,seed,role='student'):
                key=(role,seed)
                if key not in cache: cache[key]=problem(teacher_obs if role=='teacher' else student_obs,seed)
                result=cache[key].action_fn(jnp.asarray(memory,dtype=jnp.bfloat16))
                counts['actual']+=1
                if counts['actual']%128==0: print(case_id,'queries',counts['actual'],flush=True)
                return np.asarray(result.block_until_ready(),dtype=np.float32)
            repeated_a=full_action(initial,noises.optimize[0])
            repeated_b=full_action(initial,noises.optimize[0])
            repeat_calibration={'same_noise_exact':bool(np.array_equal(repeated_a,repeated_b)),
                                'maximum_absolute_difference':float(np.max(np.abs(repeated_a-repeated_b)))}
            basis=random_rank_one_basis(initial.shape,count=k,seed=args.seed+211)
            difference=teacher_memory-initial
            used_basis='independent_random_rank_one'
            if args.basis=='history_difference' and np.linalg.norm(difference)>0:
                basis=normalize_basis(np.concatenate([difference[None],basis[:k-1]],axis=0))
                used_basis='retrieved_teacher_difference_plus_independent_random_rank_one'
            result=repair_memory(initial,basis,lambda memory,seed:full_action(memory,seed)[:,:8],
                                 lambda seed:full_action(teacher_memory,seed,'teacher')[:,:8],quantize,noises,config)
            delta=result.memory-initial; norm=float(np.linalg.norm(delta)); rng=np.random.default_rng(args.seed+case['episode'])
            memories={'interfered':initial,'repaired':result.memory,'teacher':teacher_memory}; matched={}
            for name,direction in [('negative',-delta),('random',rng.normal(size=initial.shape)),('interpolation',difference)]:
                if norm==0 or np.linalg.norm(direction)==0:
                    memories[name]=initial.copy(); matched[name]={'applied_norm':0.,'relative_error':0. if norm==0 else 1.}
                else:
                    control=quantized_norm_matched_memory(jnp.asarray(initial,dtype=jnp.bfloat16),direction,
                                                         target_norm=norm,relative_tolerance=.03)
                    memories[name]=np.asarray(control.quantized_memory,dtype=np.float32)
                    matched[name]={'applied_norm':control.applied_norm,'relative_error':control.relative_error}
            outputs={}; physical={}
            for name,memory in memories.items():
                full=np.stack([full_action(memory,seed,'teacher' if name=='teacher' else 'student') for seed in noises.evaluation])
                outputs[name]=full[:,:,:8]
                physical[name]=np.stack([policy._output_transform({'state':np.asarray(student_obs.state[0]),'actions':row})['actions'] for row in full])
            baseline=np.linalg.norm((outputs['interfered']-outputs['teacher']).reshape(neval,-1),axis=1)
            metrics={}
            for name,actions in outputs.items():
                distance=np.linalg.norm((actions-outputs['teacher']).reshape(neval,-1),axis=1)
                metrics[name]={'recovery_mean':float(np.mean((baseline-distance)/(baseline+1e-12))),
                               'loss_mean':float(np.mean((actions-outputs['teacher'])**2)),**matched.get(name,{})}
            persistence={}
            for horizon,query in case['persistence_queries'].items():
                def extend(segments):
                    return [*segments[:-1],{**segments[-1],'end_exclusive':query['step']+1}]
                later_student=encode(extend(case['student_history']))
                later_teacher=encode(extend(case['teacher_history']))
                later_obs=observation(query['query_index'],later_student)
                target_obs=observation(query['query_index'],later_teacher)
                later_initial=quantize(later_student.history['static_image_emb'])
                transferred,retention=transfer_memory_delta(student.token_sources,later_student.token_sources,delta)
                later_repaired=quantize(later_initial+transferred)
                later_actions={'interfered':[],'repaired':[],'teacher':[]}
                for seed in noises.evaluation:
                    action_problem=problem(later_obs,seed); target_problem=problem(target_obs,seed)
                    for name,memory,pr in [('interfered',later_initial,action_problem),('repaired',later_repaired,action_problem),
                                           ('teacher',quantize(later_teacher.history['static_image_emb']),target_problem)]:
                        value=pr.action_fn(jnp.asarray(memory,dtype=jnp.bfloat16))
                        later_actions[name].append(np.asarray(value.block_until_ready(),dtype=np.float32)[:,:8]); counts['actual']+=1
                base=np.asarray(later_actions['interfered'])-later_actions['teacher']
                repaired=np.asarray(later_actions['repaired'])-later_actions['teacher']
                before=np.linalg.norm(base.reshape(neval,-1),axis=1); after=np.linalg.norm(repaired.reshape(neval,-1),axis=1)
                persistence[horizon]={'recovery_mean':float(np.mean((before-after)/(before+1e-12))),**retention,
                                      'applied_norm':float(np.linalg.norm(later_repaired-later_initial)),**query}
            report={'kind':'native_interference_development_pilot','case':case,'config':asdict(config),'noises':asdict(noises),
                    'checkpoint':str(args.checkpoint.resolve()),'basis_source':used_basis,'requested_basis':args.basis,
                    'basis_directions':k,'implementation_sha256':implementation_hashes,'history_config':runtime.history_config_name,
                    'manifest_sha256':manifest_hash,
                    'query_budget':planned,'actual_policy_queries':counts['actual'],'repair_query_counts':result.query_counts,
                    'repair_trace':result.trace,'fresh_noise_metrics':metrics,'persistence':persistence,
                    'persistence_auc':float(np.mean([row['recovery_mean'] for row in persistence.values()])) if persistence else None,
                    'persistence_recovery_mean':float(np.mean([row['recovery_mean'] for row in persistence.values()])) if persistence else None,
                    'persistence_auc_definition':'Supplied discrete metric: arithmetic mean of recovery at available listed horizons; not trapezoidal time integration',
                    'teacher_view':'provenance-selected relevant demo plus recent frames; no learned retrieval evaluated',
                    'persistence_protocol':'offline subsequent recorded observations; identity-aligned edit transfer; no refit or teacher access in repaired branch',
                    'student_history_frames':student.frame_count,'teacher_history_frames':teacher.frame_count,
                    'student_token_sources':student.token_sources,'teacher_token_sources':teacher.token_sources,
                    'repeat_calibration':repeat_calibration,
                    'applied_norm':norm,'seconds':time.time()-started,'continuation_policy':'exploratory_no_fixed_metric_gate',
                    'limitation':'Development/train sources. Saved physical actions have not yet been executed. Offline persistence is not closed-loop success.'}
            if counts['actual']!=planned['total_policy_queries']: raise RuntimeError('actual policy-query budget mismatch')
            with archive.open('xb') as f:
                np.savez_compressed(f,student_memory=initial,repaired_memory=result.memory,teacher_memory=teacher_memory,
                                    **{f'normalized_{key}':value for key,value in outputs.items()},
                                    **{f'physical_{key}':value for key,value in physical.items()})
            with output.open('x') as f: f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
            all_status.append({'case_id':case_id,'status':'complete','seconds':report['seconds']})
            (args.output_dir/'progress.json').write_text(json.dumps(all_status,indent=2)+'\n')
            print('DONE',case_id,'recovery',metrics['repaired']['recovery_mean'],flush=True)
    finally:
        for handle in handles.values(): handle.close()


if __name__=='__main__': main()
