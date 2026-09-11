#!/usr/bin/env python3
"""Persistent policy worker for the separate benchmark Python environment.

The local stdin/stdout protocol exchanges NPZ file paths, never network requests.
Library logs go to stderr; responses use the FRMD_RPC prefix.
"""
import argparse
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args()
    with redirect_stdout(sys.stderr):
        import h5py,jax,jax.numpy as jnp
        from trajmem_ot.robomme_runtime import load_runtime_state
        from trajmem_ot.robomme_jax import prepare_policy_observation,build_fixed_noise_action_problem
        from trajmem_ot.native_history import encode_native_history,encode_native_frame_sampling,transfer_memory_delta
        from trajmem_ot.live_history import extend_with_live_segment
        from trajmem_ot.finite_response import quantized_norm_matched_memory
        report=json.loads(args.report.read_text());case=report['case']
        seed=report['noises']['optimize'][0]-1000
        runtime=load_runtime_state(checkpoint=report['checkpoint'],data=args.data,index=case['query_index'],seed=seed)
        policy=runtime.policy
        if runtime.history_config_name!=report['history_config']:
            raise ValueError('source report and loaded checkpoint history configurations differ')
        if runtime.history_config_name not in ('perceptual-tokendrop-modul.yaml','perceptual-framesamp-modul.yaml'):
            raise ValueError('unsupported native memory backend')
        quantize=lambda x:np.asarray(jnp.asarray(x,dtype=jnp.bfloat16),dtype=np.float32)
        with np.load(args.report.with_suffix('.npz'),allow_pickle=False) as archive:
            initial=archive['student_memory'].copy();repaired=archive['repaired_memory'].copy()
            teacher_memory=archive['teacher_memory'].copy()
        norm=float(np.linalg.norm(repaired-initial))
        memories={'interfered':initial,'repaired':repaired,'teacher':teacher_memory}
        rng=np.random.default_rng(seed+case['episode'])
        for label,direction in [('negative',initial-repaired),('random',rng.normal(size=initial.shape)),
                                ('interpolation',teacher_memory-initial)]:
            memories[label]=(initial.copy() if norm==0 or np.linalg.norm(direction)==0 else
                np.asarray(quantized_norm_matched_memory(jnp.asarray(initial,dtype=jnp.bfloat16),direction,
                    target_norm=norm,relative_tolerance=.03).quantized_memory,dtype=np.float32))
        deltas={label:memory-initial for label,memory in memories.items() if label!='teacher'}
        handles={};live_count=0;label=None;branch_identity=None
        live_buffer=policy.mem_buffer;live_buffer.compute_token_drop_score=False
        @lru_cache(maxsize=512)
        def raw_feature(episode,step):
            return np.load(args.data/'features'/f'episode_{episode}'/f'token_emb_{step}.npy',allow_pickle=True).item()
        def feature(ref):
            if ref['source_revision'].startswith('live:'):
                return live_buffer.get_history_feats(ref['step'])
            return raw_feature(ref['episode'],ref['step'])
        def pixels(ref):
            if ref['source_revision'].startswith('live:'):
                return live_buffer._history_feats[ref['step']]['image_pixels'][0]
            path=ref['raw_file']
            if path not in handles:handles[path]=h5py.File(path,'r')
            return handles[path][f"episode_{ref['raw_episode']}/timestep_{ref['step']}/obs/front_rgb"][()]
        def encode(segments):
            if runtime.history_config_name=='perceptual-framesamp-modul.yaml':
                return encode_native_frame_sampling(segments,feature_loader=feature)
            return encode_native_history(segments,feature_loader=feature,pixel_loader=pixels)
        # Validate that the saved edit is being applied to exactly its source memory.
        encoded=encode(case['student_history'])
        np.testing.assert_array_equal(quantize(encoded.history['static_image_emb']),initial)
        del encoded
    print('FRMD_RPC '+json.dumps({'ready':True,'history_config':runtime.history_config_name}),flush=True)
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line)
                with redirect_stdout(sys.stderr):
                    if request['op']=='close':break
                    if request['op']=='reset':
                        label=request['condition']
                        if label not in memories:raise ValueError('unknown condition')
                        branch_identity=request['branch_identity'];live_count=0
                        live_buffer._history_feats={}
                        response={'reset':True}
                    elif request['op']=='act':
                        if label is None:raise ValueError('reset required before action')
                        with np.load(request['observation'],allow_pickle=False) as arrays:
                            item={key:arrays[key].copy() for key in ('image','wrist_image','state')}
                            item['prompt']=str(arrays['prompt'].item())
                            images=arrays['update_images'];states=arrays['update_states']
                            if len(images)!=len(states):raise ValueError('unaligned live frame update')
                            for offset in range(0,len(images),4):
                                n=min(4,len(images)-offset)
                                live_buffer.add_buffer(images[offset:offset+n],states[offset:offset+n],
                                                       list(range(live_count,live_count+n)))
                                live_count+=n
                        base_segments=case['teacher_history'] if label=='teacher' else case['student_history']
                        encoded=encode(extend_with_live_segment(base_segments,live_count,branch_identity))
                        current=quantize(encoded.history['static_image_emb'])
                        if label=='teacher':
                            memory=current;retention={'teacher_condition':True}
                        else:
                            transferred,retention=transfer_memory_delta(report['student_token_sources'],
                                                                         encoded.token_sources,deltas[label])
                            memory=quantize(current+transferred)
                            retention['applied_norm']=float(np.linalg.norm(memory-current))
                        item.update(encoded.history)
                        observation=prepare_policy_observation(policy,item)
                        noise=jax.random.normal(jax.random.key(int(request['noise_seed'])),
                            (1,int(policy._model.action_horizon),int(policy._model.action_dim)))
                        problem=build_fixed_noise_action_problem(policy,observation,noise=noise,
                                                                 rng=jax.random.key(seed),num_steps=10)
                        normalized=np.asarray(problem.action_fn(jnp.asarray(memory,dtype=jnp.bfloat16)).block_until_ready(),dtype=np.float32)
                        physical=policy._output_transform({'state':np.asarray(observation.state[0]),'actions':normalized})['actions']
                        output=Path(request['output'])
                        with output.open('xb') as f:np.savez_compressed(f,actions=physical,normalized_actions=normalized)
                        response={'output':str(output),'live_frames':live_count,'new_vision_frames':len(images),
                                  'trajectory_queries':1,'retention':retention,
                                  'action_sha256':hashlib.sha256(np.asarray(physical).tobytes()).hexdigest()}
                    else:raise ValueError('unknown RPC operation')
                print('FRMD_RPC '+json.dumps(response,allow_nan=False),flush=True)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                print('FRMD_RPC '+json.dumps({'error':repr(error)}),flush=True)
                return 1
    finally:
        for handle in handles.values():handle.close()
    return 0


if __name__=='__main__':sys.exit(main())
