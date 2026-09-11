"""Runtime-only diagnostic: no benchmark claims or repair optimization."""
import json,time,pathlib
import jax
import jax.numpy as jnp
import numpy as np
from trajmem_ot.robomme_runtime import load_runtime_state
from trajmem_ot.robomme_jax import build_fixed_noise_action_problem
out=pathlib.Path('/root/trajmem-preparation')
started=time.time()
ckpt=pathlib.Path('/root/trajmem-ot/checkpoints/mme_vla_suite/perceptual-tokendrop-modul/79999')
runtime=load_runtime_state(checkpoint=ckpt,data='/root/trajmem-ot/data/robomme_preprocessed_data_sample',index=0,seed=7)
model=runtime.policy._model
noise=jax.random.normal(jax.random.key(108),(1,int(model.action_horizon),int(model.action_dim)))
problem=build_fixed_noise_action_problem(runtime.policy,runtime.observation,noise=noise,rng=jax.random.key(109),num_steps=10)
memory=jnp.asarray(problem.memory,dtype=jnp.bfloat16)
a=np.asarray(problem.action_fn(memory).block_until_ready(),dtype=np.float32)
a_repeat=np.asarray(problem.action_fn(memory).block_until_ready(),dtype=np.float32)
assert np.isfinite(a).all()
assert np.array_equal(a,a_repeat), 'Identical-noise forward calls are not deterministic'
direction=jax.random.normal(jax.random.key(110),memory.shape,dtype=jnp.float32)
direction=direction/jnp.linalg.norm(direction)
h=2.5e-4*float(jnp.linalg.norm(memory.astype(jnp.float32)))
plus=(memory.astype(jnp.float32)+h*direction).astype(jnp.bfloat16)
minus=(memory.astype(jnp.float32)-h*direction).astype(jnp.bfloat16)
ap=np.asarray(problem.action_fn(plus).block_until_ready(),dtype=np.float32)
am=np.asarray(problem.action_fn(minus).block_until_ready(),dtype=np.float32)
assert np.isfinite(ap).all() and np.isfinite(am).all()
report={'kind':'runtime_only_smoke','checkpoint':str(ckpt),'history_config':runtime.history_config_name,'sample_index':0,'memory_shape':list(memory.shape),'memory_dtype':str(memory.dtype),'action_shape':list(a.shape),'all_actions_finite':True,'same_noise_deterministic':True,'input_half_chord_norm':float(np.linalg.norm((np.asarray(plus,dtype=np.float32)-np.asarray(minus,dtype=np.float32))/2)),'action_half_chord_norm':float(np.linalg.norm((ap-am)/2)),'policy_queries':4,'seconds':time.time()-started,'limitation':'No teacher/student repair, persistence, or physical success tested; FRMD implementation is absent from fetched main.'}
np.savez(out/'runtime-smoke-actions.npz',baseline=a,plus=ap,minus=am)
(out/'runtime-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
