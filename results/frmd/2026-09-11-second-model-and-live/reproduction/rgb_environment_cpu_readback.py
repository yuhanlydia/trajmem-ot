import json,time,random
from pathlib import Path
import numpy as np,torch
import gymnasium as gym
original_make=gym.make
def cpu_readback_make(*args,**kwargs):
 kwargs.update(sim_backend="cpu",render_backend="cpu")
 return original_make(*args,**kwargs)
gym.make=cpu_readback_make
from robomme.env_record_wrapper import BenchmarkEnvBuilder
import robomme.robomme_env
from trajmem_ot.robomme_branch import task_progress_snapshot
out=Path('/root/trajmem-preparation/rgb-environment-smoke.json');results=[]
for task in ['PatternLock','RouteStick','VideoPlaceButton']:
 builder=BenchmarkEnvBuilder(task,dataset='train',action_space='joint_angle',gui_render=False,max_steps=200)
 seed,_=builder.resolve_episode(0);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 start=time.time();env=builder.make_env_for_episode(0)
 try:
  obs,info=env.reset()
  front=np.asarray(obs['front_rgb_list'][-1]);wrist=np.asarray(obs['wrist_rgb_list'][-1])
  assert front.ndim==3 and front.shape[-1]==3 and np.ptp(front)>0
  assert wrist.ndim==3 and wrist.shape[-1]==3 and np.ptp(wrist)>0
  row={'task':task,'seed':seed,'split':'train','front_shape':list(front.shape),'wrist_shape':list(wrist.shape),'front_min':int(front.min()),'front_max':int(front.max()),'reset_frames':len(obs['front_rgb_list']),'progress':task_progress_snapshot(env.unwrapped),'seconds':time.time()-start,'renderer':'NVIDIA EGL Vulkan ICD, SAPIEN default visual geometry'}
  results.append(row);out.write_text(json.dumps(results,indent=2)+'\n');print('RGB PASS',task,flush=True)
 finally:env.close()
