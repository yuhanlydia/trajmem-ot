import json, random, time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import sapien
from trajmem_ot.headless_physics import patch_sapien_for_physics_only, physics_only_gym_make
from trajmem_ot.robomme_branch import ReplayBranchRunner
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
import robomme.robomme_env
patch_sapien_for_physics_only(sapien)
gym.make = physics_only_gym_make(gym.make)
DemonstrationWrapper._augment_obs_and_info = lambda self, obs, info, action: (obs, info)
started = time.time()
reports = []
for task in ['PatternLock', 'RouteStick', 'VideoPlaceButton']:
    def factory():
        builder = BenchmarkEnvBuilder(task, dataset='test', action_space='joint_angle', gui_render=False, max_steps=100)
        seed, _ = builder.resolve_episode(0)
        seed = 0 if seed is None else seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        return builder.make_env_for_episode(0)
    try:
        fingerprint = ReplayBranchRunner(factory).assert_deterministic([])
        reports.append({'task': task, 'fingerprint': fingerprint, 'deterministic': True})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        reports.append({'task': task, 'error': repr(exc), 'deterministic': False})
    print(json.dumps(reports[-1]), flush=True)
Path('/root/trajmem-preparation/physics-smoke.json').write_text(json.dumps({'reports': reports, 'seconds':time.time()-started}, indent=2))
