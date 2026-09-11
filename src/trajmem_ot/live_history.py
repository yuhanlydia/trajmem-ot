"""Explicit live-frame accounting for native FRMD closed-loop evaluation."""
from __future__ import annotations
import numpy as np


def collect_step_frames(observation):
    images=observation['front_rgb_list']
    joints=observation['joint_state_list']; grippers=observation['gripper_state_list']
    if not images or len(images)!=len(joints) or len(images)!=len(grippers):
        raise ValueError('live image, joint and gripper lists must be nonempty and aligned')
    images=np.stack([np.asarray(image) for image in images])
    states=np.stack([np.concatenate([np.asarray(joint).reshape(-1),
                                   np.asarray(gripper).reshape(-1)[:1]])
                     for joint,gripper in zip(joints,grippers)]).astype(np.float32)
    if images.ndim!=4 or images.shape[-1]!=3 or states.shape[1]!=8:
        raise ValueError('expected RGB frames and seven joints plus one gripper dimension')
    if not np.isfinite(images).all() or not np.isfinite(states).all():
        raise ValueError('live frames and states must be finite')
    return {'images':images[:,None],'states':states}


def extend_with_live_segment(segments, frame_count, branch_identity):
    if not segments or not isinstance(frame_count,int) or frame_count<0:
        raise ValueError('nonempty base history and nonnegative live frame count required')
    copied=[dict(segment) for segment in segments]
    if frame_count:
        copied.append({**copied[-1],'start':0,'end_exclusive':frame_count,
                       'raw_file':f'live://{branch_identity}',
                       'source_revision':f'live:{branch_identity}'})
    return copied
