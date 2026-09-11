import numpy as np
from trajmem_ot.live_history import collect_step_frames, extend_with_live_segment


def test_collects_every_new_frame_and_uses_last_state_not_only_chunk_endpoint():
    frames=collect_step_frames({'front_rgb_list':[np.zeros((2,2,3)),np.ones((2,2,3))],
       'joint_state_list':[np.arange(7),np.arange(7)+1],
       'gripper_state_list':[np.array([.1,.1]),np.array([.2,.2])]})
    assert frames['images'].shape==(2,1,2,2,3)
    assert frames['states'].shape==(2,8)
    np.testing.assert_allclose(frames['states'][:,-1],[.1,.2])


def test_live_extension_preserves_raw_ranges_and_has_branch_specific_identity():
    source=[{'task':'A','episode':0,'raw_episode':0,'raw_file':'A.h5','source_revision':'v1','start':0,'end_exclusive':5}]
    result=extend_with_live_segment(source,3,'case-A/repaired')
    assert result[:-1]==source
    assert result[-1]['start']==0 and result[-1]['end_exclusive']==3
    assert result[-1]['source_revision']=='live:case-A/repaired'
    assert extend_with_live_segment(source,0,'case-A/repaired')==source
    assert source[0]['end_exclusive']==5
