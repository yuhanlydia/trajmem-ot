import numpy as np
from trajmem_ot.native_history import encode_native_history, transfer_memory_delta


class Buffer:
    def __init__(self, **kwargs):
        self._history_feats={}
        self.token_drop_last_frame=-1
        self.frames=[]
    def _process_token_drop_score(self, step):
        self.frames.append(int(self._history_feats[step]['image_pixels'][0,0,0,0]))
        self.token_drop_last_frame=step
    def get_token_dropping_indices(self):
        return [(i,0,0) for i in range(len(self.frames))]
    def filter_token_dropping_indices(self, indices, step, budget, is_sorted=True):
        return indices[-budget:]
    def _prepare_token_dropping(self, features, selected, budget):
        image=np.zeros((budget,1),np.float32); pos=image.copy(); state=image.copy(); mask=np.zeros(budget,bool)
        for i,(step,view,patch) in enumerate(selected):
            image[i]=features[step]['image_emb_8x8'][view,patch]
            pos[i]=features[step]['pos_emb_8x8'][view,patch]
            state[i]=features[step]['state_emb']
            mask[i]=True
        return image,pos,state,mask


def test_history_concatenation_reindexes_time_and_preserves_token_sources():
    seen=[]; buffers=[]
    def factory(**kw):
        buffer=Buffer(**kw); buffers.append(buffer); return buffer
    def feature(ref):
        seen.append((ref['episode'],ref['step']))
        return {'image_emb_8x8':np.array([[[ref['episode']*10+ref['step']]]]),
                'state_emb':np.array([ref['step']])}
    segments=[{'task':'A','episode':0,'raw_episode':0,'raw_file':'A.h5','source_revision':'v1','start':0,'end_exclusive':2},
              {'task':'B','episode':1,'raw_episode':0,'raw_file':'B.h5','source_revision':'v1','start':3,'end_exclusive':5}]
    result=encode_native_history(segments,feature_loader=feature,
                                 pixel_loader=lambda ref:np.full((2,2,3),ref['episode']*10+ref['step']),
                                 token_budget=2,buffer_factory=factory,
                                 position_factory=lambda steps:np.array(steps).reshape(-1,1,1))
    assert buffers[0].frames==[0,1,13,14]
    assert seen==[(1,3),(1,4)]
    np.testing.assert_array_equal(result.history['static_image_emb'].ravel(),[13,14])
    np.testing.assert_array_equal(result.history['static_pos_emb'].ravel(),[2,3])
    assert result.frame_count==4
    assert result.token_sources[0]['step']==3
    assert result.token_sources[0]['task']=='B'


def test_persistence_tracks_tokens_after_reordering_and_eviction():
    def token(name):
        return {'task':name,'raw_file':name+'.h5','raw_episode':0,'step':0,'view':0,'patch':0,'source_revision':'v1'}
    delta=np.array([[1.,2.],[3.,4.],[8.,8.]])
    transferred,report=transfer_memory_delta([token('A'),token('B'),None],
                                             [token('B'),token('C'),None],delta)
    np.testing.assert_array_equal(transferred,[[3,4],[0,0],[0,0]])
    assert report['retained_tokens']==1
    assert report['source_tokens']==2


def test_same_episode_number_in_different_raw_shards_is_not_same_token():
    source={'task':'A','raw_episode':0,'step':0,'view':0,'patch':0,'source_revision':'v1','raw_file':'shard1.h5'}
    target={**source,'raw_file':'shard2.h5'}
    result,report=transfer_memory_delta([source],[target],np.ones((1,2)))
    np.testing.assert_array_equal(result,[[0,0]])
    assert report['retained_tokens']==0
