import json
from pathlib import Path
import numpy as np
from trajmem_ot.native_history import encode_native_frame_sampling
from mme_vla_suite.shared.mem_buffer import MemoryBuffer
root=Path('/root/trajmem-ot/data/robomme_preprocessed_pilot30')
ledger=json.loads(Path('/root/trajmem-preparation/pilot-preprocessing-ledger.json').read_text())
checks=[]
for task in ['PatternLock','RouteStick','VideoPlaceButton']:
 row=next(r for r in ledger if r['task']==task and r['status']=='complete')
 for end in [3,row['execution_start']+1]:
  segment={**row,'start':0,'end_exclusive':end}
  def feature(ref):return np.load(root/'features'/f"episode_{row['episode']}"/f"token_emb_{ref['step']}.npy",allow_pickle=True).item()
  actual=encode_native_frame_sampling([segment],feature_loader=feature)
  buffer=MemoryBuffer(prepare_buffer=False)
  indices=buffer.get_frame_sampling_indices(end-1,512,16)
  expected=buffer._prepare_frame_sampling({i:feature({'step':i}) for i in indices},indices,512,16)
  errors={}
  for key,value in zip(['static_image_emb','static_pos_emb','static_state_emb','static_mask'],expected):
   observed=actual.history[key]
   error=float(np.max(np.abs(observed.astype(np.float32)-np.asarray(value,dtype=np.float32))))
   np.testing.assert_array_equal(observed,value)
   errors[key]=error
  assert len(actual.token_sources)==512
  assert sum(t is not None for t in actual.token_sources)==int(actual.history['static_mask'].sum())
  checks.append({'task':task,'episode':row['episode'],'frames':end,'valid_tokens':int(actual.history['static_mask'].sum()),'maximum_absolute_errors':errors})
report={'kind':'released_framesamp_native_prefix_equivalence','source_revision':row['source_revision'],'checks':checks}
Path('/root/trajmem-preparation/framesamp-encoder-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
