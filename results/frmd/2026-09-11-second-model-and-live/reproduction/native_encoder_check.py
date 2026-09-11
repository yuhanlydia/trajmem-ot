import json
from pathlib import Path
import h5py
import numpy as np
from trajmem_ot.native_history import encode_native_history
from mme_vla_suite.shared.mem_buffer import MemoryBuffer
raw=Path('/root/trajmem-ot/data/robomme_raw_pilot/selected_10/record_dataset_PatternLock.h5')
data=Path('/root/trajmem-ot/data/robomme_preprocessed_pilot30')
ledger=json.loads(Path('/root/trajmem-preparation/pilot-preprocessing-ledger.json').read_text())
row=next(r for r in ledger if r['episode']==0 and r['status']=='complete')
segment={**row,'start':0,'end_exclusive':row['execution_start']+1}
with h5py.File(raw,'r') as f:
 def pixels(ref): return f[f"episode_0/timestep_{ref['step']}/obs/front_rgb"][()]
 def features(ref): return np.load(data/'features/episode_0'/f"token_emb_{ref['step']}.npy",allow_pickle=True).item()
 actual=encode_native_history([segment],pixel_loader=pixels,feature_loader=features)
 reference=MemoryBuffer(prepare_buffer=False,compute_token_drop_score=True)
 for step in range(segment['end_exclusive']):
  reference._history_feats[step]={'image_pixels':pixels({'step':step})[None],**features({'step':step})}
  reference._process_token_drop_score(step)
 expected=reference.prepare_token_dropping(segment['end_exclusive']-1,512,reference.default_history_feats_gather_fn)
 report={}
 for key,expected_value in zip(['static_image_emb','static_pos_emb','static_state_emb','static_mask'],expected):
  observed=actual.history[key]
  error=float(np.max(np.abs(observed.astype(np.float32)-np.asarray(expected_value,dtype=np.float32))))
  np.testing.assert_allclose(observed,expected_value,atol=1e-6)
  report[key]={'shape':list(observed.shape),'maximum_absolute_error':error}
 report.update(kind='real_raw_prefix_encoder_equivalence',raw_file=str(raw),episode=0,frames=actual.frame_count,
               valid_tokens=int(actual.history['static_mask'].sum()),source_revision=row['source_revision'])
 Path('/root/trajmem-preparation/native-encoder-check.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
