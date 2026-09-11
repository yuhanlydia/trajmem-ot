"""Guarded upstream preprocessing with an actual per-episode provenance ledger."""
import json, os, pathlib, time
import h5py
from mme_vla_suite.dataset_builder.build_robomme_dataset import DatasetProcessor
raw=pathlib.Path('/root/trajmem-ot/data/robomme_raw_pilot/selected_10')
out=pathlib.Path('/root/trajmem-ot/data/robomme_preprocessed_pilot30')
ledger=pathlib.Path('/root/trajmem-preparation/pilot-preprocessing-ledger.json')
if out.exists():
 raise FileExistsError('Refusing to let upstream delete an existing output directory: '+str(out))
tasks=['PatternLock','RouteStick','VideoPlaceButton']
for task in tasks:
 with h5py.File(raw/f'record_dataset_{task}.h5','r') as f:
  assert set(f)=={f'episode_{i}' for i in range(10)}
records=[]
class AuditedProcessor(DatasetProcessor):
 def _process_episode(self,data,episode_idx,global_episode_idx,mem_buffer,exec_sample_id,total_sample_id):
  group=data[f'episode_{episode_idx}']; file=pathlib.Path(data.filename)
  task=file.name.removeprefix('record_dataset_').removesuffix('.h5')
  start=self._first_execution_step(group)
  timesteps=sum(k.startswith('timestep_') for k in group)
  record={'task':task,'episode':global_episode_idx,'raw_episode':episode_idx,'raw_file':str(file),
          'split':'train','seed':int(group['setup/seed'][()]),'execution_start':start,
          'timesteps':timesteps,'first_sample':exec_sample_id,
          'source_revision':'a5e4e25ffe8af34f64944f9533d06455ce5f8337','status':'processing'}
  metadata=pathlib.Path('/root/trajmem-ot/third_party/robomme_policy_learning/third_party/robomme_benchmark/src/robomme/env_metadata/train')/f'record_dataset_{task}_metadata.json'
  rows=json.loads(metadata.read_text())['records']
  match=[r for r in rows if int(r['episode'])==episode_idx]
  assert len(match)==1 and int(match[0]['seed'])==record['seed'], 'raw seed/metadata mismatch'
  records.append(record); ledger.write_text(json.dumps(records,indent=2)+'\n')
  result=super()._process_episode(data,episode_idx,global_episode_idx,mem_buffer,exec_sample_id,total_sample_id)
  record.update(status='complete',end_sample_exclusive=result[2])
  assert result[2]-exec_sample_id==timesteps-start
  ledger.write_text(json.dumps(records,indent=2)+'\n')
  return result
processor=AuditedProcessor(raw_data_path=str(raw),preprocessed_data_path=str(out),max_episodes=10)
processor.run()
assert len(records)==30 and all(r['status']=='complete' for r in records)
(out/'meta/provenance.json').write_text(json.dumps(records,indent=2)+'\n')
print('COMPLETED',len(records),'verified source episodes',flush=True)
