"""Exercise process orchestration and frame accounting without a policy/GPU."""
import importlib.util,json,sys,types
from collections import deque
from pathlib import Path
import numpy as np


def test_closed_loop_updates_all_frames_and_pairs_noise_and_start_state(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('closed_loop',root/'scripts/run_e14r_closed_loop.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Env:
        task_list=[{}];current_task_index=0;timestep=0
        @property
        def unwrapped(self):return self
        def observation(self):
            return {'front_rgb_list':[np.full((2,2,3),self.n,np.uint8)],
                    'wrist_rgb_list':[np.zeros((2,2,3),np.uint8)],
                    'joint_state_list':[np.zeros(7)],'gripper_state_list':[np.zeros(2)]}
        def reset(self):self.n=0;return self.observation(),{'task_goal':['move'],'status':'ongoing'}
        def step(self,action):
            self.n+=1
            return self.observation(),0.,False,False,{'task_goal':['move'],'status':'ongoing'}
        def get_state_dict(self):return {'n':self.n}
        def close(self):pass
    class Builder:
        def __init__(self,*a,**kw):pass
        def resolve_episode(self,index):return 7,None
        def make_env_for_episode(self,index):return Env()
    class Raw:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __getitem__(self,key):return self if key!=() else np.zeros(7)
    monkeypatch.setitem(sys.modules,'gymnasium',types.SimpleNamespace(make=lambda *a,**kw:None))
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(manual_seed=lambda seed:None))
    monkeypatch.setitem(sys.modules,'h5py',types.SimpleNamespace(File=Raw))
    monkeypatch.setitem(sys.modules,'robomme',types.ModuleType('robomme'))
    monkeypatch.setitem(sys.modules,'robomme.env_record_wrapper',types.SimpleNamespace(BenchmarkEnvBuilder=Builder))
    monkeypatch.setitem(sys.modules,'robomme.robomme_env',types.ModuleType('robomme.robomme_env'))
    calls=[]
    class Worker:
        def __init__(self,*a,**kw):
            self.messages=deque([{'ready':True}]);self.stdout=self;self.stdin=self;self.code=None
        def readline(self):return 'FRMD_RPC '+json.dumps(self.messages.popleft())+'\n'
        def write(self,line):
            request=json.loads(line)
            if request['op']=='close':self.code=0;return
            if request['op']=='reset':
                self.label=request['condition'];self.live_count=0;self.messages.append({'reset':True});return
            with np.load(request['observation']) as obs:
                count=len(obs['update_images']);self.live_count+=count
                calls.append((self.label,count,request['noise_seed'],int(obs['image'][0,0,0])))
                if count:np.testing.assert_array_equal(obs['update_images'][:,0,0,0,0],np.arange(1,21))
            with open(request['output'],'xb') as f:np.savez(f,actions=np.zeros((20,8)))
            self.messages.append({'live_frames':self.live_count,'new_vision_frames':count,'trajectory_queries':1,'retention':{}})
        def flush(self):pass
        def poll(self):return self.code
        def wait(self,timeout):return self.code
    monkeypatch.setattr(module,'subprocess',types.SimpleNamespace(Popen=Worker,
        PIPE=module.subprocess.PIPE,TimeoutExpired=module.subprocess.TimeoutExpired))
    monkeypatch.setattr(module.select,'select',lambda read,*a:(read,[],[]))
    report=tmp_path/'repair.json'
    report.write_text(json.dumps({'kind':'native_interference_development_pilot','case':{
        'case_id':'A-0-k3','task':'PatternLock','query_step':0,'execution_start':0,'split':'train','seed':7,
        'raw_episode':0,'raw_file':'fake.h5'},'checkpoint':'model','history_config':'tokendrop',
        'noises':{'optimize':[1],'validation':[2],'evaluation':[3]}}))
    report.with_suffix('.npz').write_bytes(b'original source tensor artifact')
    output=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['closed_loop','--report',str(report),'--data',str(tmp_path),
        '--policy-python',sys.executable,'--policy-cwd',str(tmp_path),'--output',str(output),
        '--queries','2','--conditions','interfered','repaired'])
    module.main()
    result=json.loads(output.read_text())
    assert calls==[('interfered',0,200007,0),('interfered',20,201007,20),
                   ('repaired',0,200007,0),('repaired',20,201007,20)]
    assert len({r['start_fingerprint'] for r in result['outcomes']})==1
    assert all(r['steps']==40 and r['queries']==2 for r in result['outcomes'])
    assert all(r['status']=='ongoing' and not r['terminated_or_truncated'] for r in result['outcomes'])
