"""One atomic work unit. Scientific arrays contain no wall-clock metadata."""
from __future__ import annotations
import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import core as C


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def environment():
    return {'host':platform.node(),'platform':platform.platform(),'python':platform.python_version(),
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','psutil','shap','tabulate']}}


def fingerprint(config_path):
    files=[C.ROOT/'src'/name for name in ['core.py','worker.py','run.py']]
    files += [C.PROJECT/'data'/f'{name}.pkl' for name in ['adult','electricity']]
    for directory in (C.ROOT/'vendor').iterdir():
        if directory.is_dir():
            files.extend(sorted(directory.rglob('*.py')))
    records={p.relative_to(C.PROJECT).as_posix():C.digest(p) for p in files}
    config=json.loads(Path(config_path).read_text())
    operational={'max_seconds','total_max_seconds','unit_timeout_seconds','heartbeat_seconds'}
    scientific={k:v for k,v in config.items() if k not in operational and not k.startswith('_')}
    records['scientific_config']=C.hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
    return {'files':records,'digest':C.hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
            'operational_config_sha256':C.digest(config_path),'scientific_config':scientific}


def run(unit, config, attempt, work, progress):
    dataset,split,model,scenario=unit['dataset'],unit['split'],unit['model'],unit['scenario']
    env=environment()
    cachekey=C.seed(config['_fingerprint'],dataset,split,env)
    cache=work/'cache'/f'{dataset}-{split}-{cachekey}.joblib'
    receipt=cache.with_suffix('.json')
    progress(stage='training', detail=f'{dataset}/{split}')
    if cache.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==C.digest(cache):
        ctx=joblib.load(cache)
    else:
        ctx=C.prepare(dataset,split)
        cache.parent.mkdir(parents=True,exist_ok=True)
        tmp=cache.with_suffix('.tmp');joblib.dump(ctx,tmp);os.replace(tmp,cache)
        atomic(receipt,{'sha256':C.digest(cache),'fingerprint':config['_fingerprint'],'environment':env,
                        'train_indices':ctx['train_indices'],'test_indices':ctx['test_indices']})
    p=len(ctx['features']); arrays={}; records=[]
    progress(stage='oracle', detail=unit['id'])
    if unit['arm']=='E1':
        ids=np.random.default_rng(C.seed('E1-instances',dataset,split)).choice(len(ctx['test']),config['instances'],replace=False)
        ix=int(ids[unit['instance']])
        left=ctx['test'].iloc[[ix]]
        right=C.shift(left,ctx,scenario,C.seed('E1-shift',dataset,split,scenario,ix))
        g0=C.Game(ctx,ctx['models'][model],left,progress.query)
        g1=C.Game(ctx,ctx['models'][model],right,progress.query)
        phi0,v0,o0=C.exact(g0);phi1,v1,o1=C.exact(g1)
        arrays.update(oracle_phi0=phi0,oracle_phi1=phi1,oracle_values0=v0,oracle_values1=v1)
        budgets=config['budgets'][dataset]
        for method in config['methods']:
            for budget in budgets:
                outputs=np.empty((config['draws'],3,p))
                for draw in range(config['draws']):
                    key=C.seed('E1-MC',dataset,split,model,scenario,ix,method,budget,draw)
                    s0,s1=C.seed(key,0),C.seed(key,1)
                    progress(stage='E1', detail=f'{method}/{budget}/{draw}', completed_replicates=draw)
                    for j,(game,s) in enumerate([(g0,s0),(g1,s0),(g1,s1)]):
                        outputs[draw,j],rec=C.estimate(game,method,budget,s)
                        records.append({'method':method,'budget':budget,'draw':draw,'role':j,'seed':s,**rec})
                arrays[f'{method}__{budget}']=outputs
        extra={'instance_index':ix,'oracle_receipts':[o0,o1]}
    else:
        window=unit['window'];n=config['window_size']
        rng=np.random.default_rng(C.seed('E2-windows',dataset,split,scenario,window))
        left_ids=rng.choice(len(ctx['test']),n,replace=False)
        right_ids=rng.choice(len(ctx['test']),n,replace=False)
        left=ctx['test'].iloc[left_ids]
        right=C.shift(ctx['test'].iloc[right_ids],ctx,scenario,C.seed('E2-shift',dataset,split,scenario,window))
        games0=[C.Game(ctx,ctx['models'][model],left.iloc[[i]],progress.query) for i in range(n)]
        games1=[C.Game(ctx,ctx['models'][model],right.iloc[[i]],progress.query) for i in range(n)]
        phi0=[];phi1=[];oracles=[]
        for i,(g0,g1) in enumerate(zip(games0,games1)):
            progress(stage='oracle', detail=f'window={window}/position={i}')
            a,v,o=C.exact(g0);phi0.append(a); arrays[f'oracle_values0_{i}']=v;oracles.append(o)
            a,v,o=C.exact(g1);phi1.append(a); arrays[f'oracle_values1_{i}']=v;oracles.append(o)
        arrays['oracle_phi0']=np.array(phi0);arrays['oracle_phi1']=np.array(phi1)
        budget=config['budgets'][dataset][0]
        for method in config['window_methods']:
            outputs=np.empty((config['window_draws'],n,3,p))
            for draw in range(config['window_draws']):
                for i,(g0,g1) in enumerate(zip(games0,games1)):
                    key=C.seed('E2-MC',dataset,split,model,scenario,window,method,budget,draw,i)
                    s0,s1=C.seed(key,0),C.seed(key,1)
                    progress(stage='E2', detail=f'{method}/{draw}/{i}', completed_replicates=draw)
                    for j,(g,s) in enumerate([(g0,s0),(g1,s0),(g1,s1)]):
                        outputs[draw,i,j],rec=C.estimate(g,method,budget,s)
                        records.append({'method':method,'budget':budget,'draw':draw,'position':i,'role':j,'seed':s,**rec})
            arrays[f'{method}__{budget}']=outputs
        extra={'left_indices':left_ids.tolist(),'right_indices':right_ids.tolist(),'oracle_receipts':oracles}
    path=attempt/'raw.npz';temp=attempt/'raw.tmp'
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    atomic(attempt/'records.json',{'unit':unit,'records':records,'extra':extra,
                                 'shapes':{k:list(v.shape) for k,v in arrays.items()}})
    atomic(attempt/'completed.json',{'status':'completed','unit':unit,'fingerprint':config['_fingerprint'],
        'environment':env,'outputs':{x:C.digest(attempt/x) for x in ['raw.npz','records.json']}})


class Progress:
    def __init__(self,path): self.path=path;self.count=0;self.last=0;self.state={}
    def __call__(self,**kwargs):
        if kwargs.get('stage',self.state.get('stage'))!=self.state.get('stage'):
            self.state['phase_started_at']=time.time()
        self.state.update(kwargs)
        if time.time()-self.last>1:
            atomic(self.path,{**self.state,'model_query_rows':self.count,'time':time.time(),'pid':os.getpid()})
            self.last=time.time()
    def query(self,n):self.count+=n;self()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',required=True);p.add_argument('--config',required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--work',required=True);a=p.parse_args()
    unit=json.loads(Path(a.unit).read_text()); config=json.loads(Path(a.config).read_text())
    attempt=Path(a.attempt);attempt.mkdir(parents=True,exist_ok=True)
    progress=Progress(attempt/'progress.json')
    run(unit,config,attempt,Path(a.work),progress)
    print(json.dumps({'unit':unit['id'],'status':'completed','queries':progress.count}),flush=True)
