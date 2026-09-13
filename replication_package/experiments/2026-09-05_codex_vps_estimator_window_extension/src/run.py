"""Resumable, one-writer supervisor with 30-second bounded liveness."""
from __future__ import annotations
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
import core as C
from worker import atomic, environment, fingerprint


def units(config):
    result=[]
    for d in config['datasets']:
        for s in config['splits']:
            for m in config['models']:
                for scenario in ['null','mean_shift','noise']:
                    for i in range(config['instances']):
                        result.append(dict(arm='E1',dataset=d,split=s,model=m,scenario=scenario,instance=i,
                            id=f'E1-{d}-{s}-{m}-{scenario}-{i}'))
                for scenario in ['null','mean_shift']:
                    for w in range(config['windows']):
                        result.append(dict(arm='E2',dataset=d,split=s,model=m,scenario=scenario,window=w,
                            id=f'E2-{d}-{s}-{m}-{scenario}-{w}'))
    return sorted(result,key=lambda u:(u['split'],u['model'],u['arm'],
                  {'null':0,'mean_shift':1,'noise':2}[u['scenario']],u.get('instance',u.get('window')),u['dataset']))


def valid(work,unit,fp):
    root=work/'units'/unit['id']
    for receipt in sorted(root.glob('attempt-*/completed.json'),reverse=True):
        try:
            doc=json.loads(receipt.read_text())
            if doc['status']=='completed' and doc['fingerprint']==fp and doc['unit']==unit:
                if all(C.digest(receipt.parent/f)==h for f,h in doc['outputs'].items()):
                    import numpy as np
                    with np.load(receipt.parent/'raw.npz') as data:
                        shapes=json.loads((receipt.parent/'records.json').read_text())['shapes']
                        if set(data.files)==set(shapes) and all(list(data[k].shape)==v for k,v in shapes.items()):
                            return receipt.parent
        except (OSError,ValueError,KeyError):pass
    return None


def notify(message, work):
    """No external notifications in the portable release."""
    return


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--work',required=True)
    ap.add_argument('--stop-after',type=int);ap.add_argument('--limit-units',type=int)
    ap.add_argument('--max-seconds',type=int);ap.add_argument('--total-max-seconds',type=int)
    ap.add_argument('--unit-id');ap.add_argument('--interrupt-after-seconds',type=float)
    ap.add_argument('--plan',action='store_true');a=ap.parse_args()
    path=Path(a.config).resolve();config=json.loads(path.read_text()); work=Path(a.work).resolve()
    fp=fingerprint(path);config['_fingerprint']=fp['digest'];todo=units(config)
    if a.unit_id:todo=[u for u in todo if u['id']==a.unit_id]
    if not todo:raise ValueError('empty unit selection')
    if a.limit_units:todo=todo[:a.limit_units]
    work.mkdir(parents=True,exist_ok=True)
    if a.plan:
        print(json.dumps([{'unit':u['id'],'action':'skipped_validated' if valid(work,u,fp['digest']) else 'run'} for u in todo]));return 0
    if (work/'input_fingerprint.json').exists():
        if json.loads((work/'input_fingerprint.json').read_text())['digest']!=fp['digest']:
            raise RuntimeError('scientific fingerprint changed; use a new run directory')
    if (work/'environment.json').exists():
        if json.loads((work/'environment.json').read_text())!=environment():
            raise RuntimeError('environment changed; mixed-environment resume forbidden')
    lock=work/'supervisor.lock'
    if lock.exists():
        old=json.loads(lock.read_text())
        if old['host']!=environment()['host'] or psutil.pid_exists(old['pid']):
            raise RuntimeError('live/foreign lock; refuse duplicate launch')
        lock.rename(work/f'stale-lock-{int(time.time())}.json')
    with lock.open('x') as f:json.dump({'host':environment()['host'],'pid':os.getpid()},f)
    cfg=work/'resolved_config.json';atomic(cfg,config);atomic(work/'input_fingerprint.json',fp)
    start=time.time();final='FAILED';rc=1;child=None;completed=0;stop=False;resumed=0
    peak=0;last_heartbeat=0;last_checkpoint=None
    max_seconds=a.max_seconds or config['max_seconds']
    total_max=a.total_max_seconds or config.get('total_max_seconds',86400)
    budgetpath=work/'resource_consumption.json'
    consumed=json.loads(budgetpath.read_text())['seconds'] if budgetpath.exists() else 0
    def interrupted(_sig,_frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    logfile=work/f'supervisor-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}.jsonl'
    def event(**row):
        row.update(time=time.time(),host=environment()['host'])
        with logfile.open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(row),flush=True)
    atomic(work/'environment.json',environment());notify('STARTED',work)
    try:
        for unit in todo:
            prior=valid(work,unit,fp['digest'])
            if prior:
                last_checkpoint=datetime.datetime.fromtimestamp((prior/'completed.json').stat().st_mtime).astimezone().isoformat()
                completed+=1;resumed+=1;event(status='skipped_validated',unit=unit['id']);continue
            if stop:final='CANCELLED';rc=130;break
            if time.time()-start>max_seconds or time.time()-start+consumed>total_max:final='TIMED_OUT';rc=124;break
            attempt=work/'units'/unit['id']/('attempt-'+uuid.uuid4().hex[:12]);attempt.mkdir(parents=True)
            atomic(attempt/'unit.json',unit)
            command=[sys.executable,str(C.ROOT/'src/worker.py'),'--unit',str(attempt/'unit.json'),
                     '--config',str(cfg),'--attempt',str(attempt),'--work',str(work)]
            began=time.time();event(status='running',unit=unit['id'],attempt=attempt.name,command=command)
            with (attempt/'cli.log').open('w',encoding='utf-8') as transcript:
                child=subprocess.Popen(command,stdout=transcript,stderr=subprocess.STDOUT)
                atomic(attempt/'execution.json',{'command':command,'pid':child.pid,'started':began,'unit':unit,'fingerprint':fp['digest']})
                while child.poll() is None:
                    now=time.time()
                    fixture_stop=a.interrupt_after_seconds is not None and now-began>a.interrupt_after_seconds
                    if fixture_stop:stop=True
                    if stop or now-began>config['unit_timeout_seconds'] or now-start>max_seconds or now-start+consumed>total_max:
                        child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        final='CANCELLED' if stop else 'TIMED_OUT';rc=130 if stop else 124;break
                    try:proc=psutil.Process(child.pid);tree=[proc]+proc.children(recursive=True)
                    except psutil.NoSuchProcess:continue
                    rss=0;cpu_seconds=0;sampled=[];disappeared=[]
                    for member in tree:
                        try:
                            rss+=member.memory_info().rss;t=member.cpu_times();cpu_seconds+=t.user+t.system;sampled.append(member.pid)
                        except psutil.NoSuchProcess:disappeared.append(member.pid)
                    peak=max(peak,rss)
                    if rss>16*1024**3:
                        child.terminate();child.wait();raise RuntimeError('memory ceiling')
                    if now-last_heartbeat>=config.get('heartbeat_seconds',30):
                        progress={}
                        try:progress=json.loads((attempt/'progress.json').read_text())
                        except (OSError,ValueError):pass
                        hb={'status':'RUNNING','pid':os.getpid(),'child_pid':child.pid,'unit':unit['id'],
                            'completed':completed,'total':len(todo),'time':now,'rss':rss,'cpu_seconds':cpu_seconds,
                            'progress':progress,'elapsed':now-start,'timestamp':datetime.datetime.now().astimezone().isoformat(),
                            'run_id':work.name,'unit_id':unit['id'],'attempt_id':attempt.name,'worker_pid':child.pid,
                            'phase':progress.get('stage','worker_startup'),'phase_started_at':progress.get('phase_started_at',began),
                            'unit_elapsed_seconds':now-began,'completed_atomic_units':completed,'planned_atomic_units':len(todo),
                            'last_durable_checkpoint_at':last_checkpoint,'process_tree_cpu_seconds':cpu_seconds,
                            'sampled_pids':sampled,'disappeared_pids':disappeared}
                        atomic(work/'heartbeat.json',hb);event(**hb);last_heartbeat=now
                        # Reserve the next heartbeat interval so an abrupt loss cannot
                        # silently replenish the cumulative resource allowance.
                        atomic(budgetpath,{'seconds':consumed+now-start+config.get('heartbeat_seconds',30),
                                          'total_max_seconds':total_max,'reservation':True})
                    time.sleep(1)
            exit_code=child.returncode;child=None
            if final in ('CANCELLED','TIMED_OUT'):
                atomic(attempt/'terminal.json',{'status':final,'exit_code':rc,'ended':time.time()});break
            if exit_code!=0 or not valid(work,unit,fp['digest']):
                atomic(attempt/'terminal.json',{'status':'FAILED','exit_code':exit_code,'ended':time.time()})
                raise RuntimeError(f'unit failed {unit["id"]}: {exit_code}; see {attempt}/cli.log')
            completed+=1;event(status='completed',unit=unit['id'],seconds=time.time()-began)
            last_checkpoint=datetime.datetime.now().astimezone().isoformat()
            if completed%20==0:notify(f'PROGRESS {completed}/{len(todo)}',work)
            atomic(attempt/'terminal.json',{'status':'COMPLETED','exit_code':0,'ended':time.time(),'seconds':time.time()-began})
            if a.stop_after and resumed==0 and completed>=a.stop_after:
                final='CONTROLLED_STOP';rc=75;break
        else:final='COMPLETED';rc=0
    except Exception as exc:
        event(status='FAILED',error=str(exc));final='FAILED';rc=1
    finally:
        if child and child.poll() is None:child.terminate();child.wait(timeout=10)
        atomic(work/'run_status.json',{'status':final,'exit_code':rc,'completed':completed,'total':len(todo),
                                      'skipped_validated':resumed,'seconds':time.time()-start,'peak_rss':peak,
                                      'fingerprint':fp['digest'],'ended_at':datetime.datetime.now().astimezone().isoformat()})
        atomic(budgetpath,{'seconds':consumed+time.time()-start,'total_max_seconds':total_max})
        notify(final,work);lock.unlink();event(status=final,exit_code=rc,completed=completed)
    return rc


if __name__=='__main__':raise SystemExit(main())
