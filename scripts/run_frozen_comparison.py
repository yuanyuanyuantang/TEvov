"""Bounded two-suite supervisor; owns only its GPU1 model process group."""
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capture_model_artifacts import inventory, signature
from tevo.experiment.suite import atomic_json, fingerprints, load_plan
from tevo.memory.evidence import write_private_json
from tevo.reproducibility import sha256_file


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def stop(child):
    if child is None or child.poll() is not None:
        return
    if os.getpgid(child.pid) != child.pid:
        raise RuntimeError('Refusing to signal a process group not owned by this launcher')
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def check(receipt):
    for filename, expected in receipt['inputs_sha256'].items():
        if sha256_file(ROOT / filename) != expected:
            raise ValueError('Locked input changed: ' + filename)
    for name, filename in receipt['configs'].items():
        if fingerprints(load_plan(ROOT / filename)) != receipt['fingerprints'][name]:
            raise ValueError('Run identity changed: ' + name)
    current = {n: [list(s) for s in signature(p)]
               for n,p in inventory(Path(receipt['model_path'])).items()}
    if current != read(ROOT / receipt['model_signatures']):
        raise ValueError('Model assets changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.chdir(ROOT)
    receipt = read(args.receipt)
    out = args.receipt.resolve().parent
    env = os.environ.copy()
    env.update(receipt['environment'])
    server, children = None, {}
    start = time.monotonic()
    state = dict(phase='preflight', supervisor_pid=os.getpid(), started_at=now())
    def save():
        state['updated_at'] = now()
        atomic_json(out/'state.json',state)
    def launch(command, log):
        with open(out/log,'x') as stream:
            return subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL,start_new_session=True)
    try:
        check(receipt)
        gpu = subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(gpu.strip()) > 1000:
            raise RuntimeError('GPU1 is occupied; no processes were terminated')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',8010)) == 0:
                raise RuntimeError('Port8010 occupied; refusing to replace existing service')
        server = launch(receipt['server_command'],'server.log')
        state.update(phase='starting_model',server_pid=server.pid);save()
        import requests
        deadline=time.monotonic()+360
        health=[]
        while True:
            if server.poll() is not None:
                raise RuntimeError('Owned model server exited during startup')
            try:
                response=requests.get('http://127.0.0.1:8010/health',timeout=3)
                health.append(dict(time=now(),status=response.status_code))
                if response.ok:
                    break
            except requests.RequestException as exc:
                health.append(dict(time=now(),error=type(exc).__name__))
            if time.monotonic()>deadline:
                raise TimeoutError('Model startup deadline reached')
            time.sleep(5)
        write_private_json(out/'startup_probes.json',health,exclusive=True)
        check(receipt)
        for name,config in receipt['configs'].items():
            children[name]=launch([sys.executable,'scripts/run_experiment_suite.py','--config',config,'--run','--resume'],name+'.log')
        state.update(phase='actor',child_pids={n:p.pid for n,p in children.items()});save()
        last_check=0
        while any(p.poll() is None for p in children.values()):
            if server.poll() is not None:
                raise RuntimeError('Owned model server exited during evaluation')
            if time.monotonic()-start > 64800:
                raise TimeoutError('18-hour run bound reached; preserve partial results')
            for name,child in children.items():
                if child.poll() not in (None,0):
                    raise RuntimeError(name+' failed; preserve rows without automatic retries')
            if time.monotonic()-last_check>60:
                check(receipt);last_check=time.monotonic()
                state['completed']={name:len(list((ROOT/load_plan(config)['config']['output_dir']/'trials').glob('*.json')))
                                    for name,config in receipt['configs'].items()}
                save()
            time.sleep(5)
        if any(p.returncode for p in children.values()):
            raise RuntimeError('Actor suite failed')
        stop(server)
        state.update(phase='offline_regrade',server_returncode=server.returncode);save()
        check(receipt)
        for name,config in receipt['configs'].items():
            suite=load_plan(config)['config']['output_dir']
            child=launch([sys.executable,'scripts/regrade_suite.py','--config',config,'--suite-dir',suite,
                          '--output',str(Path(suite)/'regrade.json')],name+'_regrade.log')
            children['regrade_'+name]=child
            if child.wait():
                raise RuntimeError(name+' offline regrade failed')
            check(receipt)
        state.update(phase='final_verification');save()
        child=launch([sys.executable,'scripts/analyze_frozen_comparison.py','--control',receipt['configs']['control'],
                      '--variant',receipt['configs']['variant'],'--output',str(out/'results.json')],'analysis.log')
        children['analysis']=child
        if child.wait():
            raise RuntimeError('Final offline verification failed')
        check(receipt)
        result=read(out/'results.json')
        lines=['# 冻结两臂 BFCL 四次评测结果','','| 臂 | Avg@4 | Pass@4 | trials |','| --- | ---: | ---: | ---: |']
        for name,values in result['arms'].items():
            lines.append(f"| {name} | {values['avg_at_4']:.2%} | {values['pass_at_4']:.2%} | 600 |")
        for metric,label in [('trial_mean','Avg@4'),('best_of_observed','Pass@4')]:
            delta=result['variant_minus_control'][metric]
            lo,hi=delta['task_bootstrap_ci95']
            lines+=['',f"{label} 新−旧：{delta['difference']*100:.2f}pp；逐题配对95% bootstrap区间 [{lo*100:.2f}, {hi*100:.2f}]pp。"]
        lines+=['','固定seed=0、temperature=0.9；同150题，每题每臂4次；1200条均完成官方离线重判与注入核验。',
                '对照为136条普通记忆+首轮检索；新流程为123条冻结记忆+当前用户轮检索。比较整套流程，不能单独归因某一改动。',
                '该验证题集此前已使用；本轮不主张优于ReMe或独立未见任务泛化。结果无论正负均保留，不据此自动扩实验。',
                '实际请求、token未知项、逐请求核验计数和配对统计见results.json；固定种子不保证服务端逐位确定性。']
        with open(out/'结果汇总.md','x') as stream:
            stream.write('\n'.join(lines)+'\n')
        state.update(phase='complete',exit_code=0,completed={'control':600,'variant':600})
    except BaseException as exc:
        state.update(phase='stopped',exit_code=2,error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        for child in children.values():
            stop(child)
        stop(server)
        state.update(finished_at=now(),wall_seconds=time.monotonic()-start,
                     server_returncode=None if server is None else server.returncode)
        save()
        write_private_json(out/'exit.json',state,exclusive=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
