"""Run the four pending experiments on whichever GPU becomes idle first."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from refresh import HERE, ROOT, atomic_write, stamp

GPU_LOCKS = (ROOT / 'codex/model_compare_xyz50/gpu0_wait.lock', ROOT / 'codex/gpu1_wait.lock')
QUEUE = HERE / 'dispatch.status.json'


def state(path, **values):
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(values, updated_at=stamp())
    atomic_write(path, json.dumps(current, ensure_ascii=False, indent=2) + '\n')


def idle(gpu):
    row = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=15)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'], text=True, timeout=15)
    uuid, memory, utilization = next(csv.reader([row]))
    return uuid.strip() not in processes.split() and int(memory) < 512 and int(utilization) < 5


def reserve(gpu, checks):
    lock = GPU_LOCKS[gpu].open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks[gpu] = checks[gpu] + 1 if idle(gpu) else 0
        if checks[gpu] >= 3:
            return lock
    except BlockingIOError:
        checks[gpu] = 0
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
        checks[gpu] = 0
        print(f'{stamp()} GPU{gpu} query failed: {error}', flush=True)
    lock.close()
    return None


def launch(task, gpu, lock, controller):
    import yaml

    config = ROOT / task['config']
    cfg = yaml.safe_load(config.read_text())
    cfg['task']['device'] = gpu
    atomic_write(config, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    command = [sys.executable, '-B', '-u', *shlex.split(task['process_match'])]
    if config.name == 'config.yaml':
        command.append('worker')
        with (config.parent / 'started.json').open('x') as marker:
            json.dump(dict(started_at=stamp(), gpu=gpu), marker)
    state(ROOT / task['state'], status='running', stage='initializing', gpu=gpu)
    environment = os.environ.copy()
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    with (ROOT / task['console']).open('a') as output:
        return subprocess.Popen(command, cwd=config.parent / 'source', env=environment,
            stdout=output, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(), controller.fileno()))


def pending_tasks():
    tasks = json.loads((HERE / 'tasks.json').read_text())
    pending = []
    for task in tasks:
        if task.get('queue_state') != 'supervise/dispatch.status.json':
            continue
        current = json.loads((ROOT / task['state']).read_text())
        if current['status'] in ('complete', 'completed', 'failed'):
            continue
        if current['status'] != 'queued':
            raise RuntimeError(f"Refusing to relaunch non-queued task: {task['id']}")
        pending.append(task)
    # Keep the old model-comparison order, then the xyzv continuation.
    pending.sort(key=lambda task: (not task['id'].startswith('modelcompare_'), tasks.index(task)))
    for folder in {Path(ROOT / task['config']).parent for task in pending}:
        plan = json.loads((folder / 'plan.json').read_text())
        for relative, expected in plan['source_sha256'].items():
            if hashlib.sha256((folder / 'source' / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen source changed: {folder}/source/{relative}')
    return pending


def main():
    with (HERE / 'dispatch.lock').open('a+') as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            pending = pending_tasks()
            active, checks = {}, {0: 0, 1: 0}
            while pending or active:
                for gpu, (child, lock, task) in list(active.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    result = json.loads((ROOT / task['state']).read_text())
                    if code or result['status'] not in ('complete', 'completed'):
                        state(ROOT / task['state'], status='failed', stage='failed', exit_code=code)
                    print(f"{stamp()} GPU{gpu} finished {task['id']}, exit={code}", flush=True)
                    lock.close()
                    del active[gpu]
                    checks[gpu] = 0
                for gpu in (0, 1):
                    if not pending or gpu in active:
                        continue
                    lock = reserve(gpu, checks)
                    if lock is None:
                        continue
                    task = pending[0]
                    try:
                        child = launch(task, gpu, lock, controller)
                    except BaseException:
                        lock.close()
                        state(ROOT / task['state'], status='failed', stage='launch_failed')
                        raise
                    pending.pop(0)
                    active[gpu] = (child, lock, task)
                    print(f"{stamp()} GPU{gpu} started {task['id']}, pid={child.pid}", flush=True)
                state(QUEUE, status='running' if active else 'waiting', stage='dispatching',
                      waiter_pid=os.getpid(), pending=[task['id'] for task in pending],
                      active={str(gpu): child.pid for gpu, (child, _, _) in active.items()}, idle_checks=checks)
                if pending or active:
                    time.sleep(30)
            state(QUEUE, status='complete', stage='complete')
        except BaseException as error:
            state(QUEUE, status='failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
