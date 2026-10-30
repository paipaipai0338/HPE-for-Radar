"""Watch existing training status files; notify Bark only on completion/failure.

Save https://api.day.app/DEVICE_KEY in .env.bark, then run:
    python -B -u supervise/bark_monitor.py
"""
import argparse
import fcntl
from http.client import HTTPException
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'complete', 'failed'}


def snapshot(root):
    records = {}
    for base in (root / 'experiments', root / 'codex'):
        for directory, folders, files in os.walk(base):
            folders[:] = [name for name in folders if not name.startswith(('.', 'backup'))
                          and name not in {'source', 'config', 'revisions', 'cache',
                                           'checkpoint', 'log', 'fig', '__pycache__'}]
            for name in files:
                if name not in {'status.json', 'job.json'} and not name.endswith('.status.json'):
                    continue
                path = Path(directory) / name
                try:
                    data = json.loads(path.read_text())
                    if isinstance(data, dict) and isinstance(data.get('status'), str):
                        records[str(path.relative_to(root))] = data
                except (OSError, ValueError):
                    # A temporarily incomplete status file is retried on the next poll.
                    continue
    for key, data in records.items():
        if Path(key).name == 'queue.status.json':
            model = data.get('failed_model') or data.get('current_model')
            worker = str(Path(key).with_name(f'{model}.status.json'))
            if data['status'] != 'failed' or records.get(worker, {}).get('status') == 'failed':
                records[key] = dict(data, status='suppressed')
    return records


def message(key, data):
    path = Path(key)
    if path.name == 'queue.status.json':
        model = data.get('failed_model') or data.get('current_model') or path.parent.name
    elif path.name.endswith('.status.json'):
        model = path.name[:-len('.status.json')]
    else:
        model = path.parent.parent.name if path.name == 'status.json' else path.parent.name
    title = f"{'训练完成' if data['status'] == 'complete' else '训练出错'} · {model}"
    lines = [f'任务：{path.parent}']
    if 'completed_epochs' in data:
        lines.append(f"轮次：{data['completed_epochs']}/{data.get('target_epochs', '?')}")
    for field, label in (('val_mpjpe_mm', '验证 MPJPE'), ('phase_best_mpjpe_mm', '阶段最佳 MPJPE')):
        value = data.get(field)
        if isinstance(value, (int, float)):
            lines.append(f'{label}：{value:.2f} mm')
    if data['status'] == 'failed':
        lines.append(f"错误：{str(data.get('error') or '请查看训练日志')[:800]}")
        if 'exit_code' in data:
            lines.append(f"退出码：{data['exit_code']}")
    return title, '\n'.join(lines)


def bark_target(address):
    parsed = urlsplit(address.strip())
    parts = parsed.path.strip('/').split('/')
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or len(parts) != 1 or not parts[0] or parsed.query or parsed.fragment):
        raise ValueError('Bark 地址必须是 https://服务器/设备Key')
    return f'{parsed.scheme}://{parsed.netloc}/push', parts[0]


def notify(target, title, body):
    endpoint, device_key = target
    payload = json.dumps(dict(device_key=device_key, title=title, body=body,
                              group='模型训练'), ensure_ascii=False).encode('utf-8')
    request = Request(endpoint, data=payload, headers={'Content-Type': 'application/json; charset=utf-8'})
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    if result.get('code') != 200:
        raise ValueError('Bark 未确认推送成功')


def save(path, statuses):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(statuses, ensure_ascii=False, indent=2))
    temporary.replace(path)


def poll(root, path, statuses, send):
    for key, data in snapshot(root).items():
        current = data['status']
        if current in TERMINAL and statuses.get(key) != current:
            try:
                send(*message(key, data))
            except (OSError, ValueError, HTTPException) as error:
                # Never log URLs or response bodies, which may contain the device key.
                print(f'Bark 推送失败（{type(error).__name__}），下次重试：{key}', flush=True)
                continue
            statuses[key] = current
            save(path, statuses)
            print(f'已推送 {current}：{key}', flush=True)
        else:
            statuses[key] = current
    save(path, statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='检查一次后退出')
    args = parser.parse_args()
    target = bark_target((ROOT / '.env.bark').read_text())
    directory = ROOT / 'experiments/.bark-monitor'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'state.json'
        if path.exists():
            statuses = json.loads(path.read_text())
            if not isinstance(statuses, dict):
                raise ValueError('Bark 监控状态文件损坏')
        else:
            statuses = {key: data['status'] for key, data in snapshot(ROOT).items()}
            save(path, statuses)
            print(f'已建立基线：{len(statuses)} 个状态文件，不补发历史通知', flush=True)
        print('Bark 监控就绪：仅训练完成/出错通知，每 30 秒检查', flush=True)
        while True:
            poll(ROOT, path, statuses, lambda title, body: notify(target, title, body))
            if args.once:
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
