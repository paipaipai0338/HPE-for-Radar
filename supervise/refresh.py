"""Read training evidence and atomically replace the monitor; never launch training."""
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TZ = timezone(timedelta(hours=8))
EPOCH = re.compile(r"^\[([^\]]+)\] Epoch (\d+)/(\d+).*?train=(\{.*?\}) \| val=(\{.*?\})", re.M)


def stamp(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, TZ).strftime('%Y-%m-%d %H:%M:%S +08:00')


def atomic_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def command(arguments):
    return subprocess.check_output(arguments, text=True, stderr=subprocess.PIPE, timeout=20).strip()


def processes(output):
    result = {}
    for line in output.splitlines():
        user, pid, parent, elapsed, args = line.split(None, 4)
        result[int(pid)] = dict(user=user, parent=int(parent), elapsed=int(elapsed), args=args)
    return result


def metric(text, key):
    match = re.search(r"['\"]" + re.escape(key) + r"['\"]:\s*([^,}]+)", text)
    return float(match[1]) if match else None


def epoch_rows(text):
    rows = {}
    for match in EPOCH.finditer(text):
        date, epoch, total, train, val = match.groups()
        rows[int(epoch)] = dict(epoch=int(epoch), total=int(total),
                                time=datetime.strptime(date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ).timestamp(),
                                train=metric(train, 'positive_mpjpe'), val=metric(val, 'positive_mpjpe'),
                                loss=metric(val, 'loss'))
    return [rows[epoch] for epoch in sorted(rows)]


def task_status(state, task, procs, queue=None):
    status = state.get('status')
    if status in ('complete', 'completed', 'failed'):
        return ('已失败' if status == 'failed' else '已完成'), None
    if status == 'queued' and queue is not None:
        if queue.get('status') == 'failed':
            return '队列已停止（尚未启动）', None
        if queue.get('status') in ('waiting', 'running', 'paused'):
            parent_status, pid = task_status(dict(queue, status='waiting'),
                                            {'process_match': task['queue_match']}, procs)
            if parent_status == '排队中':
                return '排队中', pid
        return '状态待核实（队列进程未匹配）', None
    keys = ('waiter_pid', 'pid') if status == 'waiting' else ('worker_pid', 'train_pid', 'pid', 'waiter_pid')
    for key in keys:
        pid = state.get(key)
        if pid in procs and task['process_match'] in procs[pid]['args']:
            label = {'waiting': '排队中', 'paused': '已暂停'}.get(status, '运行中')
            return label, pid
    return '状态待核实（进程未匹配）', None


def link(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return f'[{path.name}](<{path}>)'


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def cm(value):
    return '未记录' if value is None else f'{value * 100:.3f} cm'


def tail(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 16000))
        return stream.read().decode('utf-8', errors='replace')


def render(tasks, procs, gpu_rows, gpu_apps, warnings, errors, now):
    running, queued, finished, details = [], [], [], []
    registered = set()
    for task in tasks:
        try:
            state = json.loads((ROOT / task['state']).read_text())
            queue = (json.loads((ROOT / task['queue_state']).read_text())
                     if state.get('status') == 'queued' and task.get('queue_state') else None)
            status, pid = task_status(state, task, procs, queue)
            registered.update(p for key in ('pid', 'worker_pid', 'train_pid', 'waiter_pid')
                              if (p := state.get(key)) in procs and task['process_match'] in procs[p]['args'])
            experiment_path = state.get('experiment') or task.get('experiment')
            experiment = ROOT / experiment_path if experiment_path else None
            log = experiment / 'log/log.txt' if experiment else None
            text = log.read_text() if log and log.exists() else ''
            if task.get('log_after'):
                text = text.partition(task['log_after'])[2]
            if task.get('log_before'):
                text = text.partition(task['log_before'])[0]
            rows = epoch_rows(text)
            latest = rows[-1] if rows else None
            completed = latest['epoch'] if latest else state.get('completed_epochs', 0)
            total = state.get('target_epochs', latest['total'] if latest else task['epochs'])
            stage = state.get('stage', queue.get('stage', 'queued') if queue else '未记录')
            val = latest['val'] if latest else None
            if val is None and 'val_mpjpe_cm' in state:
                val = state['val_mpjpe_cm'] / 100
            if val is None and 'val_mpjpe_mm' in state:
                val = state['val_mpjpe_mm'] / 1000
            estimate = None
            recent = rows[-11:]
            intervals = [b['time'] - a['time'] for a, b in zip(recent, recent[1:])
                         if b['epoch'] == a['epoch'] + 1 and b['time'] > a['time']]
            estimate_basis = '最近完整 epoch 耗时中位数'
            if len(rows) == 1 and pid:
                # shortcut: 单条日志的耗时包含进程初始化，出现相邻 epoch 后改用轮间耗时。
                duration = latest['time'] - (now - procs[pid]['elapsed'])
                if duration > 0:
                    intervals = [duration]
                    estimate_basis = '单条 epoch 日志与进程启动时间之差，包含初始化'
            if status == '运行中' and stage in ('training', 'validation') and intervals and completed < total:
                estimate = latest['time'] + statistics.median(intervals) * (total - completed)
            summary = f"| {cell(task['name'])} | {status} | {task['gpu']} | {completed}/{total} | {cm(val)} |"
            if status == '运行中':
                running.append(summary + f" {stamp(estimate) if estimate is not None else '暂无法估算'} |")
            elif status in ('排队中', '已暂停'):
                condition = state.get('pause_reason') or task.get('condition', state.get('idle_rule', '见状态文件'))
                queued.append(f"| {cell(task['name'])} | {status} | {task['gpu']} | {cell(condition)} |")
            else:
                finished.append(summary)
            if status.startswith('状态待核实'):
                warnings.append(f"{task['name']}：状态文件为 {state.get('status')}，但宿主机未找到对应进程；不能判为完成。")
            if status.startswith('队列已停止'):
                warnings.append(f"{task['name']}：所属队列已失败，本模型尚未启动；请检查队列状态文件。")
            if status == '已失败':
                warnings.append(f"{task['name']}：{state.get('error', '退出码 ' + str(state.get('exit_code', '未记录')))}")
            if task.get('next_step') and state.get('status') not in ('complete', 'completed'):
                next_status = '执行中' if stage == 'comparison' and pid else ('等待本任务恢复' if not pid else '等待训练完成')
                queued.insert(0, f"| {cell(task['name'])} 后续评估 | {next_status} | {task['gpu']} | {cell(task['next_step'])} |")
            details += [f"### {task['name']}", '', task['purpose'], '',
                        f"- 状态：{status}；阶段：{stage}；已完成 {completed}/{total} epoch。",
                        f"- 配置：{link(task['config'])}；状态依据：{link(task['state'])}。"]
            if pid:
                proc = procs[pid]
                label = '队列等待进程 PID' if state.get('status') == 'queued' else 'PID'
                details.append(f"- {label}：{pid}；用户：{proc['user']}；进程开始：{stamp(now - proc['elapsed'])}。")
            if queue is not None:
                details.append(f"- 所属队列：{link(task['queue_state'])}；启动条件：{task.get('condition', '见队列状态文件')}")
            if latest:
                details.append(f"- 最近完整 epoch：{latest['epoch']}（{stamp(latest['time'])}）；训练 MPJPE：{cm(latest['train'])}；验证 MPJPE：{cm(val)}；验证 loss：{latest['loss']}。")
                finite = [r for r in rows if r['val'] is not None and math.isfinite(r['val'])]
                if finite:
                    best = min(finite, key=lambda row: row['val'])
                    details.append(f"- 已记录的最佳验证 MPJPE：{cm(best['val'])}（epoch {best['epoch']}）。")
                if estimate is not None:
                    details.append(f"- 预计训练结束：{stamp(estimate)}（{estimate_basis}，仅供参考，不含后续评估）。")
                if any(v is not None and not math.isfinite(v) for row in rows for v in (row['train'], row['val'], row['loss'])):
                    warnings.append(f"{task['name']}：训练日志中存在 NaN / Inf 指标。")
            evidence = [ROOT / task['state']]
            if queue is not None:
                evidence.append(ROOT / task['queue_state'])
            for candidate in (log, ROOT / task['console'], ROOT / task.get('waiting_log', task['console'])):
                if candidate and candidate.is_file():
                    evidence.append(candidate)
                    if candidate == log:
                        details.append(f"- 训练日志：{link(candidate)}。")
            console = ROOT / task['console']
            details.append(f"- 控制台：{link(console)}。" if console.exists() else '- 控制台：尚未生成。')
            if status == '排队中' and task.get('waiting_log'):
                details.append(f"- 等待日志：{link(task['waiting_log'])}。")
            activity = max(path.stat().st_mtime for path in evidence)
            details.append(f"- 最近证据更新时间：{stamp(activity)}。")
            if status == '运行中' and pid and now - activity > 1800:
                warnings.append(f"{task['name']}：进程仍在，但状态 / 日志超过 30 分钟未更新，需检查。")
            if console.exists() and status == '运行中' and re.search(r'Traceback \(most recent call last\)|OutOfMemoryError|CUDA out of memory', tail(console)):
                warnings.append(f"{task['name']}：控制台末尾有异常记录，请核查。")
            if experiment:
                details.append(f"- 实验目录：{link(experiment)}。")
                for checkpoint in ('last.pth', 'best.pth'):
                    path = experiment / 'checkpoint' / checkpoint
                    if path.exists():
                        details.append(f"- {checkpoint}：{link(path)}（更新于 {stamp(path.stat().st_mtime)}）。")
            details.append('')
        except (OSError, ValueError, KeyError, TypeError) as error:
            message = f"{task.get('name', '未命名任务')}：采集失败：{error}"
            errors.append(message)
            warnings.append(message)

    lines = ['# 训练监控', '', f'最后刷新：{stamp(now)}；刷新间隔：10 分钟；采集状态：' + ('部分失败。' if errors else '成功。') + f'任务 / 采集提示：{len(warnings)} 项，见下文。', '',
             f'下次计划刷新：{stamp((int(now) // 600 + 1) * 600)}。超过 20 分钟未更新请检查 {link(HERE / "health.json")} 和 {link(HERE / "monitor.log")}。', '',
             '范围：tasks.json 登记的项目任务；GPU 表列出全机计算占用。MPJPE 统一为 cm；不同骨架与数据划分的指标不宜直接横比。', '',
             '## 当前任务', '', '| 任务 | 状态 | GPU | 已完成 epoch | 最近验证 MPJPE | 预计结束时间 |', '| --- | --- | --- | --- | --- | --- |',
             *(running or ['| 暂无已确认运行的登记任务 | — | — | — | — | — |']), '',
             '预计结束时间为北京时间的训练结束估计，不含后续评估；最多取最近 11 条完整 epoch 日志，有多少用多少。只有一条时按进程启动至该轮完成的耗时估算（含初始化）；无有效耗时数据或正在评估时显示“暂无法估算”。', '',
             '## 排队与后续任务', '', '| 任务 | 状态 | GPU | 启动条件 / 内容 |', '| --- | --- | --- | --- |',
             *(queued or ['| 暂无已登记排队任务 | — | — | — |']), '',
             '顺序来自现有任务和启动条件；本监控不负责启动、重试或调整训练。', '',
             '## GPU 占用', '', '| GPU | 型号 | 利用率 | 显存 MiB（已用 / 总计） |', '| --- | --- | --- | --- |']
    gpu_ids = {}
    for row in gpu_rows:
        index, uuid, name, utilization, used, total = (part.strip() for part in row)
        gpu_ids[uuid] = index
        lines.append(f'| {index} | {name} | {utilization}% | {used} / {total} |')
    lines += ['', '| GPU | PID | 用户 | 归属 | 命令 |', '| --- | --- | --- | --- | --- |']
    for pid, uuid, memory in gpu_apps:
        pid = int(pid.strip())
        proc = procs.get(pid, {})
        owner = '已登记任务' if pid in registered else '未登记占用（不推断实验配置）'
        lines.append(f"| {gpu_ids.get(uuid.strip(), uuid.strip())} | {pid} | {proc.get('user', '未知')} | {owner} | {cell(proc.get('args', '进程已退出或不可读'))} |")
    lines += ['', '## 异常与待处理', '', *(['- ' + cell(warning) for warning in warnings] or ['- 本次未发现已登记任务异常。']), '',
              '## 已完成、失败或待核实的登记任务', '', '| 任务 | 状态 | GPU | 已完成 epoch | 最近验证 MPJPE |', '| --- | --- | --- | --- | --- |',
              *(finished or ['| 无 | — | — | — | — |']), '', '## 任务详情与证据', '', *details]
    return '\n'.join(lines) + '\n'


def main():
    with (HERE / 'refresh.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            if b'codex-linux-sandbox' in Path('/proc/1/cmdline').read_bytes():
                raise RuntimeError('请在宿主机执行；沙箱看不到真实训练进程和 GPU，已保留原文档。')
            tasks = json.loads((HERE / 'tasks.json').read_text())
            if not isinstance(tasks, list) or len({task['id'] for task in tasks}) != len(tasks):
                raise ValueError('tasks.json 必须是任务列表，且 id 唯一')
            procs = processes(command(['ps', '-eo', 'user=,pid=,ppid=,etimes=,args=']))
            warnings, errors = [], []
            gpu_rows, gpu_apps = [], []
            try:
                gpu_rows = list(csv.reader(command(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']).splitlines()))
                gpu_apps = list(csv.reader(command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits']).splitlines()))
            except (OSError, subprocess.SubprocessError) as error:
                message = f'GPU 采集失败：{error}；GPU 占用无法确认。'
                errors.append(message)
                warnings.append(message)
            now = time.time()
            document = render(tasks, procs, gpu_rows, gpu_apps, warnings, errors, now)
            atomic_write(HERE / 'training_monitor.md', document)
            atomic_write(HERE / 'health.json', json.dumps(dict(updated_at=stamp(), status='partial' if errors else 'ok', warnings=warnings), ensure_ascii=False, indent=2) + '\n')
            print(f'{stamp()} refreshed; warnings={len(warnings)}', flush=True)
        except Exception as error:
            atomic_write(HERE / 'health.json', json.dumps(dict(updated_at=stamp(), status='failed', error=str(error)), ensure_ascii=False, indent=2) + '\n')
            raise


if __name__ == '__main__':
    main()
