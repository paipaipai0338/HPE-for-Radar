import json
from datetime import datetime
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import refresh


class MonitorTests(unittest.TestCase):
    def test_epoch_metrics_units_and_resume(self):
        text = """[2026-10-28 10:00:00] Epoch 1/300 | lr=0.001 | train={'positive_mpjpe': 0.2} | val={'positive_mpjpe': 0.15, 'loss': 0.15}
[2026-10-28 10:02:00] Epoch 2/300 | lr=0.001 | train={'positive_mpjpe': 0.1} | val={'positive_mpjpe': 1.4e-1, 'loss': 0.14}
[2026-10-28 10:04:00] Epoch 2/300 | lr=0.001 | train={'positive_mpjpe': 0.1} | val={'positive_mpjpe': 0.13, 'loss': 0.13}
"""
        rows = refresh.epoch_rows(text)
        self.assertEqual([row['epoch'] for row in rows], [1, 2])
        self.assertEqual(rows[-1]['val'], 0.13)
        self.assertEqual(refresh.cm(rows[-1]['val']), '13.000 cm')
        self.assertEqual(refresh.epoch_rows('Epoch 3/300 still training'), [])
        self.assertIsNone(refresh.metric("{'loss': 0.1}", 'positive_mpjpe'))

    def test_live_queue_terminal_and_reused_pid(self):
        procs = refresh.processes('pai 20 1 120 python train.py --worker\npai 21 20 30 python train.py --worker')
        task = {'process_match': 'train.py'}
        self.assertEqual(refresh.task_status({'status': 'running', 'worker_pid': 20}, task, procs), ('运行中', 20))
        self.assertEqual(refresh.task_status({'status': 'waiting', 'waiter_pid': 20}, task, procs), ('排队中', 20))
        self.assertEqual(refresh.task_status({'status': 'paused', 'pid': 20}, task, procs), ('已暂停', 20))
        self.assertEqual(refresh.task_status({'status': 'complete', 'pid': 20}, task, procs), ('已完成', None))
        self.assertEqual(refresh.task_status({'status': 'failed'}, task, procs), ('已失败', None))
        procs[20]['args'] = 'python unrelated.py'
        self.assertIn('待核实', refresh.task_status({'status': 'running', 'pid': 20}, task, procs)[0])

    def test_queued_model_follows_live_parent_queue(self):
        task = {'process_match': 'queue.py --worker PCT', 'queue_match': 'queue.py'}
        state = {'status': 'queued'}
        queue = {'status': 'waiting', 'waiter_pid': 20}
        procs = refresh.processes('pai 20 1 3600 python queue.py')
        self.assertEqual(refresh.task_status(state, task, procs, queue), ('排队中', 20))
        queue['status'] = 'running'
        self.assertEqual(refresh.task_status(state, task, procs, queue), ('排队中', 20))
        queue['status'] = 'paused'
        self.assertEqual(refresh.task_status(state, task, procs, queue), ('排队中', 20))
        self.assertIn('待核实', refresh.task_status(state, task, {}, queue)[0])
        queue['status'] = 'failed'
        self.assertEqual(refresh.task_status(state, task, procs, queue), ('队列已停止（尚未启动）', None))

    def test_dense_resume_filters_old_metrics_and_uses_target_epochs(self):
        with tempfile.TemporaryDirectory(dir=refresh.HERE) as directory:
            root = Path(directory)
            task = dict(id='resume', name='密集滑窗续训', purpose='续训100轮', gpu=0, epochs=150,
                        state='state.json', process_match='queue.py --worker PCT',
                        queue_state='queue.json', queue_match='queue.py', config='config.yaml',
                        console='console.log', condition='等待前序模型完成', log_after='PHASE dense')
            state = dict(status='queued', completed_epochs=50, target_epochs=150, experiment=str(root / 'exp'))
            queue = dict(status='waiting', waiter_pid=20, stage='existing_gpu0_queue')
            (root / 'state.json').write_text(json.dumps(state))
            (root / 'queue.json').write_text(json.dumps(queue))
            log = root / 'exp/log/log.txt'
            log.parent.mkdir(parents=True)
            old = "[2026-10-28 20:00:00] Epoch 50/50 | train={'positive_mpjpe': 0.1} | val={'positive_mpjpe': 0.13, 'loss': 0.13}\n"
            log.write_text(old)
            procs = refresh.processes('pai 20 1 3600 python queue.py')
            with patch.object(refresh, 'ROOT', root):
                warnings, errors = [], []
                document = refresh.render([task], procs, [], [], warnings, errors, time.time() + 3600)
                self.assertIn('| 密集滑窗续训 | 排队中 | 0 | 等待前序模型完成 |', document)
                self.assertIn('50/150', document)
                self.assertNotIn('13.000 cm', document)
                self.assertEqual(warnings, [])
                self.assertEqual(errors, [])
                procs.update(refresh.processes('pai 21 20 60 python queue.py --worker PCT'))
                state.update(status='paused', pid=21, pause_reason='优先微调完成后自动恢复')
                (root / 'state.json').write_text(json.dumps(state))
                document = refresh.render([task], procs, [], [], [], [], time.time() + 3600)
                self.assertIn('| 密集滑窗续训 | 已暂停 | 0 | 优先微调完成后自动恢复 |', document)
                state.update(status='running', pid=21, stage='training')
                (root / 'state.json').write_text(json.dumps(state))
                procs.update(refresh.processes('pai 21 20 60 python queue.py --worker PCT'))
                log.write_text(old + '[2026-10-28 21:00:00] PHASE dense\n'
                               "[2026-10-28 23:30:00] Epoch 51/150 | train={'positive_mpjpe': 0.1} | val={'positive_mpjpe': 0.14, 'loss': 0.14}\n")
                document = refresh.render([task], procs, [], [], [], [], time.time())
                self.assertIn('| 密集滑窗续训 | 运行中 | 0 | 51/150 | 14.000 cm |', document)
                self.assertIn('最佳验证 MPJPE：14.000 cm', document)
                task.pop('log_after')
                task['log_before'] = 'PHASE dense'
                state.update(status='complete', target_epochs=50)
                (root / 'state.json').write_text(json.dumps(state))
                document = refresh.render([task], procs, [], [], [], [], time.time())
                self.assertIn('50/50', document)
                self.assertIn('最佳验证 MPJPE：13.000 cm', document)
                self.assertNotIn('14.000 cm', document)

    def test_snapshot_failure_preserves_previous_and_marks_health(self):
        with tempfile.TemporaryDirectory(dir=refresh.HERE) as directory:
            root = Path(directory)
            (root / 'tasks.json').write_text('invalid json')
            document = root / 'training_monitor.md'
            document.write_text('last valid snapshot')
            with patch.object(refresh, 'HERE', root), patch.object(Path, 'read_bytes', return_value=b'/sbin/init'):
                with self.assertRaises(ValueError):
                    refresh.main()
            self.assertEqual(document.read_text(), 'last valid snapshot')
            self.assertEqual(json.loads((root / 'health.json').read_text())['status'], 'failed')

    def test_current_task_eta_matches_details_and_handles_missing_samples(self):
        with tempfile.TemporaryDirectory(dir=refresh.HERE) as directory:
            root = Path(directory)
            task = dict(id='eta', name='预计时间测试', purpose='检查时间估计', gpu=0, epochs=20,
                        state='state.json', process_match='train.py', config='config.yaml', console='console.log')
            state = dict(status='running', stage='training', pid=20, target_epochs=10, experiment='exp')
            (root / 'state.json').write_text(json.dumps(state))
            log = root / 'exp/log/log.txt'
            log.parent.mkdir(parents=True)
            procs = refresh.processes('pai 20 1 120 python train.py')
            with patch.object(refresh, 'ROOT', root):
                document = refresh.render([task], procs, [], [], [], [], time.time())
                self.assertIn('| 0/10 | 未记录 | 暂无法估算 |', document)
                log.write_text('\n'.join(
                    f"[2026-10-29 10:{minute:02d}:00] Epoch {epoch}/20 | train={{'positive_mpjpe': 0.1}} | val={{'positive_mpjpe': 0.14, 'loss': 0.14}}"
                    for epoch, minute in enumerate((0, 2, 6, 8), start=1)))
                complete_log = log.read_text()
                for count, expected in ((1, '2026-10-29 10:18:00 +08:00'),
                                        (2, '2026-10-29 10:18:00 +08:00'),
                                        (3, '2026-10-29 10:27:00 +08:00')):
                    with self.subTest(epoch_records=count):
                        log.write_text('\n'.join(complete_log.splitlines()[:count]))
                        now = datetime(2026, 10, 29, 10, 0, tzinfo=refresh.TZ).timestamp()
                        document = refresh.render([task], procs, [], [], [], [], now)
                        self.assertIn(f'| {count}/10 | 14.000 cm | {expected} |', document)
                        self.assertIn(f'预计训练结束：{expected}', document)
                log.write_text(complete_log.splitlines()[0])
                procs[20]['elapsed'] = 0
                document = refresh.render([task], procs, [], [], [], [], now)
                self.assertIn('| 1/10 | 14.000 cm | 暂无法估算 |', document)
                procs[20]['elapsed'] = 120
                log.write_text(complete_log)
                document = refresh.render([task], procs, [], [], [], [], time.time())
                expected = '2026-10-29 10:20:00 +08:00'
                self.assertIn(f'| 4/10 | 14.000 cm | {expected} |', document)
                self.assertIn(f'预计训练结束：{expected}', document)
                state['stage'] = 'comparison'
                (root / 'state.json').write_text(json.dumps(state))
                document = refresh.render([task], procs, [], [], [], [], time.time())
                self.assertIn('| 4/10 | 14.000 cm | 暂无法估算 |', document)
                self.assertNotIn('预计训练结束：', document)
                (root / 'state.json').write_text(json.dumps(dict(state, status='complete')))
                document = refresh.render([task], procs, [], [], [], [], time.time())
                self.assertIn('| 暂无已确认运行的登记任务 | — | — | — | — | — |', document)

    def test_waiting_predecessor_and_registered_experiment(self):
        with tempfile.TemporaryDirectory(dir=refresh.HERE) as directory:
            root = Path(directory)
            task = dict(id='resume', name='续训50轮', purpose='补齐步长1对照', gpu=1, epochs=350,
                        state='state.json', process_match='train.py', experiment='experiment',
                        config='config.yaml', console='console.log', condition='等待前序任务完成')
            state = dict(status='waiting', stage='waiting_predecessor', waiter_pid=20, completed_epochs=300)
            (root / 'state.json').write_text(json.dumps(state))
            log = root / 'experiment/log/log.txt'
            log.parent.mkdir(parents=True)
            log.write_text('')
            procs = refresh.processes('pai 20 1 3600 python train.py --wait')
            warnings, errors = [], []
            with patch.object(refresh, 'ROOT', root):
                document = refresh.render([task], procs, [], [], warnings, errors, time.time() + 3600)
                self.assertIn('| 续训50轮 | 排队中 | 1 | 等待前序任务完成 |', document)
                self.assertIn('300/350', document)
                self.assertIn(str(log), document)
                self.assertEqual(warnings, [])
                self.assertEqual(errors, [])
                state.update(status='running', stage='training', worker_pid=20)
                (root / 'state.json').write_text(json.dumps(state))
                log.write_text("[2026-10-28 23:30:00] Epoch 301/350 | train={'positive_mpjpe': 0.1} | val={'positive_mpjpe': 0.14, 'loss': 0.14}\n")
                document = refresh.render([task], procs, [], [], warnings, errors, time.time() + 3600)
                self.assertIn('| 续训50轮 | 运行中 | 1 | 301/350 | 14.000 cm |', document)
                self.assertTrue(any('超过 30 分钟' in warning for warning in warnings))

    def test_snapshot_and_transition_into_comparison(self):
        with tempfile.TemporaryDirectory(dir=refresh.HERE) as directory:
            root = Path(directory)
            task = dict(id='test', name='实验', purpose='比较精度', gpu=0, epochs=3, state='state.json',
                        process_match='train.py', config='config.yaml', console='console.log', next_step='完成训练后评估')
            (root / 'tasks.json').write_text(json.dumps([task]))
            state = dict(status='running', stage='comparison', worker_pid=20, completed_epochs=3)
            (root / 'state.json').write_text(json.dumps(state))
            def query(args):
                if args[0] == 'ps':
                    return 'pai 20 1 120 python train.py\norange 30 1 60 python other.py'
                if 'compute-apps' in args[1]:
                    return '20, GPU-test, 100\n30, GPU-test, 200'
                return '0, GPU-test, RTX, 90, 300, 1000'
            with patch.object(refresh, 'HERE', root), patch.object(refresh, 'ROOT', root), \
                    patch.object(refresh, 'command', side_effect=query), patch.object(Path, 'read_bytes', return_value=b'/sbin/init'):
                refresh.main()
                document = (root / 'training_monitor.md').read_text()
                self.assertIn('执行中', document)
                self.assertIn('3/3', document)
                self.assertIn('orange', document)
                self.assertIn('未登记占用', document)
                self.assertEqual(json.loads((root / 'health.json').read_text())['status'], 'ok')
                state.update(status='failed', error='OutOfMemoryError')
                (root / 'state.json').write_text(json.dumps(state))
                refresh.main()
                self.assertIn('OutOfMemoryError', (root / 'training_monitor.md').read_text())
                self.assertEqual(json.loads((root / 'health.json').read_text())['status'], 'ok')
                (root / 'state.json').write_text('{')
                refresh.main()
                self.assertEqual(json.loads((root / 'health.json').read_text())['status'], 'partial')
                self.assertIn('采集失败', (root / 'training_monitor.md').read_text())


if __name__ == '__main__':
    unittest.main()
