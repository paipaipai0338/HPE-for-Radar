import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import bark_monitor as monitor


class BarkMonitorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = self.root / 'experiments/model/queue/status.json'
        self.file.parent.mkdir(parents=True)
        self.state = self.root / 'state.json'

    def write(self, status, path=None, **values):
        path = path or self.file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status=status, **values)))

    def test_terminal_only_no_history_and_no_duplicates_after_restart(self):
        self.write('complete')
        statuses = {key: data['status'] for key, data in monitor.snapshot(self.root).items()}
        send = MagicMock()
        monitor.poll(self.root, self.state, statuses, send)
        send.assert_not_called()
        for status in ('queued', 'running', 'paused', 'cancelled', 'running'):
            self.write(status)
            monitor.poll(self.root, self.state, statuses, send)
        send.assert_not_called()
        self.write('complete', completed_epochs=350, target_epochs=350, val_mpjpe_mm=123.456)
        monitor.poll(self.root, self.state, statuses, send)
        self.assertEqual(send.call_count, 1)
        self.assertIn('训练完成', send.call_args.args[0])
        self.assertIn('350/350', send.call_args.args[1])
        self.assertIn('123.46 mm', send.call_args.args[1])
        monitor.poll(self.root, self.state, json.loads(self.state.read_text()), send)
        self.assertEqual(send.call_count, 1)
        self.write('running')
        monitor.poll(self.root, self.state, statuses, send)
        self.write('failed', error='CUDA out of memory')
        monitor.poll(self.root, self.state, statuses, send)
        self.assertEqual(send.call_count, 2)
        self.assertIn('CUDA out of memory', send.call_args.args[1])

    def test_network_failure_is_retried_without_marking_sent(self):
        self.write('failed', error='bad input')
        statuses = {}
        send = MagicMock(side_effect=OSError('offline'))
        monitor.poll(self.root, self.state, statuses, send)
        self.assertEqual(statuses, {})
        send.side_effect = None
        monitor.poll(self.root, self.state, statuses, send)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(next(iter(statuses.values())), 'failed')

    def test_ignore_archives_and_partial_status_files(self):
        self.write('running')
        for folder in ('source', 'config', 'revisions', 'backup_before_today', 'cache'):
            self.write('failed', self.file.parent / folder / 'job.json')
        (self.file.parent / 'broken.status.json').write_text('{')
        self.assertEqual(len(monitor.snapshot(self.root)), 1)

    def test_queue_failure_fallback_without_duplicate_model_alert(self):
        queue = self.root / 'experiments/compare/queue.status.json'
        worker = queue.with_name('PointNet.status.json')
        self.write('running', worker)
        self.write('failed', queue, failed_model='PointNet', exit_code=-9)
        send = MagicMock()
        monitor.poll(self.root, self.state, {}, send)
        self.assertEqual(send.call_count, 1)
        self.assertIn('PointNet', send.call_args.args[0])
        self.assertIn('-9', send.call_args.args[1])
        self.write('failed', worker, error='bad input')
        send.reset_mock()
        monitor.poll(self.root, self.state, {}, send)
        self.assertEqual(send.call_count, 1)
        self.write('complete', worker)
        self.write('complete', queue)
        send.reset_mock()
        monitor.poll(self.root, self.state, {}, send)
        self.assertEqual(send.call_count, 1)

    def test_bark_https_json_api_checks_response(self):
        target = monitor.bark_target('https://api.day.app/example-key/')
        self.assertEqual(target, ('https://api.day.app/push', 'example-key'))
        for value in ('http://api.day.app/key', 'https://api.day.app/',
                      'https://api.day.app/key/title/body', 'https://api.day.app/key?x=y',
                      'https://user:password@api.day.app/key'):
            with self.assertRaises(ValueError):
                monitor.bark_target(value)
        with patch.object(monitor, 'urlopen') as open_url:
            response = open_url.return_value.__enter__.return_value
            response.read.return_value = b'{"code":200}'
            monitor.notify(target, '训练完成', 'PointNet')
            request = open_url.call_args.args[0]
            self.assertEqual(request.full_url, target[0])
            self.assertEqual(json.loads(request.data)['device_key'], 'example-key')
            response.read.return_value = b'{"code":400}'
            with self.assertRaises(ValueError):
                monitor.notify(target, '训练出错', 'PointNet')


if __name__ == '__main__':
    unittest.main()
