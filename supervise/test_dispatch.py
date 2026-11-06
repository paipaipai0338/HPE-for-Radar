import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dispatch


class RetryQueueTests(unittest.TestCase):
    def test_retry_selects_its_queued_tasks_and_checks_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            here = root / 'supervise'
            here.mkdir()
            source = root / 'control/source'
            source.mkdir(parents=True)
            (source / 'model.py').write_text('original')
            (source.parent / 'plan.json').write_text(json.dumps({
                'source_sha256': {'model.py': hashlib.sha256(b'original').hexdigest()}}))
            tasks = []
            for name, queue, state in [('active', 'dispatch', 'running'),
                                        ('modelcompare_retry', 'retry', 'queued'),
                                        ('xyzv_retry', 'retry', 'queued'),
                                        ('old_failure', 'retry', 'failed')]:
                (root / f'{name}.json').write_text(json.dumps({'status': state}))
                tasks.append(dict(id=name, queue_state=f'supervise/{queue}.status.json',
                                  state=f'{name}.json', config='control/config.yaml'))
            (here / 'tasks.json').write_text(json.dumps(tasks))
            with patch.multiple(dispatch, ROOT=root, HERE=here, QUEUE=here / 'retry.status.json'):
                self.assertEqual([t['id'] for t in dispatch.pending_tasks()],
                                 ['modelcompare_retry', 'xyzv_retry'])
                (source / 'model.py').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'Frozen source changed'):
                    dispatch.pending_tasks()

    def test_retry_respects_existing_gpu_lock_and_idle_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.lock'
            checks = {0: 0}
            with patch.object(dispatch, 'GPU_LOCKS', (path,)), patch.object(dispatch, 'idle', return_value=True):
                with path.open('a+') as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.assertIsNone(dispatch.reserve(0, checks))
                    self.assertEqual(checks[0], 0)
                self.assertIsNone(dispatch.reserve(0, checks))
                self.assertIsNone(dispatch.reserve(0, checks))
                lock = dispatch.reserve(0, checks)
                self.assertIsNotNone(lock)
                lock.close()
            with patch.object(dispatch, 'GPU_LOCKS', (path,)), patch.object(dispatch, 'idle', return_value=False):
                self.assertIsNone(dispatch.reserve(0, checks))
                self.assertEqual(checks[0], 0)


if __name__ == '__main__':
    unittest.main()
