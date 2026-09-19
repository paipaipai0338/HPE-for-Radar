"""运行：python -m run.test_init_checkpoint"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn

from run.utils.checkpoint import load_init_checkpoint


def test_init_checkpoint():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    weights = {key: torch.full_like(value, 7) for key, value in original.items()}

    def assert_unchanged():
        assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())

    with TemporaryDirectory() as directory:
        path = Path(directory) / 'weights.pth'
        for payload in (weights, {'model_state_dict': weights, 'epoch': 10}):
            torch.save(payload, path)
            assert load_init_checkpoint(path, model)
            assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
        model.load_state_dict(original)

        for invalid in (
            {key: value for key, value in weights.items() if key != '1.bias'},
            {**weights, 'extra': torch.zeros(1)},
            {**weights, '1.bias': torch.zeros(2)},
            {**weights, '1.bias': weights['1.bias'].double()},
            {**weights, '1.bias': 'invalid'},
            None,
        ):
            torch.save({'model_state_dict': invalid}, path)
            assert not load_init_checkpoint(path, model)
            assert_unchanged()

        path.write_bytes(b'corrupt checkpoint')
        assert not load_init_checkpoint(path, model)
        assert not load_init_checkpoint(Path(directory) / 'missing.pth', model)
        assert_unchanged()

        torch.save(weights, path)
        real_load = model.load_state_dict
        calls = 0

        def fail_after_loading(state, strict=True):
            nonlocal calls
            calls += 1
            real_load(state, strict=strict)
            if calls == 1:
                raise RuntimeError('simulated failure after copying weights')

        # 第一次加载模拟写入后失败，第二次回退使用真实加载。
        with patch.object(model, 'load_state_dict', side_effect=fail_after_loading):
            assert not load_init_checkpoint(path, model)
        assert_unchanged()


if __name__ == '__main__':
    test_init_checkpoint()
    print('init_checkpoint checks passed')
