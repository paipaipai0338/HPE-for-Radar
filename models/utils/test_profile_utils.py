import contextlib
import io
import itertools
import unittest

import torch
from torch import nn

from models.utils.profile_utils import _profile_macs, profile_model


class ProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_attention_macs(self):
        model = nn.MultiheadAttention(8, 2, batch_first=True)
        q, k, v = torch.randn(2, 3, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        self.assertEqual(_profile_macs(model, (q, k, v)), 2176)
        self.assertTrue(model.training)
        self.assertFalse(any('total_ops' in m._buffers for m in model.modules()))

    def test_profile_parameters_and_modes(self):
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)
                self.extra = nn.Parameter(torch.ones(7))

            def forward(self, x):
                return self.linear(x.sort(dim=0).values) + self.extra.sum()

        model = SmallModel().train()
        model.linear.eval()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            profile_model('small', model, torch.ones(4, 3), warmup=0, repeat=1)
        text = output.getvalue()
        self.assertIn('Params = 0.000015 M', text)
        self.assertNotIn('Data operations:', text)
        self.assertNotIn('aten::', text)
        self.assertIn('ms/batch', text)
        self.assertTrue(model.training)
        self.assertFalse(model.linear.training)

    def test_failure_cleans_up_thop(self):
        class Broken(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                self.linear(x)
                raise ValueError('expected failure')

        model = Broken()
        hook = model.linear.register_forward_hook(lambda *args: None)
        original_hooks = dict(model.linear._forward_hooks)
        with self.assertRaisesRegex(ValueError, 'expected failure'):
            _profile_macs(model, (torch.ones(1, 2),))
        self.assertEqual(model.linear._forward_hooks, original_hooks)
        self.assertTrue(model.training)
        self.assertFalse(any('total_ops' in m._buffers for m in model.modules()))
        hook.remove()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA/spconv required')
    def test_sparse_active_pairs(self):
        from models.VoxelNeXt.VoxelNeXt_helper import spconv, SPCONV_ALGO
        device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
        coordinates = torch.tensor([[0, *p] for p in itertools.product(range(5), repeat=3)],
                                   dtype=torch.int32, device=device)
        x = spconv.SparseConvTensor(torch.randn(125, 2, device=device), coordinates, [5, 5, 5], 1)
        for cls, kernel, stride, expected in (
                (spconv.SubMConv3d, 3, 1, 13 ** 3 * 2 * 3),
                (spconv.SparseConv3d, 3, 2, 7 ** 3 * 2 * 3),
                (spconv.SubMConv3d, 1, 1, 125 * 2 * 3)):
            layer = cls(2, 3, kernel, stride=stride, padding=kernel // 2,
                        bias=False, algo=SPCONV_ALGO).to(device)
            self.assertEqual(_profile_macs(layer, (x,)), expected)
            self.assertIsNone(layer.indice_key)
            self.assertFalse(layer._forward_hooks)
        isolated = spconv.SparseConvTensor(x.features[:1], coordinates[:1], [5, 5, 5], 1)
        layer = spconv.SubMConv3d(2, 3, 3, padding=1, bias=False, algo=SPCONV_ALGO).to(device)
        self.assertEqual(_profile_macs(layer, (isolated,)), 6)


if __name__ == '__main__':
    unittest.main()
