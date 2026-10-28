import unittest

import torch

from run.utils.build_metric import Metric


class GroupMetricTest(unittest.TestCase):
    def test_group_statistics_keep_masks_regions_counts_and_accumulation(self):
        errors = torch.arange(12).reshape(3, 2, 2).float() / 4
        centered = errors / 2
        mask = torch.tensor([[[True, True], [True, False]],
                             [[True, False], [True, True]],
                             [[False, False], [False, False]]])
        roots = torch.tensor([[[[2., 0.], [5., 0.]], [[2., 2.4], [2., 0.]]],
                              [[[2., 3.], [2., 0.]], [[2., 0.], [5., 0.]]],
                              [[[2., 0.], [2., 0.]], [[2., 0.], [2., 0.]]]])
        gt = torch.zeros(3, 2, 2, 17, 3)
        gt[..., 11, :2] = roots
        gt[..., 12, :2] = roots
        inside = (((roots - torch.tensor([2., 0.])) / torch.tensor([1.6, 2.4])).square().sum(-1) <= 1) & mask
        expected = [errors[mask].sum().item(), mask.sum().item(),
                    errors[inside].sum().item(), inside.sum().item(),
                    centered[mask].sum().item(), centered[inside].sum().item()]
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            with self.subTest(device=device):
                metric = Metric({'Positive_MPJPE': 1}, None, 5., 2.)
                for _ in range(2):
                    metric._record_group_mpjpe(errors.to(device), mask.to(device),
                                               ['repeat', 'repeat', 'empty'], gt, centered.to(device))
                self.assertEqual(metric.group_mpjpe, {'repeat': [value * 2 for value in expected]})
                self.assertIsInstance(metric.group_mpjpe['repeat'][1], int)
                self.assertIsInstance(metric.group_mpjpe['repeat'][3], int)


if __name__ == '__main__':
    unittest.main()
