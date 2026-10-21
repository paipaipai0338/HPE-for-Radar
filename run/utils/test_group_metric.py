import unittest

import torch

from run.utils.build_metric import Metric


class GroupMetricTest(unittest.TestCase):
    def test_all_and_inside_use_valid_people_and_person_weighting(self):
        metric = Metric({'positive_mpjpe': 1}, None, 5, 2)
        low = torch.zeros(1, 1, 4, 17, 3)
        low[..., 0] = 2
        low[:, :, 1, :, 1] = 2.4  # 椭圆边界也计入。
        low[:, :, 2, :, 0] = 4
        errors = torch.tensor([[[1., 3., 8., 100.]]])
        mask = torch.tensor([[[True, True, True, False]]])
        metric._record_group_mpjpe(errors, mask, ['test'], low)
        metric._record_group_mpjpe(errors[:, :, :1], mask[:, :, :1], ['test'], low[:, :, :1])
        metric._record_group_mpjpe(errors[:, :, 2:3], mask[:, :, 2:3], ['outside'], low[:, :, 2:3])
        groups = metric.epoch_end()['mpjpe_by_group']
        self.assertEqual(groups['test']['mpjpe'], 13 / 4)
        self.assertEqual(groups['test']['mpjpe_inside'], 5 / 3)
        self.assertEqual(groups['test']['count_inside'], 3)
        self.assertIsNone(groups['outside']['mpjpe_inside'])
        self.assertEqual(metric.group_mpjpe, {})

    def test_matching_and_pose_valid_preserve_gt_region(self):
        metric = Metric({'positive_mpjpe': 1}, None, 5, 2, pose_matching_by_hip=True)
        low = torch.zeros(1, 1, 2, 17, 3)
        low[:, :, 0, :, 0] = 2
        low[:, :, 1, :, 0] = 4
        pred = low.flip(2).clone()
        pred[..., 2] += 0.25
        gt = {'padded': low, 'mask': torch.ones(1, 1, 2, dtype=torch.bool)}
        pre = {'pose': pred, 'pose_valid': torch.tensor([[[False, True]]])}
        metric.calculate_batch(pre, gt, ['test'], group_gt_low=low)
        group = metric.epoch_end()['mpjpe_by_group']['test']
        self.assertEqual(group['count'], 1)
        self.assertEqual(group['count_inside'], 1)
        self.assertAlmostEqual(group['mpjpe_inside'], 0.25)


if __name__ == '__main__':
    unittest.main()
