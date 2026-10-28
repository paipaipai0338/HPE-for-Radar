import unittest

import torch

from data2datasets.dataset_for_all_task import HPE_Dataset
from preprocess.gtprocess import get_gt_boxes
from run.utils.process_one_epoch import adapt_coordinate_frame, resample_cropped_pointcloud


class PointcloudProcessingTest(unittest.TestCase):
    def devices(self):
        return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

    def test_sampling_preserves_valid_points_repeats_and_empty_frames(self):
        counts = [0, 1, 17, 199, 200, 201, 413]
        points = torch.arange(2 * len(counts) * 420 * 6).reshape(2, len(counts), 420, 6).float()
        mask = torch.zeros(2, len(counts), 420, dtype=torch.bool)
        for frame, count in enumerate(counts):
            mask[:, frame, torch.randperm(420)[:count]] = True
        points.masked_fill_(~mask[..., None], float('nan'))
        for device in self.devices():
            for random in (False, True):
                with self.subTest(device=device, random=random):
                    output, valid = resample_cropped_pointcloud(points.to(device), mask.to(device), 200, random)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(tuple(output.shape), (2, len(counts), 200, 6))
                    self.assertEqual(valid[:, 0].count_nonzero().item(), 0)
                    self.assertEqual(output[:, 0].count_nonzero().item(), 0)
                    self.assertTrue(valid[:, 1:].all())
                    for instance in range(2):
                        for frame, count in enumerate(counts[1:], 1):
                            original = points[instance, frame, mask[instance, frame]]
                            actual = output[instance, frame].cpu()
                            if random:
                                selected = actual[:, 0].unique()
                                self.assertTrue(torch.isin(selected, original[:, 0]).all())
                                if count <= 200:
                                    torch.testing.assert_close(selected.sort().values, original[:, 0].sort().values)
                                else:
                                    self.assertEqual(len(selected), 200)
                            else:
                                indices = (torch.linspace(0, count - 1, 200, device=device).long()
                                           if count >= 200 else torch.arange(200, device=device) % count)
                                torch.testing.assert_close(actual, original[indices.cpu()])
                    # Dropout updates this mask in place in the shared training pipeline.
                    valid &= torch.ones_like(valid)

    def test_sampling_handles_empty_inputs_and_single_output(self):
        for shape in ((0, 16, 10, 6), (2, 3, 0, 6)):
            points = torch.empty(shape)
            output, mask = resample_cropped_pointcloud(points, torch.zeros(shape[:-1], dtype=torch.bool))
            self.assertEqual(tuple(output.shape), (*shape[:-2], 200, 6))
            self.assertEqual(mask.count_nonzero().item(), 0)
        points = torch.arange(18).reshape(1, 1, 3, 6).float()
        output, _ = resample_cropped_pointcloud(points, torch.ones(1, 1, 3, dtype=torch.bool), 1, False)
        torch.testing.assert_close(output, points[:, :, :1])

    def test_composed_transform_matches_original_order_and_preserves_inputs(self):
        torch.manual_seed(42)
        points = torch.randn(2, 3, 11, 6)
        pose = torch.randn(2, 3, 2, 17, 3)
        point_mask = torch.rand(2, 3, 11) > .2
        person_mask = torch.rand(2, 3, 2) > .2
        rotation = torch.from_numpy(HPE_Dataset._rotation_matrix_from_euler_deg(12., -9., 31.)).expand(2, 3, 3, 3)
        translation = torch.randn(2, 3, 3)
        samples = {
            'radar_high_pc': {'padded': points, 'mask': point_mask},
            'gt_for_high': {'padded': pose, 'mask': person_mask},
            'high_to_low_R': rotation, 'high_to_low_t': translation,
        }
        before_points, before_pose = points.clone(), pose.clone()
        for coordinate in ('low', 'high'):
            for augment in (False, True):
                data = {'augmentation': {
                    'radar_rotation': dict(enabled_train=augment, roll_range_deg=[7., 7.],
                                           pitch_range_deg=[-3., -3.], yaw_range_deg=[5., 5.]),
                    'radar_height': dict(enabled_train=augment, delta_height_range_m=[.1, .1]),
                }}
                matrix = (torch.from_numpy(HPE_Dataset._rotation_matrix_from_euler_deg(7., -3., 5.))
                          if augment else torch.eye(3))
                delta = torch.tensor([0., 0., .1 if augment else 0.])

                def reference(xyz, mask):
                    shape = xyz.shape
                    low = xyz.reshape(2, 3, -1, 3) @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)
                    transformed = (low - delta - translation.unsqueeze(-2)) @ rotation
                    transformed = transformed @ matrix.T
                    if coordinate == 'low':
                        transformed = transformed @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)
                    return transformed.reshape(shape).masked_fill(~mask, 0.)

                expected_points = reference(points[..., :3], point_mask[..., None])
                expected_pose = reference(pose, person_mask[..., None, None])
                task = dict(input='radar_high_pc', output='gt_for_high', coordinate=coordinate)
                for device in self.devices():
                    with self.subTest(coordinate=coordinate, augment=augment, device=device):
                        actual = adapt_coordinate_frame(samples, task, data, 'train', device=device)
                        cloud, target = actual['radar_high_pc'], actual['gt_for_high']
                        torch.testing.assert_close(cloud['padded'][..., :3].cpu(), expected_points, atol=2e-6, rtol=2e-6)
                        torch.testing.assert_close(cloud['padded'][..., 3:].cpu(), points[..., 3:])
                        torch.testing.assert_close(target['padded'].cpu(), expected_pose, atol=2e-6, rtol=2e-6)
                        expected_bbox = get_gt_boxes(expected_pose, person_mask, threshold=.3).masked_fill(~person_mask[..., None], 0.)
                        torch.testing.assert_close(target['bbox'].cpu(), expected_bbox, atol=2e-6, rtol=2e-6)
                        self.assertEqual(cloud['mask'].device.type, device)
                        self.assertEqual(target['mask'].device.type, device)
        torch.testing.assert_close(points, before_points)
        torch.testing.assert_close(pose, before_pose)


if __name__ == '__main__':
    unittest.main()
