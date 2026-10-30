import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
import numpy as np

from data2datasets.dataset_for_all_task import HPE_Dataset, PackedFrame, collate_fn

from run.utils.process_one_epoch import (
    jitter_bbox_mask,
    prepare_person_pointcloud,
    validate_coordinate_frame,
    adapt_coordinate_frame,
)


class BboxJitterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.config = dict(probability=1., translation_max_m=[0.] * 3,
                           scale_range=[1., 1.], rotation_max_deg=[0.] * 3,
                           min_point_iou=.7, max_attempts=1)
        self.task = dict(input='radar_high_pc', output='gt_for_high', coordinate='high')

    def fixture(self, cloud=None):
        if cloud is None:
            cloud = torch.zeros(11, 3)
            cloud[:, 0] = torch.arange(11) / 4 - 1.25
            cloud = cloud.repeat_interleave(10, dim=0)
        offset = torch.tensor([[2., 0., 0.], [2.25, .25, 0.], [2.5, .5, .25]])
        xyz = (cloud[None] + offset[:, None])[None, None]
        bounds = torch.tensor([-1., -.5, -.5, 1., .5, .5])[None, None, None]
        bounds = bounds + offset.repeat(1, 2)[None, None]
        valid = torch.ones(xyz.shape[:-1], dtype=torch.bool)
        original = ((xyz >= bounds[..., :3].unsqueeze(-2)) &
                    (xyz <= bounds[..., 3:].unsqueeze(-2))).all(-1)
        return xyz, bounds, valid, original

    def perturb(self, fixture, values, **changes):
        config = {**self.config, **changes}
        with patch('run.utils.process_one_epoch.torch.rand', side_effect=[
                torch.zeros(fixture[3].shape[:-1]), torch.tensor([values]).expand(int((fixture[3].sum(-1) >= 10).sum()), -1)]):
            return jitter_bbox_mask(*fixture, config)

    def test_translation_and_axis_scaling(self):
        fixture = self.fixture()
        before = [value.clone() for value in fixture]
        translated = self.perturb(fixture, [1.] + [.5] * 8, translation_max_m=[.25, 0., 0.])
        expected = fixture[3].clone()
        expected[..., 10:20] = False
        expected[..., 100:110] = True
        torch.testing.assert_close(translated, expected)
        shrunk = self.perturb(fixture, [.5] * 9, scale_range=[.75, .75])
        expected = torch.zeros_like(fixture[3])
        expected[..., 20:90] = True
        torch.testing.assert_close(shrunk, expected)
        expanded = self.perturb(fixture, [.5] * 9, scale_range=[1.25, 1.25])
        self.assertTrue(expanded.all())
        for original, current in zip(before, fixture):
            torch.testing.assert_close(original, current, rtol=0, atol=0)

    def test_rotation_uses_oriented_box_instead_of_enclosing_box(self):
        cloud = torch.tensor([[0., 0., 0.], [.8, .4, 0.], [.8, -.4, 0.],
                              [0., .6, 0.], [.9, .9, 0.]])
        fixture = self.fixture(cloud.repeat_interleave(torch.tensor([80, 10, 10, 10, 10]), dim=0))
        rotated = self.perturb(fixture, [.5] * 8 + [1.], rotation_max_deg=[0., 0., 45.])
        expected = torch.zeros_like(fixture[3])
        expected[..., :90] = True
        expected[..., 100:110] = True
        torch.testing.assert_close(rotated, expected)

    def test_each_frame_has_independent_probability_and_parameters(self):
        fixture = self.fixture()
        config = {**self.config, 'probability': .7, 'translation_max_m': [.25, 0., 0.]}
        with patch('run.utils.process_one_epoch.torch.rand', side_effect=[
                torch.tensor([[[.69, .7, .1]]]),
                torch.tensor([[1.] + [.5] * 8, [0.] + [.5] * 8])]):
            result = jitter_bbox_mask(*fixture, config)
        expected = fixture[3].clone()
        expected[:, :, 0, 10:20] = False
        expected[:, :, 0, 100:110] = True
        expected[:, :, 2, :10] = True
        expected[:, :, 2, 90:100] = False
        torch.testing.assert_close(result, expected)

    def test_sparse_frame_does_not_disable_other_frames(self):
        xyz, bounds, valid, original = self.fixture()
        valid[:, :, 1, 9:] = False
        original &= valid
        result = self.perturb((xyz, bounds, valid, original), [1.] + [.5] * 8,
                              translation_max_m=[.25, 0., 0.])
        torch.testing.assert_close(result[:, :, 1], original[:, :, 1])
        self.assertFalse(torch.equal(result[:, :, 0], original[:, :, 0]))

    def test_tracking_is_final_candidate_and_rejected_frames_keep_gt(self):
        fixture = self.fixture()
        tracking = fixture[3].clone()
        tracking[..., 10:20] = False
        tracking[..., 100:110] = True
        tracking[:, :, 1, :90] = False
        config = {**self.config, 'tracking_probability': 1., 'translation_max_m': [10.] * 3}
        with patch('run.utils.process_one_epoch.torch.rand', side_effect=[
                torch.zeros(1, 1, 3), torch.zeros(1, 1, 3)]) as draw:
            result = jitter_bbox_mask(*fixture, config, tracking)
        expected = tracking.clone()
        expected[:, :, 1] = fixture[3][:, :, 1]
        torch.testing.assert_close(result, expected)
        self.assertEqual(draw.call_count, 2)

    def test_missing_tracking_uses_random_candidate(self):
        fixture = self.fixture()
        tracking = fixture[3].clone()
        tracking[:, :, 1] = False
        config = {**self.config, 'tracking_probability': 1., 'translation_max_m': [.25, 0., 0.]}
        with patch('run.utils.process_one_epoch.torch.rand', side_effect=[
                torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), torch.tensor([[1.] + [.5] * 8])]):
            result = jitter_bbox_mask(*fixture, config, tracking)
        expected = fixture[3].clone()
        expected[:, :, 1, 10:20] = False
        expected[:, :, 1, 100:110] = True
        torch.testing.assert_close(result, expected)

    def test_rejection_and_retry_only_affect_failed_frame(self):
        xyz, bounds, valid, _ = self.fixture()
        xyz = xyz.clone()
        xyz[:, :, 1, :, 0] = 2.25 - .875
        original = ((xyz >= bounds[..., :3].unsqueeze(-2)) &
                    (xyz <= bounds[..., 3:].unsqueeze(-2))).all(-1)
        config = {**self.config, 'translation_max_m': [.25, 0., 0.], 'max_attempts': 2}
        with patch('run.utils.process_one_epoch.torch.rand', side_effect=[
                torch.zeros(1, 1, 3), torch.tensor([[1.] + [.5] * 8] * 3), torch.full((1, 9), .5)]) as draw:
            result = jitter_bbox_mask(xyz, bounds, valid, original, config)
        expected = original.clone()
        expected[:, :, [0, 2], 10:20] = False
        expected[:, :, [0, 2], 100:110] = True
        torch.testing.assert_close(result, expected)
        self.assertEqual(draw.call_count, 3)
        self.assertEqual(draw.call_args_list[-1].args, (1, 9))

    def test_exhausted_attempts_fall_back_without_filling_empty_or_padded_frames(self):
        xyz, bounds, valid, original = self.fixture()
        valid = valid.expand(1, 2, -1, -1).clone()
        valid[:, 1] = False
        valid[:, :, 2] = False
        original = original.expand_as(valid) & valid
        bounds = bounds.expand(1, 2, -1, -1)
        config = {**self.config, 'translation_max_m': [10., 0., 0.], 'max_attempts': 2}
        with patch('run.utils.process_one_epoch.torch.rand', side_effect=[
                torch.zeros(1, 2, 3), torch.tensor([[1.] + [.5] * 8] * 2),
                torch.tensor([[1.] + [.5] * 8] * 2)]) as draw:
            result = jitter_bbox_mask(xyz, bounds, valid, original, config)
        torch.testing.assert_close(result, original)
        self.assertEqual(draw.call_count, 3)

    def test_too_few_points_skip_jitter_without_consuming_random_numbers(self):
        xyz, bounds, valid, original = self.fixture(torch.zeros(9, 3))
        with patch('run.utils.process_one_epoch.torch.rand', side_effect=AssertionError('Unexpected RNG')):
            torch.testing.assert_close(jitter_bbox_mask(xyz, bounds, valid, original, self.config), original)
            torch.testing.assert_close(jitter_bbox_mask(xyz, bounds, valid, original,
                                                      {**self.config, 'probability': 0.}), original)

    def test_candidate_with_fewer_than_ten_points_is_rejected_even_at_high_iou(self):
        cloud = torch.zeros(10, 3)
        cloud[0, 0] = -.9
        fixture = self.fixture(cloud)
        result = self.perturb(fixture, [1.] + [.5] * 8, translation_max_m=[.2, 0., 0.])
        torch.testing.assert_close(result, fixture[3])

    def test_point_iou_and_minimum_count_hold_for_random_candidates(self):
        fixture = self.fixture()
        config = dict(probability=1., translation_max_m=[.4] * 3,
                      scale_range=[.5, 1.5], rotation_max_deg=[20.] * 3,
                      min_point_iou=.7, max_attempts=4)
        torch.manual_seed(42)
        for _ in range(20):
            result = jitter_bbox_mask(*fixture, config)
            intersection = (result & fixture[3]).sum(-1)
            union = (result | fixture[3]).sum(-1)
            self.assertTrue((intersection >= .7 * union).all())
            self.assertTrue((result.sum(-1) >= 10).all())

    def test_preparation_preserves_labels_points_and_disabled_sampling(self):
        xyz, bounds, valid, _ = self.fixture()
        points = torch.cat((xyz[:, 0], torch.arange(xyz.shape[-2]).view(1, 1, -1, 1).expand(1, 3, -1, 3)), -1)
        pose = bounds[..., :3].permute(0, 2, 1, 3).unsqueeze(-2).expand(1, 3, 1, 17, 3).clone()
        samples = {'radar_high_pc': {'padded': points, 'mask': valid[:, 0]},
                   'gt_for_high': {'padded': pose, 'mask': torch.ones(1, 3, 1, dtype=torch.bool),
                                   'bbox': bounds.permute(0, 2, 1, 3)}}
        before = copy.deepcopy(samples)
        base = dict(person_max_points=200)
        jittered = {**base, 'augmentation': {'bbox_jitter': {**self.config, 'enabled_train': True,
                    'enabled_val': False, 'translation_max_m': [.25, 0., 0.]}}}
        for stage in ('train', 'val'):
            torch.manual_seed(42)
            reference = prepare_person_pointcloud(samples, self.task, base, 'cpu', stage)
            rng_reference = torch.random.get_rng_state()
            disabled = copy.deepcopy(jittered)
            disabled['augmentation']['bbox_jitter']['enabled_train'] = False
            torch.manual_seed(42)
            actual = prepare_person_pointcloud(samples, self.task, disabled, 'cpu', stage)
            torch.testing.assert_close(actual[0]['input'], reference[0]['input'], rtol=0, atol=0)
            torch.testing.assert_close(actual[4], reference[4], rtol=0, atol=0)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_reference))
        validation = prepare_person_pointcloud(samples, self.task, jittered, 'cpu', 'val')
        torch.testing.assert_close(validation[4], reference[4])
        with patch('run.utils.process_one_epoch.torch.rand', side_effect=[
                torch.zeros(1, 1, 3), torch.tensor([[1.] + [.5] * 8] * 3)]):
            actual = prepare_person_pointcloud(samples, self.task, jittered, 'cpu', 'train')
        self.assertFalse(torch.equal(actual[4], reference[4]))
        torch.testing.assert_close(actual[1]['padded'], pose, rtol=0, atol=0)
        torch.testing.assert_close(actual[1]['bbox'], validation[1]['bbox'], rtol=0, atol=0)
        for name in samples:
            for key in samples[name]:
                torch.testing.assert_close(samples[name][key], before[name][key], rtol=0, atol=0)
        self.assertEqual(actual[0]['input'].shape, (1, 3, 200, 6))
        self.assertTrue(actual[0]['mask'].all())

    def test_config_validation_and_main_defaults(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config.yaml').read_text())
        validate_coordinate_frame(cfg['task'], cfg['data'])
        defaults = cfg['data']['augmentation']['bbox_jitter']
        self.assertTrue(defaults['enabled_train'])
        self.assertFalse(defaults['enabled_val'])
        self.assertEqual(defaults['min_point_iou'], .7)
        self.assertEqual(defaults['probability'], .7)
        for key, value in [('translation_max_m', [-.1, 0., 0.]), ('translation_max_m', [0.]),
                           ('rotation_max_deg', [float('inf')] * 3), ('scale_range', [0., 1.]),
                           ('scale_range', [1.1, .9]), ('probability', float('nan')),
                           ('probability', True), ('tracking_probability', 1.1), ('min_point_iou', 0.), ('min_point_iou', 1.1),
                           ('max_attempts', 0), ('max_attempts', True), ('enabled_train', 'true')]:
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                validate_coordinate_frame(self.task, {'augmentation': {'bbox_jitter': {**defaults, key: value}}})

    def test_tracking_cache_matches_exact_radar_gt_and_single_person(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = HPE_Dataset.__new__(HPE_Dataset)
            dataset.tracking_bbox_cache_dir = Path(directory)
            dataset.tracking_boxes = {}
            dataset.sample_groups = ['20260703/group_011']
            dataset.data_path_list = dict(radar_high_pc=[['1.npy', PackedFrame('pc.pack', '2.npy', 0, 0, 0),
                                                         '3.npy', '4.npy']],
                                          gt=[['11.pkl', '12.pkl', '13.pkl', '14.pkl']])
            box = np.array([1., -1., -1., 3., 1., 1.], dtype=np.float32)
            metadata = dict(group=dataset.sample_groups[0], rotation=np.eye(3).tolist(), translation=[0., 0., 0.])
            np.savez(Path(directory) / '20260703_group_011.npz', boxes=np.tile(box, (3, 1)),
                     stems=['1', '2', '3'], gt_stems=['11', '99', '13'], metadata=json.dumps(metadata))
            poses = [np.zeros((n, 17, 3)) for n in (1, 1, 2, 1)]
            calib = dict(R=np.eye(3), t=np.zeros(3))
            boxes = dataset._tracking_bbox_sequence(0, poses, calib)
            np.testing.assert_array_equal(boxes[0][0], box)
            for other in boxes[1:]:
                self.assertFalse(other.any())
            dataset.tracking_boxes.clear()
            with self.assertRaisesRegex(ValueError, '外参'):
                dataset._tracking_bbox_sequence(0, poses, dict(R=np.eye(3), t=np.ones(3)))

    def test_collated_tracking_survives_scene_transform_and_only_changes_train_crop(self):
        xyz, bounds, valid, original = self.fixture()
        points = torch.cat((xyz[0, 0], torch.zeros(3, xyz.shape[-2], 3)), -1).numpy()
        tracking_boxes = bounds[0].permute(1, 0, 2).numpy().copy()
        tracking_boxes[..., [0, 3]] += .25
        poses = np.zeros((3, 1, 17, 3), dtype=np.float32)
        poses[..., 0, :] = bounds[0, 0, :, :3].numpy()[:, None] + .3
        poses[..., 1:, :] = (bounds[0, 0, :, 3:].numpy()[:, None, None] - .3)
        sample = dict(radar_high_pc=list(points), gt_for_high=list(poses), tracking_bbox=list(tracking_boxes),
                      high_to_low_R=[np.eye(3, dtype=np.float32)] * 3,
                      high_to_low_t=[np.zeros(3, dtype=np.float32)] * 3)
        batch = collate_fn([sample], max_people=2)
        expected = original.clone()
        expected[..., 10:20] = False
        expected[..., 100:110] = True
        torch.testing.assert_close(batch['radar_high_pc']['tracking_mask'][:, :1], expected)
        self.assertFalse(batch['radar_high_pc']['tracking_mask'][:, 1].any())
        cfg = dict(person_max_points=200, augmentation=dict(
            radar_rotation=dict(enabled_train=True, roll_range_deg=[0., 0.], pitch_range_deg=[0., 0.], yaw_range_deg=[90., 90.]),
            radar_height=dict(enabled_train=True, delta_height_range_m=[.1, .1]),
            bbox_jitter={**self.config, 'enabled_train': True, 'enabled_val': False, 'tracking_probability': 1.}))
        transformed = adapt_coordinate_frame(batch, self.task, cfg, 'train')
        torch.testing.assert_close(transformed['radar_high_pc']['tracking_mask'], batch['radar_high_pc']['tracking_mask'])
        self.assertFalse(torch.equal(transformed['radar_high_pc']['padded'], batch['radar_high_pc']['padded']))
        training = prepare_person_pointcloud(transformed, self.task, cfg, 'cpu', 'train')
        torch.testing.assert_close(training[4][:, :1], expected)
        torch.testing.assert_close(training[1]['padded'], transformed['gt_for_high']['padded'])
        validation = prepare_person_pointcloud(batch, self.task, cfg, 'cpu', 'val')
        torch.testing.assert_close(validation[4][:, :1], original)


if __name__ == '__main__':
    unittest.main()
