import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml
from torch import nn

from metrics.pose import (get_bone_length, get_center_mpjpe, get_mpjpe, get_pampjpe,
                          get_pose_direction, get_pose_hungarian_match,
                          get_relative_pose_mpjpe, get_time_smoothness)
from preprocess.gtprocess import get_gt_boxes
from run.utils.build_metric import Metric
from run.utils.build_model import build_model
from run.utils.process_one_epoch import (prepare_person_pointcloud, prepare_pose_target,
                                         predict_valid_person_frames, train_one_epoch, val_one_epoch)
from utils.MOCAP21 import coco_to_mocap
from utils.skeleton import get_pose_layout


class TubeStub(nn.Module):
    def forward(self, xyz, features):
        return xyz[:, :, :2], xyz.new_zeros(xyz.shape[0], xyz.shape[1], 128, 2)


class PoseFormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def sparse_samples(self):
        counts = torch.tensor([0, 9, 10, 12, 0, 10, 0])
        pose = torch.zeros(1, 7, 2, 17, 3)
        pose[..., 0] = 2.
        pose[:, :, 1, :, 0] = 20.
        mask = torch.ones(1, 7, 2, dtype=torch.bool)
        target = dict(padded=pose, mask=mask, bbox=get_gt_boxes(pose, mask, .3))
        points = torch.zeros(1, 7, 12, 6)
        points[..., 0] = 2.
        points[..., 3] = torch.arange(7)[None, :, None]
        return dict(group_id=['test'], gt_for_high=target, gt_for_low=target,
                    radar_high_pc=dict(padded=points, mask=torch.arange(12)[None, None] < counts[None, :, None]),
                    high_to_low_R=torch.eye(3)[None, None].expand(1, 7, 3, 3),
                    high_to_low_t=torch.zeros(1, 7, 3))

    def test_sparse_frames_are_excluded_before_sampling_for_train_and_val(self):
        samples = self.sparse_samples()
        task = dict(input='radar_high_pc', output='gt_for_high')
        expected = torch.tensor([[False, False, True, True, False, True, False]])
        for stage in ('train', 'val'):
            with self.subTest(stage=stage):
                inputs, gt, instances, _, _, _ = prepare_person_pointcloud(
                    samples, task, dict(person_max_points=200), 'cpu', stage)
                torch.testing.assert_close(gt['mask'][..., 0], expected)
                self.assertFalse(gt['mask'][..., 1].any())
                self.assertEqual(instances.tolist(), [[True, False]])
                torch.testing.assert_close(inputs['mask'].any(-1), expected)
                self.assertEqual(inputs['input'][:, :2].count_nonzero().item(), 0)
                self.assertEqual(gt['padded'][~gt['mask']].count_nonzero().item(), 0)
        self.assertTrue(samples['gt_for_high']['mask'].all())

    def test_training_only_forwards_contiguous_valid_frames_and_preserves_gradients(self):
        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(.1))
                self.frames = []

            def forward(self, inputs):
                self.assert_valid(inputs)
                self.frames.append(inputs['input'][0, :, 0, 3].tolist())
                pose = inputs['input'][..., :3].mean(-2)[:, :, None, None, :].expand(-1, -1, 1, 17, -1)
                return dict(pose=pose + self.bias,
                            transform_regularization=self.bias.square() * inputs['input'].shape[1])

            def assert_valid(self, inputs):
                assert inputs['mask'].any(-1).all(), 'Invalid frame reached model.forward'

        task = dict(input='radar_high_pc', output='gt_for_high', precision='FP32')
        model = RecordingModel()
        metric = Metric({'positive_mpjpe': 1}, None, 5., 2.)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = model.bias.detach().clone()
        result, _ = train_one_epoch(model, [self.sparse_samples()], optimizer, metric,
                                   torch.device('cpu'), dict(person_max_points=20), task,
                                   dict(name='RecordingModel'), None)
        self.assertEqual(model.frames, [[2., 3.], [5.]])
        self.assertEqual(result['mpjpe_by_group']['test']['count'], 3)
        self.assertFalse(torch.equal(model.bias, before))
        self.assertTrue(torch.isfinite(model.bias.grad))

        model.frames.clear()
        inputs = dict(input=torch.ones(2, 4, 12, 6),
                      mask=torch.tensor([[True] * 4, [False, True, False, True]])[..., None].expand(2, 4, 12))
        prediction = predict_valid_person_frames(model, inputs)
        torch.testing.assert_close(prediction['transform_regularization'], model.bias.square() * 3)
        prediction['transform_regularization'].backward()

    def test_all_invalid_or_fully_dropped_batches_skip_forward_loss_and_step(self):
        task = dict(input='radar_high_pc', output='gt_for_high', precision='FP32')
        for dropout in (False, True):
            with self.subTest(dropout=dropout):
                samples = self.sparse_samples()
                data = dict(person_max_points=20)
                if dropout:
                    data['augmentation'] = {'point_dropout': dict(enabled_train=True, probability=1.)}
                else:
                    samples['radar_high_pc']['mask'].zero_()
                model = nn.Linear(1, 1)
                metric = Metric({'positive_mpjpe': 1}, None, 5., 2.)
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                with patch.object(model, 'forward', side_effect=AssertionError('Unexpected forward')), \
                     patch.object(metric, 'calculate_batch', side_effect=AssertionError('Unexpected loss')), \
                     patch.object(optimizer, 'step', side_effect=AssertionError('Unexpected step')):
                    result, _ = train_one_epoch(model, [samples], optimizer, metric,
                                               torch.device('cpu'), data, task, dict(name='Linear'), None)
                self.assertEqual(result, {})

    def test_interpolation_matches_independent_weights_and_keeps_gradients(self):
        x = torch.randn(2, 3, 17, 3, dtype=torch.float64, requires_grad=True)
        original = x.detach().clone()
        weights = torch.zeros(21, 17, dtype=x.dtype)
        weights[0, [11, 12]] = .5
        weights[4, [5, 6]] = .5
        weights[6, :5] = .2
        for i, t in enumerate((.25, .5, .75), 1):
            weights[i] = (1-t)*weights[0] + t*weights[4]
        weights[5] = .55*weights[4] + .45*weights[6]
        weights[7] = .5*weights[5]
        weights[7, 5] += .5
        weights[11] = .5*weights[5]
        weights[11, 6] += .5
        copied = {8: 5, 9: 7, 10: 9, 12: 6, 13: 8, 14: 10,
                  15: 11, 16: 13, 17: 15, 18: 12, 19: 14, 20: 16}
        for dst, src in copied.items():
            weights[dst, src] = 1
        actual = coco_to_mocap(x)
        torch.testing.assert_close(actual, torch.einsum('ji,...ic->...jc', weights, x))
        for dst, src in copied.items():
            torch.testing.assert_close(actual[..., dst, :], x[..., src, :], rtol=0, atol=0)
        shift = x.new_tensor([3., 4., 5.])
        torch.testing.assert_close(coco_to_mocap(x+shift), actual+shift)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, weights.sum(0)[None, None, :, None].expand_as(x))
        torch.testing.assert_close(x, original)
        self.assertEqual(coco_to_mocap(torch.zeros(0, 17, 3)).shape, (0, 21, 3))
        with self.assertRaises(ValueError):
            coco_to_mocap(torch.zeros(21, 3))

    def test_target_conversion_masks_padding_and_keeps_crop_and_action_labels(self):
        pose = torch.randn(1, 2, 2, 17, 3)
        pose[:, :, 1] = torch.nan
        gt = dict(padded=pose, mask=torch.tensor([[[True, False], [True, False]]]),
                  bbox=torch.randn(1, 2, 2, 6), action=torch.zeros(1, 2, 2, 4))
        self.assertIs(prepare_pose_target(gt, 'coco'), gt)
        converted = prepare_pose_target(gt, 'mocap')
        self.assertEqual(converted['padded'].shape, (1, 2, 2, 21, 3))
        self.assertTrue(torch.isfinite(converted['padded']).all())
        self.assertEqual(converted['padded'][:, :, 1].count_nonzero().item(), 0)
        for key in ('mask', 'bbox', 'action'):
            self.assertIs(converted[key], gt[key])
        self.assertTrue(torch.isnan(gt['padded'][:, :, 1]).all())
        with self.assertRaises(ValueError):
            prepare_pose_target(gt, 'unknown')

    def test_metrics_handle_both_formats_and_mocap_pelvis_root(self):
        for fmt in ('coco', 'mocap'):
            with self.subTest(fmt=fmt):
                gt = torch.randn(1, 3, 1, len(get_pose_layout(fmt)['names']), 3, dtype=torch.float64)
                pre = gt + gt.new_tensor([3., 4., 0.])
                torch.testing.assert_close(get_mpjpe(pre, gt, fmt), torch.full((1, 3, 1), 5., dtype=gt.dtype))
                torch.testing.assert_close(get_center_mpjpe(pre, gt, fmt), torch.full((1, 3, 1), 5., dtype=gt.dtype))
                self.assertLess(get_relative_pose_mpjpe(pre, gt, fmt).max().item(), 1e-12)
                self.assertLess(get_pampjpe(pre, gt, fmt).max().item(), 1e-12)
                self.assertLess(get_bone_length(pre, gt, fmt).max().item(), 1e-12)
                self.assertLess(get_pose_direction(pre, gt, fmt).max().item(), 1e-12)
                static = gt[:, :1].expand(-1, 3, -1, -1, -1)
                moved = static + torch.arange(3, dtype=gt.dtype)[None, :, None, None, None]*gt.new_tensor([3., 4., 0.])
                torch.testing.assert_close(get_time_smoothness(moved, static, fmt), torch.full((1, 2, 1), 25., dtype=gt.dtype))
        gt = torch.zeros(1, 1, 1, 21, 3)
        pre = gt.clone()
        pre[..., 0, 0] = 2.
        self.assertEqual(get_center_mpjpe(pre, gt, 'mocap').item(), 2.)
        self.assertGreater(get_relative_pose_mpjpe(pre, gt, 'mocap').item(), 0)

    def test_matching_uses_mocap_root_and_metrics_preserve_group_statistics(self):
        gt = torch.zeros(1, 1, 2, 21, 3)
        gt[:, :, 1, 0, 0] = 5.
        pre = gt.flip(2).clone().requires_grad_()
        mask = torch.ones(1, 1, 2, dtype=torch.bool)
        matches = get_pose_hungarian_match(pre, gt, mask, hip_joint_indices=(0,))
        self.assertEqual(matches[0][1].tolist(), [1, 0])
        metric = Metric({'positive_mpjpe': 1}, None, 5., 2.,
                        pose_matching_by_hip=True, pose_format='mocap')
        loss, _ = metric.calculate_batch({'pose': pre}, {'padded': gt, 'mask': mask},
                                         group_ids=['test'], group_gt_low=torch.zeros(1, 1, 2, 17, 3))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(pre.grad).all())
        result = metric.epoch_end()
        self.assertEqual(result['mpjpe_by_group']['test']['count'], 2)
        self.assertEqual(result['mpjpe_by_group']['test']['mpjpe'], 0.)

    def test_train_and_val_epochs_run_with_both_formats(self):
        pose = torch.randn(1, 2, 2, 17, 3)*.05
        pose[..., 0] += 2.
        pose[:, :, 1] = 0.
        mask = torch.tensor([[[True, False], [True, False]]])
        target = dict(padded=pose, mask=mask,
                      bbox=get_gt_boxes(pose, mask, .3).masked_fill(~mask[..., None], 0.))
        points = torch.randn(1, 2, 32, 6)*.02
        points[..., 0] += 2.
        samples = dict(group_id=['test'], gt_for_high=target, gt_for_low=target,
                       radar_high_pc=dict(padded=points, mask=torch.ones(1, 2, 32, dtype=torch.bool)),
                       high_to_low_R=torch.eye(3)[None, None].expand(1, 2, 3, 3),
                       high_to_low_t=torch.zeros(1, 2, 3))
        for fmt in ('coco', 'mocap'):
            with self.subTest(fmt=fmt):
                model = build_model('P4Transformer', num_joints=len(get_pose_layout(fmt)['names']))
                model.tube_embedding = TubeStub()
                task = dict(input='radar_high_pc', output='gt_for_high', coordinate='high',
                            pose_format=fmt, precision='FP32')
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                before = model.mlp_head_pose[-1].weight.detach().clone()
                train_metric = Metric({'positive_mpjpe': 1}, None, 5., 2., pose_format=fmt)
                val_metric = Metric({'positive_mpjpe': 1}, None, 5., 2., pose_format=fmt)
                args = (torch.device('cpu'), dict(person_max_points=32), task, dict(name='P4Transformer'), None)
                trained, _ = train_one_epoch(model, [samples], optimizer, train_metric, *args)
                validated, _ = val_one_epoch(model, [samples], val_metric, *args)
                self.assertGreater(trained['positive_mpjpe'], 0)
                self.assertGreater(validated['positive_mpjpe'], 0)
                self.assertFalse(torch.equal(model.mlp_head_pose[-1].weight, before))
                self.assertTrue(torch.isfinite(model.mlp_head_pose[-1].weight.grad).all())
                self.assertEqual(target['padded'].shape[-2], 17)

    def test_main_derives_output_size_from_format_before_model_creation(self):
        import run.main as entry
        config = yaml.safe_load((Path(__file__).resolve().parents[1]/'config.yaml').read_text())
        architecture = yaml.safe_load((Path(__file__).resolve().parents[2]/'models/P4Transformer/model_config.yaml').read_text())
        self.assertIsNone(config['model']['init_checkpoint'])
        self.assertFalse(config['task']['train']['resume']['enabled'])
        for fmt, count in [('coco', 17), ('mocap', 21)]:
            cfg = copy.deepcopy(config)
            cfg['task']['pose_format'] = fmt
            with patch.object(entry, 'parse_args', return_value=SimpleNamespace(config='test.yaml')), \
                 patch.object(entry, 'load_config', side_effect=[cfg, architecture.copy()]), \
                 patch.object(entry, 'set_device', return_value=torch.device('cpu')), \
                 patch.object(entry, 'build_model', side_effect=RuntimeError('checked_model_options')) as build:
                with self.assertRaisesRegex(RuntimeError, 'checked_model_options'):
                    entry.main()
                self.assertEqual(build.call_args.kwargs['num_joints'], count)

    def test_plot_uses_mocap_edges(self):
        import numpy as np
        from run.utils.plot_fig import _draw_pose
        axis = Mock()
        pose = np.arange(63).reshape(1, 21, 3)
        _draw_pose(axis, pose, np.array([True]), 'blue', 'GT')
        self.assertEqual(axis.plot.call_count, 20)
        self.assertEqual(axis.plot.call_args_list[-1].args[0], [pose[0, 19, 0], pose[0, 20, 0]])

    def test_mocap_centering_and_roi_mirroring_follow_joint_names(self):
        from models.ResNet3D.ResNet3D import ResNet3D
        model = build_model('P4Transformer', num_joints=21, center_output_on_hip=True)
        model.tube_embedding = TubeStub()
        inputs = dict(input=torch.randn(1, 2, 8, 6), mask=torch.ones(1, 2, 8, dtype=torch.bool))
        centered = model(inputs)['pose']
        torch.testing.assert_close(centered[..., 0, :], torch.zeros(1, 2, 1, 3))

        model = ResNet3D(21, 1, 1, [4], [1], [1], [0], [0],
                         [[-2, 2]]*3, [4, 4, 4], roi_mirror_probability=1.)
        local = torch.arange(63, dtype=torch.float32).reshape(21, 3)/100
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.pose_head[-1].bias.copy_(local.flatten())
        names = get_pose_layout('mocap')['names']
        opposite = [name.replace('left_', 'right_') if name.startswith('left_') else
                    name.replace('right_', 'left_') if name.startswith('right_') else name for name in names]
        expected = local[[names.index(name) for name in opposite]]*torch.tensor([1., -1., 1.])
        cube = dict(input=torch.ones(1, 2, 1, 8, 8, 8),
                    bbox=torch.tensor([-1., -1., -1., 1., 1., 1.]).expand(1, 2, 1, 6),
                    bbox_mask=torch.ones(1, 2, 1, dtype=torch.bool))
        actual = model(cube)['pose']
        torch.testing.assert_close(actual, expected[None, None, None].expand_as(actual))
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(model.pose_head[-1].bias.grad).all())

    def test_dense_heatmap_encoder_uses_mocap_pelvis(self):
        from models.HRRadarPose.HRRadarPose import HRRadarPose
        pose = torch.ones(1, 1, 2, 21, 3)
        pose[..., 0, :] = torch.tensor([.5, 0., 0.])
        mask = torch.tensor([[[True, False]]])
        heatmap, indices, offsets, valid = HRRadarPose.output_encoder(
            None, pose, mask, [[-2, 2]]*3, [8, 8, 8],
            center_joint_indices=get_pose_layout('mocap')['root_indices'])
        self.assertEqual(indices.tolist(), [[[5*64+4*8+4, 0]]])
        self.assertEqual(heatmap[0, 0, 0, 5, 4, 4].item(), 1.)
        self.assertEqual(offsets.shape, pose.shape)
        torch.testing.assert_close(valid, mask)


if __name__ == '__main__':
    unittest.main()
