import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import yaml

from run.utils.build_model import build_model
from run.utils.build_experiment import build_experiment
from run.utils.build_metric import Metric
from run.utils.load_config import load_config
from run.utils.process_one_epoch import prepare_person_pointcloud


class TubeStub(nn.Module):
    def forward(self, xyz, features):
        self.xyz = xyz.detach().clone()
        return xyz[:, :, :2], xyz.new_zeros(xyz.shape[0], xyz.shape[1], 128, 2)


class CenteredOutputTest(unittest.TestCase):
    def test_model_centers_output_and_preserves_absolute_input_and_gradients(self):
        torch.manual_seed(42)
        model = build_model('P4Transformer', center_output_on_hip=True)
        model.tube_embedding = TubeStub()
        points = torch.randn(2, 3, 8, 6) + torch.tensor([3., .5, 1., 0., 0., 0.])
        inputs = {'input': points, 'mask': torch.ones(2, 3, 8, dtype=torch.bool)}
        pose = model(inputs)['pose']
        torch.testing.assert_close(pose[..., [11, 12], :].mean(-2), torch.zeros(2, 3, 1, 3), atol=1e-6, rtol=0)
        torch.testing.assert_close(model.tube_embedding.xyz, points[..., :3])
        loss = pose.square().mean()
        loss.backward()
        gradient = model.mlp_head_pose[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        model.center_output_on_hip = False
        raw = model(inputs)['pose']
        torch.testing.assert_close(pose, raw - raw[..., [11, 12], :].mean(-2, keepdim=True))

    def test_pointcloud_preparation_does_not_translate_points_or_gt(self):
        points = torch.zeros(1, 2, 16, 6)
        points[..., 0] = torch.linspace(2., 3., 16)
        points[..., 1:3] = .5
        pose = torch.ones(1, 2, 1, 17, 3)
        pose[..., 0] = 2.5
        mask = torch.ones(1, 2, 1, dtype=torch.bool)
        samples = {'radar_high_pc': {'padded': points, 'mask': torch.ones(1, 2, 16, dtype=torch.bool)},
                   'gt_for_high': {'padded': pose, 'mask': mask,
                                   'bbox': torch.tensor([1., -2., -2., 4., 2., 2.]).expand(1, 2, 1, 6)}}
        task = dict(input='radar_high_pc', output='gt_for_high', center_on_pointcloud=False, center_xy_in_low=False)
        inputs, gt, _, _, _, center = prepare_person_pointcloud(
            samples, task, {'person_max_points': 16}, torch.device('cpu'), 'val')
        torch.testing.assert_close(inputs['input'], points)
        torch.testing.assert_close(gt['padded'], pose)
        self.assertEqual(center.count_nonzero().item(), 0)

    def test_centered_loss_ignores_translation_and_padding_and_records_groups(self):
        gt_pose = torch.zeros(1, 1, 2, 17, 3)
        prediction = gt_pose.clone()
        prediction[..., 0, 0] = 1.7
        prediction += torch.tensor([3., 2., 1.])
        prediction.requires_grad_()
        gt_pose += torch.tensor([10., -4., 8.])
        mask = torch.tensor([[[True, False]]])
        metric = Metric({'relative_pose_mpjpe': 1}, None, 5., 2.)
        loss, _ = metric.calculate_batch({'pose': prediction}, {'padded': gt_pose, 'mask': mask},
                                         ['test'], group_gt_low=gt_pose)
        self.assertAlmostEqual(loss.item(), .1, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[:, :, 1].count_nonzero().item(), 0)
        result = metric.epoch_end()
        self.assertAlmostEqual(result['relative_pose_mpjpe'], .1, places=6)
        self.assertAlmostEqual(result['mpjpe_by_group']['test']['centered_mpjpe'], .1, places=6)

    def test_saved_model_config_keeps_output_centering_option(self):
        model = build_model('P4Transformer', center_output_on_hip=True)
        arch = load_config(Path(__file__).with_name('model_config.yaml'))
        arch['center_output_on_hip'] = True
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.yaml'
            source.write_text('model:\n  name: P4Transformer\n')
            paths = build_experiment(tmp, 'P4Transformer', source, model, model_config=arch)
            saved = yaml.safe_load((paths['config'] / 'model_config.yaml').read_text())
            self.assertTrue(saved['center_output_on_hip'])
            restored = type(model)(**saved)
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertTrue(restored.center_output_on_hip)


if __name__ == '__main__':
    unittest.main()
