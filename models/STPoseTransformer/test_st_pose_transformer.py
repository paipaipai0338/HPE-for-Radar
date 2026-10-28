import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.STPoseTransformer.STPoseTransformer import _voxel_local_pooling
from preprocess.gtprocess import get_gt_boxes
from run.utils.build_experiment import build_experiment
from run.utils.build_metric import Metric
from run.utils.build_model import build_model
from run.utils.load_config import load_config
from run.utils.model_init import model_init
from run.utils.process_one_epoch import train_one_epoch, val_one_epoch


class STPoseTransformerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def inputs(self, batch=1):
        points = torch.randn(batch, 16, 200, 6)
        points[..., :3] = points[..., :3] * .2 + torch.tensor([2., -.5, 1.])
        return {'input': points, 'mask': torch.rand(batch, 16, 200) > .2}

    # Native CPU kernels also exercise BF16 backward on hosts without AVX512.
    @torch.backends.mkldnn.flags(enabled=False)
    def test_training_masks_and_gradients_in_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = model_init(build_model('STPoseTransformer')).train()
                inputs = self.inputs(batch=2)
                inputs['mask'][0, 5] = False
                inputs['mask'][1] = False
                inputs['input'].masked_fill_(~inputs['mask'][..., None], float('nan'))
                with torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
                    pose = model(inputs)['pose']
                    loss = pose.square().mean()
                self.assertEqual(pose.shape, (2, 16, 1, 17, 3))
                self.assertTrue(torch.isfinite(pose).all())
                self.assertEqual(pose[0, 5].count_nonzero().item(), 0)
                self.assertEqual(pose[1].count_nonzero().item(), 0)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                self.assertGreater(model.velocity_scale_raw.grad.abs().item(), 0.)

    def test_absolute_coordinates_padding_and_unused_features(self):
        model = build_model('STPoseTransformer').eval()
        inputs = self.inputs()
        original_points = inputs['input'].clone()
        captured = {}
        hooks = [model.point_mlp.register_forward_pre_hook(
                     lambda module, args: captured.update(mlp=args[0].detach().clone())),
                 model.spatial_pos_enc.register_forward_pre_hook(
                     lambda module, args: captured.update(xyz=args[0].detach().clone())),
                 model.head.register_forward_hook(
                     lambda module, args, output: captured.update(head=output.detach().clone()))]
        with torch.no_grad(), patch('models.STPoseTransformer.STPoseTransformer._voxel_local_pooling',
                                    wraps=_voxel_local_pooling) as voxel_pool:
            expected = model(inputs)['pose']
            points = original_points[..., :4].masked_fill(~inputs['mask'][..., None], 0.)
            torch.testing.assert_close(captured['mlp'], points.reshape(16, 200, 4))
            torch.testing.assert_close(captured['xyz'], points[..., :3].reshape(16, 200, 3))
            torch.testing.assert_close(expected, captured['head'].reshape(1, 16, 1, 17, 3))
            flat = points.reshape(16, 200, 4)
            scores = flat[..., 3].abs().masked_fill(~inputs['mask'].reshape(16, 200), float('-inf'))
            selected = scores.topk(model.spatial_max_points, dim=1, sorted=False).indices
            xyz = flat[..., :3].gather(1, selected[..., None].expand(-1, -1, 3))
            self.assertEqual(voxel_pool.call_count, 3)
            for call in voxel_pool.call_args_list:
                torch.testing.assert_close(call.args[2], xyz)
                self.assertEqual(call.args[4], ([0, 6], [-3, 3], [-2, 2]))
            for hook in hooks:
                hook.remove()
            changed = original_points.clone()
            changed[..., 4:] = float('nan')
            changed.masked_fill_(~inputs['mask'][..., None], float('nan'))
            actual = model({'input': changed, 'mask': inputs['mask']})['pose']
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(inputs['input'], original_points)
        invalid = original_points.clone()
        invalid[0, 0, 0, 0] = float('inf')
        with self.assertRaises(ValueError):
            model({'input': invalid, 'mask': torch.ones_like(inputs['mask'])})

    def test_voxel_pooling_keeps_frames_masks_and_outside_points_separate(self):
        features = torch.tensor([[[1., 3.], [2., 1.], [9., 9.], [-1., -1.]],
                                 [[4., 0.], [0., 6.], [8., 8.], [10., 10.]]], requires_grad=True)
        xyz = torch.tensor([[0., 0., 0.], [1., 1., 1.], [.2, .2, .2], [1.1, 0., 0.]]).expand(2, -1, -1)
        padding = torch.tensor([[False, False, True, False]]).expand(2, -1)
        pooled = _voxel_local_pooling(features, padding, xyz, 1., ((0., 1.),) * 3)
        expected = features.detach().clone()
        expected[:, :2] += torch.tensor([[[2., 3.]], [[4., 6.]]])
        torch.testing.assert_close(pooled, expected)
        pooled.sum().backward()
        torch.testing.assert_close(features.grad, torch.tensor([
            [[1., 3.], [3., 1.], [1., 1.], [1., 1.]],
            [[3., 1.], [1., 3.], [1., 1.], [1., 1.]],
        ]))

    def test_voxel_pooling_matches_per_point_reference_with_ties_and_empty_frames(self):
        xyz = torch.rand(3, 20, 3) * 2 - .5
        xyz[:, :2] = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        valid = torch.rand(3, 20) > .2
        valid[1] = False
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    features = torch.randint(-2, 3, (3, 20, 4)).to(device=device, dtype=dtype).requires_grad_()
                    expected_frames = []
                    active = valid & ((xyz >= 0) & (xyz <= 1)).all(-1)
                    cells = (xyz.clamp(0, 1 - 1e-5) / .5).floor().long()
                    for frame in range(3):
                        rows = []
                        for point in range(20):
                            value = features[frame, point]
                            if active[frame, point]:
                                neighbors = active[frame] & (cells[frame] == cells[frame, point]).all(-1)
                                value = value + features[frame, neighbors.to(device)].amax(0)
                            rows.append(value)
                        expected_frames.append(torch.stack(rows))
                    expected = torch.stack(expected_frames)
                    actual = _voxel_local_pooling(features, ~valid.to(device), xyz.to(device), .5, ((0., 1.),) * 3)
                    torch.testing.assert_close(actual, expected)
                    actual_grad = torch.autograd.grad(actual.float().square().sum(), features, retain_graph=True)[0]
                    expected_grad = torch.autograd.grad(expected.float().square().sum(), features)[0]
                    torch.testing.assert_close(actual_grad, expected_grad)

    @torch.backends.mkldnn.flags(enabled=False)
    def test_shared_training_and_validation_with_missing_person_frame(self):
        points = torch.randn(1, 16, 200, 6) * .03
        points[:, :, :100, :3] += torch.tensor([2., 0., 1.])
        points[:, :, 100:, :3] += torch.tensor([4., 1., 1.])
        point_mask = torch.ones(1, 16, 200, dtype=torch.bool)
        point_mask[:, 5, :100] = False
        pose = torch.zeros(1, 16, 2, 17, 3)
        pose[:, :, 0] = torch.tensor([2., 0., 1.])
        pose[:, :, 1] = torch.tensor([4., 1., 1.])
        person_mask = torch.ones(1, 16, 2, dtype=torch.bool)
        target = {'padded': pose, 'mask': person_mask,
                  'bbox': get_gt_boxes(pose, person_mask, threshold=.3)}
        samples = {
            'radar_high_pc': {'padded': points, 'mask': point_mask},
            'gt_for_high': target,
            'gt_for_low': {'padded': pose - torch.tensor([0., 0., .5])},
            'high_to_low_R': torch.eye(3).expand(1, 16, 3, 3),
            'high_to_low_t': torch.tensor([0., 0., -.5]).expand(1, 16, 3),
        }
        task = dict(coordinate='low', input='radar_high_pc', output='gt_for_high', precision='BF16')
        data = dict(person_max_points=200, augmentation={})
        model_cfg = {'name': 'STPoseTransformer'}
        model = model_init(build_model('STPoseTransformer'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        device = torch.device('cpu')
        train_metric = Metric({'Positive_MPJPE': 1}, None, 5., 2.)
        val_metric = Metric({'Positive_MPJPE': 1}, None, 5., 2.)
        trained, _ = train_one_epoch(model, [samples], optimizer, train_metric, device,
                                     data, task, model_cfg, None)
        validated, _ = val_one_epoch(model, [samples], val_metric, device,
                                     data, task, model_cfg, None)
        self.assertGreater(trained['positive_mpjpe'], 0.)
        self.assertGreater(validated['positive_mpjpe'], 0.)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_experiment_source_snapshot_loads_without_holosense(self):
        model = build_model('STPoseTransformer').eval()
        arch = load_config(Path(__file__).with_name('model_config.yaml'))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'config.yaml'
            source.write_text('model:\n  name: STPoseTransformer\n')
            paths = build_experiment(directory, 'STPoseTransformer', source, model, arch)
            spec = importlib.util.spec_from_file_location('saved_st_pose', paths['config'] / 'STPoseTransformer.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            restored = module.STPoseTransformer(**load_config(paths['config'] / 'model_config.yaml')).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            inputs = self.inputs()
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs)['pose'], model(inputs)['pose'])


if __name__ == '__main__':
    unittest.main()
