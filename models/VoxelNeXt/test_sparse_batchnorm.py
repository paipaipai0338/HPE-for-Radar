import unittest

import torch
from torch import nn

from models.VoxelNeXt.VoxelNeXt_helper import SparseBatchNorm1d


class SparseBatchNormTests(unittest.TestCase):
    def test_single_voxel_keeps_gradients_and_running_statistics(self):
        original = nn.BatchNorm1d(8)
        original(torch.randn(12, 8))
        norm = SparseBatchNorm1d(8)
        norm.load_state_dict(original.state_dict(), strict=True)
        before = {k: v.clone() for k, v in norm.state_dict().items()}
        features = torch.randn(1, 8, requires_grad=True)
        result = norm(features)
        torch.testing.assert_close(result, original.eval()(features))
        result.square().sum().backward()
        for gradient in (features.grad, norm.weight.grad, norm.bias.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        for key, value in before.items():
            torch.testing.assert_close(norm.state_dict()[key], value)

    def test_multiple_voxels_preserve_batchnorm_updates(self):
        original, norm = nn.BatchNorm1d(8), SparseBatchNorm1d(8)
        norm.load_state_dict(original.state_dict(), strict=True)
        features = torch.randn(9, 8)
        torch.testing.assert_close(norm(features), original(features))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(norm.state_dict()[key], value)
        self.assertEqual(norm(torch.empty(0, 8)).shape, (0, 8))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for spconv integration')
    def test_full_model_single_voxel_training(self):
        from models.VoxelNeXt.VoxelNeXt import VoxelNeXt

        model = VoxelNeXt(num_point_features=0, backbone_channels=(8, 16, 32, 32, 32),
                          backbone_out_channel=64, head_shared_conv_channel=64,
                          voxel_size=(.05, .05, .1), query_ffn_dim=512).cuda().train()
        for points in (torch.tensor([2., 0., 0.], device='cuda').expand(1, 1, 32, 3).clone(),
                       torch.rand(2, 2, 32, 3, device='cuda') + torch.tensor([1., -1., -1.], device='cuda')):
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pose = model({'input': points, 'mask': torch.ones(points.shape[:3], device='cuda', dtype=torch.bool)})['pose']
            self.assertTrue(torch.isfinite(pose).all())
            pose.float().square().mean().backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))


if __name__ == '__main__':
    unittest.main()
