import unittest

import torch

from models.ResNet3D.ResNet3D import ResNet3D
from models.ResNet3D.roi_align import crop_spectra_to_gt


class RoiMirrorTest(unittest.TestCase):
    def test_probability_selects_people_consistently_across_frames(self):
        model = ResNet3D(17, 1, 1, [4], [1], [1], [0], [0],
                         [[-2, 2]] * 3, [4, 4, 4], roi_mirror_probability=.5)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.pose_head[-1].bias.copy_(torch.arange(51) / 100)
        boxes = torch.tensor([-1., -.5, -1., 1., 1.5, 1.]).reshape(1, 1, 1, 6).repeat(1, 2, 3, 1)
        inputs = {'input': torch.ones(1, 2, 1, 8, 8, 8), 'bbox': boxes,
                  'bbox_mask': torch.ones(1, 2, 3, dtype=torch.bool)}
        model.eval()
        expected = model(inputs)['pose'].detach()
        torch.manual_seed(0)
        selected = torch.rand(3) < .5
        self.assertTrue(selected.any() and (~selected).any())
        permutation = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        reflected = expected[:, :, selected][..., permutation, :].clone()
        reflected[..., 1] = 1. - reflected[..., 1]
        expected[:, :, selected] = reflected
        model.train()
        torch.manual_seed(0)
        torch.testing.assert_close(model(inputs)['pose'], expected)

    def test_mirror_restores_anatomy_and_shifted_scene_coordinates(self):
        model = ResNet3D(17, 1, 1, [4], [1], [1], [0], [0],
                         [[-2, 2]] * 3, [4, 4, 4], roi_shift_m=.1,
                         roi_mirror_probability=1.)
        local = torch.arange(51, dtype=torch.float32).reshape(17, 3) / 100
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.pose_head[-1].bias.copy_(local.flatten())
        cube = torch.arange(2 * 8**3, dtype=torch.float32).reshape(1, 2, 1, 8, 8, 8)
        boxes = torch.tensor([-1., -.5, -1., 1., 1.5, 1.]).reshape(1, 1, 1, 6).repeat(1, 2, 2, 1)
        boxes[:, :, 1] = 0.
        mask = torch.tensor([[[True, False], [True, False]]])
        inputs = {'input': cube, 'bbox': boxes, 'bbox_mask': mask}
        original_cube, original_boxes = cube.clone(), boxes.clone()
        torch.manual_seed(11)
        shift = torch.empty_like(boxes[..., :3]).uniform_(-.1, .1)
        shifted = boxes + shift.repeat(1, 1, 1, 2)
        expected_crops, _ = crop_spectra_to_gt(
            cube, {'bbox': shifted, 'mask': mask}, model.xyz_limits, model.roi_cube_size)
        observed = []
        handle = model.layers[0].register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone()))
        torch.manual_seed(11)
        result = model(inputs)
        handle.remove()
        torch.testing.assert_close(observed[0], expected_crops['input'].reshape(2, 1, 4, 4, 4).flip(-2))
        permutation = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        centers = (shifted[:, :, 0, :3] + shifted[:, :, 0, 3:]) / 2
        half_sizes = (shifted[:, :, 0, 3:] - shifted[:, :, 0, :3]) / 2
        expected = local[permutation] * torch.tensor([1., -1., 1.])
        expected = expected * half_sizes[..., None, :] + centers[..., None, :]
        torch.testing.assert_close(result['pose'][:, :, 0], expected)
        self.assertEqual(int(result['pose'][:, :, 1].count_nonzero()), 0)
        torch.testing.assert_close(result['pose_valid'], mask)
        torch.testing.assert_close(cube, original_cube)
        torch.testing.assert_close(boxes, original_boxes)
        result['pose'].square().sum().backward()
        self.assertTrue(torch.isfinite(model.pose_head[-1].bias.grad).all())
        self.assertGreater(float(model.pose_head[-1].bias.grad.abs().sum()), 0.)
        model.eval()
        torch.testing.assert_close(model(inputs)['pose'][:, :, 0],
                                   local.reshape(1, 1, 17, 3).expand(1, 2, -1, -1)
                                   + torch.tensor([0., .5, 0.]))

    def test_rejects_invalid_probability_and_joint_order(self):
        args = (17, 1, 1, [4], [1], [1], [0], [0], [[-2, 2]] * 3, [4, 4, 4])
        for probability in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                ResNet3D(*args, roi_mirror_probability=probability)
        with self.assertRaises(ValueError):
            ResNet3D(1, *args[1:], roi_mirror_probability=.5)


if __name__ == '__main__':
    unittest.main()
