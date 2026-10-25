import unittest

import torch

from models.ResNet3D.ResNet3D import ResNet3D
from models.ResNet3D.roi_align import crop_spectra_to_gt


class RoiPowerGainTest(unittest.TestCase):
    def test_physical_gain_shared_across_frames_and_disabled_in_eval(self):
        args = (17, 1, 1, [4], [1], [1], [0], [0], [[-2, 2]] * 3, [4, 4, 4])
        model = ResNet3D(*args, roi_power_gain=True)
        plain = ResNet3D(*args)
        plain.load_state_dict(model.state_dict(), strict=True)
        cube = torch.linspace(0., 2., 2 * 8**3).reshape(1, 2, 1, 8, 8, 8)
        boxes = torch.tensor([-1., -1., -1., 1., 1., 1.]).reshape(1, 1, 1, 6).repeat(1, 2, 2, 1)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        inputs = {'input': cube, 'bbox': boxes, 'bbox_mask': mask}
        original_cube, original_boxes = cube.clone(), boxes.clone()
        crops, _ = crop_spectra_to_gt(cube, {'bbox': boxes, 'mask': mask}, model.xyz_limits, model.roi_cube_size)
        torch.manual_seed(37)
        gain = torch.empty(2).uniform_(-1., 1.).exp2()
        self.assertTrue(((gain >= .5) & (gain <= 2.)).all())
        self.assertNotEqual(float(gain[0]), float(gain[1]))
        expected = (crops['input'].expm1() * gain[:, None, None, None, None, None]).log1p().reshape(4, 1, 4, 4, 4)
        seen = []
        hook = model.layers[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        torch.manual_seed(37)
        result = model(inputs)
        hook.remove()
        torch.testing.assert_close(seen[0], expected)
        torch.testing.assert_close(result['pose_valid'], mask)
        result['pose'].square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        torch.testing.assert_close(cube, original_cube)
        torch.testing.assert_close(boxes, original_boxes)
        model.eval(); plain.eval()
        torch.testing.assert_close(model(inputs)['pose'], plain(inputs)['pose'])
        for value in [1e-8, 1000.]:
            inputs['input'] = torch.full_like(cube, value)
            model.train()
            self.assertTrue(torch.isfinite(model(inputs)['pose']).all())
        with self.assertRaises(TypeError):
            ResNet3D(*args, roi_power_gain='true')


if __name__ == '__main__':
    unittest.main()
