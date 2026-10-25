import unittest

import torch

from models.ResNet3D.ResNet3D import ResNet3D
from models.ResNet3D.roi_align import crop_spectra_to_gt, restore_spectra_people


class RoiShiftTest(unittest.TestCase):
    def test_outside_boxes_are_predicted_and_padding_is_excluded(self):
        boxes = torch.tensor([[[[10., 10., 10., 12., 12., 12.],
                                 [0., 0., 0., 0., 0., 0.]]]])
        mask = torch.tensor([[[True, False]]])
        crops, context = crop_spectra_to_gt(
            torch.ones(1, 1, 1, 4, 4, 4), {'bbox': boxes, 'mask': mask},
            [[-2, 2]] * 3, [4, 4, 4])
        self.assertEqual(tuple(crops['input'].shape), (1, 1, 1, 4, 4, 4))
        self.assertEqual(int(crops['input'].count_nonzero()), 0)
        output = restore_spectra_people({'pose': torch.zeros(1, 1, 1, 1, 3)}, context)
        torch.testing.assert_close(output['pose_valid'], mask)
        torch.testing.assert_close(output['pose'][0, 0, 0, 0], torch.full((3,), 11.))
        torch.testing.assert_close(output['pose'][0, 0, 1, 0], torch.zeros(3))

    def test_shift_preserves_frame_and_ground_truth(self):
        torch.manual_seed(42)
        model = ResNet3D(1, 1, 1, [4], [1], [1], [0], [0],
                         [[-2, 2]] * 3, [4, 4, 4], roi_shift_m=0.2)
        for parameter in model.parameters():
            parameter.data.zero_()
        boxes = torch.tensor([-1., -1., -1., 1., 1., 1.]).reshape(1, 1, 1, 6)
        original = boxes.clone()
        inputs = {'input': torch.ones(1, 1, 1, 8, 8, 8), 'bbox': boxes,
                  'bbox_mask': torch.ones(1, 1, 1, dtype=torch.bool)}
        # A zero local pose must restore to the shifted ROI centre.
        torch.manual_seed(7)
        shift = torch.empty(1, 1, 1, 3).uniform_(-0.2, 0.2)
        torch.manual_seed(7)
        result = model(inputs)
        torch.testing.assert_close(result['pose'][..., 0, :], shift)
        torch.testing.assert_close(boxes, original)
        result['pose'].sum().backward()
        self.assertIsNotNone(model.pose_head[-1].bias.grad)
        model.eval()
        torch.testing.assert_close(model(inputs)['pose'], torch.zeros(1, 1, 1, 1, 3))
        model.train()
        model.roi_shift_m = 0
        torch.testing.assert_close(model(inputs)['pose'], torch.zeros(1, 1, 1, 1, 3))


if __name__ == '__main__':
    unittest.main()
