import unittest
import torch
from run.utils.pose_evaluation import evaluate_pose_formats
from utils.MOCAP21 import coco_to_mocap

class PoseEvaluationTest(unittest.TestCase):
    def test_translation_alignment_and_main_ellipse(self):
        torch.manual_seed(42)
        y=torch.randn(3,17,3)*.1
        y[:,:,:2]-=y[:,[11,12],:2].mean(1,keepdim=True)
        y[:,:,:2]+=torch.tensor([[2.,0.],[3.6,0.],[4.,0.]])[:,None]
        p=y+torch.tensor([.3,0.,0.])
        r=evaluate_pose_formats(p,y)
        self.assertEqual(r['coco/inside']['frames'],2)
        for layout in ['coco','mocap']:
            self.assertAlmostEqual(r[layout+'/all']['mpjpe']/3,.3,places=5)
            self.assertLess(r[layout+'/all']['centered_mpjpe'],1e-5)
            self.assertLess(r[layout+'/all']['pa_mpjpe'],1e-5)
        torch.testing.assert_close(coco_to_mocap(y)[:,0],y[:,[11,12]].mean(1))
    def test_pa_removes_rotation_and_scale(self):
        y=torch.randn(4,17,3)
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        r=evaluate_pose_formats(2*y@rotation+3,y)
        self.assertLess(r['coco/all']['pa_mpjpe'],1e-4)

if __name__=='__main__':unittest.main()
