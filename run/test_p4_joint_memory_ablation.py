import unittest

import numpy as np
import torch

from models.P4JointMemory.P4JointMemory import P4JointMemory
from models.P4JointMemory.test_joint_memory import inputs
from run.analyze_p4_joint_memory_ablation import predict, masks_for, accumulate, finish


class AblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = P4JointMemory().eval()
        with torch.no_grad():
            self.model.feature_update.weight.normal_(std=.05)
            self.model.feature_update.bias.fill_(.03)
        self.stream = {k:v[0] for k,v in inputs(batch=1,length=24).items()}
        self.device = torch.device('cpu')

    def test_no_history_matches_independent_frames_and_hooks_restore(self):
        full = predict(self.model,self.stream,'full',self.device)
        independent=[]
        for index in range(24):
            batch={k:v[index:index+1][None] for k,v in self.stream.items()}
            with torch.inference_mode():
                independent.append(self.model(batch)[0][0].numpy())
        actual=predict(self.model,self.stream,'no_history',self.device)
        np.testing.assert_allclose(actual,np.concatenate(independent),atol=1e-6)
        self.assertGreater(float(np.abs(full-actual).max()),1e-5)
        np.testing.assert_array_equal(full,predict(self.model,self.stream,'full',self.device))

    def test_controls_branch_removal_chunk_resets_and_metrics(self):
        full=predict(self.model,self.stream,'full',self.device)
        np.testing.assert_allclose(full,predict(self.model,self.stream,'paired_permutation',self.device),atol=1e-6)
        with torch.inference_mode():
            f=self.stream['features'];h=torch.relu(self.model.head_hidden(self.model.head_norm(f)))
            high=torch.einsum('th,jch->tjc',h,self.model.joint_weight)+self.model.joint_bias
            expected=high@self.stream['rotation'].transpose(-1,-2)+self.stream['translation'][:,None]
        np.testing.assert_allclose(predict(self.model,self.stream,'trained_head_only',self.device),expected.numpy(),atol=1e-6)
        for name in ('no_token_features','no_xyz_encoding','no_quality','no_attention_evidence','no_global_features'):
            changed=predict(self.model,self.stream,name,self.device)
            self.assertGreater(float(np.abs(full-changed).max()),1e-5,name)
        reset=predict(self.model,self.stream,'reset_every_16',self.device)
        np.testing.assert_allclose(reset[:16],full[:16],atol=1e-6)
        fresh={k:v[16:] for k,v in self.stream.items()}
        np.testing.assert_allclose(reset[16:],predict(self.model,fresh,'full',self.device),atol=1e-6)
        self.stream.update(target=torch.from_numpy(full),baseline=torch.from_numpy(full),group='test')
        target,masks,horizontal=masks_for(self.stream,4)
        result=dict(conditions={},direction=np.zeros(6,np.int64),groups={},joint_error=np.zeros(17),max_prediction_difference=0.)
        accumulate(result,self.stream,full,full,target,masks,horizontal)
        report=finish(result)
        self.assertEqual(report['conditions']['all']['count'],24)
        self.assertEqual(report['conditions']['all']['mpjpe'],0.)
        self.assertEqual(report['delta_mpjpe_group_bootstrap_ci95'],[0.,0.])


if __name__=='__main__':
    unittest.main()
