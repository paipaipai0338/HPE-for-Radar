import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch import nn

from models.P4JointMemory.P4JointMemory import P4JointMemory, reset_state, detach_state
from run.train_p4_joint_memory import augment_tokens, losses, evaluate, main, FIELDS
from run.utils.p4_sequence_data import StatefulBatches, extract_group, match_people, StreamBuilder
from run.analyze_pose_refiner import load_predictor


def inputs(batch=2, length=12, count=8):
    return dict(tokens=torch.randn(batch, length, count, 128).bfloat16(),
                xyz=torch.randn(batch, length, count, 3), features=torch.randn(batch, length, 128),
                quality=torch.rand(batch, length, 6), rotation=torch.eye(3).expand(batch, length, 3, 3),
                translation=torch.tensor([0., 0., -1.]).expand(batch, length, 3))


def head_state():
    head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 51))
    return head, {f'mlp_head_pose.{key}': value for key, value in head.state_dict().items()}


class JointMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_reliability_gates_hold_context_and_joint_state(self):
        model = P4JointMemory(reliability_gating=True).eval()
        batch = inputs(batch=1, length=8)
        with torch.no_grad():
            for gate in (model.context_gate, model.joint_gate):
                gate.bias.fill_(-100.)
            out, state = model(batch)
            torch.testing.assert_close(out, out[:, :1].expand_as(out))
            torch.testing.assert_close(state[1], batch['features'][:, 0])
            first, previous = model({k:v[:, :3] for k,v in batch.items()})
            last, _ = model({k:v[:, 3:] for k,v in batch.items()}, previous)
            torch.testing.assert_close(torch.cat((first,last),1), out)
        model = P4JointMemory(reliability_gating=True)
        with torch.no_grad():model.feature_update.weight.normal_(std=.03)
        batch['features'].requires_grad_()
        model(batch)[0][:, -1].square().mean().backward()
        for gate in (model.context_gate, model.joint_gate):
            self.assertTrue(torch.isfinite(gate.weight.grad).all())
            self.assertGreater(float(gate.weight.grad.abs().sum()),0.)
        self.assertGreater(float(batch['features'].grad[:, 0].abs().sum()),0.)

    def test_pretrained_head_transfer_and_coordinates(self):
        head, weights = head_state()
        model = P4JointMemory(); model.load_p4_head(weights)
        batch = inputs()
        batch['rotation'] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).expand(2, 12, 3, 3)
        batch['quality'] = batch['quality'].double()
        actual, _ = model(batch)
        high = head(batch['features']).reshape(2, 12, 17, 3)
        expected = high @ batch['rotation'].transpose(-1, -2)+batch['translation'][..., None, :]
        torch.testing.assert_close(actual, expected)

    def test_spatial_tokens_affect_prediction_and_order_does_not(self):
        model = P4JointMemory()
        with torch.no_grad():
            model.feature_update.weight.normal_(std=.05)
        batch = inputs(batch=1)
        actual, _ = model(batch)
        permuted = dict(batch, tokens=batch['tokens'].flip(2), xyz=batch['xyz'].flip(2))
        torch.testing.assert_close(model(permuted)[0], actual)
        changed = dict(batch, tokens=batch['tokens']+2)
        # LayerNorm removes uniform shifts; perturb spatial evidence non-uniformly.
        changed['xyz'] = batch['xyz']+torch.tensor([2., 0., 0.])
        self.assertGreater(float((model(changed)[0]-actual).detach().abs().max()), 1e-5)

    def test_causality_chunk_equivalence_reset_and_missing_observations(self):
        model = P4JointMemory()
        with torch.no_grad():
            model.feature_update.weight.normal_(std=.05)
        batch = inputs()
        actual, _ = model(batch)
        first = {k:v[:, :5] for k,v in batch.items()}
        second = {k:v[:, 5:] for k,v in batch.items()}
        a, state = model(first)
        b, _ = model(second, detach_state(state))
        torch.testing.assert_close(torch.cat((a,b),1), actual)
        fresh, _ = model({k:v[:1] for k,v in second.items()})
        reset = reset_state(state, torch.tensor([True,False]))
        torch.testing.assert_close(model(second,reset)[0][:1], fresh)
        changed = {k:v.clone() for k,v in batch.items()}; changed['tokens'][:, 8:] = 100
        torch.testing.assert_close(model(changed)[0][:,:8], actual[:,:8])
        batch['token_mask'] = torch.ones(2,12,8,dtype=torch.bool); batch['token_mask'][:,5:] = False
        batch['tokens'][:,5:] = float('nan');batch['xyz'][:,5:] = float('nan')
        out,_=model(batch)
        torch.testing.assert_close(out[:,5:],out[:,4:5].expand(-1,7,-1,-1))
        batch['token_mask'][:] = False
        self.assertEqual(model(batch)[0].count_nonzero().item(),0)

    def test_masks_augmentation_and_training_gradients(self):
        model=P4JointMemory(); batch=inputs(length=32)
        valid=torch.ones(2,32,dtype=torch.bool)
        config=dict(point_token_dropout=.2,missing_block_probability=1.,missing_block_max_frames=16,
                    horizontal_pose_weight=2.,direction_loss_weight=.02)
        augmented,n=augment_tokens(batch,valid,config)
        self.assertGreater(n,0)
        self.assertTrue(augmented['token_mask'][:,:8].any(-1).all())
        self.assertNotIn('token_mask',batch)
        pred,_=model(augmented)
        loss,_,count=losses(pred,torch.randn_like(pred),valid,config)
        loss.backward()
        self.assertEqual(count,64)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.feature_update.weight.grad.abs().sum()),0)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        optimizer.step()
        optimizer.zero_grad()
        losses(model(augmented)[0], torch.randn_like(pred), valid, config)[0].backward()
        self.assertGreater(float(model.attention.in_proj_weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.memory.weight_hh.grad.abs().sum()), 0)

    def test_evidence_extraction_keeps_point_tokens_and_low_xyz(self):
        class Tube(nn.Module):
            def forward(self, p):
                return p[:,:,::4,:3], p.new_zeros(p.shape[0],p.shape[1],128,p[:, :, ::4].shape[2])
        class Frontend(nn.Module):
            def __init__(self):
                super().__init__();self.tube_embedding=Tube();self.transformer=nn.Identity()
                self.mlp_head_pose=nn.Linear(128,51)
                nn.init.zeros_(self.mlp_head_pose.weight);nn.init.ones_(self.mlp_head_pose.bias)
            def forward(self, data):
                p=data['input'];b,t=p.shape[:2];xyz,f=self.tube_embedding(p)
                tokens=self.transformer(f.permute(0,1,3,2).reshape(b,-1,128))
                pooled=tokens.reshape(b,t,-1,128).amax(2)
                return {'pose':self.mlp_head_pose(pooled).reshape(b,t,1,17,3)}
        class Dataset(torch.utils.data.Dataset):
            sample_groups=['test/group']*2
            data_path_list={'radar_high_pc':[['0_000000000.npy','0_120000000.npy'],['0_120000000.npy','0_240000000.npy']]}
            @staticmethod
            def _frame_timestamp_ns(ref):return int(Path(ref).stem.split('_')[1])
            def __len__(self):return 2
            def __getitem__(self,index):
                return dict(group_id='test/group',radar_high_pc=[np.ones((20,6),np.float32)]*2,
                            gt_for_high=[np.ones((1,17,3),np.float32)]*2,
                            high_to_low_R=[np.eye(3,dtype=np.float32)]*2,
                            high_to_low_t=[np.array([0.,0.,-1.],np.float32)]*2)
        config=dict(extraction_batch_size=2,num_workers=0,max_frame_gap_seconds=.25,
                    track_max_center_distance=.5,track_max_joint_distance=.4)
        training=dict(data=dict(max_people=4,person_max_points=200),
                      task=dict(input='radar_high_pc',output='gt_for_high',coordinate='high',precision='FP32'))
        with patch('run.utils.p4_sequence_data.np.load',return_value=np.ones((20,6),np.float32)):
            streams=extract_group(Frontend(),Dataset(),[0,1],training,config,torch.device('cpu'))
        self.assertEqual(streams[0]['tokens'].shape,(2,50,128))
        self.assertEqual(streams[0]['tokens'].dtype,torch.bfloat16)
        self.assertEqual(streams[0]['quality'].dtype,torch.float32)
        torch.testing.assert_close(streams[0]['xyz'],torch.tensor([1.,1.,0.]).expand(2,50,3))

    def test_training_entry_and_nonoverlapping_batches(self):
        head,weights=head_state();batch=inputs(batch=1,length=12)
        target=head(batch['features']).detach().reshape(1,12,17,3)+batch['translation'][...,None,:]
        stream={k:v[0] for k,v in batch.items()};stream.update(baseline=target[0],target=target[0],group='test')
        total=sum(int(valid.sum()) for _,valid,_ in StatefulBatches([stream],5,2,FIELDS))
        self.assertEqual(total,12)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'base/checkpoint').mkdir(parents=True)
            torch.save(dict(model_state_dict=weights),root/'base/checkpoint/best.pth')
            config=dict(baseline_experiment=str(root/'base'),checkpoint='best.pth',output_dir=str(root/'runs'),
                        device='cpu',seed=42,batch_size=1,chunk_length=12,epochs=1,missing_block_max_frames=4,
                        point_token_dropout=.2,missing_block_probability=1.,warmup_frames=4,
                        horizontal_pose_weight=2.,direction_loss_weight=.02,learning_rate=.001,
                        head_learning_rate=.00001,weight_decay=0.,early_stopping_patience=1)
            path=root/'config.yaml';path.write_text(yaml.safe_dump(config))
            with patch('sys.argv',['train','--config',str(path)]),patch('run.train_p4_joint_memory.prepare',
                       return_value=(dict(train=[stream],val=[stream]),{},'test')):
                main()
            output=next((root/'runs').iterdir());history=json.loads((output/'history.json').read_text())
            self.assertEqual(history[0]['all']['count'],12)
            self.assertEqual(history[1]['train_frames'],12)
            self.assertTrue((output/'best.pth').exists())
            prediction = load_predictor(output, 'best.pth', 'cpu')(stream)
            self.assertEqual(prediction.shape, (12, 17, 3))
            self.assertTrue(np.isfinite(prediction).all())

    def test_stream_association_rejects_crossings_and_gaps(self):
        config = dict(max_frame_gap_seconds=.25, track_max_center_distance=.5,
                      track_max_joint_distance=.4)
        pose = np.zeros((1, 17, 3), np.float32)
        self.assertEqual(match_people(pose, pose, .5, .4), [(0, 0)])
        ambiguous = np.concatenate((pose, pose+.1))
        self.assertEqual(match_people(ambiguous, ambiguous, .5, .4), [])
        builder = StreamBuilder(config)
        evidence = dict(features=torch.zeros(128), quality=torch.ones(6))
        for time in (0., .1, .5):
            builder.append(time, pose, [evidence])
        streams = builder.finish('test')
        self.assertEqual([len(s['features']) for s in streams], [2, 1])
        self.assertAlmostEqual(float(streams[0]['quality'][1, -1]), .1)
        self.assertEqual(float(streams[1]['quality'][0, -1]), 0.)


if __name__=='__main__':unittest.main()
