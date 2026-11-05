import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from collections import OrderedDict
from unittest.mock import patch
import json
import copy

import yaml

import numpy as np
import torch
from torch import nn

from models.P4JointMemory.P4JointFineTune import P4JointFineTune
from models.P4JointMemory.P4JointMemory import P4JointMemory, detach_state
from run.train_p4_joint_finetune import evaluate, train_epoch
from run.utils.p4_finetune_data import (OnlineHPE, OnlinePoints, prepare_input_index, load_streams, PointSteps, PointStepSampler,
                                        make_loader, predict_saved_joint)


class Tube(nn.Module):
    spatial_stride = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5, 128)

    def forward(self, xyz, attributes):
        points = torch.cat((xyz, attributes.permute(0, 1, 3, 2)), -1)
        return xyz[:, :, ::4], self.projection(points[:, :, ::4]).permute(0, 1, 3, 2)


class Frontend(nn.Module):
    feat_dim = 2
    emb_relu = False

    def __init__(self):
        super().__init__()
        self.tube_embedding = Tube()
        self.pos_embedding = nn.Conv1d(4, 128, 1)
        self.transformer = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        self.mlp_head_pose = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 51))

    def forward(self, points):
        b, t = points.shape[:2]
        xyz, features = self.tube_embedding(points[..., :3], points[..., 3:].permute(0, 1, 3, 2))
        n = xyz.shape[2]
        stamps = torch.arange(1, t+1).float().expand(b, n, t).transpose(1, 2)[..., None]
        position = self.pos_embedding(torch.cat((xyz, stamps), -1).reshape(b, t*n, 4).transpose(1, 2)).transpose(1, 2)
        output = self.transformer(features.permute(0, 1, 3, 2).reshape(b, t*n, -1)+position)
        last = output.reshape(b, t, n, -1)[:, -1]
        return last, self.mlp_head_pose(last.max(1).values).reshape(b, 17, 3)


def batch(b=2, length=12):
    return dict(points=torch.randn(b, length, 2, 8, 5), valid=torch.ones(b, length, dtype=torch.bool),
                quality=torch.rand(b, length, 6), rotation=torch.eye(3).expand(b, length, 3, 3),
                translation=torch.tensor([0., 0., -1.]).expand(b, length, 3))


class FineTuneTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_encoder_matches_original_and_gradients_reach_each_stage(self):
        frontend = Frontend()
        decoder = P4JointMemory()
        decoder.load_p4_head({f'mlp_head_pose.{k}':v for k,v in frontend.mlp_head_pose.state_dict().items()})
        model = P4JointFineTune(frontend, decoder)
        data = batch()
        tokens, original = frontend(data['points'].flatten(0, 1))
        encoded = model.encode(data)
        torch.testing.assert_close(encoded['tokens'].flatten(0, 1), tokens)
        expected = original.reshape(2, 12, 17, 3)+data['translation'][..., None, :]
        torch.testing.assert_close(model(data)[0], expected)
        for train_points in (False, True):
            model.zero_grad(set_to_none=True)
            model.set_encoder_stage(train_points)
            model(data)[0].square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.transformer.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.transformer.parameters()),0)
            self.assertEqual(any(p.grad is not None for p in model.tube_embedding.parameters()),train_points)
            if train_points:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.tube_embedding.parameters()),0)
        data['valid'][:, -1] = False
        encoded = model.encode(data)
        self.assertFalse(encoded['token_mask'][:, -1].any())
        self.assertEqual(int(encoded['tokens'][:, -1].count_nonzero()),0)

    def test_gradient_boundaries_preserve_state_without_resetting_history(self):
        model = P4JointFineTune(Frontend(), P4JointMemory()).eval()
        with torch.no_grad():
            model.decoder.feature_update.weight.normal_(std=.05)
        data = batch()
        with torch.no_grad():
            full, _ = model(data)
            first, state = model({k:v[:, :5] for k,v in data.items()})
            last, _ = model({k:v[:, 5:] for k,v in data.items()},detach_state(state))
        torch.testing.assert_close(torch.cat((first,last),1),full)

    def test_checkpoint_microbatches_preserve_predictions_and_parameter_gradients(self):
        reference=P4JointFineTune(Frontend(),P4JointMemory(),encoder_batch_size=3,checkpoint_encoder=False)
        with torch.no_grad():
            reference.decoder.feature_update.weight.normal_(std=.03)
        recomputed=copy.deepcopy(reference);recomputed.checkpoint_encoder=True
        data=batch();data['valid'][0,-1]=False
        for train_points in (False,True):
            predictions=[];gradients=[]
            for model in (reference,recomputed):
                model.set_encoder_stage(train_points);model.zero_grad(set_to_none=True)
                prediction=model(data)[0]
                prediction[data['valid']].square().mean().backward()
                predictions.append(prediction.detach())
                gradients.append({name:p.grad.detach().clone() if p.grad is not None else None
                                  for name,p in model.named_parameters()})
            torch.testing.assert_close(*predictions)
            for name in gradients[0]:
                a,b=gradients[0][name],gradients[1][name]
                if a is None:self.assertIsNone(b)
                else:torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7,msg=name)

    def test_one_optimizer_step_and_gradient_reaches_first_of_128_frames(self):
        model=P4JointFineTune(Frontend(),P4JointMemory(),encoder_batch_size=8)
        with torch.no_grad():model.decoder.feature_update.weight.normal_(std=.03)
        data=batch(b=1,length=128);data['points'].requires_grad_()
        data.update(target=torch.zeros(1,128,17,3),reset=torch.tensor([True]),
                    baseline=torch.zeros(1,128,17,3),position=torch.arange(128)[None])
        class Loader:
            dataset=type('Metadata',(),{'streams':[{'target':torch.zeros(128,17,3)}]})()
            def __iter__(self):yield data
        def hold_after_first(encoded,valid,config):
            encoded=dict(encoded);mask=torch.zeros_like(encoded['token_mask']);mask[:,0]=True
            encoded['token_mask']=mask
            return encoded,127
        def final_frame_loss(pose,target,valid,config):
            return pose[:,-1].square().mean(),pose.norm(dim=-1).mean(-1)[valid].sum(),int(valid.sum())
        config=dict(chunk_length=128)
        with patch('run.train_p4_joint_finetune.augment_tokens',side_effect=hold_after_first), \
             patch('run.train_p4_joint_finetune.losses',side_effect=final_frame_loss),redirect_stderr(StringIO()):
            report=train_epoch(model,Loader(),torch.optim.SGD(model.parameters(),lr=1e-4),torch.device('cpu'),config)
        self.assertEqual(report['optimizer_steps'],1)
        self.assertEqual(report['train_frames'],128)
        self.assertGreater(float(data['points'].grad[:,0].abs().sum()),0)
        self.assertEqual(float(data['points'].grad[:,1:].abs().sum()),0)

    def test_online_points_match_deterministic_sampling_and_reject_wrong_identity(self):
        class Data(torch.utils.data.Dataset):
            sample_groups=['test']*3
            data_path_list = {'radar_high_pc':[['0_000000000.npy','0_120000000.npy'],
                ['0_120000000.npy','0_240000000.npy'],['0_240000000.npy','0_360000000.npy']]}
            @staticmethod
            def _frame_timestamp_ns(ref):return int(Path(ref).stem.split('_')[1])
            def __len__(self):return 3
            def __getitem__(self,index):
                return dict(group_id='test',radar_high_pc=[np.ones((20,6),np.float32)]*2,
                    gt_for_high=[np.ones((1,17,3),np.float32)]*2,
                    high_to_low_R=[np.eye(3,dtype=np.float32)]*2,
                    high_to_low_t=[np.array([0.,0.,-1.],np.float32)]*2)
        stream=dict(group='test',window_indices=torch.arange(3),target=torch.tensor([1.,1.,0.]).expand(3,17,3).clone(),
            rotation=torch.eye(3).expand(3,3,3),translation=torch.tensor([0.,0.,-1.]).expand(3,3),
            timestamp=torch.tensor([n*1e-9 for n in (120000000,240000000,360000000)],dtype=torch.float64),
            quality=torch.ones(3,6),baseline=torch.zeros(3,17,3))
        training=dict(data=dict(T=2,person_max_points=200,max_people=4),task=dict(input='radar_high_pc',output='gt_for_high'))
        context=dict(training=training,feature_dim=2,input_batch_size=2,mode='val',tracking=dict(
            max_frame_gap_seconds=.25,track_max_center_distance=.5,track_max_joint_distance=.4))
        reader=OnlinePoints(context,Data())
        points=reader.read(stream,0,3)
        self.assertEqual(tuple(points.shape),(3,2,200,5));self.assertEqual(points.dtype,torch.float32)
        torch.testing.assert_close(points,torch.ones_like(points))
        stream['online']=context
        config=dict(chunk_length=2,batch_size=1,num_workers=2)
        results=list(make_loader([stream],config,torch.device('cpu'),False,Data()))
        self.assertEqual([int(b['valid'].sum()) for b in results],[2,1])
        torch.testing.assert_close(results[0]['points'][0],points[:2])
        stream['target'][1] += 1
        with self.assertRaisesRegex(ValueError,'identity'):
            reader.read(stream,0,3)
        stream['target'][1] -= 1
        stream['translation'][1,0] += 1
        with self.assertRaisesRegex(ValueError,'Calibration'):
            reader.read(stream,0,3)

    def test_raw_frame_memory_is_bounded_and_returns_copies(self):
        dataset=OnlineHPE.__new__(OnlineHPE);dataset.recent_frames=OrderedDict()
        calls=[]
        def load(path):
            calls.append(path)
            return np.ones(3,dtype=np.float32)
        a=dataset._get_cached_frame_data('radar_high_pc',Path('frame'),load);a[0]=9
        np.testing.assert_array_equal(dataset._get_cached_frame_data('radar_high_pc',Path('frame'),load),np.ones(3))
        self.assertEqual(len(calls),1)
        for i in range(300):dataset._get_cached_frame_data('radar_high_pc',Path(str(i)),load)
        self.assertEqual(len(dataset.recent_frames),256)

    def test_online_identity_survives_reordered_people(self):
        class Data:
            sample_groups=['test']
            data_path_list={'radar_high_pc':[['0_000000000.npy','0_120000000.npy']]}
            @staticmethod
            def _frame_timestamp_ns(ref):return int(Path(ref).stem.split('_')[1])
            def __getitem__(self,index):
                a=np.ones((17,3),np.float32);b=a.copy();b[:,0]=4
                points=np.ones((40,6),np.float32);points[20:,0]=4
                return dict(group_id='test',radar_high_pc=[points]*2,
                    gt_for_high=[np.stack((a,b)),np.stack((b,a))],
                    high_to_low_R=[np.eye(3,dtype=np.float32)]*2,
                    high_to_low_t=[np.array([0.,0.,-1.],np.float32)]*2)
        context=dict(training=dict(data=dict(T=2,person_max_points=200,max_people=4),
            task=dict(input='radar_high_pc',output='gt_for_high')),feature_dim=2,input_batch_size=2,
            mode='val',tracking=dict(max_frame_gap_seconds=.25,track_max_center_distance=.5,track_max_joint_distance=.4))
        stream=dict(group='test',window_indices=torch.tensor([0]),target=torch.tensor([1.,1.,0.]).expand(1,17,3).clone(),
            rotation=torch.eye(3)[None],translation=torch.tensor([[0.,0.,-1.]]),timestamp=torch.tensor([120000000*1e-9],dtype=torch.float64))
        reader=OnlinePoints(context,Data())
        torch.testing.assert_close(reader.read(stream,0,1),torch.ones(1,2,200,5))
        stream['target'][...,0]=4
        selected=reader.read(stream,0,1)
        torch.testing.assert_close(selected[...,0],torch.full((1,2,200),4.))
        torch.testing.assert_close(selected[...,1:],torch.ones(1,2,200,4))

    def test_prepare_writes_only_compact_indices_and_keeps_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'config').mkdir();(p/'init').mkdir()
            training=dict(data=dict(T=2,person_max_points=200,max_people=4,pointcloud_source='fftgen',split_method='person'),
                          task=dict(input='radar_high_pc',output='gt_for_high'))
            (p/'config/config.yaml').write_text(yaml.safe_dump(training));(p/'config/model_config.yaml').write_text('feat_dim: 2\n')
            source={};datasets={}
            for mode in ('train','val'):
                group=mode+'/test'
                source_stream=dict(group=group,target=torch.ones(3,17,3),baseline=torch.zeros(3,17,3),
                    quality=torch.rand(3,6),rotation=torch.eye(3).expand(3,3,3),translation=torch.zeros(3,3),
                    timestamp=torch.tensor([n*1e-9 for n in (120000000,240000000,360000000)],dtype=torch.float64))
                torch.save([source_stream],p/(mode+'.pt'))
                source[mode]={group:dict(cache=str(p/(mode+'.pt')),frames=3)}
                class Data:
                    sample_groups=[group]*3
                    data_path_list={'radar_high_pc':[[f'0_{i:09d}.npy'] for i in (120000000,240000000,360000000)]}
                    @staticmethod
                    def _frame_timestamp_ns(ref):return int(Path(ref).stem.split('_')[1])
                datasets[mode]=Data()
            (p/'init/evidence_manifest.json').write_text(json.dumps(source))
            config=dict(baseline_experiment=str(p),init_experiment=str(p/'init'),index_dir=str(p/'indices'),input_batch_size=2,
                max_frame_gap_seconds=.25,track_max_center_distance=.5,track_max_joint_distance=.4)
            with patch('run.utils.p4_finetune_data.make_raw_dataset',side_effect=lambda training,mode:datasets[mode]), \
                 patch('run.utils.p4_finetune_data.group_fingerprint',return_value='test'):
                manifest,_=prepare_input_index(config)
            loaded=load_streams(manifest['val'])
            self.assertEqual(len(loaded),1)
            torch.testing.assert_close(loaded[0]['window_indices'],torch.arange(3))
            self.assertFalse(list(p.rglob('*.npy')))
            self.assertLess(sum(f.stat().st_size for f in (p/'indices').rglob('*.pt')),32000)

    def test_lane_order_padding_evaluation_and_training(self):
        data=batch(b=1,length=19)
        class Points:
            shape=(2,8,5)
            def read(self,stream,start,size):
                offset=int(stream['group'])+start
                return data['points'][0,offset:offset+size].clone()
        reader=Points()
        streams=[]
        for offset,length in ((0,7),(7,3),(10,9)):
            streams.append({**{k:v[0,offset:offset+length] for k,v in data.items() if k not in ('points','valid')},
                'baseline':torch.zeros(length,17,3),'target':torch.zeros(length,17,3),
                'online':{},'group':str(offset)})
        config=dict(chunk_length=8,batch_size=2,num_workers=0,warmup_frames=4,
            point_token_dropout=.2,missing_block_probability=1.,missing_block_max_frames=4,
            horizontal_pose_weight=2.,direction_loss_weight=.02)
        steps=list(PointStepSampler(streams,8,2,False))
        self.assertEqual(steps[0],((0,0),(1,0)))
        self.assertEqual(steps[1],((2,0),None))
        self.assertEqual(steps[2],((2,8),None))
        with patch('run.utils.p4_finetune_data.OnlinePoints',return_value=reader):
            dataset=PointSteps(streams,8)
            self.assertEqual(sum(int(dataset[s]['valid'].sum()) for s in steps),19)
            self.assertFalse(dataset[steps[2]]['reset'][0])
            model=P4JointFineTune(Frontend(),P4JointMemory()).eval()
            expected=sum(np.linalg.norm(predict_saved_joint(model,s,'cpu',online=reader),axis=-1).mean(-1).sum() for s in streams)/19
            loader=make_loader(streams,config,torch.device('cpu'),False)
            validation_display=StringIO()
            with redirect_stderr(validation_display):
                report=evaluate(model,loader,torch.device('cpu'),config)
            self.assertIn('19/19',validation_display.getvalue())
            self.assertIn('100%',validation_display.getvalue())
            self.assertEqual(report['all']['count'],19)
            self.assertEqual(report['cold']['count'],11)
            self.assertAlmostEqual(report['all']['decoded']['mpjpe'],float(expected),places=6)
            model.set_encoder_stage(True)
            training_display=StringIO()
            with redirect_stderr(training_display):
                training=train_epoch(model,loader,torch.optim.AdamW(model.parameters(),lr=1e-4),torch.device('cpu'),config)
            self.assertIn('19/19',training_display.getvalue())
            self.assertIn('100%',training_display.getvalue())
            self.assertIn('mpjpe_cm',training_display.getvalue())
            self.assertEqual(training['train_frames'],19)
            self.assertEqual(training['optimizer_steps'],len(steps))
            self.assertGreater(training['gradient_norms']['transformer'],0)
            self.assertGreater(training['gradient_norms']['point_encoder'],0)
            config['num_workers']=2
            self.assertEqual([b['valid'].sum().item() for b in make_loader(streams,config,torch.device('cpu'),False)],[10,8,1])


if __name__=='__main__':
    unittest.main()
