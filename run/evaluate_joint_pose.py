"""Evaluate a joint-finetuned snapshot with main's COCO/MOCAP and ellipse metrics."""
import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from data2datasets.load_json import get_meta_info
from models.P4JointMemory.P4JointMemory import reset_state
from run.train_p4_joint_finetune import encode
from run.utils.p4_finetune_data import load_streams, PointSteps, PointStepSampler, make_raw_dataset, load_saved_joint_model
from run.utils.pose_evaluation import evaluate_pose_formats


class EvaluationSteps(PointSteps):
    def __getitem__(self, requests):
        batch = super().__getitem__(requests)
        batch['stream_ids'] = [r[0] if r is not None else -1 for r in requests]
        return batch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('experiments/P4JointFineTune/20261103_112734'))
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();p=args.experiment
    saved=torch.load(p/'best.pth',map_location='cpu',weights_only=False)
    device=torch.device(args.device);torch.cuda.set_device(device)
    model=load_saved_joint_model(p,saved,device).eval()
    manifest=json.loads((p/'evidence_manifest.json').read_text());streams=load_streams(manifest['val'])
    # Reconstruct the original split rather than silently scoring a changed exclusion list.
    split_path=p/'evaluation_split.json'
    split_path.write_bytes(subprocess.check_output(['git','show','b14289c:data2datasets/data description.json']))
    def original_meta(*unused,**kwargs):return get_meta_info(json_path=split_path,**kwargs)
    with patch('data2datasets.dataset_for_all_task.get_meta_info',side_effect=original_meta):
        raw=make_raw_dataset(streams[0]['online']['training'],'val')
    endpoints={(group,raw._frame_timestamp_ns(raw.data_path_list['radar_high_pc'][i][-1])*1e-9):i
               for i,group in enumerate(raw.sample_groups)}
    for stream in streams:
        stream['window_indices']=torch.tensor([endpoints[(stream['group'],t)] for t in stream['timestamp'].tolist()])
    c=saved['config'];loader=DataLoader(EvaluationSteps(streams,c['chunk_length'],raw),
        sampler=PointStepSampler(streams,c['chunk_length'],c['batch_size'],False),batch_size=None,
        num_workers=c['num_workers'],persistent_workers=c['num_workers']>0,pin_memory=True)
    totals={};state=None;frames=0
    with torch.inference_mode():
        for batch in loader:
            state=reset_state(state,batch['reset'].to(device))
            inputs={k:v.to(device) for k,v in batch.items() if torch.is_tensor(v) and k!='reset'}
            pose,state=model.decoder(encode(model,inputs,device),state)
            valid=inputs['valid'];report=evaluate_pose_formats(pose[valid].cpu(),inputs['target'][valid].cpu())
            for key,row in report.items():
                total=totals.setdefault(key,dict.fromkeys(row,0))
                for name,value in row.items():total[name]+=value
            frames+=int(valid.sum());print('evaluated',frames,flush=True)
    expected=sum(len(s['target']) for s in streams);assert frames==expected
    result={key:dict(frames=row['frames'],**{k:v/row['frames'] if row['frames'] else None
                for k,v in row.items() if k!='frames'}) for key,row in totals.items()}
    history=json.loads((p/'history.json').read_text());logged=next(e for e in history if e['epoch']==saved['epoch'])['all']['decoded']['mpjpe']
    assert abs(result['coco/all']['mpjpe']-logged)<1e-5
    report=dict(checkpoint='best.pth',epoch=saved['epoch'],units='metres',
        protocol='Unique T16 endpoints, causal state, GT crop/association; no keeper or status-based rejection',
        ellipse=dict(center=[2,0],radii=[1.6,2.4],coordinate='low',selection='original GT hip midpoint'),
        mocap='main COCO17-derived 21 joints, not independent motion-capture ground truth',metrics=result)
    path=p/'pose_format_metrics.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
