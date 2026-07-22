import argparse
from pathlib import Path
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from run.utils.write_log import write_log
from run.utils.load_config import load_config
from run.utils.set_seed import set_seed
from run.utils.set_device import set_device
from run.utils.build_model import build_model
from run.utils.model_init import model_init
from run.utils.build_metric import Metric
from run.utils.build_experiment import build_experiment
from run.utils.process_one_epoch import train_one_epoch, val_one_epoch
from run.utils.checkpoint import save_checkpoint, load_training_checkpoint, load_model_checkpoint
from run.utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup

# from data2datasets.dataset import HPE_Dataset
from data2datasets.dataset_for_single import HPE_Dataset
from data2datasets.utils import collate_pc_gt_fn

# nohup /home/pai/miniconda3/envs/pytorch/bin/python /home/pai/Huawei/run/main.py &

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default='/home/pai/Huawei/run/config.yaml',
    )
    return parser.parse_args()


def main():
    # 加载 cfg
    args = parse_args()
    cfg = load_config(args.config)
    cfg_experiment = cfg['experiment']
    cfg_data = cfg['data']
    cfg_model = cfg['model']
    cfg_task = cfg['task']

    # 固定随机种子
    set_seed(cfg_task['seed'])

    # 获取device
    device_id = cfg_task['device']
    device = set_device(device_id)

    # 获取dataloader
    dataset = {
        'train': HPE_Dataset(root_path=cfg_data['root_path'], sensor_config=cfg_data['sensor_config'], mode='train', base_source=cfg_data['base_source'], split_method=cfg_data['split_method'], ratio=cfg_data['ratio'], T=cfg_data['T'], preload_cache=cfg_data.get('preload_cache', False)),
        'val': HPE_Dataset(root_path=cfg_data['root_path'], sensor_config=cfg_data['sensor_config'], mode='val', base_source=cfg_data['base_source'], split_method=cfg_data['split_method'], ratio=cfg_data['ratio'], T=cfg_data['T'], preload_cache=cfg_data.get('preload_cache', False)),
    }
    collate_fn = partial(collate_pc_gt_fn, max_points=cfg_data['max_points'], max_people=cfg_data['max_people'])
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=cfg_task['batch_size'], collate_fn=collate_fn, shuffle=cfg_task['train']['shuffle'], num_workers=cfg_data['num_workers'], pin_memory=True, persistent_workers=True, prefetch_factor=2),
        'val': DataLoader(dataset['val'], batch_size=cfg_task['batch_size'], collate_fn=collate_fn, shuffle=cfg_task['val']['shuffle'], num_workers=cfg_data['num_workers'], pin_memory=True, persistent_workers=True, prefetch_factor=2)
    }

    # 获取模型
    model = build_model(cfg_model['name'])
    model = model.to(device)
    model = model_init(model)

    # train
    if cfg_task['stage'] == 'train':

        # 指标构建
        metric = {
            'train': Metric(cfg_task['train']['metrics']),
            'val':  Metric(cfg_task['val']['metrics'])
        }
        best_metric_name = cfg_task['train']['best_metric'].lower()
        if best_metric_name not in metric['val'].cfg_metrics:
            raise ValueError(f"best_metric 不在 val metrics 中: {cfg_task['train']['best_metric']}")

        # 优化器与学习率调度
        num_epoch = cfg_task['train']['epoch']
        warmup_epoch = cfg_task['train']['warmup_epoch']
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=cfg_task['train']['init_lr'],
            betas=(0.9, 0.999)
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_epochs=num_epoch,
            warmup_epoch=warmup_epoch,
            min_lr=1e-10,
        )

        # retraining checkpoint, metric, start_epoch, best_metric 加载
        if cfg_task['train']['resume']['enabled']:
            checkpoint_path = Path(cfg_task['train']['resume']['checkpoint_path'])
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
            experiment_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == 'checkpoint' else checkpoint_path.parent
            paths = {
                'root': experiment_dir,
                'checkpoint': experiment_dir / 'checkpoint',
                'log': experiment_dir / 'log',
                'fig': experiment_dir / 'fig',
                'config': experiment_dir / 'config',
            }
            start_epoch, best_metric = load_training_checkpoint(checkpoint_path, model, optimizer, scheduler, metric, device)
        else:
            start_epoch = 0
            best_metric = float('inf')
            paths = build_experiment(
                output_root=cfg_experiment['output_path'],
                model_name=cfg_model['name'],
                source_config_path=args.config,
                model=model,
            )

        # 记录日志
        log_path = paths['log'] / 'log.txt'
        fig_path = paths['fig'] / 'fig.png'
        write_log(log_path, "=" * 80)

        if cfg_task['train']['resume']['enabled']:
            write_log(log_path, f"Resume training from: {checkpoint_path}")
            write_log(log_path, f"Start epoch: {start_epoch + 1}")
        else:
            write_log(log_path, "Start new training")

        write_log(log_path, f"Experiment description: {cfg_experiment['description']}")
        write_log(log_path, f"Best metric: {best_metric_name}: {best_metric}")

        for epoch in range(start_epoch, cfg_task['train']['epoch']):
            epoch_lr = optimizer.param_groups[0]['lr']
            train_metrics, metric['train'] = train_one_epoch(
                model,
                dataloader['train'],
                optimizer,
                metric['train'],
                device,
                cfg_task,
            )
            
            scheduler.step()

            val_metrics, metric['val'] = val_one_epoch(
                model,
                dataloader['val'],
                metric['val'],
                device,
                cfg_task,
                if_plot=(epoch == 0),
                fig_path=fig_path,
            )

            message = (
                f"Epoch {epoch + 1}/{num_epoch} | "
                f"lr={epoch_lr:.10f} | "
                f"train={train_metrics} | "
                f"val={val_metrics}"
            )

            write_log(log_path, message)
            print(message)

            current = val_metrics[best_metric_name]
            if current < best_metric:
                previous_best = best_metric
                best_metric = current
                save_checkpoint(paths['checkpoint'] / 'best.pth', epoch, model, optimizer, scheduler, metric, best_metric)
                message = (
                        f"Best checkpoint updated | "
                        f"{best_metric_name}: "
                        f"{previous_best:.6f} -> "
                        f"{best_metric:.6f} | "
                        f"path={paths['checkpoint'] / 'best.pth'}"
                    )
                write_log(log_path, message)
                print(message)
            save_checkpoint(paths['checkpoint'] / 'last.pth', epoch, model, optimizer, scheduler, metric, best_metric)

        
    elif cfg_task['stage'] == 'val':
        metric = Metric(cfg_task['val']['metrics'])
        # best/last checkpoint 加载 
        load_model_checkpoint(cfg_task['val']['checkpoint_path'], model, device)

        pose_pre = []
        pose_gt = []
        pose_confidence = []
        gt_valid = []
        pc = []
        pc_valid = []
        high_to_low_R = []
        high_to_low_t = []

        model.eval()
        with torch.no_grad():
            for samples in tqdm(dataloader['val'], total=len(dataloader['val'])):
                input_key = cfg_task['input']
                model_input = {}
                if 'pc' in input_key:
                    model_input['input'] = samples[input_key]['padded'].to(device, non_blocking=True)
                    model_input['mask'] = samples[input_key]['mask'].to(device, non_blocking=True)
                else:
                    model_input['input'] = samples[input_key].to(device, non_blocking=True)

                target_key = cfg_task['output']
                gt = {
                    'padded': samples[target_key]['padded'].to(device, non_blocking=True),
                    'mask': samples[target_key]['mask'].to(device, non_blocking=True),
                }

                pre = model(model_input)

                pc.append(model_input['input'].detach().cpu())
                pc_valid.append(model_input['mask'].detach().cpu())
                pose_pre.append(pre['pose'].detach().cpu())
                pose_confidence.append(pre['confidence'].detach().cpu())
                pose_gt.append(gt['padded'].detach().cpu())
                gt_valid.append(gt['mask'].detach().cpu())
                high_to_low_R.append(samples['high_to_low_R'].detach().cpu())
                high_to_low_t.append(samples['high_to_low_t'].detach().cpu())
        pc = torch.concatenate(pc, dim=0)
        pc_valid = torch.concatenate(pc_valid, dim=0)
        pose_pre = torch.concatenate(pose_pre, dim=0)
        pose_confidence = torch.concatenate(pose_confidence, dim=0)
        pose_gt = torch.concatenate(pose_gt, dim=0)
        gt_valid = torch.concatenate(gt_valid, dim=0)
        high_to_low_R = torch.concatenate(high_to_low_R, dim=0)
        high_to_low_t = torch.concatenate(high_to_low_t, dim=0)

        results = {
            'pc': pc,
            'pc_valid': pc_valid,
            'pose_pre': pose_pre,
            'pose_confidence': pose_confidence,
            'pose_gt': pose_gt,
            'gt_valid': gt_valid,
            'high_to_low_R': high_to_low_R,
            'high_to_low_t': high_to_low_t,
        }
        torch.save(results, '/home/pai/Huawei/temp/result.pkl')

        from metrics.pose import get_bce, get_detection_metric, get_mpjpe, get_pampjpe
        ratio = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        bce = get_bce(pose_confidence, gt_valid).mean()
        mpjpe = get_mpjpe(pose_pre, pose_gt, type='coco')
        pampjpe = get_pampjpe(pose_pre, pose_gt, type='coco')
        gt_mask = gt_valid.bool()
        gt_num = int(gt_mask.sum().item())
        if gt_num > 0:
            mpjpe_gt = mpjpe.masked_select(gt_mask).mean().item()
            pampjpe_gt = pampjpe.masked_select(gt_mask).mean().item()
        else:
            mpjpe_gt = float('nan')
            pampjpe_gt = float('nan')

        print(f"val bce: {bce.item():.6f}")
        print(f"val mpjpe(gt): {mpjpe_gt:.6f}")
        print(f"val pampjpe(gt): {pampjpe_gt:.6f}")
        print("ratio | acc | precision | recall | f1 | mpjpe@success | pampjpe@success | success_num")
        for r in ratio + [1.0]:
            acc, precision, recall, f1 = get_detection_metric(pose_confidence, gt_valid, ratio=r)
            success_mask = (pose_confidence >= r) & gt_valid.bool()
            success_num = int(success_mask.sum().item())

            if success_num > 0:
                mpjpe_success = mpjpe.masked_select(success_mask).mean().item()
                pampjpe_success = pampjpe.masked_select(success_mask).mean().item()
                mpjpe_text = f"{mpjpe_success:.6f}"
                pampjpe_text = f"{pampjpe_success:.6f}"
            else:
                mpjpe_text = "nan"
                pampjpe_text = "nan"

            print(
                f"{r:.1f} | "
                f"{acc.item():.6f} | "
                f"{precision.item():.6f} | "
                f"{recall.item():.6f} | "
                f"{f1.item():.6f} | "
                f"{mpjpe_text} | "
                f"{pampjpe_text} | "
                f"{success_num}"
            )

        '''temp 测试使用'''
        from matplotlib import pyplot as plt
        from utils.COCO import COCO_SKELETON
        frame_mask = gt_valid.sum(dim=-1).bool()
        pc_frame = pc[frame_mask]
        pc_valid_frame = pc_valid[frame_mask]
        pose_gt_frame = pose_gt[frame_mask]
        pose_pre_frame = pose_pre[frame_mask]
        gt_valid_frame = gt_valid[frame_mask]
        mpjpe_frame = mpjpe[frame_mask]
        pose_confidence_frame = pose_confidence[frame_mask]
        high_to_low_R = high_to_low_R[frame_mask]
        high_to_low_t = high_to_low_t[frame_mask]

        for k in range(pose_gt_frame.shape[0]):
            pc_xyz = pc_frame[k][pc_valid_frame[k]][:, :3]
            pc_xyz = pc_xyz[torch.isfinite(pc_xyz).all(dim=-1)]
            person_mask = gt_valid_frame[k].bool()
            pose_gt_valid = pose_gt_frame[k][person_mask]
            pose_pre_valid = pose_pre_frame[k][person_mask]
            mpjpe_valid = mpjpe_frame[k][person_mask]
            pose_confidence_valid = pose_confidence_frame[k][person_mask]
            high_to_low_R_valid = high_to_low_R[k]
            high_to_low_t_valid = high_to_low_t[k]

            pc_xyz = (high_to_low_R_valid @ pc_xyz.T + high_to_low_t_valid.reshape(3, 1)).T

            fig = plt.figure()
            ax = plt.subplot(111, projection='3d')
            if pc_xyz.shape[0] > 0:
                ax.scatter(
                    pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], s=2,
                    c='green', label='PC'
                )

            for person_idx in range(pose_gt_valid.shape[0]):
                gt_person = pose_gt_valid[person_idx]
                pre_person = pose_pre_valid[person_idx]

                gt_person = (high_to_low_R_valid @ gt_person.T + high_to_low_t_valid.reshape(3, 1)).T
                pre_person = (high_to_low_R_valid @ pre_person.T + high_to_low_t_valid.reshape(3, 1)).T

                gt_label = 'GT' if person_idx == 0 else None
                pre_label = 'PRE' if person_idx == 0 else None

                ax.scatter(
                    gt_person[:, 0], gt_person[:, 1], gt_person[:, 2], s=5,
                    c='red', label=gt_label
                )
                for joint_a, joint_b in COCO_SKELETON:
                    ax.plot(
                        [gt_person[joint_a, 0], gt_person[joint_b, 0]],
                        [gt_person[joint_a, 1], gt_person[joint_b, 1]],
                        [gt_person[joint_a, 2], gt_person[joint_b, 2]],
                        color='red',
                        linewidth=1.5,
                    )

                ax.scatter(
                    pre_person[:, 0], pre_person[:, 1], pre_person[:, 2], s=5,
                    c='blue', label=pre_label
                )
                for joint_a, joint_b in COCO_SKELETON:
                    ax.plot(
                        [pre_person[joint_a, 0], pre_person[joint_b, 0]],
                        [pre_person[joint_a, 1], pre_person[joint_b, 1]],
                        [pre_person[joint_a, 2], pre_person[joint_b, 2]],
                        color='blue',
                        linewidth=1.5,
                    )

            ax.set_xlim(0.0, 6.0)
            ax.set_ylim(-3.0, 3.0)
            ax.set_zlim(-3.0, 3.0)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            mpjpe_text = ', '.join([f'{value.item():.4f}' for value in mpjpe_valid])
            confidence_text = ', '.join([f'{value.item():.4f}' for value in pose_confidence_valid])
            ax.set_title(f"Frame:{k} People:{pose_gt_valid.shape[0]}\nMPJPE:{mpjpe_text}\nConfidence:{confidence_text}")
            ax.legend()

            fig.tight_layout()
            fig.savefig('/home/pai/Huawei/temp.png', dpi=400)
            plt.show(block=False)
            plt.pause(0.1)
            plt.close(fig)
    else:
        raise ValueError(f"cfg_task['stage'] dismatched, got {cfg_task['stage']}")


if __name__ == "__main__":
    main()
