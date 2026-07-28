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
from data2datasets.utils_data import collate_pc_gt_fn
# from data2datasets.dataset_for_detection import HPE_Dataset, collate_detection_fn

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
    # 判断重复值是否相等
    model_config_path = (
        Path(__file__).resolve().parents[1]
        / 'models'
        / cfg_model['name']
        / 'model_config.yaml'
    )
    cfg_model_arch = load_config(model_config_path)
    if 'max_people' in cfg_model_arch:
        assert cfg_data['max_people'] == cfg_model_arch['max_people'], (
            'max_people mismatch: '
            f'data={cfg_data["max_people"]}, '
            f'model={cfg_model_arch["max_people"]}'
        )
    if 'point_cloud_range' in cfg_model_arch:
        assert (
            cfg_data['point_cloud_range']
            == cfg_model_arch['point_cloud_range']
        ), (
            'point_cloud_range mismatch: '
            f'data={cfg_data["point_cloud_range"]}, '
            f'model={cfg_model_arch["point_cloud_range"]}'
        )

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
    # collate_fn = partial(collate_detection_fn, max_points=cfg_data['max_points'], max_people=cfg_data['max_people'])
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
        cfg_matching = cfg_task['matching_for_hungarian']
        metric = {
            'train': Metric(
                cfg_task['train']['metrics'],
                cfg_data['point_cloud_range'],
                cfg_matching['bbox_l1_weight'],
                cfg_matching['bbox_iou_weight'],
            ),
            'val': Metric(
                cfg_task['val']['metrics'],
                cfg_data['point_cloud_range'],
                cfg_matching['bbox_l1_weight'],
                cfg_matching['bbox_iou_weight'],
            ),
        }
        best_metric_name = cfg_task['train']['best_metric'].lower()
        if best_metric_name != 'loss':
            raise ValueError(
                "best_metric 必须设置为 loss，"
                f"当前为: {cfg_task['train']['best_metric']}"
            )

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
                if_plot=False,
                fig_path=fig_path,
            )
            val_metrics['loss'] = sum(
                weight * val_metrics[name]
                for name, weight in metric['val'].cfg_metrics.items()
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
        # 只做结果保存 后续分析见 /home/pai/Huawei/run/check.py

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
        bbox_pre = []
        objectness_logits = []
        bbox_gt = []

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
                    'bbox': samples[target_key]['bbox'].to(device, non_blocking=True)
                }

                pre = model(model_input)

                pc.append(model_input['input'].detach().cpu())
                pc_valid.append(model_input['mask'].detach().cpu())

                pose = pre.get('pose')
                if pose is not None:
                    pose_pre.append(pose.detach().cpu())

                confidence = pre.get('confidence')
                if confidence is not None:
                    pose_confidence.append(confidence.detach().cpu())

                bbox = pre.get('bbox')
                if bbox is not None:
                    bbox_pre.append(bbox.detach().cpu())

                logits = pre.get('objectness_logits')
                if logits is not None:
                    objectness_logits.append(logits.detach().cpu())

                pose_gt.append(gt['padded'].detach().cpu())
                bbox_gt.append(gt['bbox'].detach().cpu())
                gt_valid.append(gt['mask'].detach().cpu())

                transform_R = samples.get('high_to_low_R')
                if transform_R is not None:
                    high_to_low_R.append(transform_R.detach().cpu())

                transform_t = samples.get('high_to_low_t')
                if transform_t is not None:
                    high_to_low_t.append(transform_t.detach().cpu())

        pc = torch.concatenate(pc, dim=0)
        pc_valid = torch.concatenate(pc_valid, dim=0)
        pose_pre = torch.concatenate(pose_pre, dim=0) if pose_pre else None
        pose_confidence = (
            torch.concatenate(pose_confidence, dim=0)
            if pose_confidence else None
        )
        pose_gt = torch.concatenate(pose_gt, dim=0)
        gt_valid = torch.concatenate(gt_valid, dim=0)
        high_to_low_R = (
            torch.concatenate(high_to_low_R, dim=0)
            if high_to_low_R else None
        )
        high_to_low_t = (
            torch.concatenate(high_to_low_t, dim=0)
            if high_to_low_t else None
        )
        bbox_pre = torch.concatenate(bbox_pre, dim=0) if bbox_pre else None
        objectness_logits = (
            torch.concatenate(objectness_logits, dim=0)
            if objectness_logits else None
        )
        bbox_gt = torch.concatenate(bbox_gt, dim=0)

        results = {
            'pc': pc,
            'pc_valid': pc_valid,
            'pose_pre': pose_pre,
            'bbox_pre': bbox_pre,
            'objectness_logits': objectness_logits,
            'pose_gt': pose_gt,
            'bbox_gt': bbox_gt,
            'gt_valid': gt_valid,
            'high_to_low_R': high_to_low_R,
            'high_to_low_t': high_to_low_t,
        }
        torch.save(results, '/home/pai/Huawei/run/result.pkl')
    else:
        raise ValueError(f"cfg_task['stage'] dismatched, got {cfg_task['stage']}")


if __name__ == "__main__":
    main()
