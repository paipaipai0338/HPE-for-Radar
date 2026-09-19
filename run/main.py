import argparse
from pathlib import Path
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess.radarprocess import Radar_Config

from run.utils.write_log import write_log
from run.utils.load_config import load_config
from run.utils.set_seed import set_seed
from run.utils.set_device import set_device
from run.utils.build_model import build_model
from run.utils.model_init import model_init
from run.utils.build_metric import Metric
from run.utils.build_experiment import build_experiment, save_radar_config
from run.utils.process_one_epoch import train_one_epoch, val_one_epoch, prepare_bin_input, get_autocast_dtype, center_cropped_pointcloud
from run.utils.checkpoint import save_checkpoint, load_training_checkpoint, load_model_checkpoint, load_init_checkpoint
from run.utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup

# from data2datasets.dataset import HPE_Dataset, collate_fn as dataset_collate_fn
# from data2datasets.dataset_for_single import HPE_Dataset, collate_fn as dataset_collate_fn
# from data2datasets.dataset_for_detection import HPE_Dataset, collate_fn as dataset_collate_fn
from data2datasets.dataset_for_all_task import HPE_Dataset, collate_fn as dataset_collate_fn

# nohup /home/pai/miniconda3/envs/pytorch/bin/python /home/pai/Huawei/run/main.py &

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/home/pai/Huawei/run/config.yaml',)
    return parser.parse_args()


def main():
    # 加载 cfg
    args = parse_args()
    cfg = load_config(args.config)
    precision = str(cfg['task']['precision']).upper()
    get_autocast_dtype(precision)
    cfg['task']['precision'] = precision
    # 判断模型重复值是否相等
    model_config_path = (Path(__file__).resolve().parents[1] / 'models' / cfg['model']['name'] / 'model_config.yaml')
    cfg_model_arch = load_config(model_config_path)
    if 'max_people' in cfg_model_arch:
        assert cfg['data']['max_people'] == cfg_model_arch['max_people'], 'max_people mismatch: ' f'data={cfg["data"]["max_people"]}, model={cfg_model_arch["max_people"]}'

    if 'xyz_limits' in cfg_model_arch:
        assert cfg['data']['xyz_limits'] == cfg_model_arch['xyz_limits'], 'xyz_limits mismatch: ' f'data={cfg["data"]["xyz_limits"]}, model={cfg_model_arch["xyz_limits"]}'

    if 'map_size' in cfg_model_arch:
        assert cfg['data']['map_size'] == cfg_model_arch['map_size'], 'map_size mismatch: ' f'data={cfg["data"]["map_size"]}, model={cfg_model_arch["map_size"]}'

    if 'cube_size' in cfg_model_arch:
        assert cfg['data']['cube_size'] == cfg_model_arch['cube_size'], 'cube_size mismatch: ' f'data={cfg["data"]["cube_size"]}, model={cfg_model_arch["cube_size"]}'

    # 固定随机种子
    set_seed(cfg['task']['seed'])

    # 获取device
    device = set_device(cfg['task']['device'])

    # 配置雷达config
    radar_config = Radar_Config()
    for k, v in cfg['radar'].items():
        if hasattr(radar_config, k):
            setattr(radar_config, k, v)
        else:
            raise ValueError(f"Radar_Config 不存在字段: {k}")
    radar_config.__post_init__()

    # 获取模型
    model = build_model(cfg['model']['name'])
    model = model.to(device)
    model = model_init(model)

    # train
    if cfg['task']['stage'] == 'train':
        init_checkpoint = cfg['model'].get('init_checkpoint')
        if init_checkpoint and not cfg['task']['train']['resume']['enabled']:
            load_init_checkpoint(init_checkpoint, model)

        # 获取dataloader
        dataset = {
            'train': HPE_Dataset(root_path=cfg['data']['root_path'], sensor_config=cfg['data']['sensor_config'], mode='train', base_source=cfg['data']['base_source'], split_method=cfg['data']['split_method'], ratio=cfg['data']['ratio'], T=cfg['data']['T'], acc_frame=cfg['data']['acc_frame'], preload_cache=cfg['data']['preload_cache'], enable_action=cfg['data']['enable_action'], enable_rotation=cfg['data']['enable_rotation_train'], radar_config=radar_config, packed_data_root=cfg['data']['packed_data_root']),
            'val': HPE_Dataset(root_path=cfg['data']['root_path'], sensor_config=cfg['data']['sensor_config'], mode='val', base_source=cfg['data']['base_source'], split_method=cfg['data']['split_method'], ratio=cfg['data']['ratio'], T=cfg['data']['T'], acc_frame=cfg['data']['acc_frame'], preload_cache=cfg['data']['preload_cache'], enable_action=cfg['data']['enable_action'], enable_rotation=cfg['data']['enable_rotation_val'], radar_config=radar_config, packed_data_root=cfg['data']['packed_data_root']),
        }
        for split in ('train', 'val'):
            if len(dataset[split]) == 0:
                raise ValueError(
                    f"{split} 数据集为空，无法训练或验证。请检查 data.root_path、"
                    "data description.json、split_method 对应的人员/组划分，"
                    "以及数据过滤后是否仍有有效样本。"
                )
        collate_fn = partial(dataset_collate_fn, max_points=cfg['data']['max_points'], max_people=cfg['data']['max_people'])
        dataloader = {
            'train': DataLoader(dataset['train'], batch_size=cfg['task']['batch_size'], collate_fn=collate_fn, shuffle=cfg['task']['train']['shuffle'], num_workers=cfg['data']['num_workers'], pin_memory=True, persistent_workers=True, prefetch_factor=2),
            'val': DataLoader(dataset['val'], batch_size=cfg['task']['batch_size'], collate_fn=collate_fn, shuffle=cfg['task']['val']['shuffle'], num_workers=cfg['data']['num_workers'], pin_memory=True, persistent_workers=True, prefetch_factor=2)
        }
        # 指标构建
        metric = {
            'train': Metric(
                cfg['task']['train']['metrics'],
                cfg['data']['xyz_limits'],
                cfg['task']['matching_for_hungarian']['bbox_l1_weight'],
                cfg['task']['matching_for_hungarian']['bbox_iou_weight'],
                pose_matching_by_hip=cfg['task']['matching_for_hungarian']['pose_matching_by_hip'],
                pose_confidence_weight=cfg['task']['matching_for_hungarian']['pose_confidence_weight'],
            ),
            'val': Metric(
                cfg['task']['val']['metrics'],
                cfg['data']['xyz_limits'],
                cfg['task']['matching_for_hungarian']['bbox_l1_weight'],
                cfg['task']['matching_for_hungarian']['bbox_iou_weight'],
                pose_matching_by_hip=cfg['task']['matching_for_hungarian']['pose_matching_by_hip'],
                pose_confidence_weight=cfg['task']['matching_for_hungarian']['pose_confidence_weight'],
            ),
        }
        best_metric_name = cfg['task']['train']['best_metric'].lower()
        if best_metric_name != 'loss':
            raise ValueError(
                "best_metric 必须设置为 loss，"
                f"当前为: {cfg['task']['train']['best_metric']}"
            )

        # 优化器与学习率调度
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=cfg['task']['train']['init_lr'],
            betas=(0.9, 0.999)
        )
        scaler = (
            torch.amp.GradScaler(device.type)
            if precision == 'FP16'
            else None
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_epochs=cfg['task']['train']['epoch'],
            warmup_epoch=cfg['task']['train']['warmup_epoch'],
            min_lr=1e-10,
        )

        # retraining checkpoint, metric, start_epoch, best_metric 加载
        if cfg['task']['train']['resume']['enabled']:
            checkpoint_path = Path(cfg['task']['train']['resume']['checkpoint_path'])
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
            start_epoch, best_metric = load_training_checkpoint(checkpoint_path, model, optimizer, scheduler, metric, device, scaler)
        else:
            start_epoch = 0
            best_metric = float('inf')
            paths = build_experiment(
                output_root=cfg['experiment']['output_path'],
                model_name=cfg['experiment'].get('group', cfg['model']['name']),
                source_config_path=args.config,
                model=model,
            )

        save_radar_config(paths['config'], radar_config)

        # 记录日志
        log_path = paths['log'] / 'log.txt'
        fig_path = paths['fig'] / 'fig.png'
        write_log(log_path, "=" * 80)

        if cfg['task']['train']['resume']['enabled']:
            write_log(log_path, f"Resume training from: {checkpoint_path}")
            write_log(log_path, f"Start epoch: {start_epoch + 1}")
        else:
            write_log(log_path, "Start new training")

        write_log(log_path, f"Experiment description: {cfg['experiment']['description']}")
        write_log(log_path, f"Precision: {precision}")
        write_log(log_path, f"Best metric: {best_metric_name}: {best_metric}")

        for epoch in range(start_epoch, cfg['task']['train']['epoch']):
            epoch_lr = optimizer.param_groups[0]['lr']
            train_metrics, metric['train'] = train_one_epoch(
                model,
                dataloader['train'],
                optimizer,
                metric['train'],
                device,
                cfg['data'],
                cfg['task'],
                cfg['model'],
                radar_config,
                scaler,
            )
            
            scheduler.step()

            val_metrics, metric['val'] = val_one_epoch(
                model,
                dataloader['val'],
                metric['val'],
                device,
                cfg['data'],
                cfg['task'],
                cfg['model'],
                radar_config,
            )
            val_metrics['loss'] = sum(
                weight * val_metrics.get(name, 0.0)
                for name, weight in metric['val'].cfg_metrics.items()
            )

            message = (
                f"Epoch {epoch + 1}/{cfg['task']['train']['epoch']} | "
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
                save_checkpoint(paths['checkpoint'] / 'best.pth', epoch, model, optimizer, scheduler, metric, best_metric, scaler)
                message = (
                        f"Best checkpoint updated | "
                        f"{best_metric_name}: "
                        f"{previous_best:.6f} -> "
                        f"{best_metric:.6f} | "
                        f"path={paths['checkpoint'] / 'best.pth'}"
                    )
                write_log(log_path, message)
                print(message)
            save_checkpoint(paths['checkpoint'] / 'last.pth', epoch, model, optimizer, scheduler, metric, best_metric, scaler)


    elif cfg['task']['stage'] == 'val':
        # 只做结果保存 后续分析见 /home/pai/Huawei/run/check.py

        # 获取dataloader
        dataset = {
            'val': HPE_Dataset(root_path=cfg['data']['root_path'], sensor_config=cfg['data']['sensor_config'], mode='val', base_source=cfg['data']['base_source'], split_method=cfg['data']['split_method'], ratio=cfg['data']['ratio'], T=cfg['data']['T'], acc_frame=cfg['data']['acc_frame'], preload_cache=cfg['data']['preload_cache'], enable_action=cfg['data']['enable_action'], enable_rotation=cfg['data']['enable_rotation_val'], radar_config=radar_config, packed_data_root=cfg['data']['packed_data_root']),
        }
        collate_fn = partial(dataset_collate_fn, max_points=cfg['data']['max_points'], max_people=cfg['data']['max_people'])
        dataloader = {
            'val': DataLoader(dataset['val'], batch_size=cfg['task']['batch_size'], collate_fn=collate_fn, shuffle=cfg['task']['val']['shuffle'], num_workers=cfg['data']['num_workers'], pin_memory=True, persistent_workers=True, prefetch_factor=2)
        }

        # best/last checkpoint 加载 
        load_model_checkpoint(cfg['task']['val']['checkpoint_path'], model, device)

        pose_pre = []
        confidence_pre = []
        pose_gt = []
        gt_valid = []
        pc = []
        pc_valid = []
        bin_inputs = []
        high_to_low_R = []
        high_to_low_t = []
        bbox_pre = []
        objectness_logits = []
        bbox_gt = []
        action_logits = []
        action_gt = []

        model.eval()
        with torch.no_grad():
            for samples in tqdm(dataloader['val'], total=len(dataloader['val'])):
                model_input = {}
                if 'pc' in cfg['task']['input']:
                    model_input['input'] = samples[cfg['task']['input']]['padded'].to(device, non_blocking=True)
                    model_input['mask'] = samples[cfg['task']['input']]['mask'].to(device, non_blocking=True)

                    # wrapper 将dataset取出的多人按照 mask 进行筛选，有效 mask 则按照bbox筛选点云，无效略过；将多人维度合并到batch中构建全新的batch
                    person_mask = samples[cfg['task']['output']]['mask'].to(device, non_blocking=True)
                    person_bbox = samples[cfg['task']['output']]['bbox'].to(device, non_blocking=True)

                    points = model_input['input']
                    point_mask = model_input['mask']
                    B, T, N, D = points.shape
                    K = person_mask.shape[2]

                    # [B,T,K,6] -> [B,K,T,6]，为每个人生成独立点云实例。
                    bbox = person_bbox.permute(0, 2, 1, 3)
                    min_xyz = bbox[..., :3].unsqueeze(3)
                    max_xyz = bbox[..., 3:].unsqueeze(3)
                    xyz = points[:, None, :, :, :3]
                    inside_bbox = ((xyz >= min_xyz) & (xyz <= max_xyz)).all(dim=-1)

                    person_frame_mask = person_mask.permute(0, 2, 1)
                    cropped_mask = (
                        inside_bbox
                        & point_mask[:, None, :, :]
                        & person_frame_mask.unsqueeze(-1)
                    )
                    cropped_points = points[:, None, :, :, :].expand(B, K, T, N, D)
                    # 仅保留 T 帧内至少一帧存在的人员，并将 B、K 合并为新 batch。
                    valid_instance_mask = person_frame_mask.any(dim=2)
                    if not valid_instance_mask.any():
                        continue

                    if cfg['task']['center_on_pointcloud']:
                        cropped_points, pointcloud_center = center_cropped_pointcloud(
                            cropped_points, cropped_mask
                        )
                    cropped_points = cropped_points.masked_fill(~cropped_mask.unsqueeze(-1), 0.0)
                    pc_for_save = cropped_points.permute(0, 2, 1, 3, 4).contiguous()
                    pc_valid_for_save = cropped_mask.permute(0, 2, 1, 3).contiguous()
                    model_input['input'] = cropped_points[valid_instance_mask].contiguous()
                    model_input['mask'] = cropped_mask[valid_instance_mask].contiguous()
                else:
                    model_input['input'] = prepare_bin_input(
                        samples,
                        cfg['task']['input'],
                        device,
                        model,
                        cfg['data'],
                        cfg['model'],
                        radar_config,
                    )
                    bin_for_save = model_input['input']

                gt = {
                    'padded': samples[cfg['task']['output']]['padded'].to(device, non_blocking=True),
                    'mask': samples[cfg['task']['output']]['mask'].to(device, non_blocking=True),
                    'bbox': samples[cfg['task']['output']]['bbox'].to(device, non_blocking=True),
                }
                if cfg['task']['center_on_pointcloud'] and 'pc' in cfg['task']['input']:
                    gt['padded'] = (gt['padded'] - pointcloud_center.unsqueeze(-2)).masked_fill(
                        ~gt['mask'][..., None, None], 0.0
                    )
                    gt['bbox'] = (gt['bbox'] - pointcloud_center.repeat(1, 1, 1, 2)).masked_fill(
                        ~gt['mask'].unsqueeze(-1), 0.0
                    )
                if 'action' in samples[cfg['task']['output']]:
                    gt['action'] = samples[cfg['task']['output']]['action'].to(
                        device, non_blocking=True
                    )
                model_input['gt'] = gt

                pre = model(model_input)

                if 'pc' in cfg['task']['input']:
                    instance_pose = pre['pose']
                    if instance_pose.shape[2] != 1:
                        raise ValueError(
                            '按人裁剪后的 pose 模型必须为每个实例只输出一个人，'
                            f'实际 shape={tuple(instance_pose.shape)}'
                        )
                    instance_pose = instance_pose.squeeze(2)
                    pose = instance_pose.new_zeros(B, K, T, instance_pose.shape[2], instance_pose.shape[3])
                    pose[valid_instance_mask] = instance_pose
                    pre['pose'] = pose.permute(0, 2, 1, 3, 4).contiguous()
                if 'pc' in cfg['task']['input']:
                    pc.append(pc_for_save.detach().cpu())
                    pc_valid.append(pc_valid_for_save.detach().cpu())
                else:
                    bin_inputs.append(bin_for_save.detach().cpu())

                pose = pre.get('pose')
                if pose is not None:
                    pose_pre.append(pose.detach().cpu())

                confidence = pre.get('confidence')
                if confidence is not None:
                    confidence_pre.append(confidence.detach().cpu())

                bbox = pre.get('bbox')
                if bbox is not None:
                    bbox_pre.append(bbox.detach().cpu())

                logits = pre.get('objectness_logits')
                if logits is not None:
                    objectness_logits.append(logits.detach().cpu())

                batch_action_logits = pre.get('action_logits')
                if batch_action_logits is not None:
                    action_logits.append(
                        batch_action_logits.detach().cpu()
                    )

                pose_gt.append(gt['padded'].detach().cpu())
                bbox_gt.append(gt['bbox'].detach().cpu())
                gt_valid.append(gt['mask'].detach().cpu())
                if gt.get('action') is not None:
                    action_gt.append(gt['action'].detach().cpu())

                transform_R = samples.get('high_to_low_R')
                if transform_R is not None:
                    high_to_low_R.append(transform_R.detach().cpu())

                transform_t = samples.get('high_to_low_t')
                if transform_t is not None:
                    high_to_low_t.append(transform_t.detach().cpu())

        pc = torch.concatenate(pc, dim=0) if pc else None
        pc_valid = (
            torch.concatenate(pc_valid, dim=0)
            if pc_valid else None
        )
        bin_inputs = (
            torch.concatenate(bin_inputs, dim=0)
            if bin_inputs else None
        )
        pose_pre = torch.concatenate(pose_pre, dim=0) if pose_pre else None
        confidence_pre = (torch.concatenate(confidence_pre, dim=0) if confidence_pre else None)
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
        action_logits = (
            torch.concatenate(action_logits, dim=0)
            if action_logits else None
        )
        action_gt = (
            torch.concatenate(action_gt, dim=0)
            if action_gt else None
        )

        results = {
            'input_key': cfg['task']['input'],
            'target_key': cfg['task']['output'],
            'pose_pre': pose_pre,
            'confidence_pre': confidence_pre,
            'bbox_pre': bbox_pre,
            'objectness_logits': objectness_logits,
            'action_logits': action_logits,
            'pose_gt': pose_gt,
            'bbox_gt': bbox_gt,
            'action_gt': action_gt,
            'action_label': getattr(
                dataset['val'],
                'action_label',
                None,
            ),
            'gt_valid': gt_valid,
            'high_to_low_R': high_to_low_R,
            'high_to_low_t': high_to_low_t,
        }
        if 'pc' in cfg['task']['input']:
            results['pc'] = pc
            results['pc_valid'] = pc_valid
        else:
            results['bin'] = bin_inputs
        torch.save(results, '/home/pai/Huawei/run/result.pkl')
    else:
        raise ValueError(f"cfg['task']['stage'] dismatched, got {cfg['task']['stage']}")


if __name__ == "__main__":
    main()
