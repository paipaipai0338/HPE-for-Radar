import argparse
from pathlib import Path
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

from preprocess.radarprocess import Radar_Config
from preprocess.gtprocess import get_gt_boxes

from run.utils.write_log import write_log
from run.utils.load_config import load_config
from run.utils.set_seed import set_seed
from run.utils.set_device import set_device
from run.utils.build_model import build_model
from run.utils.model_init import model_init
from run.utils.build_metric import Metric
from run.utils.build_experiment import build_experiment, save_radar_config
from run.utils.process_one_epoch import train_one_epoch, val_one_epoch, prepare_bin_input, get_autocast_dtype, prepare_person_pointcloud, validate_center_xy_in_low, validate_coordinate_frame, adapt_coordinate_frame
from run.utils.process_one_epoch import predict_valid_person_frames
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


def get_train_windows(dataset, dense_window_start_epoch, epoch):
    if dense_window_start_epoch is None:
        return dataset
    if type(dense_window_start_epoch) is not int or dense_window_start_epoch < 1:
        raise ValueError('dense_window_start_epoch 必须为从 1 开始的正整数')
    if dataset.window_stride != 1 or dataset.sky_history or dataset.spectra_history:
        raise ValueError('分阶段滑窗要求基础步长为 1，且没有额外 BIN 历史帧')
    if epoch + 1 >= dense_window_start_epoch:
        return dataset
    stride = max(1, dataset.T // 2)
    return Subset(dataset, [i for i, start in enumerate(dataset.sky_start_indices) if start % stride == 0])


def main():
    # 加载 cfg
    args = parse_args()
    cfg = load_config(args.config)
    validate_coordinate_frame(cfg['task'], cfg['data'])
    validate_center_xy_in_low(cfg['task'], cfg['data'])
    precision = str(cfg['task']['precision']).upper()
    get_autocast_dtype(precision)
    cfg['task']['precision'] = precision
    # 判断模型重复值是否相等
    model_config_path = (Path(__file__).resolve().parents[1] / 'models' / cfg['model']['name'] / 'model_config.yaml')
    cfg_model_arch = load_config(model_config_path)
    cfg_model_arch.update(cfg['model'].get('options', {}))
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

    if cfg['data'].get('pointcloud_source', 'files') == 'skyrelay':
        if cfg['task']['input'] != 'radar_high_pc' or cfg['task']['output'] != 'gt_for_high':
            raise ValueError('Sky 点云基线要求 task.input=radar_high_pc / output=gt_for_high')
        if cfg['model']['name'] != 'P4Transformer':
            raise ValueError('Sky 点云基线当前使用 P4Transformer')

    if cfg['data'].get('include_kxyz', False) and cfg['model']['name'] != 'ResNet3D':
        raise ValueError('include_kxyz 当前仅支持 ResNet3D')
    if cfg['model']['name'] == 'ResNet3D':
        channels = radar_config.num_chirp + (6 if cfg['data'].get('include_kxyz', False) else 0)
        if cfg_model_arch['inchannels'] != channels:
            raise ValueError(f'ResNet3D inchannels 应为 {channels}（D + 可选 6 组 K）')

    # 获取模型
    model = build_model(cfg['model']['name'], **cfg['model'].get('options', {}))
    model = model.to(device)
    model = model_init(model)

    # train
    if cfg['task']['stage'] == 'train':
        init_checkpoint = cfg['model'].get('init_checkpoint')
        if init_checkpoint and not cfg['task']['train']['resume']['enabled']:
            load_init_checkpoint(init_checkpoint, model)

        if not cfg['task']['train']['resume']['enabled']:
            paths = build_experiment(
                output_root=cfg['experiment']['output_path'],
                model_name=cfg['experiment'].get('group', cfg['model']['name']),
                source_config_path=args.config,
                model=model,
                model_config=cfg_model_arch,
            )
            save_radar_config(paths['config'], radar_config)

        # 获取dataloader
        dataset = {
            'train': HPE_Dataset(root_path=cfg['data']['root_path'], sensor_config=cfg['data']['sensor_config'], mode='train', base_source=cfg['data']['base_source'], split_method=cfg['data']['split_method'], ratio=cfg['data']['ratio'], T=cfg['data']['T'], acc_frame=cfg['data']['acc_frame'], preload_cache=cfg['data']['preload_cache'], enable_action=cfg['data']['enable_action'], radar_config=radar_config, packed_data_root=cfg['data']['packed_data_root'], pointcloud_source=cfg['data'].get('pointcloud_source', 'files'), include_kxyz=cfg['data'].get('include_kxyz', False), val_person_id=cfg['data'].get('val_person_id', '4'), window_stride=cfg['data'].get('window_stride'), train_groups=cfg['data'].get('train_groups')),
            'val': HPE_Dataset(root_path=cfg['data']['root_path'], sensor_config=cfg['data']['sensor_config'], mode='val', base_source=cfg['data']['base_source'], split_method=cfg['data']['split_method'], ratio=cfg['data']['ratio'], T=cfg['data']['T'], acc_frame=cfg['data']['acc_frame'], preload_cache=cfg['data']['preload_cache'], enable_action=cfg['data']['enable_action'], radar_config=radar_config, packed_data_root=cfg['data']['packed_data_root'], pointcloud_source=cfg['data'].get('pointcloud_source', 'files'), include_kxyz=cfg['data'].get('include_kxyz', False), val_person_id=cfg['data'].get('val_person_id', '4'), window_stride=cfg['data'].get('window_stride')),
        }
        for split in ('train', 'val'):
            if len(dataset[split]) == 0:
                raise ValueError(
                    f"{split} 数据集为空，无法训练或验证。请检查 data.root_path、"
                    "data description.json、split_method 对应的人员/组划分，"
                    "以及数据过滤后是否仍有有效样本。"
                )
        collate_fn = partial(dataset_collate_fn, max_people=cfg['data']['max_people'])
        steps_per_epoch = cfg['task']['train'].get('steps_per_epoch')
        if steps_per_epoch is not None:
            if type(steps_per_epoch) is not int or steps_per_epoch <= 0:
                raise ValueError("task.train.steps_per_epoch 必须是正整数")
        dense_window_start_epoch = cfg['task']['train'].get('dense_window_start_epoch')

        def build_train_loader(epoch):
            windows = get_train_windows(dataset['train'], dense_window_start_epoch, epoch)
            if not len(windows):
                raise ValueError('当前滑窗步长下训练集为空')
            train_sampler = None if steps_per_epoch is None else RandomSampler(
                windows, replacement=True, num_samples=steps_per_epoch * cfg['task']['batch_size'],
            )
            return DataLoader(windows, batch_size=cfg['task']['batch_size'], collate_fn=collate_fn,
                              shuffle=cfg['task']['train']['shuffle'] if train_sampler is None else False,
                              sampler=train_sampler, num_workers=cfg['data']['num_workers'],
                              pin_memory=True, persistent_workers=True, prefetch_factor=2)

        dataloader = {
            'train': build_train_loader(0),
            'val': DataLoader(dataset['val'], batch_size=cfg['task']['batch_size'], collate_fn=collate_fn, shuffle=cfg['task']['val']['shuffle'], num_workers=cfg['data']['num_workers'], pin_memory=True, persistent_workers=True, prefetch_factor=2)
        }
        # 指标构建
        metric = {
            'train': Metric(
                cfg['task']['train']['metrics'],
                cfg['data']['xyz_limits'],
                cfg['task'].get('matching_for_hungarian', {}).get('bbox_l1_weight', 5.0),
                cfg['task'].get('matching_for_hungarian', {}).get('bbox_iou_weight', 2.0),
                pose_matching_by_hip=cfg['task'].get('matching_for_hungarian', {}).get('pose_matching_by_hip', False),
                pose_confidence_weight=cfg['task'].get('matching_for_hungarian', {}).get('pose_confidence_weight', 0.0),
            ),
            'val': Metric(
                cfg['task']['val']['metrics'],
                cfg['data']['xyz_limits'],
                cfg['task'].get('matching_for_hungarian', {}).get('bbox_l1_weight', 5.0),
                cfg['task'].get('matching_for_hungarian', {}).get('bbox_iou_weight', 2.0),
                pose_matching_by_hip=cfg['task'].get('matching_for_hungarian', {}).get('pose_matching_by_hip', False),
                pose_confidence_weight=cfg['task'].get('matching_for_hungarian', {}).get('pose_confidence_weight', 0.0),
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
            min_lr=1e-20,
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
            if cfg['task']['train']['resume'].get('reset_scheduler', False):
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] = cfg['task']['train']['init_lr']
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_epochs=cfg['task']['train']['epoch'] - start_epoch,
                    warmup_epoch=cfg['task']['train']['warmup_epoch'],
                    min_lr=1e-20,
                )
        else:
            start_epoch = 0
            best_metric = float('inf')

        save_radar_config(paths['config'], radar_config)

        # 记录日志
        log_path = paths['log'] / 'log.txt'
        group_log_path = paths['log'] / 'group_metrics.txt'
        fig_path = paths['fig'] / 'fig.png'
        write_log(log_path, "=" * 80)

        if cfg['task']['train']['resume']['enabled']:
            write_log(log_path, f"Resume training from: {checkpoint_path}")
            write_log(log_path, f"Start epoch: {start_epoch + 1}")
            if cfg['task']['train']['resume'].get('reset_scheduler', False):
                write_log(log_path, f"Reset scheduler: lr={optimizer.param_groups[0]['lr']}, warmup_epoch={cfg['task']['train']['warmup_epoch']}, remaining_epochs={cfg['task']['train']['epoch'] - start_epoch}")
        else:
            write_log(log_path, "Start new training")

        write_log(log_path, f"Experiment description: {cfg['experiment']['description']}")
        write_log(log_path, f"Precision: {precision}")
        if 'pc' in cfg['task']['input']:
            write_log(log_path, 'Val policy: no ellipse restriction; '
                      'accumulated GT-crop points >= 10 before sampling; continuous valid frames only')
        write_log(log_path, f"Best metric: {best_metric_name}: {best_metric}")
        write_log(log_path, f"Batches per epoch: train={len(dataloader['train'])}, val={len(dataloader['val'])}")
        if dense_window_start_epoch is not None:
            write_log(log_path, f'Train window stride: {max(1, dataset["train"].T // 2)} before epoch '
                      f'{dense_window_start_epoch}, then 1; val={dataset["val"].window_stride}')

        for epoch in range(start_epoch, cfg['task']['train']['epoch']):
            if (dense_window_start_epoch is not None and epoch + 1 >= dense_window_start_epoch
                    and isinstance(dataloader['train'].dataset, Subset)):
                del dataloader['train']
                dataloader['train'] = build_train_loader(epoch)
                write_log(log_path, f'Epoch {epoch + 1}: train window stride switched to 1; '
                          f'batches={len(dataloader["train"])}')
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
            train_labels = train_metrics.pop('mpjpe_by_label', {})
            val_labels = val_metrics.pop('mpjpe_by_label', {})
            train_metrics.pop('mpjpe_by_group', None)
            val_groups = val_metrics.pop('mpjpe_by_group', {})
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

            write_log(group_log_path, f"Epoch {epoch + 1}/{cfg['task']['train']['epoch']}")
            write_log(group_log_path, f"MPJPE by label (m) | train={train_labels} | val={val_labels}")
            for group, values in val_groups.items():
                inside_mpjpe = 'N/A' if values['mpjpe_inside'] is None else f"{values['mpjpe_inside']:.6f}"
                inside_centered = 'N/A' if values['centered_mpjpe_inside'] is None else f"{values['centered_mpjpe_inside']:.6f}"
                write_log(group_log_path, f"Val group {group} | label={values['label']} | "
                          f"MPJPE={values['mpjpe']:.6f} m / {inside_mpjpe} m | "
                          f"centered MPJPE={values['centered_mpjpe']:.6f} m / {inside_centered} m | count={values['count']}")

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
            'val': HPE_Dataset(root_path=cfg['data']['root_path'], sensor_config=cfg['data']['sensor_config'], mode='val', base_source=cfg['data']['base_source'], split_method=cfg['data']['split_method'], ratio=cfg['data']['ratio'], T=cfg['data']['T'], acc_frame=cfg['data']['acc_frame'], preload_cache=cfg['data']['preload_cache'], enable_action=cfg['data']['enable_action'], radar_config=radar_config, packed_data_root=cfg['data']['packed_data_root'], pointcloud_source=cfg['data'].get('pointcloud_source', 'files'), include_kxyz=cfg['data'].get('include_kxyz', False), val_person_id=cfg['data'].get('val_person_id', '4'), window_stride=cfg['data'].get('window_stride')),
        }
        collate_fn = partial(dataset_collate_fn, max_people=cfg['data']['max_people'])
        dataloader = {
            'val': DataLoader(dataset['val'], batch_size=cfg['task']['batch_size'], collate_fn=collate_fn, shuffle=cfg['task']['val']['shuffle'], num_workers=cfg['data']['num_workers'], pin_memory=True, persistent_workers=True, prefetch_factor=2)
        }

        # best/last checkpoint 加载 
        load_model_checkpoint(cfg['task']['val']['checkpoint_path'], model, device)

        pose_pre = []
        pose_valid = []
        confidence_pre = []
        pose_gt = []
        gt_valid = []
        pc = []
        pc_valid = []
        bin_inputs = []
        pointcloud_centers = []
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
                samples = adapt_coordinate_frame(samples, cfg['task'], cfg['data'], 'val')
                model_input = {}
                if 'pc' in cfg['task']['input']:
                    model_input, gt, valid_instance_mask, cropped_points, cropped_mask, pointcloud_center = prepare_person_pointcloud(
                        samples, cfg['task'], cfg['data'], device, 'val')
                    if not valid_instance_mask.any():
                        continue
                    B, T, K = gt['mask'].shape
                    pc_for_save = cropped_points.permute(0, 2, 1, 3, 4).contiguous()
                    pc_valid_for_save = cropped_mask.permute(0, 2, 1, 3).contiguous()
                    if cfg['task'].get('center_xy_in_low', False) or cfg['task'].get('center_on_pointcloud', False):
                        pointcloud_centers.append(pointcloud_center.cpu())
                else:
                    model_input['input'] = prepare_bin_input(samples, cfg['task']['input'], device, model,
                                                           cfg['data'], cfg['model'], radar_config)
                    bin_for_save = model_input['input']
                    gt = {key: samples[cfg['task']['output']][key].to(device)
                          for key in ('padded', 'mask', 'bbox')}

                if cfg['model']['name'] == 'ResNet3D':
                    model_input.update(bbox=gt['bbox'], bbox_mask=gt['mask'])

                if 'action' in samples[cfg['task']['output']]:
                    gt['action'] = samples[cfg['task']['output']]['action'].to(
                        device, non_blocking=True
                    )
                model_input['gt'] = gt

                pre = (predict_valid_person_frames(model, model_input)
                       if 'pc' in cfg['task']['input'] else model(model_input))

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
                    pre['pose_valid'] = gt['mask'].clone()
                if 'pc' in cfg['task']['input']:
                    pc.append(pc_for_save.detach().cpu())
                    pc_valid.append(pc_valid_for_save.detach().cpu())
                else:
                    bin_inputs.append(bin_for_save.detach().cpu())

                pose = pre.get('pose')
                if pose is not None:
                    pose_pre.append(pose.detach().cpu())
                    if 'pose_valid' in pre:
                        pose_valid.append(pre['pose_valid'].detach().cpu())

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

        if pc:
            max_scene_points = max(cloud.shape[-2] for cloud in pc)
            pc = [torch.nn.functional.pad(cloud, (0, 0, 0, max_scene_points - cloud.shape[-2])) for cloud in pc]
            pc_valid = [torch.nn.functional.pad(mask, (0, max_scene_points - mask.shape[-1])) for mask in pc_valid]
        pc = torch.concatenate(pc, dim=0) if pc else None
        pc_valid = torch.concatenate(pc_valid, dim=0) if pc_valid else None
        bin_inputs = (
            torch.concatenate(bin_inputs, dim=0)
            if bin_inputs else None
        )
        pose_pre = torch.concatenate(pose_pre, dim=0) if pose_pre else None
        confidence_pre = (torch.concatenate(confidence_pre, dim=0) if confidence_pre else None)
        if not pose_gt:
            raise ValueError('验证集没有满足区域及点数条件的样本，无法导出')
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
            'pose_valid': torch.cat(pose_valid) if pose_valid else None,
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
        results['coordinate_frame'] = cfg['task']['coordinate']
        if pointcloud_centers:
            results['coordinate_frame'] = ('low_xy_centered' if cfg['task'].get('center_xy_in_low', False)
                                           else cfg['task']['coordinate'] + '_centered')
            results['pointcloud_center'] = torch.concatenate(pointcloud_centers, dim=0)
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
