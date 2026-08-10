import torch
from tqdm import tqdm

from metrics.detection import get_hungarian_match

def train_one_epoch(model, dataloader, optimizer, metric, device, cfg_task):
    model.train()

    for samples in tqdm(dataloader, total=len(dataloader)):
        # 获取模型输入
        input_key = cfg_task['input']
        model_input = {}
        if 'pc' in input_key:
            model_input['input'] = samples[input_key]['padded'].to(device, non_blocking=True)
            model_input['mask'] = samples[input_key]['mask'].to(device, non_blocking=True)

            # wrapper 将dataset取出的多人按照 mask 进行筛选，有效 mask 则按照bbox筛选点云，无效略过；将多人维度合并到batch中构建全新的batch
            person_mask = samples[cfg_task['output']]['mask'].to(device, non_blocking=True)
            person_bbox = samples[cfg_task['output']]['bbox'].to(device, non_blocking=True)

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
            cropped_points = points[:, None, :, :, :].expand(
                B, K, T, N, D
            ).masked_fill(~cropped_mask.unsqueeze(-1), 0.0)

            # 仅保留 T 帧内至少一帧存在的人员，并将 B、K 合并为新 batch。
            valid_instance_mask = person_frame_mask.any(dim=2)
            if not valid_instance_mask.any():
                continue
            model_input['input'] = cropped_points[valid_instance_mask].contiguous()
            model_input['mask'] = cropped_mask[valid_instance_mask].contiguous()

        else:
            # 待后续谱图整理
            model_input['input'] = samples[input_key].to(device, non_blocking=True)

        # 获取监督对象
        target_key = cfg_task['output']
        gt = {
            'padded': samples[target_key]['padded'].to(device, non_blocking=True),
            'mask': samples[target_key]['mask'].to(device, non_blocking=True),
            'bbox': samples[target_key]['bbox'].to(device, non_blocking=True),
            'action': samples[target_key]['action'].to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)

        pre = model(model_input)

        if 'pc' in input_key:
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

        loss, _ = metric.calculate_batch(pre, gt)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f'训练 loss 出现 NaN 或 Inf: {loss.item()}'
            )

        loss.backward()
        optimizer.step()

    epoch_metric = metric.epoch_end()

    return epoch_metric, metric

def val_one_epoch(model, dataloader, metric, device, cfg_task):
    model.eval()
    with torch.no_grad():
        for samples in tqdm(dataloader, total=len(dataloader)):
            input_key = cfg_task['input']
            model_input = {}
            if 'pc' in input_key:
                model_input['input'] = samples[input_key]['padded'].to(device, non_blocking=True)
                model_input['mask'] = samples[input_key]['mask'].to(device, non_blocking=True)

                # wrapper 将dataset取出的多人按照 mask 进行筛选，有效 mask 则按照bbox筛选点云，无效略过；将多人维度合并到batch中构建全新的batch
                person_mask = samples[cfg_task['output']]['mask'].to(device, non_blocking=True)
                person_bbox = samples[cfg_task['output']]['bbox'].to(device, non_blocking=True)

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
                cropped_points = points[:, None, :, :, :].expand(
                    B, K, T, N, D
                ).masked_fill(~cropped_mask.unsqueeze(-1), 0.0)

                # 仅保留 T 帧内至少一帧存在的人员，并将 B、K 合并为新 batch。
                valid_instance_mask = person_frame_mask.any(dim=2)
                if not valid_instance_mask.any():
                    continue
                model_input['input'] = cropped_points[valid_instance_mask].contiguous()
                model_input['mask'] = cropped_mask[valid_instance_mask].contiguous()
            else:
                model_input['input'] = samples[input_key].to(device, non_blocking=True)

            # 获取监督对象
            target_key = cfg_task['output']
            gt = {
                'padded': samples[target_key]['padded'].to(device, non_blocking=True),
                'mask': samples[target_key]['mask'].to(device, non_blocking=True),
                'bbox': samples[target_key]['bbox'].to(device, non_blocking=True),
                'action': samples[target_key]['action'].to(device, non_blocking=True),
            }

            pre = model(model_input)
            if 'pc' in input_key:
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
            loss, _ = metric.calculate_batch(pre, gt)

            if not torch.isfinite(loss):
                raise FloatingPointError(f'验证 loss 出现 NaN 或 Inf: {loss.item()}')
        
        epoch_metric = metric.epoch_end()
    return epoch_metric, metric
