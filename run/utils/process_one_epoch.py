import torch
from tqdm import tqdm

from preprocess.gtprocess import get_gt_boxes
from preprocess.radarprocess_RPM2 import range_cube_to_rpm2_maps
from preprocess.radarpreprocess_HRRadarPose import range_cube_to_range_doppler_azi_ele, range_doppler_azi_ele_to_doppler_xyz

POSE_MAX_POINTS = 200


def resample_cropped_pointcloud(points, mask, max_points=POSE_MAX_POINTS):
    """将每人每帧的有效点随机采样或重复到固定点数。"""
    output = points.new_empty(*points.shape[:-2], max_points, points.shape[-1])
    output_mask = mask.new_ones(*mask.shape[:-1], max_points)
    for instance_idx in range(points.shape[0]):
        valid_times = torch.nonzero(mask[instance_idx].any(-1), as_tuple=True)[0]
        if not len(valid_times):
            raise ValueError("裁剪后的点云实例没有任何有效点")
        for time_idx in range(points.shape[1]):
            source_time = time_idx
            if not mask[instance_idx, time_idx].any():
                source_time = valid_times[(valid_times - time_idx).abs().argmin()].item()
            valid = points[instance_idx, source_time, mask[instance_idx, source_time]]
            if len(valid) >= max_points:
                indices = torch.randperm(len(valid), device=points.device)[:max_points]
            else:
                indices = torch.cat((
                    torch.arange(len(valid), device=points.device),
                    torch.randint(
                        len(valid), (max_points - len(valid),), device=points.device
                    ),
                ))[torch.randperm(max_points, device=points.device)]
            output[instance_idx, time_idx] = valid[indices]
    return output, output_mask


@torch.no_grad()
def augment_person_windows(points, point_mask, pose, pose_mask, rotation, translation,
                           yaw_deg, shift):
    """每个单人窗口共享变换；rotation/translation 指向原始低位机重力坐标系。

    points: [M,T,N,D], pose: [M,T,J,3], rotation: [M,3,3]。
    yaw_deg: [M], shift: [M,3]；第一帧 GT 无效的实例原样返回。
    """
    eligible = pose_mask[:, 0]
    result_points, result_pose = points.clone(), pose.clone()
    if not eligible.any():
        return result_points, result_pose
    r = rotation[eligible, None]
    t = translation[eligible, None, None]
    low_pose = pose[eligible] @ r.transpose(-1, -2) + t
    center = low_pose[:, 0, [11, 12]].mean(dim=1)[:, None, None]
    angle = torch.deg2rad(yaw_deg[eligible])
    yaw = torch.eye(3, device=points.device, dtype=points.dtype).repeat(len(angle), 1, 1)
    yaw[:, 0, 0] = yaw[:, 1, 1] = angle.cos()
    yaw[:, 0, 1] = -angle.sin()
    yaw[:, 1, 0] = angle.sin()

    def transform(xyz):
        low = xyz @ r.transpose(-1, -2) + t
        low = (low - center) @ yaw[:, None].transpose(-1, -2) + center + shift[eligible, None, None]
        return (low - t) @ r

    result_points[eligible, ..., :3] = transform(points[eligible, ..., :3]).masked_fill(
        ~point_mask[eligible, ..., None], 0.0
    )
    result_pose[eligible] = transform(pose[eligible]).masked_fill(
        ~pose_mask[eligible, ..., None, None], 0.0
    )
    return result_points, result_pose


@torch.no_grad()
def apply_person_geometry_augmentation(model_input, gt, samples, valid_instance_mask,
                                       cfg_data, cfg_task, device, stage):
    """训练与验证共用的增广入口，由各阶段开关分别控制。"""
    geometry = cfg_data.get('person_geometry_augmentation', {})
    if not geometry.get(f'enabled_{stage}', False):
        return
    input_key, target_key = cfg_task['input'], cfg_task['output']
    B, _, K = gt['mask'].shape
    if input_key != 'radar_high_pc' or target_key != 'gt_for_high':
        raise ValueError('人体几何增广目前要求 radar_high_pc 输入和 gt_for_high 监督')
    if cfg_task.get('center_on_pointcloud', False):
        raise ValueError('人体几何增广要求 center_on_pointcloud: false')
    instance_pose = gt['padded'].permute(0, 2, 1, 3, 4)[valid_instance_mask]
    instance_mask = gt['mask'].permute(0, 2, 1)[valid_instance_mask]
    # 每个单人窗口独立采样，参数沿时间维共享。
    count = instance_pose.shape[0]
    yaw = torch.empty(count, device=device).uniform_(*geometry['yaw_range_deg'])
    limits = torch.as_tensor(geometry['translation_ranges'], device=device, dtype=torch.float32)
    if limits.shape != (3, 2) or (limits[:, 1] < limits[:, 0]).any():
        raise ValueError('translation_ranges 必须为三个有效的 [min,max] 范围')
    shift = torch.rand(count, 3, device=device) * (limits[:, 1] - limits[:, 0]) + limits[:, 0]
    rotation = samples['high_to_gravity_R'][:, 0].to(device, non_blocking=True)
    translation = samples['high_to_gravity_t'][:, 0].to(device, non_blocking=True)
    rotation = rotation[:, None].expand(B, K, 3, 3)[valid_instance_mask]
    translation = translation[:, None].expand(B, K, 3)[valid_instance_mask]
    model_input['input'], augmented_pose = augment_person_windows(
        model_input['input'], model_input['mask'], instance_pose, instance_mask,
        rotation, translation, yaw, shift,
    )
    pose_by_person = gt['padded'].permute(0, 2, 1, 3, 4).clone()
    pose_by_person[valid_instance_mask] = augmented_pose
    gt['padded'] = pose_by_person.permute(0, 2, 1, 3, 4).contiguous()
    gt['bbox'] = get_gt_boxes(gt['padded'], gt['mask'], threshold=0.3).masked_fill(
        ~gt['mask'].unsqueeze(-1), 0.0
    )



def center_cropped_pointcloud(points, mask):
    """按每人每帧的有效点 xyz 均值平移；空帧中心为零。

    输入为 [B,K,T,N,D] 和 [B,K,T,N]，返回点云及 [B,T,K,3] 中心。
    """
    xyz = points[..., :3].masked_fill(~mask.unsqueeze(-1), 0.0)
    center = xyz.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
    centered = torch.cat(
        [points[..., :3] - center.unsqueeze(-2), points[..., 3:]], dim=-1
    ).masked_fill(~mask.unsqueeze(-1), 0.0)
    return centered, center.permute(0, 2, 1, 3)


def get_autocast_dtype(precision):
    precision = str(precision).upper()
    try:
        return {
            'FP32': None,
            'FP16': torch.float16,
            'BF16': torch.bfloat16,
        }[precision]
    except KeyError as error:
        raise ValueError(
            f"precision 必须为 FP32、FP16 或 BF16，当前为: {precision}"
        ) from error


@torch.no_grad()
def prepare_bin_input(samples, input_key, device, model, cfg_data, cfg_model, radar_config):
    radar_input = samples[input_key].to(device, non_blocking=True)
    if cfg_model['name'] == 'RPM2':
        radar_power = range_cube_to_rpm2_maps(
            range_cube=radar_input,
            radar_config=radar_config,
            xyz_limits=cfg_data['xyz_limits'],
            map_size=cfg_data['map_size'],
            remove_static=cfg_data['remove_static'],
        )
    elif cfg_model['name'] in ('HRRadarPose', 'ResNet3D'):
        (
            range_doppler_azi_ele,
            range_axis,
            velocity_axis,
            azimuth_axis_rad,
            elevation_axis_rad,
        ) = range_cube_to_range_doppler_azi_ele(
            range_cube=radar_input,
            radar_config=radar_config,
            remove_static=cfg_data['remove_static'],
        )
        doppler_xyz, x_axis, y_axis, z_axis = (
            range_doppler_azi_ele_to_doppler_xyz(
                range_doppler_azi_ele,
                range_axis,
                azimuth_axis_rad,
                elevation_axis_rad,
                xyz_limits=cfg_data['xyz_limits'],
                cube_size=cfg_data['cube_size'],
            )
        )
        radar_power = doppler_xyz
    else:
        raise ValueError(f"不支持 BIN 输入的模型: {cfg_model['name']}")

    return radar_power.clamp_min(torch.finfo(radar_power.dtype).tiny)


def train_one_epoch(model, dataloader, optimizer, metric, device, cfg_data, cfg_task, cfg_model, radar_config, scaler=None):
    model.train()
    autocast_dtype = get_autocast_dtype(cfg_task.get('precision', 'FP32'))
    for samples in tqdm(dataloader, total=len(dataloader)):
        # 获取模型输入
        input_key = cfg_task['input']
        target_key = cfg_task['output']
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
            cropped_points = points[:, None, :, :, :].expand(B, K, T, N, D)
            # 仅保留 T 帧内至少一帧存在的人员，并将 B、K 合并为新 batch。
            valid_instance_mask = person_frame_mask.any(dim=2) & cropped_mask.any(dim=(2, 3))
            if not valid_instance_mask.any():
                continue

            if cfg_task.get('center_on_pointcloud', False):
                cropped_points, pointcloud_center = center_cropped_pointcloud(
                    cropped_points, cropped_mask
                )
            model_input['input'], model_input['mask'] = resample_cropped_pointcloud(
                cropped_points[valid_instance_mask], cropped_mask[valid_instance_mask]
            )
        else:
            model_input['input'] = prepare_bin_input(samples, input_key, device, model, cfg_data, cfg_model, radar_config)

        # 获取监督对象
        gt = {
            'padded': samples[target_key]['padded'].to(device, non_blocking=True),
            'mask': samples[target_key]['mask'].to(device, non_blocking=True),
            'bbox': samples[target_key]['bbox'].to(device, non_blocking=True),
        }
        if cfg_task.get('center_on_pointcloud', False) and 'pc' in input_key:
            gt['padded'] = (gt['padded'] - pointcloud_center.unsqueeze(-2)).masked_fill(
                ~gt['mask'][..., None, None], 0.0
            )
            gt['bbox'] = (gt['bbox'] - pointcloud_center.repeat(1, 1, 1, 2)).masked_fill(
                ~gt['mask'].unsqueeze(-1), 0.0
            )
        if 'pc' in input_key:
            apply_person_geometry_augmentation(
                model_input, gt, samples, valid_instance_mask, cfg_data, cfg_task, device, 'train'
            )

        if 'action' in samples[target_key]:
            gt['action'] = samples[target_key]['action'].to(device, non_blocking=True)
        if cfg_model['name'] == 'RPM2':
            model_input['gt'] = gt
        if cfg_model['name'] == 'HRRadarPose':
            body_center, center_indices, keypoint_offset, target_valid = model.output_encoder(
                pose=gt['padded'], pose_valid=gt['mask'],
                xyz_limits=cfg_data['xyz_limits'], cube_size=cfg_data['cube_size']
                )
            gt['body_center'] = body_center
            gt['indices'] = center_indices
            gt['keypoint_offset'] = keypoint_offset
            gt['hrradarpose_valid'] = target_valid

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
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

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    epoch_metric = metric.epoch_end()

    return epoch_metric, metric

def val_one_epoch(model, dataloader, metric, device, cfg_data, cfg_task, cfg_model, radar_config):
    model.eval()
    autocast_dtype = get_autocast_dtype(cfg_task.get('precision', 'FP32'))
    with torch.no_grad():
        for samples in tqdm(dataloader, total=len(dataloader)):
            input_key = cfg_task['input']
            target_key = cfg_task['output']
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
                cropped_points = points[:, None, :, :, :].expand(B, K, T, N, D)
                # 仅保留 T 帧内至少一帧存在的人员，并将 B、K 合并为新 batch。
                valid_instance_mask = person_frame_mask.any(dim=2) & cropped_mask.any(dim=(2, 3))
                if not valid_instance_mask.any():
                    continue

                if cfg_task.get('center_on_pointcloud', False):
                    cropped_points, pointcloud_center = center_cropped_pointcloud(
                        cropped_points, cropped_mask
                    )
                model_input['input'], model_input['mask'] = resample_cropped_pointcloud(
                    cropped_points[valid_instance_mask], cropped_mask[valid_instance_mask]
                )
            else:
                model_input['input'] = prepare_bin_input(samples, input_key, device, model, cfg_data, cfg_model, radar_config)


            # 获取监督对象
            gt = {
                'padded': samples[target_key]['padded'].to(device, non_blocking=True),
                'mask': samples[target_key]['mask'].to(device, non_blocking=True),
                'bbox': samples[target_key]['bbox'].to(device, non_blocking=True),
            }
            if cfg_task.get('center_on_pointcloud', False) and 'pc' in input_key:
                gt['padded'] = (gt['padded'] - pointcloud_center.unsqueeze(-2)).masked_fill(
                    ~gt['mask'][..., None, None], 0.0
                )
                gt['bbox'] = (gt['bbox'] - pointcloud_center.repeat(1, 1, 1, 2)).masked_fill(
                    ~gt['mask'].unsqueeze(-1), 0.0
                )
            if 'pc' in input_key:
                apply_person_geometry_augmentation(
                    model_input, gt, samples, valid_instance_mask, cfg_data, cfg_task, device, 'val'
                )
            if 'action' in samples[target_key]:
                gt['action'] = samples[target_key]['action'].to(device, non_blocking=True)
            if cfg_model['name'] == 'RPM2':
                model_input['gt'] = gt
            if cfg_model['name'] == 'HRRadarPose':
                body_center, center_indices, keypoint_offset, target_valid = model.output_encoder(
                    pose=gt['padded'], pose_valid=gt['mask'],
                    xyz_limits=cfg_data['xyz_limits'], cube_size=cfg_data['cube_size']
                    )
                gt['body_center'] = body_center
                gt['indices'] = center_indices
                gt['keypoint_offset'] = keypoint_offset
                gt['hrradarpose_valid'] = target_valid

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
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
        
        epoch_metric = metric.epoch_end()
    return epoch_metric, metric
