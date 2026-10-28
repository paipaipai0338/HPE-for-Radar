import torch
from tqdm import tqdm

from preprocess.gtprocess import get_gt_boxes
from preprocess.radar_spectra import range_cube_to_training_spectra
from preprocess.radarprocess_RPM2 import range_cube_to_rpm2_maps
from preprocess.radarpreprocess_HRRadarPose import range_cube_to_range_doppler_azi_ele, range_doppler_azi_ele_to_doppler_xyz

def validate_coordinate_frame(cfg_task, cfg_data):
    if cfg_task.get('coordinate', 'high') not in ('high', 'low'):
        raise ValueError('coordinate 必须为 high 或 low')
    count = cfg_data.get('person_max_points', 200)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('person_max_points 必须为正整数')
    augmentation = cfg_data.get('augmentation', {})
    for name, keys in (('radar_rotation', ('roll_range_deg', 'pitch_range_deg', 'yaw_range_deg')),
                       ('radar_height', ('delta_height_range_m',))):
        config = augmentation.get(name, {})
        for key in keys:
            limits = torch.as_tensor(config.get(key, [0., 0.]), dtype=torch.float32)
            if limits.shape != (2,) or not torch.isfinite(limits).all() or limits[0] > limits[1]:
                raise ValueError(f'{name}.{key} 必须为有限的 [min,max]')
    drop = augmentation.get('point_dropout', {})
    probability = drop.get('probability', 0.)
    if not 0 <= probability <= 1:
        raise ValueError('point_dropout.probability 必须在 [0,1]')
    if cfg_task.get('center_on_pointcloud', False) and cfg_task.get('center_xy_in_low', False):
        raise ValueError('两种中心化不能同时开启')
    height = augmentation.get('radar_height', {})
    if cfg_task.get('center_on_pointcloud', False) and any(height.get(f'enabled_{stage}', False) for stage in ('train', 'val')):
        raise ValueError('XYZ 中心化会抵消高度增广，不能同时开启')
    enabled_augmentation = any(augmentation.get(name, {}).get(f'enabled_{stage}', False)
                               for name in ('radar_rotation', 'radar_height', 'point_dropout')
                               for stage in ('train', 'val'))
    if enabled_augmentation or cfg_task.get('coordinate', 'high') == 'low':
        if cfg_task['input'] != 'radar_high_pc' or cfg_task['output'] != 'gt_for_high':
            raise ValueError('坐标转换和点云增广要求 radar_high_pc / gt_for_high')


def validate_center_xy_in_low(cfg_task, cfg_data):
    if cfg_task.get('center_xy_in_low', False) and cfg_task.get('coordinate', 'high') != 'low':
        raise ValueError('center_xy_in_low 要求 coordinate: low')


@torch.no_grad()
def adapt_coordinate_frame(samples, cfg_task, cfg_data=None, stage='val', device=None):
    """共享场景变换：高度在原始 low 中施加，角度在 high 中施加。"""
    if 'pc' not in cfg_task['input']:
        return samples
    augmentation = (cfg_data or {}).get('augmentation', {})
    cloud, target = samples[cfg_task['input']], samples[cfg_task['output']]
    device = cloud['padded'].device if device is None else device
    rotation = samples['high_to_low_R'].to(device, non_blocking=True)
    translation = samples['high_to_low_t'].to(device, non_blocking=True)
    point_mask = cloud['mask'].to(device, non_blocking=True)
    person_mask = target['mask'].to(device, non_blocking=True)
    batch_size = cloud['padded'].shape[0]
    height = augmentation.get('radar_height', {})
    angles = augmentation.get('radar_rotation', {})
    matrices = torch.eye(3).repeat(batch_size, 1, 1)
    if angles.get(f'enabled_{stage}', False):
        from data2datasets.dataset_for_all_task import HPE_Dataset
        for index in range(batch_size):
            values = [torch.empty(()).uniform_(*angles[key]).item()
                      for key in ('roll_range_deg', 'pitch_range_deg', 'yaw_range_deg')]
            matrices[index] = torch.from_numpy(HPE_Dataset._rotation_matrix_from_euler_deg(*values))
    delta = torch.zeros(batch_size, 1, 1, 3)
    if height.get(f'enabled_{stage}', False):
        delta[..., 2].uniform_(*height['delta_height_range_m'])

    matrices = matrices.to(device, non_blocking=True)[:, None].transpose(-1, -2)
    delta = delta.to(device, non_blocking=True)
    # 合成原有坐标与增广变换，整批点云只做一次矩阵乘法。
    linear = rotation.transpose(-1, -2) @ rotation @ matrices
    offset = (-delta @ rotation) @ matrices
    if cfg_task.get('coordinate', 'high') == 'low':
        linear = linear @ rotation.transpose(-1, -2)
        offset = offset @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)

    def transform(xyz, mask):
        original_shape = xyz.shape
        xyz = xyz.reshape(batch_size, xyz.shape[1], -1, 3)
        xyz = xyz @ linear + offset
        return xyz.reshape(original_shape).masked_fill(~mask, 0.)

    points = cloud['padded'].to(device, non_blocking=True).clone()
    points[..., :3] = transform(points[..., :3], point_mask[..., None])
    pose = transform(target['padded'].to(device, non_blocking=True), person_mask[..., None, None])
    bbox = get_gt_boxes(pose, person_mask, threshold=0.3).masked_fill(~person_mask[..., None], 0.)
    return {**samples, cfg_task['input']: {**cloud, 'padded': points, 'mask': point_mask},
            cfg_task['output']: {**target, 'padded': pose, 'mask': person_mask, 'bbox': bbox}}


def resample_cropped_pointcloud(points, mask, max_points=200, random=True):
    """固定点数；空帧保留零输入，不借用其他帧。"""
    if not points.numel():
        return (points.new_zeros(*points.shape[:-2], max_points, points.shape[-1]),
                mask.new_zeros(*mask.shape[:-1], max_points))
    counts = mask.sum(-1, keepdim=True)
    slots = torch.arange(max_points, device=points.device)
    if random:
        scores = torch.rand(mask.shape, device=points.device).masked_fill(~mask, float('inf'))
        order = scores.topk(min(max_points, points.shape[-2]), dim=-1, largest=False).indices
        repeats = (torch.rand(*mask.shape[:-1], max_points, device=points.device) * counts).long()
        ranks = torch.where(slots < counts, slots, repeats)
        indices = order.gather(-1, ranks)
        # 稀疏帧先保留全部有效点，再补点并打乱顺序。
        shuffle = torch.rand(indices.shape, device=points.device).argsort(-1)
        indices = indices.gather(-1, shuffle)
    else:
        positions = torch.arange(points.shape[-2], device=points.device).expand_as(mask)
        ranks = (mask.cumsum(-1) - 1).clamp_min(0)
        order = torch.zeros_like(ranks).scatter_add_(-1, ranks, positions * mask)
        ranks = torch.where(counts >= max_points,
                            torch.div(slots * (counts - 1).clamp_min(0), max(1, max_points - 1),
                                      rounding_mode='floor'),
                            slots % counts.clamp_min(1))
        indices = order.gather(-1, ranks)
    output_mask = (counts > 0).expand(*mask.shape[:-1], max_points).clone()
    output = points.gather(-2, indices[..., None].expand(*indices.shape, points.shape[-1]))
    return output.masked_fill(~output_mask[..., None], 0.), output_mask


def center_cropped_pointcloud(points, mask):
    xyz = points[..., :3].masked_fill(~mask.unsqueeze(-1), 0.)
    center = xyz.sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)
    centered = torch.cat((points[..., :3] - center.unsqueeze(-2), points[..., 3:]), -1)
    return centered.masked_fill(~mask.unsqueeze(-1), 0.), center.permute(0, 2, 1, 3)


@torch.no_grad()
def prepare_person_pointcloud(samples, cfg_task, cfg_data, device, stage):
    """训练、验证和导出共用裁剪、中心化、采样与逐点丢弃。"""
    cloud, target = samples[cfg_task['input']], samples[cfg_task['output']]
    points, point_mask = cloud['padded'].to(device), cloud['mask'].to(device)
    pose, person_mask, bbox = (target[key].to(device) for key in ('padded', 'mask', 'bbox'))
    bounds = bbox.permute(0, 2, 1, 3)
    xyz = points[:, None, ..., :3]
    cropped_mask = ((xyz >= bounds[..., :3].unsqueeze(-2)) &
                    (xyz <= bounds[..., 3:].unsqueeze(-2))).all(-1)
    cropped_mask &= point_mask[:, None] & person_mask.permute(0, 2, 1)[..., None]
    if stage == 'val':
        person_mask = person_mask & (cropped_mask.sum(-1).permute(0, 2, 1) >= 10)
        cropped_mask &= person_mask.permute(0, 2, 1)[..., None]
    valid_instance_mask = cropped_mask.any(dim=(2, 3))
    cropped_points = points[:, None].expand(-1, person_mask.shape[2], -1, -1, -1)
    center = points.new_zeros(*person_mask.shape, 3)
    if cfg_task.get('center_on_pointcloud', False) or cfg_task.get('center_xy_in_low', False):
        _, center = center_cropped_pointcloud(cropped_points, cropped_mask)
        if cfg_task.get('center_xy_in_low', False):
            center[..., 2] = 0.
        cropped_points = torch.cat((cropped_points[..., :3] - center.permute(0, 2, 1, 3).unsqueeze(-2),
                                    cropped_points[..., 3:]), -1)
        pose = (pose - center.unsqueeze(-2)).masked_fill(~person_mask[..., None, None], 0.)
    cropped_points = cropped_points.masked_fill(~cropped_mask[..., None], 0.)
    sampled, sampled_mask = resample_cropped_pointcloud(
        cropped_points[valid_instance_mask], cropped_mask[valid_instance_mask],
        cfg_data['person_max_points'], random=stage == 'train')
    dropout = cfg_data.get('augmentation', {}).get('point_dropout', {})
    if dropout.get(f'enabled_{stage}', False):
        sampled_mask &= torch.rand(sampled_mask.shape, device=device) >= dropout['probability']
        sampled = sampled.masked_fill(~sampled_mask[..., None], 0.)
    gt = {'padded': pose, 'mask': person_mask,
          'bbox': get_gt_boxes(pose, person_mask, threshold=0.3).masked_fill(~person_mask[..., None], 0.)}
    return ({'input': sampled, 'mask': sampled_mask}, gt, valid_instance_mask,
            cropped_points, cropped_mask, center)


@torch.no_grad()
def predict_valid_person_frames(model, model_input):
    """仅对连续有效帧推理；不跨越被筛除的帧拼接时序。"""
    valid = model_input['mask'].any(-1)
    if not valid.any():
        raise ValueError('没有可推理的有效人帧')
    full = valid.all(-1)
    prediction = None

    def infer(indices, begin, end):
        nonlocal prediction
        inputs = {key: model_input[key][indices, begin:end].contiguous()
                  for key in ('input', 'mask')}
        pose = model(inputs)['pose']
        if prediction is None:
            prediction = pose.new_zeros(*valid.shape, *pose.shape[2:])
        prediction[indices, begin:end] = pose

    if full.any():
        infer(full, 0, valid.shape[1])
    for index in (~full).nonzero().flatten().tolist():
        flags = torch.cat((valid.new_zeros(1), valid[index], valid.new_zeros(1))).int()
        changes = (flags[1:] - flags[:-1]).nonzero().flatten().tolist()
        for begin, end in zip(changes[::2], changes[1::2]):
            infer(slice(index, index + 1), begin, end)
    return {'pose': prediction}


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
    if cfg_model['name'] == 'ResNet3D':
        indices = samples['bin_frame_index'] if cfg_data.get('include_kxyz', False) else None
        return range_cube_to_training_spectra(
            radar_input, radar_config, cfg_data['xyz_limits'], cfg_data['cube_size'],
            cfg_data.get('angle_bins', [128,32]), cfg_data['remove_static'], indices)
    if cfg_model['name'] == 'RPM2':
        radar_power = range_cube_to_rpm2_maps(
            range_cube=radar_input,
            radar_config=radar_config,
            xyz_limits=cfg_data['xyz_limits'],
            map_size=cfg_data['map_size'],
            remove_static=cfg_data['remove_static'],
        )
    elif cfg_model['name'] == 'HRRadarPose':
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
            num_azimuth_bins=cfg_data.get('angle_bins', [128, 128])[0],
            num_elevation_bins=cfg_data.get('angle_bins', [128, 128])[1],
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
    validate_coordinate_frame(cfg_task, cfg_data)
    validate_center_xy_in_low(cfg_task, cfg_data)
    model.train()
    autocast_dtype = get_autocast_dtype(cfg_task.get('precision', 'FP32'))
    for samples in tqdm(dataloader, total=len(dataloader)):
        samples = adapt_coordinate_frame(samples, cfg_task, cfg_data, 'train', device=device)
        # 获取模型输入
        input_key = cfg_task['input']
        target_key = cfg_task['output']
        model_input = {}
        if 'pc' in input_key:
            model_input, gt, valid_instance_mask, cropped_points, cropped_mask, pointcloud_center = prepare_person_pointcloud(
                samples, cfg_task, cfg_data, device, 'train')
            if not valid_instance_mask.any():
                continue
            B, T, K = gt['mask'].shape
        else:
            model_input['input'] = prepare_bin_input(samples, input_key, device, model, cfg_data, cfg_model, radar_config)
            gt = {key: samples[target_key][source].to(device, non_blocking=True)
                  for key, source in (('padded', 'padded'), ('mask', 'mask'), ('bbox', 'bbox'))}

        if cfg_model['name'] == 'ResNet3D':
            model_input.update(bbox=gt['bbox'], bbox_mask=gt['mask'])

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
            pre['pose_valid'] = valid_instance_mask[:, None].expand(B, T, K).contiguous()

        loss, _ = metric.calculate_batch(pre, gt, group_ids=samples.get('group_id'),
                                            group_gt_low=samples['gt_for_low']['padded'])

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f'训练 loss 出现 NaN 或 Inf: {loss.item()}'
            )

        if cfg_model['name'] == 'ResNet3D' and loss.detach().item() > 1.0:
            # Preserve difficult samples; record evidence for transient numerical spikes.
            valid = gt['mask'].bool()
            print({'event': 'resnet3d_large_train_loss', 'loss_m': loss.detach().item(),
                   'groups': samples.get('group_id'),
                   'input_max_abs': model_input['input'].abs().max().item(),
                   'input_max_abs_by_sample': model_input['input'].abs().flatten(1).amax(dim=1).tolist(),
                   'gt_max_abs': gt['padded'][valid].abs().max().item() if valid.any() else 0.,
                   'prediction_max_abs': pre['pose'][valid].abs().max().item() if valid.any() else 0.},
                  flush=True)

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
    validate_coordinate_frame(cfg_task, cfg_data)
    validate_center_xy_in_low(cfg_task, cfg_data)
    model.eval()
    autocast_dtype = get_autocast_dtype(cfg_task.get('precision', 'FP32'))
    evaluated_batches = 0
    with torch.no_grad():
        for samples in tqdm(dataloader, total=len(dataloader)):
            samples = adapt_coordinate_frame(samples, cfg_task, cfg_data, 'val', device=device)
            input_key = cfg_task['input']
            target_key = cfg_task['output']
            model_input = {}
            if 'pc' in input_key:
                model_input, gt, valid_instance_mask, cropped_points, cropped_mask, pointcloud_center = prepare_person_pointcloud(
                    samples, cfg_task, cfg_data, device, 'val')
                if not valid_instance_mask.any():
                    continue
                B, T, K = gt['mask'].shape
            else:
                model_input['input'] = prepare_bin_input(samples, input_key, device, model, cfg_data, cfg_model, radar_config)
                gt = {key: samples[target_key][source].to(device, non_blocking=True)
                      for key, source in (('padded', 'padded'), ('mask', 'mask'), ('bbox', 'bbox'))}

            if cfg_model['name'] == 'ResNet3D':
                model_input.update(bbox=gt['bbox'], bbox_mask=gt['mask'])

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
                pre = (predict_valid_person_frames(model, model_input)
                       if 'pc' in input_key else model(model_input))


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
                pre['pose_valid'] = gt['mask'].clone()

            loss, _ = metric.calculate_batch(pre, gt, group_ids=samples.get('group_id'),
                                            group_gt_low=samples['gt_for_low']['padded'])
            evaluated_batches += 1

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'训练 loss 出现 NaN 或 Inf: {loss.item()}'
                )
        
        if not evaluated_batches:
            raise ValueError('验证集没有满足区域及点数条件的样本')
        epoch_metric = metric.epoch_end()
    return epoch_metric, metric
