import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_pc_gt_fn(
    batch,
    max_points: int = 300,
    max_people: int = 4,
):
    """
    Args:
        batch:
            List[Dict[str, List[np.ndarray | torch.Tensor]]]

            每个样本为一个字典，每个键对应长度为 T 的序列。

        max_points:
            雷达点云每帧保留的最大点数。

        max_people:
            GT 每帧保留的最大人数。

    Returns:
        collated:
            radar_low_pc / radar_high_pc:
                padded: [B, T, max_points, 6]
                mask:   [B, T, max_points]

            gt:
                padded: [B, T, max_people, 17, 3]
                mask:   [B, T, max_people]

            其他固定尺寸数据:
                Tensor [B, T, ...]
    """
    if not batch:
        raise ValueError("batch 不能为空")

    if max_points <= 0:
        raise ValueError(
            f"max_points 必须大于 0，当前为 {max_points}"
        )

    if max_people <= 0:
        raise ValueError(
            f"max_people 必须大于 0，当前为 {max_people}"
        )

    variable_config = {
        'radar_low_pc': {
            'max_var': max_points,
            'fixed_dims': (6,),
            'random_sample': True,
        },
        'radar_high_pc': {
            'max_var': max_points,
            'fixed_dims': (6,),
            'random_sample': True,
        },
        'gt': {
            'max_var': max_people,
            'fixed_dims': (17, 3),
            'random_sample': False,
        },
        'gt_for_high': {
            'max_var': max_people,
            'fixed_dims': (17, 3),
            'random_sample': False,
        },
        'gt_for_low': {
            'max_var': max_people,
            'fixed_dims': (17, 3),
            'random_sample': False,
        },
    }

    all_keys = batch[0].keys()
    collated = {}

    # 检查所有样本的键是否一致
    expected_keys = set(all_keys)

    for sample_idx, sample in enumerate(batch):
        if set(sample.keys()) != expected_keys:
            raise ValueError(
                f"batch 中第 {sample_idx} 个样本的键不一致："
                f"expected={expected_keys}, "
                f"actual={set(sample.keys())}"
            )

    B = len(batch)

    for key in all_keys:
        # ============================================================
        # 可变长度数据：点云和多人 GT
        # ============================================================
        if key in variable_config:
            config = variable_config[key]

            max_var = config['max_var']
            fixed_dims = config['fixed_dims']
            random_sample = config['random_sample']

            T = len(batch[0][key])

            # 检查所有样本的时间长度是否一致
            for sample_idx, sample in enumerate(batch):
                if len(sample[key]) != T:
                    raise ValueError(
                        f"{key} 的时间长度不一致："
                        f"sample 0 的 T={T}，"
                        f"sample {sample_idx} 的 "
                        f"T={len(sample[key])}"
                    )

            padded = torch.zeros(
                B,
                T,
                max_var,
                *fixed_dims,
                dtype=torch.float32,
            )

            mask = torch.zeros(
                B,
                T,
                max_var,
                dtype=torch.bool,
            )

            for batch_idx in range(B):
                for time_idx in range(T):
                    arr = batch[batch_idx][key][time_idx]

                    if arr is None:
                        continue

                    tensor = torch.as_tensor(
                        arr,
                        dtype=torch.float32,
                    )

                    if tensor.numel() == 0:
                        continue

                    expected_ndim = 1 + len(fixed_dims)

                    if tensor.ndim != expected_ndim:
                        raise ValueError(
                            f"{key} 数据维数错误："
                            f"batch_idx={batch_idx}, "
                            f"time_idx={time_idx}, "
                            f"expected ndim={expected_ndim}, "
                            f"actual shape={tuple(tensor.shape)}"
                        )

                    if tuple(tensor.shape[1:]) != fixed_dims:
                        raise ValueError(
                            f"{key} 数据形状错误："
                            f"batch_idx={batch_idx}, "
                            f"time_idx={time_idx}, "
                            f"expected [N, {fixed_dims}], "
                            f"actual={tuple(tensor.shape)}"
                        )

                    original_n = tensor.shape[0]
                    valid_n = min(original_n, max_var)

                    if original_n > max_var:
                        if random_sample:
                            # 点云超过 max_points 时随机采样
                            selected_indices = torch.randperm(
                                original_n
                            )[:max_var]

                            tensor = tensor[selected_indices]
                        else:
                            # GT 超过 max_people 时保留前 max_people 个
                            tensor = tensor[:max_var]
                    else:
                        tensor = tensor[:valid_n]

                    padded[
                        batch_idx,
                        time_idx,
                        :valid_n,
                    ] = tensor

                    mask[
                        batch_idx,
                        time_idx,
                        :valid_n,
                    ] = True

            collated[key] = {
                'padded': padded,
                'mask': mask,
            }

        # ============================================================
        # 固定尺寸数据
        # ============================================================
        else:
            T = len(batch[0][key])

            for sample_idx, sample in enumerate(batch):
                if len(sample[key]) != T:
                    raise ValueError(
                        f"{key} 的时间长度不一致："
                        f"sample 0 的 T={T}，"
                        f"sample {sample_idx} 的 "
                        f"T={len(sample[key])}"
                    )

            # default_collate 的输出是：
            # List[T]，其中每个元素为 Tensor[B, ...]
            time_collated = default_collate([
                sample[key]
                for sample in batch
            ])

            # 转换为 Tensor[B, T, ...]
            collated[key] = torch.stack(
                list(time_collated),
                dim=1,
            )

    return collated