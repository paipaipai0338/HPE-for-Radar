import random
import sys
import os
import json
import pickle
import time
import torch
import copy
import numpy as np
from dataclasses import dataclass
from typing import *
from pathlib import Path
from torch import nn
from functools import partial
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from data2datasets.load_json import get_meta_info
from preprocess.radarprocess import (
    Radar_Config,
    bin_buffer_to_cube_range_fft,
    get_bin_data,
    get_pc_data,
)
from preprocess.lidarprocess import get_lidar_data
from preprocess.realsenseprocess import get_realsense_data
from preprocess.gtprocess import get_gt_boxes, get_gt_data


@dataclass(frozen=True)
class PackedBinFrame:
    pack_path: str
    frame_name: str
    timestamp_ns: int
    offset: int
    length: int


def _collate_base(batch, max_points=300, max_people=4):
    """统一多任务 Dataset 私有的点云与 GT padding 实现。"""
    if not batch:
        raise ValueError("batch 不能为空")
    if max_points <= 0 or max_people <= 0:
        raise ValueError("max_points 和 max_people 必须大于 0")

    variable_config = {
        'radar_low_pc': (max_points, (6,), True),
        'radar_high_pc': (max_points, (6,), True),
        'gt': (max_people, (17, 3), False),
        'gt_for_high': (max_people, (17, 3), False),
        'gt_for_low': (max_people, (17, 3), False),
    }
    expected_keys = set(batch[0])
    for sample_idx, sample in enumerate(batch):
        if set(sample) != expected_keys:
            raise ValueError(
                f"batch 中第 {sample_idx} 个样本的键不一致："
                f"expected={expected_keys}, actual={set(sample)}"
            )

    B = len(batch)
    collated = {}
    for key in batch[0]:
        T = len(batch[0][key])
        for sample_idx, sample in enumerate(batch):
            if len(sample[key]) != T:
                raise ValueError(
                    f"{key} 时间长度不一致：sample 0 T={T}, "
                    f"sample {sample_idx} T={len(sample[key])}"
                )

        if key not in variable_config:
            time_collated = default_collate(
                [sample[key] for sample in batch]
            )
            collated[key] = torch.stack(list(time_collated), dim=1)
            continue

        max_var, fixed_dims, random_sample = variable_config[key]
        padded = torch.zeros(
            B, T, max_var, *fixed_dims, dtype=torch.float32
        )
        mask = torch.zeros(B, T, max_var, dtype=torch.bool)

        for batch_idx, sample in enumerate(batch):
            for time_idx, arr in enumerate(sample[key]):
                if arr is None:
                    continue
                tensor = torch.as_tensor(arr, dtype=torch.float32)
                if tensor.numel() == 0:
                    continue
                expected_ndim = 1 + len(fixed_dims)
                if (
                    tensor.ndim != expected_ndim
                    or tuple(tensor.shape[1:]) != fixed_dims
                ):
                    raise ValueError(
                        f"{key} shape 错误：batch_idx={batch_idx}, "
                        f"time_idx={time_idx}, actual={tuple(tensor.shape)}, "
                        f"expected=[N,{fixed_dims}]"
                    )

                original_n = tensor.shape[0]
                valid_n = min(original_n, max_var)
                if original_n > max_var:
                    if random_sample:
                        indices = torch.randperm(original_n)[:max_var]
                        tensor = tensor[indices]
                    else:
                        tensor = tensor[:max_var]
                padded[batch_idx, time_idx, :valid_n] = tensor[:valid_n]
                mask[batch_idx, time_idx, :valid_n] = True

        collated[key] = {'padded': padded, 'mask': mask}

    return collated


def collate_fn(
    batch: List[Dict[str, Any]],
    max_points: int = 300,
    max_people: int = 4,
    bbox_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    对统一多任务样本进行 padding，并生成雷达坐标系下的 3D 包围盒。

    Returns:
        radar_low_pc / radar_high_pc:
            padded: [B, T, max_points, 6]
            mask:   [B, T, max_points]

        gt / gt_for_high / gt_for_low:
            padded: [B, T, max_people, 17, 3]
            mask:   [B, T, max_people]

        gt_for_high / gt_for_low 额外包含:
            bbox: [B, T, max_people, 6]
                最后一维为
                [xmin, ymin, zmin, xmax, ymax, zmax]，
                无效 person 槽位填 0。
            action: [B, T, max_people, 4]
                one-hot 类别顺序为
                [stand, sit_squat, lie, other]，
                无效 person 槽位填 0。
    """
    action_sequences = []
    has_action = 'action' in batch[0]

    for sample_idx, sample in enumerate(batch):
        gt_sequence = sample.get('gt_for_high')
        if gt_sequence is None:
            raise KeyError(
                f"统一多任务样本 {sample_idx} 缺少 gt_for_high"
            )

        action_sequence = sample.get('action')
        if (action_sequence is not None) != has_action:
            raise ValueError("同一 batch 中 action 的启用状态不一致")
        if not has_action:
            continue
        if len(action_sequence) != len(gt_sequence):
            raise ValueError(
                "action 与 GT 的时间长度不一致："
                f"sample_idx={sample_idx}, "
                f"action_T={len(action_sequence)}, "
                f"gt_T={len(gt_sequence)}"
            )

        for time_idx, frame_gt in enumerate(gt_sequence):
            num_people = np.asarray(frame_gt).shape[0]
            if num_people > max_people:
                raise ValueError(
                    "GT 人数超过 max_people，不能静默截断检测标注："
                    f"sample_idx={sample_idx}, time_idx={time_idx}, "
                    f"num_people={num_people}, max_people={max_people}"
                )

            frame_action = np.asarray(action_sequence[time_idx])
            if frame_action.shape != (num_people, 4):
                raise ValueError(
                    "action 必须与 GT 人体一一对应且 shape 为 [N,4]："
                    f"sample_idx={sample_idx}, time_idx={time_idx}, "
                    f"gt_people={num_people}, "
                    f"action_shape={frame_action.shape}"
                )

        action_sequences.append(action_sequence)

    # action 是随人数变化的标注，由本函数单独 padding；基础 collate
    # 只处理其余传感器和 GT 数据。
    batch_without_action = [
        {
            key: value
            for key, value in sample.items()
            if key != 'action'
        }
        for sample in batch
    ]

    collated = _collate_base(
        batch=batch_without_action,
        max_points=max_points,
        max_people=max_people,
    )

    B = len(batch)
    padded_action = None
    if has_action:
        T = len(action_sequences[0])
        padded_action = torch.zeros(
            B, T, max_people, 4, dtype=torch.float32,
        )
        for batch_idx, action_sequence in enumerate(action_sequences):
            for time_idx, frame_action in enumerate(action_sequence):
                action_tensor = torch.as_tensor(
                    frame_action,
                    dtype=torch.float32,
                )
                num_people = action_tensor.shape[0]
                padded_action[batch_idx, time_idx, :num_people] = (
                    action_tensor
                )

    for gt_key in ('gt_for_high', 'gt_for_low'):
        if gt_key not in collated:
            continue

        gt_data = collated[gt_key]
        valid_person = gt_data['mask']
        bbox = get_gt_boxes(
            gt=gt_data['padded'],
            gt_mask=valid_person,
            threshold=bbox_threshold,
        )
        gt_data['bbox'] = bbox.masked_fill(
            ~valid_person.unsqueeze(-1),
            0.0,
        )
        if padded_action is not None:
            gt_data['action'] = padded_action.masked_fill(
                ~valid_person.unsqueeze(-1),
                0.0,
            )

    return collated


class HPE_Dataset(Dataset):

    FILE_READ_MAX_ATTEMPTS = 5
    FILE_READ_RETRY_BASE_DELAY_SEC = 0.05
    MIN_RADAR_POINTS_PER_FRAME = 20
    _ROTATION_ROLL_RANGE_DEG = (-5.0, 5.0)
    _ROTATION_PITCH_RANGE_DEG = (-10.0, 10.0)
    _ROTATION_YAW_RANGE_DEG = (-5.0, 5.0)
    DEFAULT_BAD_BIN_FRAMES_PATH = Path(__file__).with_name(
        'bad_bin_frames.json'
    )
    
    def __init__(
        self,
        root_path='/mnt/huawei',
        sensor_config=None,
        mode='train',
        base_source='radar_high_bin',
        split_method='group',
        ratio=0.7,
        T=8,
        preload_cache=True,
        enable_rotation=False,
        enable_action=False,
        radar_config: Optional[Radar_Config] = None,
        radar_bin_root: Optional[Union[str, Path]] = None,
        bad_bin_frames_path: Optional[Union[str, Path]] = (
            DEFAULT_BAD_BIN_FRAMES_PATH
        ),
        max_groups: Optional[int] = None,
    ):
        super(HPE_Dataset, self).__init__()
        assert mode in ['train', 'val'], 'mode disnmatched'
        split_method = split_method.lower()
        assert split_method in ['person', 'group', 'sequence'], 'split_method has unmatched method'
        if not isinstance(enable_rotation, bool):
            raise TypeError(
                "enable_rotation 必须为 bool，"
                f"实际为 {type(enable_rotation).__name__}"
            )
        if not isinstance(enable_action, bool):
            raise TypeError(
                "enable_action 必须为 bool，"
                f"实际为 {type(enable_action).__name__}"
            )

        self.root_path = Path(root_path)
        self.mode = mode
        self.enable_rotation = enable_rotation
        self.enable_action = enable_action
        self.radar_config = radar_config
        self.radar_bin_root = (
            None if radar_bin_root is None else Path(radar_bin_root)
        )
        self.bad_bin_frames = self._load_bad_bin_frames(
            bad_bin_frames_path
        )
        self.base_source = base_source
        self.ratio = ratio
        self.T = T
        self.action_label = [
            'stand',
            'sit_squat',
            'lie',
            'other',
        ]
        self.preload_cache = preload_cache
        self.calib_cache = {}
        self.pointcloud_cache = {}
        self.gt_cache = {}
        self.action_cache = {}
        self.skip_bad_samples = 0
        self.bad_files = set()
        self.npy_valid_cache = {}
        self.bin_valid_cache = {}
        self.packed_bin_cache = {}
        self.packed_bin_cache_pid = os.getpid()

        # 加载元信息
        json_path = self.root_path / 'data description.json'
        self.meta_info = get_meta_info(json_path)
        if split_method == 'person':
            person_ids = (
                {'0', '1', '2', '3', '5'}
                if mode == 'train'
                else {'4', '6', '7', '8'}
            )
            self.meta_info = {
                person_id: person_data
                for person_id, person_data in self.meta_info.items()
                if person_id in person_ids
            }
        if max_groups is not None:
            if max_groups <= 0:
                raise ValueError("max_groups 必须大于 0")
            remaining = max_groups
            for person_data in self.meta_info.values():
                for entry in person_data:
                    selected = entry['valid_group'][:remaining]
                    entry['valid_group'] = selected
                    remaining -= len(selected)
            self.meta_info = {
                person_id: [
                    entry for entry in person_data
                    if entry['valid_group']
                ]
                for person_id, person_data in self.meta_info.items()
            }
            self.meta_info = {
                person_id: person_data
                for person_id, person_data in self.meta_info.items()
                if person_data
            }
        
        # 定义传感器，是否选取该传感器
        self.sensor_config = {
            'lidar': False,
            'radar_high_bin': True,
            'radar_low_bin': False,
            'radar_low_pc': False,
            'radar_high_pc': False,
            'gt': True,
            'realsense': False,
        } if sensor_config is None else sensor_config

        assert self.sensor_config[base_source] is True, 'base_source in sensor_config is False'
        self.suffix_map = {
            'lidar': '.pcd',
            'radar_low_bin': '.bin',
            'radar_high_bin': '.bin',
            'radar_low_pc': '.npy',
            'radar_high_pc': '.npy',
            'gt': '.pkl',
            'realsense': '.bin',
        }
        self.cached_sensor_names = {
            'radar_high_pc',
            'gt',
        }
        # 更新 meta_info
        self._build_aligned_data()
        
        # 数据划分 TODO：sequence
        self.meta_info_splited = {'train': copy.deepcopy(self.meta_info), 'val': copy.deepcopy(self.meta_info)}
        if split_method == 'person':
            # 按人来划分
            train_person_ids = {'0', '1', '2', '3', '5'}
            val_person_ids = {'4', '6', '7', '8'}

            self.meta_info_splited['train'] = {
                person_id: person_data
                for person_id, person_data in self.meta_info.items()
                if person_id in train_person_ids
            }
            self.meta_info_splited['val'] = {
                person_id: person_data
                for person_id, person_data in self.meta_info.items()
                if person_id in val_person_ids
            } 

        elif split_method == 'group':
            # 按照组来划分
            rng = random.Random(42)
            for person, person_data in self.meta_info.items():
                for idx, entry in enumerate(person_data):
                    valid_group = entry['valid_group'].copy()
                    rng.shuffle(valid_group)
                    self.meta_info_splited['train'][person][idx]['valid_group'] = valid_group[:int(ratio*len(valid_group))]
                    self.meta_info_splited['train'][person][idx]['group_data_path'] = {k: self.meta_info_splited['train'][person][idx]['group_data_path'][k] for k in self.meta_info_splited['train'][person][idx]['valid_group']}
                    self.meta_info_splited['val'][person][idx]['valid_group'] = valid_group[int(ratio*len(valid_group)):]
                    self.meta_info_splited['val'][person][idx]['group_data_path'] = {k: self.meta_info_splited['val'][person][idx]['group_data_path'][k] for k in self.meta_info_splited['val'][person][idx]['valid_group']}
        elif split_method == 'sequence':
            # 按照序列来划分
            pass
        self._display_meta_info(self.meta_info)
        self._display_meta_info(self.meta_info_splited['train'])
        self._display_meta_info(self.meta_info_splited['val'])

        # meta_info 展平按 T 划分
        self.mode_meta_info = self.meta_info_splited[mode]
        self.data_path_list = {k: [] for k in self.sensor_config if self.sensor_config[k]}
        for person_id, person_data in self.mode_meta_info.items():
            for entry in person_data:
                valid_group = entry['valid_group']
                group_data_path = entry['group_data_path']
                for group in valid_group:
                    frame = len(group_data_path[group][base_source])
                    starts = list(range(0, frame - T + 1, 4))
                    windows = [(start, start + T) for start in starts]
                    for start_idx, end_idx in windows:
                        window_by_sensor = {}
                        for sensor_name in self.data_path_list.keys():
                            all_sensor_files = group_data_path[group][sensor_name]
                            window_by_sensor[sensor_name] = all_sensor_files[start_idx:end_idx]

                        if not self._is_valid_window(window_by_sensor):
                            self.skip_bad_samples += 1
                            continue

                        for sensor_name, window_files in window_by_sensor.items():
                            # 将文件路径列表添加到 data_path_list 中
                            self.data_path_list[sensor_name].append(window_files)

        if self.skip_bad_samples > 0:
            print(f"跳过损坏样本窗口数: {self.skip_bad_samples}")
            print(f"损坏文件数: {len(self.bad_files)}")

        if self.preload_cache:
            self.preload_data_cache()

    @staticmethod
    def _load_bad_bin_frames(
        manifest_path: Optional[Union[str, Path]],
    ) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
        if manifest_path is None:
            return {}

        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"坏 BIN 帧清单不存在: {manifest_path}"
            )
        with manifest_path.open('r', encoding='utf-8') as file:
            manifest = json.load(file)
        if manifest.get('schema_version') != 1:
            raise ValueError(
                "坏 BIN 帧清单 schema_version 必须为 1，"
                f"实际为 {manifest.get('schema_version')}"
            )
        records = manifest.get('bad_frames')
        if not isinstance(records, list):
            raise ValueError("坏 BIN 帧清单缺少 bad_frames 列表")

        lookup = {}
        required = {
            'date', 'group', 'sensor', 'frame_name',
            'frame_index', 'bad_type',
        }
        for record_index, record in enumerate(records):
            if not isinstance(record, dict) or not required <= record.keys():
                raise ValueError(
                    f"bad_frames[{record_index}] 字段不完整"
                )
            if record['bad_type'] not in {'numeric', 'length'}:
                raise ValueError(
                    f"bad_frames[{record_index}].bad_type 无效: "
                    f"{record['bad_type']}"
                )
            if not isinstance(record['frame_index'], int) or record['frame_index'] < 0:
                raise ValueError(
                    f"bad_frames[{record_index}].frame_index 必须为非负整数"
                )
            key = (
                str(record['date']),
                str(record['group']),
                str(record['sensor']),
                str(record['frame_name']),
            )
            if key in lookup:
                raise ValueError(
                    f"坏 BIN 帧清单存在重复记录: {key}"
                )
            lookup[key] = record
        return lookup

    @staticmethod
    def _get_bin_frame_manifest_key(
        file_path: Union[str, Path, PackedBinFrame],
    ) -> Optional[Tuple[str, str, str, str]]:
        path = Path(
            file_path.pack_path
            if isinstance(file_path, PackedBinFrame)
            else file_path
        )
        try:
            marker = path.parts.index('data_collection')
            date = path.parts[marker - 1]
            group = path.parts[marker + 1]
            sensor = path.parts[marker + 2]
        except (ValueError, IndexError):
            return None
        frame_name = (
            file_path.frame_name
            if isinstance(file_path, PackedBinFrame)
            else path.name
        )
        return date, group, sensor, frame_name

    def _display_meta_info(self, meta_info: Dict) -> None:
        '''Dict[str(person_id), List]
                List[Dict]
                    Dict[
                        'date': str,
                        'valid_group': List[str]
                        'group_data_path: Dict
                            Dict[
                                'group_name: aligned_frames
                            ]
                    ]'''
        if not meta_info:
            print("没有数据可显示")
        
        print("\n" + "=" * 70)
        print(f"受试者元信息 (共 {len(meta_info)} 人)")
        print("=" * 70)
        
        for person_id, person_data in meta_info.items():
            print(f"受试者 ID: {person_id}")
            print(f"   记录数: {len(person_data)} 条")
            
            for idx, entry in enumerate(person_data, 1):
                print(f"   记录 {idx}:")
                print(f"      - 日期: {entry['date']}")
                print(f"      - 组数: {len(entry['valid_group'])} 个")
                print(f"      - 组列表: {entry['valid_group']}")
                for group in entry['valid_group']:
                    print(f"            - {group}帧数: {len(entry['group_data_path'][group][self.base_source])}")
            
            print("   " + "-" * 50)

    def _is_valid_npy(
        self,
        sensor_name: str,
        file_path: str,
    ) -> bool:
        cache_key = str(file_path)
        if cache_key in self.npy_valid_cache:
            return self.npy_valid_cache[cache_key]

        try:
            array = np.load(
                cache_key,
                mmap_mode="r",
                allow_pickle=False,
            )

            # 确保至少读取并解析 header
            _ = array.shape
            _ = array.dtype

            # 点云窗口中只要有一帧为空或形状非法，就将该文件标记为
            # 无效；_is_valid_window 会据此跳过包含它的整个 T 窗口。
            if sensor_name in {'radar_low_pc', 'radar_high_pc'}:
                if (
                    array.ndim != 2
                    or array.shape[0] < self.MIN_RADAR_POINTS_PER_FRAME
                    or array.shape[1] < 3
                ):
                    raise ValueError(
                        "雷达点云必须为 [N,C]、"
                        f"N>={self.MIN_RADAR_POINTS_PER_FRAME} 且 C>=3，"
                        f"实际形状={array.shape}"
                    )
            del array
            valid = True

        except Exception as exc:
            valid = False

            if (
                cache_key not in self.bad_files
                and len(self.bad_files) < 10
            ):
                print(
                    f"跳过损坏文件: "
                    f"sensor={sensor_name}, "
                    f"path={cache_key}, "
                    f"error={type(exc).__name__}: {exc}"
                )

            self.bad_files.add(cache_key)

        self.npy_valid_cache[cache_key] = valid

        return valid

    def _is_valid_radar_bin(
        self,
        sensor_name: str,
        file_path: Union[str, PackedBinFrame],
    ) -> bool:
        cache_key = file_path
        if cache_key in self.bin_valid_cache:
            return self.bin_valid_cache[cache_key]

        if self.radar_config is None:
            raise ValueError(
                f'{sensor_name} 已启用，但未向 HPE_Dataset 传入 radar_config'
            )

        use_range = self.radar_config.num_samp // 2
        num_ant = self.radar_config.Tx * self.radar_config.Rx
        expected_bytes = (
            use_range
            * self.radar_config.num_chirp
            * num_ant
            * 4
        )

        manifest_key = self._get_bin_frame_manifest_key(file_path)
        bad_record = self.bad_bin_frames.get(manifest_key)
        if bad_record is not None:
            valid = False
            error = (
                "listed in bad_bin_frames.json: "
                f"bad_type={bad_record['bad_type']}, "
                f"frame_index={bad_record['frame_index']}"
            )
        elif isinstance(file_path, PackedBinFrame):
            actual_bytes = file_path.length
            valid = actual_bytes == expected_bytes
            error = f'size mismatch: {actual_bytes} != {expected_bytes}'
        else:
            try:
                actual_bytes = os.path.getsize(file_path)
                valid = actual_bytes == expected_bytes
            except OSError as exc:
                actual_bytes = None
                valid = False
                error = f'{type(exc).__name__}: {exc}'
            else:
                error = f'size mismatch: {actual_bytes} != {expected_bytes}'

        if not valid:
            if file_path not in self.bad_files and len(self.bad_files) < 10:
                print(
                    f'跳过损坏文件: sensor={sensor_name}, '
                    f'path={file_path}, error={error}'
                )
            self.bad_files.add(file_path)

        self.bin_valid_cache[cache_key] = valid
        return valid

    def _is_valid_window(
        self,
        window_by_sensor: Dict[str, List[str]],
    ) -> bool:
        for sensor_name, window_files in window_by_sensor.items():
            suffix = self.suffix_map[sensor_name]
            if suffix == '.npy':
                for file_path in window_files:
                    if not self._is_valid_npy(sensor_name, file_path):
                        return False
            elif suffix == '.bin' and 'radar_' in sensor_name:
                for file_path in window_files:
                    if not self._is_valid_radar_bin(sensor_name, file_path):
                        return False

        return True
        
    def _build_aligned_data(self) -> None:
        '''
        在 meta_info 基础上添加 group_data_path
        Before:
            Dict[str(person_id), List]
                List[Dict]
                    Dict[
                        'date': str,
                        'valid_group': List[str]
                    ]
        After:
            Dict[str(person_id), List]
                List[Dict]
                    Dict[
                        'date': str,
                        'valid_group': List[str]
                        'group_data_path: Dict
                            Dict[
                                'group_name': aligned_frames
                            ]
                    ]
        '''
        for person_id, person_data in self.meta_info.items():
            for entry in person_data:
                date = entry['date']
                valid_group = entry['valid_group']
                group_data_path = {}
                for group_name in valid_group:
                    # 构建数据目录路径
                    group_dir = self.root_path / date / 'data_collection' / group_name
                    
                    if not group_dir.exists():
                        print(f"目录不存在: Person id:{person_id}, Date:{date}, Group:{group_dir}")
                        continue
                    
                    # 构建传感器路径字典
                    sensor_paths = self._build_sensor_paths(group_dir)
                    
                    # 执行对齐
                    aligned_frames = self._align_multi_sensor_files(
                        sources=sensor_paths, 
                        base_source=self.base_source, 
                        # time_offsets_sec={
                        #     "gt": -0.2
                        # },
                        )
                    
                    if not aligned_frames or not aligned_frames.get(self.base_source):
                        print(f"{group_name} 对齐后没有数据")
                        continue
                    group_data_path[f'{group_name}'] = aligned_frames
                entry['valid_group'] = list(group_data_path)
                entry['group_data_path'] = group_data_path
    
    def _align_multi_sensor_files(
        self,
        sources: Dict[str, Optional[Path]],
        max_delta_sec: Optional[float] = 0.3,
        one_to_one: bool = True,
        base_source: Optional[str] = None,
        time_offsets_sec: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[Optional[str]]]:
        """
        aligned_frames: 
        Dict[
            'sensor_name': List
        ]

        """
        def unix_to_datetime(unix_ts: float) -> datetime:
            """
            将 Unix 浮点时间戳转换为本地 datetime 对象。
            unix_ts: 例如 1719999999.123456789 这种秒级+纳秒的小数
            输入:
            unix_ts: float，Unix 时间戳，单位为秒，shape 为标量。
            输出:
            datetime，Python datetime 对象，shape 为标量对象。
            """
            return datetime.fromtimestamp(unix_ts)

        def files_to_time_list(files: List) -> List:
            """
            将 aaa_bbb.ccc 文件转化为时间 list, 前提条件 aaa, bbb 分别为 unix时间戳的 s, ns
            输入:
            files: List[str]，文件名或路径列表，shape=(N,)。
            中间变量:
            base: str，单个文件名去后缀后的时间戳字符串，shape 为标量字符串。
            sec/ns: int，Unix 秒和纳秒，shape 均为标量。
            unix_ts: float，秒级 Unix 时间戳，shape 为标量。
            输出:
            times: List[datetime]，datetime 对象列表，shape=(N,)。
            """
            times = []
            for file in files:
                if isinstance(file, PackedBinFrame):
                    times.append(unix_to_datetime(file.timestamp_ns * 1e-9))
                    continue
                base = Path(file).stem
                sec_str, ns_str = base.split('_')
                sec = int(sec_str)
                ns = int(ns_str)
                unix_ts = sec + ns * 1e-9
                times.append(unix_to_datetime(unix_ts))
            return times
        
        def list_files(
            sensor_name: str,
            dir_path: Optional[str],
            suffix: str,
        ) -> Tuple[List, List[datetime]]:
            """列出目录中匹配后缀的文件并解析时间戳"""
            if not dir_path or not suffix or not Path(dir_path).is_dir():
                return [], []

            if (
                sensor_name == 'radar_high_bin'
                and self.radar_bin_root is not None
            ):
                pack_path = Path(dir_path) / 'frames.binpack'
                index_path = Path(dir_path) / 'frames_index.npz'
                if not pack_path.is_file() or not index_path.is_file():
                    return [], []
                with np.load(index_path, allow_pickle=False) as index:
                    required = {'frame_names', 'timestamps_ns', 'offsets', 'lengths'}
                    missing = required - set(index.files)
                    if missing:
                        raise KeyError(f"{index_path} 缺少字段: {sorted(missing)}")
                    names = np.asarray(index['frame_names']).reshape(-1)
                    timestamps = np.asarray(index['timestamps_ns'], dtype=np.int64).reshape(-1)
                    offsets = np.asarray(index['offsets'], dtype=np.int64).reshape(-1)
                    lengths = np.asarray(index['lengths'], dtype=np.int64).reshape(-1)

                count = len(names)
                if not (len(timestamps) == len(offsets) == len(lengths) == count):
                    raise ValueError(f"打包 BIN 索引字段长度不一致: {index_path}")
                pack_size = pack_path.stat().st_size
                if count and (
                    offsets[0] != 0
                    or np.any(lengths < 0)
                    or np.any(offsets[1:] != offsets[:-1] + lengths[:-1])
                    or offsets[-1] + lengths[-1] != pack_size
                ):
                    raise ValueError(f"打包 BIN offset/length 非连续或越界: {index_path}")

                refs = [
                    PackedBinFrame(
                        pack_path=str(pack_path),
                        frame_name=str(names[idx]),
                        timestamp_ns=int(timestamps[idx]),
                        offset=int(offsets[idx]),
                        length=int(lengths[idx]),
                    )
                    for idx in range(count)
                ]
                return refs, files_to_time_list(refs)

            files = [f for f in os.listdir(dir_path) if f.lower().endswith(suffix)]
            files.sort()
            # 假设 files_to_time_list 函数已存在
            times = files_to_time_list(files)
            return files, times

        def time_diff(dt1: datetime, dt2: datetime) -> float:
            return abs((dt1 - dt2).total_seconds())

        def find_global_matches(base_times: List[datetime], target_times: List[datetime]) -> Dict[int, int]:
            """
            对两个时间序列进行单调、一对一的全局最优匹配。

            优化目标：
                1. 优先最大化匹配帧数量；
                2. 匹配数量相同时，最小化总时间误差。

            返回：
                Dict[base_idx, target_idx]
            """
            n = len(base_times)
            m = len(target_times)

            if n == 0 or m == 0:
                return {}

            # dp_count[i, j]:
            # base 前 i 帧与 target 前 j 帧能够获得的最大匹配数量
            dp_count = np.zeros((n + 1, m + 1), dtype=np.int32)

            # dp_cost[i, j]:
            # 在最大匹配数量下的最小总时间误差
            dp_cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)

            # action:
            # 1 = 跳过 base
            # 2 = 跳过 target
            # 3 = base 和 target 匹配
            action = np.zeros((n + 1, m + 1), dtype=np.uint8)

            # 空序列之间的匹配数量和误差均为 0
            dp_cost[0, :] = 0.0
            dp_cost[:, 0] = 0.0

            def is_better(
                candidate_count: int,
                candidate_cost: float,
                best_count: int,
                best_cost: float,
            ) -> bool:
                """匹配数量优先，其次比较总误差。"""
                if candidate_count > best_count:
                    return True

                if candidate_count == best_count and candidate_cost < best_cost:
                    return True

                return False

            for i in range(1, n + 1):
                for j in range(1, m + 1):

                    # 情况 1：跳过当前 base 帧
                    best_count = dp_count[i - 1, j]
                    best_cost = dp_cost[i - 1, j]
                    best_action = 1

                    # 情况 2：跳过当前 target 帧
                    candidate_count = dp_count[i, j - 1]
                    candidate_cost = dp_cost[i, j - 1]

                    if is_better(
                        candidate_count,
                        candidate_cost,
                        best_count,
                        best_cost,
                    ):
                        best_count = candidate_count
                        best_cost = candidate_cost
                        best_action = 2

                    # 情况 3：匹配当前 base 帧和 target 帧
                    current_error = time_diff(
                        base_times[i - 1],
                        target_times[j - 1],
                    )

                    valid_match = (
                        max_delta_sec is None
                        or current_error <= max_delta_sec
                    )

                    if valid_match:
                        candidate_count = dp_count[i - 1, j - 1] + 1
                        candidate_cost = (
                            dp_cost[i - 1, j - 1]
                            + current_error
                        )

                        if is_better(
                            candidate_count,
                            candidate_cost,
                            best_count,
                            best_cost,
                        ):
                            best_count = candidate_count
                            best_cost = candidate_cost
                            best_action = 3

                    dp_count[i, j] = best_count
                    dp_cost[i, j] = best_cost
                    action[i, j] = best_action

            # 从右下角回溯得到完整匹配关系
            matches = {}

            i = n
            j = m

            while i > 0 and j > 0:
                current_action = action[i, j]

                if current_action == 3:
                    matches[i - 1] = j - 1
                    i -= 1
                    j -= 1

                elif current_action == 1:
                    i -= 1

                elif current_action == 2:
                    j -= 1

                else:
                    break

            return matches
        # 过滤掉路径为None的传感器
        sources = {k: v for k, v in sources.items() if v is not None}
        if not sources:
            return {}

        # 读取所有传感器的文件列表和时间戳
        sensor_data = {}
        for name, path in sources.items():
            suffix = self.suffix_map.get(name)
            files, times = list_files(name, path, suffix)
            if time_offsets_sec and name in time_offsets_sec:
                offset = timedelta(seconds=float(time_offsets_sec[name]))
                times = [t + offset for t in times]
            if files:  # 只保留非空的传感器
                sensor_data[name] = {
                    'path': path,
                    'files': files,
                    'times': times
                }

        if not sensor_data:
            return {name: [] for name in sources.keys()}

        # 选择基准传感器
        if base_source is None or base_source not in sensor_data:
            base_source = list(sensor_data.keys())[0]

        base_times = sensor_data[base_source]['times']
        base_files = sensor_data[base_source]['files']

        if not one_to_one:
            raise ValueError(
                "当前全局匹配实现要求 one_to_one=True"
            )

        result = {
            name: []
            for name in sources.keys()
        }

        # 如果某个启用传感器没有有效文件，
        # 则无法满足“所有传感器全部匹配成功”的要求。
        missing_sensors = set(sources.keys()) - set(sensor_data.keys())

        if missing_sensors:
            print(
                f"以下传感器没有有效数据，当前组无法完成全传感器对齐: "
                f"{sorted(missing_sensors)}"
            )
            return result

        # match_maps[name][base_idx] = 该传感器对应的文件索引
        match_maps: Dict[str, Dict[int, int]] = {
            base_source: {
                base_idx: base_idx
                for base_idx in range(len(base_times))
            }
        }

        # 基准传感器分别与其他每个传感器进行全局匹配
        for name, data in sensor_data.items():
            if name == base_source:
                continue

            match_maps[name] = find_global_matches(
                base_times=base_times,
                target_times=data['times'],
            )

        # 逐个检查基准帧
        for base_idx in range(len(base_times)):
            frame_paths = {}
            all_matched = True

            # 必须遍历所有启用的传感器
            for name in sources.keys():
                data = sensor_data[name]

                if name == base_source:
                    sensor_idx = base_idx
                else:
                    sensor_idx = match_maps[name].get(base_idx)

                    if sensor_idx is None:
                        all_matched = False
                        break

                sensor_file = data['files'][sensor_idx]
                frame_paths[name] = (
                    sensor_file
                    if isinstance(sensor_file, PackedBinFrame)
                    else os.path.join(data['path'], sensor_file)
                )

            # 只有当前基准帧在所有传感器中均成功匹配，
            # 才整体写入 result。
            if all_matched:
                for name in sources.keys():
                    result[name].append(frame_paths[name])

        return result

    def _build_sensor_paths(self, group_dir: Path) -> Dict[str, Optional[Path]]:
        """
        构建传感器路径字典
        """
        sensor_paths = {}
        radar_low_path = group_dir / 'dpct低位机'
        radar_low_bin_path = radar_low_path / 'Bin'
        radar_low_pc_path = radar_low_path / 'PC'
        radar_high_path = group_dir / 'dpct高位机'
        if self.radar_bin_root is None:
            radar_high_bin_path = radar_high_path / 'Bin'
        else:
            relative_group = group_dir.relative_to(self.root_path)
            radar_high_bin_path = (
                self.radar_bin_root
                / relative_group
                / 'dpct高位机'
                / 'Bin'
            )
            pack_path = radar_high_bin_path / 'frames.binpack'
            index_path = radar_high_bin_path / 'frames_index.npz'
            if not pack_path.is_file() or not index_path.is_file():
                print(
                    "SSD 打包文件缺失，跳过当前数据组: "
                    f"pack={pack_path}, index={index_path}"
                )
        radar_high_pc_path = radar_high_path / 'PC'
        lidar_path = group_dir / 'robosense'
        realsense_path = group_dir / 'realsense' / 'undistorted_depth'
        gt_path = group_dir / 'camera results' / 'smoothed 3D'

        sensor_paths = {
            'lidar': lidar_path if self.sensor_config['lidar'] else None,
            'radar_low_bin': radar_low_bin_path if self.sensor_config['radar_low_bin'] else None,
            'radar_high_bin': radar_high_bin_path if self.sensor_config['radar_high_bin'] else None,
            'radar_low_pc': radar_low_pc_path if self.sensor_config['radar_low_pc'] else None,
            'radar_high_pc': radar_high_pc_path if self.sensor_config['radar_high_pc'] else None,
            'gt': gt_path if self.sensor_config['gt'] else None,
            'realsense': realsense_path if self.sensor_config['realsense'] else None,
        }
        
        return sensor_paths

    def _copy_cached_data(self, data: Any) -> Any:
        """
        缓存中保存原始读取结果；取出时复制 ndarray，避免下游原地修改污染缓存。
        """
        if isinstance(data, np.ndarray):
            return data.copy()

        return copy.deepcopy(data)

    def _get_cache_for_sensor(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        if sensor_name == 'gt':
            return self.gt_cache

        if sensor_name in self.cached_sensor_names:
            return self.pointcloud_cache

        return None

    def _cache_frame_data(
        self,
        sensor_name: str,
        path: Path | str,
        load_fn: Callable[[Path | str], Any],
    ) -> Optional[Any]:
        path_key = str(path)
        cache = self._get_cache_for_sensor(sensor_name)

        if cache is None:
            return None

        if path_key not in cache:
            data = load_fn(path)
            if isinstance(data, np.ndarray):
                data.setflags(write=False)
            cache[path_key] = data

        return cache[path_key]

    def _get_cached_frame_data(
        self,
        sensor_name: str,
        path: Path | str,
        load_fn: Callable[[Path | str], Any],
    ) -> Any:
        cached_data = self._cache_frame_data(
            sensor_name=sensor_name,
            path=path,
            load_fn=load_fn,
        )

        if cached_data is None:
            return load_fn(path)

        return self._copy_cached_data(cached_data)

    def _get_sensor_loader(self, sensor_name: str) -> Callable[[Path | str], Any]:
        requires_radar_config = sensor_name in {'radar_low_bin', 'radar_high_bin'}
        if requires_radar_config and self.radar_config is None:
            raise ValueError(f'{sensor_name} 已启用，但未向 HPE_Dataset 传入 radar_config')

        get_data_function_dict = {
            'lidar': get_lidar_data,
            'radar_low_bin': partial(get_bin_data, radar_config=self.radar_config),
            'radar_high_bin': partial(get_bin_data, radar_config=self.radar_config),
            'radar_low_pc': get_pc_data,
            'radar_high_pc': get_pc_data,
            'gt': get_gt_data,
            'realsense': get_realsense_data,
        }

        return get_data_function_dict[sensor_name]

    def _get_action_path(self, gt_path: Path | str) -> Path:
        """
        action 文件由对应 GT 文件生成，因此二者使用相同文件名。

        GT:
            camera results/smoothed 3D/<timestamp>.pkl
        action:
            camera results/action label/<timestamp>.pkl

        同时兼容已有数据中的 ``<timestamp>.npz``。
        """
        gt_path = Path(gt_path)
        action_dir = gt_path.parent.parent / 'action label'
        pkl_path = action_dir / f'{gt_path.stem}.pkl'
        npz_path = action_dir / f'{gt_path.stem}.npz'

        if pkl_path.exists():
            return pkl_path
        if npz_path.exists():
            return npz_path
        return pkl_path

    def _load_gt_action_frame(
        self,
        gt_path: Path | str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        同步读取一帧 GT 和 action，并用 GT 的有效人体 mask 同步筛选。

        action pkl 支持 ``{'labels': ..., 'valid': ...}``；其中 valid
        不参与筛选，人体有效性统一由 GT 是否缺失/包含非有限关节决定。
        """
        gt_path = Path(gt_path)
        action_path = self._get_action_path(gt_path)

        if not action_path.exists():
            raise FileNotFoundError(
                f"GT 对应的 action 文件不存在: {action_path}"
            )

        def load_pickle(path: Path) -> Any:
            def load_once() -> Any:
                with open(path, 'rb') as file:
                    return pickle.load(file)

            return self._load_with_permission_retry(
                path=path,
                load_once=load_once,
            )

        raw_gt = load_pickle(gt_path)

        if action_path.suffix.lower() == '.npz':
            def load_action_npz() -> Dict[str, np.ndarray]:
                with np.load(action_path) as action_npz:
                    return {
                        key: action_npz[key].copy()
                        for key in action_npz.files
                    }

            action_data = self._load_with_permission_retry(
                path=action_path,
                load_once=load_action_npz,
            )
        else:
            action_data = load_pickle(action_path)

        if not isinstance(action_data, dict) or 'labels' not in action_data:
            raise ValueError(
                "action 标注必须包含 labels，"
                f"path={action_path}, type={type(action_data).__name__}"
            )

        gt = np.asarray(raw_gt, dtype=np.float32)
        labels = np.asarray(action_data['labels'], dtype=np.float32)

        if gt.ndim != 3 or gt.shape[-1] != 3:
            raise ValueError(
                f"GT 必须为 [N,J,3]，path={gt_path}, shape={gt.shape}"
            )

        num_people = gt.shape[0]
        if labels.shape != (num_people, 4):
            raise ValueError(
                "action labels 必须为 [N,4] 且与 GT 人数一致，"
                f"path={action_path}, gt_people={num_people}, "
                f"labels_shape={labels.shape}"
            )

        if not np.isfinite(labels).all():
            raise ValueError(
                f"action labels 包含 NaN 或 Inf: {action_path}"
            )

        # action_data['valid'] 本质上是 GT 缺失关节的派生信息。
        # 统一直接由原始 GT 计算 mask，保证此后的人体筛选同步。
        person_valid_mask = np.isfinite(gt).all(axis=(1, 2))
        gt = gt[person_valid_mask]
        labels = labels[person_valid_mask]

        gt.setflags(write=False)
        labels.setflags(write=False)
        return gt, labels

    def _load_with_permission_retry(
        self,
        path: Path | str,
        load_once: Callable[[], Any],
    ) -> Any:
        """
        对 NFS 偶发 PermissionError 进行有限次数指数退避重试。

        只重试 PermissionError；文件不存在、格式损坏等其他异常立即抛出。
        """
        path = Path(path)

        for attempt_idx in range(self.FILE_READ_MAX_ATTEMPTS):
            try:
                return load_once()
            except PermissionError as exc:
                is_last_attempt = (
                    attempt_idx + 1
                    == self.FILE_READ_MAX_ATTEMPTS
                )
                if is_last_attempt:
                    raise PermissionError(
                        "NFS 文件在多次重试后仍无读取权限："
                        f"path={path}, "
                        f"attempts={self.FILE_READ_MAX_ATTEMPTS}"
                    ) from exc

                delay = (
                    self.FILE_READ_RETRY_BASE_DELAY_SEC
                    * (2 ** attempt_idx)
                )
                time.sleep(delay)

    def _get_gt_action_frame(
        self,
        gt_path: Path | str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        path_key = str(gt_path)

        if (
            path_key not in self.gt_cache
            or path_key not in self.action_cache
        ):
            gt, action = self._load_gt_action_frame(gt_path)
            self.gt_cache[path_key] = gt
            self.action_cache[path_key] = action

        return (
            self.gt_cache[path_key].copy(),
            self.action_cache[path_key].copy(),
        )

    def _get_gt_action_sequence(
        self,
        gt_paths: List[Path | str],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        gt_sequence = []
        action_sequence = []

        for gt_path in gt_paths:
            gt, action = self._get_gt_action_frame(gt_path)
            gt_sequence.append(gt)
            action_sequence.append(action)

        return gt_sequence, action_sequence

    def preload_data_cache(self) -> None:
        """
        在主进程中预热点云和 GT 缓存。

        Linux 默认 fork worker 时，这些只读缓存的数据 buffer 可以被子进程共享。
        """
        for sensor_name, path_windows in self.data_path_list.items():
            if sensor_name not in self.cached_sensor_names:
                continue

            seen_paths = set()
            for window_paths in path_windows:
                for path in window_paths:
                    path_key = str(path)
                    if path_key in seen_paths:
                        continue

                    if sensor_name == 'gt' and self.enable_action:
                        self._get_gt_action_frame(path)
                    else:
                        load_fn = self._get_sensor_loader(sensor_name)
                        self._cache_frame_data(
                            sensor_name=sensor_name,
                            path=path,
                            load_fn=load_fn,
                        )
                    seen_paths.add(path_key)

    def _get_sensor_data_from_path(self, sensor_name: str, sensor_path: List[Path|str]) -> List:
        if sensor_path is None:
            return None

        if (
            sensor_name == 'radar_high_bin'
            and sensor_path
            and all(isinstance(path, PackedBinFrame) for path in sensor_path)
        ):
            return self._get_packed_bin_sequence(sensor_path)

        load_fn = self._get_sensor_loader(sensor_name)
        data = []
        for path in sensor_path:
            data.append(
                self._get_cached_frame_data(
                    sensor_name=sensor_name,
                    path=path,
                    load_fn=load_fn,
                )
            )
        return data

    def _get_packed_bin_sequence(
        self,
        frame_refs: List[PackedBinFrame],
    ) -> List[np.ndarray]:
        pack_paths = {ref.pack_path for ref in frame_refs}
        if len(pack_paths) != 1:
            raise ValueError("同一个时间窗口不能跨越多个 BIN 打包文件")

        current_pid = os.getpid()
        if self.packed_bin_cache_pid != current_pid:
            self.packed_bin_cache.clear()
            self.packed_bin_cache_pid = current_pid

        pack_path = frame_refs[0].pack_path
        if pack_path not in self.packed_bin_cache:
            self.packed_bin_cache[pack_path] = np.memmap(
                pack_path, mode='r', dtype=np.uint8,
            )
        packed = self.packed_bin_cache[pack_path]

        output = []
        for ref in frame_refs:
            raw = packed[ref.offset:ref.offset + ref.length]
            frame = bin_buffer_to_cube_range_fft(
                raw,
                self.radar_config,
                source_name=ref.frame_name,
            )
            if frame is None:
                raise ValueError(f"打包 BIN 帧大小错误: {ref.frame_name}")
            output.append(frame)
        return output

    def clear_data_cache(self) -> None:
        """
        清空点云和 GT 的内存缓存；不会影响标定和 npy 有效性缓存。
        """
        self.pointcloud_cache.clear()
        self.gt_cache.clear()
        self.action_cache.clear()
        self.packed_bin_cache.clear()

    def _load_calib_T(self, date: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        加载指定日期对应的 GT/image -> 高位/低位雷达外参。

        同一个日期在当前 Dataset 实例中只加载一次。
        """
        if date in self.calib_cache:
            return self.calib_cache[date]

        calib_path = self.root_path / date / 'calib'

        if not calib_path.exists():
            raise FileNotFoundError(
                f"标定目录不存在: {calib_path}"
            )

        low_path = (
            calib_path
            / 'extrinsic_img_to_radar_low.npz'
        )
        high_path = (
            calib_path
            / 'extrinsic_img_to_radar_high.npz'
        )

        if not low_path.exists():
            raise FileNotFoundError(
                f"低位雷达标定文件不存在: {low_path}"
            )

        if not high_path.exists():
            raise FileNotFoundError(
                f"高位雷达标定文件不存在: {high_path}"
            )

        with np.load(low_path) as low_calib:
            if 'R_est' not in low_calib or 't_est' not in low_calib:
                raise KeyError(
                    f"{low_path} 中缺少 R_est 或 t_est"
                )

            R_low = np.asarray(
                low_calib['R_est'],
                dtype=np.float32,
            )

            t_low = np.asarray(
                low_calib['t_est'],
                dtype=np.float32,
            ).reshape(-1)

        with np.load(high_path) as high_calib:
            if 'R_est' not in high_calib or 't_est' not in high_calib:
                raise KeyError(
                    f"{high_path} 中缺少 R_est 或 t_est"
                )

            R_high = np.asarray(
                high_calib['R_est'],
                dtype=np.float32,
            )

            t_high = np.asarray(
                high_calib['t_est'],
                dtype=np.float32,
            ).reshape(-1)

        if R_low.shape != (3, 3):
            raise ValueError(
                f"R_low 应为 [3,3]，实际为 {R_low.shape}"
            )

        if t_low.shape != (3,):
            raise ValueError(
                f"t_low 应为 [3]，实际为 {t_low.shape}"
            )

        if R_high.shape != (3, 3):
            raise ValueError(
                f"R_high 应为 [3,3]，实际为 {R_high.shape}"
            )

        if t_high.shape != (3,):
            raise ValueError(
                f"t_high 应为 [3]，实际为 {t_high.shape}"
            )

        R_hl = R_low @ R_high.T
        t_hl = t_low - R_hl @ t_high

        calib = {
            'gt_to_low': {
                'R': R_low,
                't': t_low,
            },
            'gt_to_high': {
                'R': R_high,
                't': t_high,
            },
            'high_to_low': {
                'R': R_hl,
                't': t_hl,
            },
        }

        self.calib_cache[date] = calib

        return calib

    def _transform_gt_sequence(self, gt_sequence: List, R: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
        """
        将长度为 T 的 GT 序列转换到目标雷达坐标系。

        原始 gt_sequence 不会被修改。

        对于单帧 GT：
            [P, J, 3]：
                删除任意关节含 NaN/Inf 的 person，
                返回 [P_valid, J, 3]。

            [J, 3]：
                若任意关节含 NaN/Inf，则返回空数组 [0, J, 3]；
                否则按单人数据处理，返回 [1, J, 3]。

        Args:
            gt_sequence:
                长度为 T 的 GT 列表。

            R:
                源坐标系到目标坐标系的旋转矩阵 [3,3]。

            t:
                源坐标系到目标坐标系的平移向量 [3] 或 [3,1]。

        Returns:
            transformed_sequence:
                转换后的 GT 序列，不修改原始 GT。
        """
        R = np.asarray(R, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32).reshape(3)

        if R.shape != (3, 3):
            raise ValueError(
                f"R 应为 [3,3]，实际为 {R.shape}"
            )

        transformed_sequence = []

        for frame_idx, gt_frame in enumerate(gt_sequence):
            if gt_frame is None:
                transformed_sequence.append(None)
                continue

            # copy=True，保证不会修改 samples['gt'] 中的原始数据
            gt_array = np.array(
                gt_frame,
                dtype=np.float32,
                copy=True,
            )

            if gt_array.size == 0:
                transformed_sequence.append(gt_array)
                continue

            if gt_array.shape[-1] != 3:
                raise ValueError(
                    f"GT 最后一维必须为 3，"
                    f"frame_idx={frame_idx}, "
                    f"实际形状={gt_array.shape}"
                )

            # ---------------------------------------------------------
            # 统一为 [P, J, 3]
            # ---------------------------------------------------------
            if gt_array.ndim == 2:
                # 单人 GT：[J,3] -> [1,J,3]
                gt_array = gt_array[None, ...]

            elif gt_array.ndim != 3:
                raise ValueError(
                    f"GT 应为 [J,3] 或 [P,J,3]，"
                    f"frame_idx={frame_idx}, "
                    f"实际形状={gt_array.shape}"
                )

            num_people, num_joints, _ = gt_array.shape

            # ---------------------------------------------------------
            # 删除包含 NaN/Inf 的整个人
            # person_valid_mask: [P]
            # ---------------------------------------------------------
            person_valid_mask = np.isfinite(
                gt_array
            ).all(axis=(1, 2))

            invalid_people = (
                num_people - int(person_valid_mask.sum())
            )

            if invalid_people > 0:
                print(
                    f"Warning: frame {frame_idx} ignored "
                    f"{invalid_people} person(s) containing NaN/Inf"
                )

            valid_gt = gt_array[person_valid_mask]

            if valid_gt.shape[0] == 0:
                transformed_sequence.append(
                    np.empty(
                        (0, num_joints, 3),
                        dtype=np.float32,
                    )
                )
                continue

            # ---------------------------------------------------------
            # [P,J,3] -> [P*J,3]，执行刚体变换
            # ---------------------------------------------------------
            flat_gt = valid_gt.reshape(-1, 3)

            transformed_flat = (
                R @ flat_gt.T
                + t.reshape(3, 1)
            ).T

            transformed_gt = transformed_flat.reshape(
                valid_gt.shape
            )

            transformed_sequence.append(
                transformed_gt.astype(
                    np.float32,
                    copy=False,
                )
            )

        return transformed_sequence

    @staticmethod
    def _rotation_matrix_from_euler_deg(
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> np.ndarray:
        """
        根据雷达安装姿态偏移生成旧雷达坐标到增强坐标的旋转矩阵。

        雷达坐标系采用 x 前、y 左、z 上的右手系。安装姿态旋转定义为
        D = Rz(yaw) @ Ry(pitch) @ Rx(roll)，同一物理点在扰动后的雷达
        坐标系中表示为 p_aug = D.T @ p，因此返回 A = D.T。
        """
        roll, pitch, yaw = np.deg2rad(
            np.asarray(
                [roll_deg, pitch_deg, yaw_deg],
                dtype=np.float64,
            )
        )

        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        R_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_roll, -sin_roll],
                [0.0, sin_roll, cos_roll],
            ],
            dtype=np.float64,
        )
        R_y = np.asarray(
            [
                [cos_pitch, 0.0, sin_pitch],
                [0.0, 1.0, 0.0],
                [-sin_pitch, 0.0, cos_pitch],
            ],
            dtype=np.float64,
        )
        R_z = np.asarray(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        mounting_rotation = R_z @ R_y @ R_x
        return mounting_rotation.T.astype(
            np.float32,
            copy=False,
        )

    def _sample_rotation_matrix(self) -> np.ndarray:
        """
        为一个 T 帧窗口采样一次安装角扰动。

        使用 PyTorch 随机数生成器，使 DataLoader worker 的标准随机种子
        机制能够控制增强的可复现性。
        """
        roll_deg = torch.empty((), dtype=torch.float32).uniform_(
            *self._ROTATION_ROLL_RANGE_DEG
        ).item()
        pitch_deg = torch.empty((), dtype=torch.float32).uniform_(
            *self._ROTATION_PITCH_RANGE_DEG
        ).item()
        yaw_deg = torch.empty((), dtype=torch.float32).uniform_(
            *self._ROTATION_YAW_RANGE_DEG
        ).item()

        return self._rotation_matrix_from_euler_deg(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )

    @staticmethod
    def _rotate_pointcloud_sequence(
        pointcloud_sequence: List,
        rotation: np.ndarray,
    ) -> List[np.ndarray]:
        """
        绕雷达原点旋转点云序列的 xyz，保持其余点特征不变。
        """
        rotation = np.asarray(rotation, dtype=np.float32)
        if rotation.shape != (3, 3):
            raise ValueError(
                f"rotation 应为 [3,3]，实际为 {rotation.shape}"
            )

        rotated_sequence = []

        for frame_idx, pointcloud_frame in enumerate(
            pointcloud_sequence
        ):
            if pointcloud_frame is None:
                rotated_sequence.append(None)
                continue

            pointcloud = np.array(
                pointcloud_frame,
                dtype=np.float32,
                copy=True,
            )

            if (
                pointcloud.ndim != 2
                or pointcloud.shape[-1] < 3
            ):
                raise ValueError(
                    "雷达点云必须为 [N,C] 且 C>=3，"
                    f"frame_idx={frame_idx}, "
                    f"实际形状={pointcloud.shape}"
                )

            if pointcloud.shape[0] > 0:
                pointcloud[:, :3] = (
                    rotation @ pointcloud[:, :3].T
                ).T

            rotated_sequence.append(pointcloud)

        return rotated_sequence

    def __len__(self) -> int:
        return len(self.data_path_list[self.base_source])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        samples = {}
        # 获取数据
        for sensor_name, flag in self.sensor_config.items():
            if not flag:
                continue

            paths = self.data_path_list.get(sensor_name)
            paths = paths[idx]

            if sensor_name == 'gt':
                if self.enable_action:
                    gt_sequence, action_sequence = (
                        self._get_gt_action_sequence(paths)
                    )
                    samples['gt'] = gt_sequence
                    samples['action'] = action_sequence
                else:
                    samples['gt'] = self._get_sensor_data_from_path(
                        sensor_name,
                        paths,
                    )
                date = paths[0].split('/')[3]
            else:
                data = self._get_sensor_data_from_path(
                    sensor_name,
                    paths,
                )
                samples[sensor_name] = data
        
        calib = self._load_calib_T(date)
        raw_gt = samples['gt']

        R_high = calib['gt_to_high']['R']
        t_high = calib['gt_to_high']['t']
        R_low = calib['gt_to_low']['R']
        t_low = calib['gt_to_low']['t']
        R_high_to_low = calib['high_to_low']['R']
        t_high_to_low = calib['high_to_low']['t']

        if self.enable_rotation:
            # 雷达安装角在一个 T 帧窗口内固定；高低雷达使用同一个 A。
            rotation = self._sample_rotation_matrix()

            for radar_key in (
                'radar_high_pc',
                'radar_low_pc',
            ):
                if radar_key in samples:
                    samples[radar_key] = (
                        self._rotate_pointcloud_sequence(
                            pointcloud_sequence=samples[radar_key],
                            rotation=rotation,
                        )
                    )

            # p_radar_aug = A @ (R @ p_gt + t)
            R_high = rotation @ R_high
            t_high = rotation @ t_high
            R_low = rotation @ R_low
            t_low = rotation @ t_low

            # 高低雷达坐标同时使用 A 后：
            # p_low_aug = A R_hl A.T p_high_aug + A t_hl
            R_high_to_low = (
                rotation
                @ R_high_to_low
                @ rotation.T
            )
            t_high_to_low = rotation @ t_high_to_low

        samples['gt_for_high'] = (
            self._transform_gt_sequence(
                gt_sequence=raw_gt,
                R=R_high,
                t=t_high,
            )
        )

        samples['gt_for_low'] = (
            self._transform_gt_sequence(
                gt_sequence=raw_gt,
                R=R_low,
                t=t_low,
            )
        )

        samples['high_to_low_R'] = [
            R_high_to_low.copy()
            for _ in range(self.T)
        ]
        samples['high_to_low_t'] = [
            t_high_to_low.copy()
            for _ in range(self.T)
        ]
              
        return samples



if __name__ == '__main__':
    """
    radar_high_bin: torch.Size([B, T, 256, 64, 16])

    radar_high_pc
        padded: torch.Size([B, T, N, 6])
        mask:   torch.Size([B, T, N])

    gt
        padded: torch.Size([B, T, K, 17, 3])
        mask:   torch.Size([B, T, K])

    gt_for_high
        padded: torch.Size([B, T, K, 17, 3])
        mask:   torch.Size([B, T, K])
        bbox:   torch.Size([B, T, K, 6])
        action: torch.Size([B, T, K, A])

    gt_for_low
        padded: torch.Size([B, T, K, 17, 3])
        mask:   torch.Size([B, T, K])
        bbox:   torch.Size([B, T, K, 6])
        action: torch.Size([B, T, K, A])

    high_to_low_R: torch.Size([B, T, 3, 3])
    high_to_low_t: torch.Size([B, T, 3])
    """
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from preprocess.radarprocess import Radar_Config, get_radar_res
    from preprocess.radarprocess_RPM2 import (
        range_cube_to_rd_accumulated_angle_power,
        range_cube_to_range_angle_power,
        range_angle_power_to_cartesian_map,
    )
    from utils.COCO import COCO_SKELETON

    root_path = '/mnt/huawei'
    T = 4
    b = 0
    t = -1
    cartesian_size = (256, 256)
    xyz_limits = ((0.0, 6.0), (-3.0, 3.0), (-2.0, 2.0))
    output_path = 'radar_three_methods_with_gt_bbox.png'

    radar_config = Radar_Config()
    dataset = HPE_Dataset(root_path, T=T, radar_config=radar_config)
    collate_fn = partial(
        collate_fn,
        max_points=300,
        max_people=4,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
    )

    for batch_idx, samples in enumerate(dataloader):
        # 可视化一个样本、一个时刻，避免三种角度算法同时处理整个 batch。
        range_cube = samples['radar_high_bin'][b:b + 1, t:t + 1 if t != -1 else None]
        gt_pose = samples['gt_for_high']['padded'][b, t].cpu().numpy()
        bbox = samples['gt_for_high']['bbox'][b, t].cpu().numpy()
        bbox_mask = samples['gt_for_high']['mask'][b, t].cpu().numpy()

        range_res, _, _, _ = get_radar_res(radar_config)
        range_axis = torch.arange(
            range_cube.shape[2],
            dtype=range_cube.real.dtype,
            device=range_cube.device,
        ) * range_res

        fig, axes = plt.subplots(6, 4, figsize=(22, 27))
        with torch.no_grad():
            for row, method in enumerate(('bartlett', 'mvdr', 'music')):
                radar_config.angle_method = method
                projection_maps = []
                for remove_static in (False, True):
                    range_azi_power, azi_axis_rad = (
                        range_cube_to_range_angle_power(
                            range_cube,
                            radar_config,
                            plane='azi',
                            remove_static=remove_static,
                        )
                    )
                    horizontal_power, x_axis, y_axis = (
                        range_angle_power_to_cartesian_map(
                            range_azi_power,
                            range_axis,
                            azi_axis_rad,
                            xyz_limits,
                            cartesian_size,
                            plane='horizontal',
                        )
                    )
                    range_ele_power, ele_axis_rad = (
                        range_cube_to_range_angle_power(
                            range_cube,
                            radar_config,
                            plane='ele',
                            remove_static=remove_static,
                        )
                    )
                    vertical_power, vertical_x_axis, z_axis = (
                        range_angle_power_to_cartesian_map(
                            range_ele_power,
                            range_axis,
                            ele_axis_rad,
                            xyz_limits,
                            cartesian_size,
                            plane='vertical',
                        )
                    )
                    projection_maps.extend((
                        horizontal_power[0, 0],
                        vertical_power[0, 0],
                    ))

                # 三种算法的绝对尺度不同。每种算法内部用四幅图共同的
                # 99.5% 分位值转相对 dB，兼顾前后可比性与异常峰值鲁棒性。
                eps = torch.finfo(projection_maps[0].dtype).eps
                method_reference = torch.quantile(
                    torch.cat([power.reshape(-1) for power in projection_maps]),
                    0.995,
                ).clamp_min(eps)
                projection_db = [
                    (10.0 * torch.log10(
                        power.clamp_min(method_reference * 1e-4)
                        / method_reference
                    ))
                    .clamp(-40.0, 0.0)
                    .cpu().numpy()
                    for power in projection_maps
                ]

                x_values = x_axis.cpu().numpy()
                y_values = y_axis.cpu().numpy()
                vertical_x_values = vertical_x_axis.cpu().numpy()
                z_values = z_axis.cpu().numpy()
                titles = (
                    'Raw XY', 'Raw XZ',
                    'Clutter-removed XY', 'Clutter-removed XZ',
                )

                for col, (power_db, title) in enumerate(zip(projection_db, titles)):
                    is_xy = col % 2 == 0
                    if is_xy:
                        extent = [
                            x_values[0], x_values[-1],
                            y_values[0], y_values[-1],
                        ]
                        ylabel = 'Lateral Y (m)'
                    else:
                        extent = [
                            vertical_x_values[0], vertical_x_values[-1],
                            z_values[0], z_values[-1],
                        ]
                        ylabel = 'Height Z (m)'

                    # map 的维度依次为 [X,Y] 或 [X,Z]；转置后让 X
                    # 对应图像横轴，Y/Z 对应纵轴。
                    image = axes[row, col].imshow(
                        power_db.T,
                        extent=extent,
                        origin='lower',
                        aspect='auto',
                        cmap='turbo',
                        vmin=-40.0,
                        vmax=0.0,
                    )
                    axes[row, col].set(
                        title=f'{method.upper()} {title}',
                        xlabel='Forward X (m)',
                        ylabel=ylabel,
                    )

                    for person_idx in np.flatnonzero(bbox_mask):
                        person_bbox = bbox[person_idx]
                        joints = gt_pose[person_idx]
                        joint_valid = np.isfinite(joints).all(axis=1)
                        color = plt.get_cmap('tab10')(person_idx % 10)
                        xmin, ymin, zmin, xmax, ymax, zmax = person_bbox
                        lower = ymin if is_xy else zmin
                        upper = ymax if is_xy else zmax
                        axes[row, col].add_patch(Rectangle(
                            (xmin, lower), xmax - xmin, upper - lower,
                            fill=False, edgecolor=color, linewidth=2,
                            linestyle='--',
                        ))
                        vertical_coord = joints[:, 1] if is_xy else joints[:, 2]
                        axes[row, col].scatter(
                            joints[joint_valid, 0],
                            vertical_coord[joint_valid],
                            s=14,
                            color=color,
                            edgecolors='white',
                            linewidths=0.4,
                            zorder=3,
                        )
                        for joint_a, joint_b in COCO_SKELETON:
                            if not (joint_valid[joint_a] and joint_valid[joint_b]):
                                continue
                            axes[row, col].plot(
                                [joints[joint_a, 0], joints[joint_b, 0]],
                                [vertical_coord[joint_a], vertical_coord[joint_b]],
                                color=color,
                                linewidth=1.5,
                                zorder=3,
                            )

                colorbar = fig.colorbar(
                    image,
                    ax=axes[row, :].tolist(),
                    fraction=0.015,
                    pad=0.01,
                )
                colorbar.set_label('Power relative to joint 99.5% level (dB)')

            for method_idx, method in enumerate(('bartlett', 'mvdr', 'music')):
                radar_config.angle_method = method
                rd_projection_maps = []
                for zero_doppler_weight in (1.0, 0.1):
                    for plane, cartesian_plane in (('azi', 'horizontal'), ('ele', 'vertical')):
                        rd_angle_power, rd_angle_axis = range_cube_to_rd_accumulated_angle_power(
                            range_cube, radar_config, plane=plane, snr_threshold=4.0,
                            topk_doppler=12, zero_doppler_weight=zero_doppler_weight,
                            normalization_gamma=0.5,
                        )
                        rd_map, rd_x_axis, rd_vertical_axis = range_angle_power_to_cartesian_map(
                            rd_angle_power, range_axis, rd_angle_axis, xyz_limits,
                            cartesian_size, plane=cartesian_plane,
                        )
                        rd_projection_maps.append((rd_map[0, 0], rd_x_axis, rd_vertical_axis))

                rd_reference = torch.quantile(torch.cat([item[0].reshape(-1) for item in rd_projection_maps]), 0.995).clamp_min(torch.finfo(rd_projection_maps[0][0].dtype).eps)
                for col, (rd_map, rd_x_axis, rd_vertical_axis) in enumerate(rd_projection_maps):
                    rd_db = (10.0 * torch.log10(rd_map.clamp_min(rd_reference * 1e-4) / rd_reference)).clamp(-40.0, 0.0).cpu().numpy()
                    is_xy = col % 2 == 0
                    rd_x_values, rd_vertical_values = rd_x_axis.cpu().numpy(), rd_vertical_axis.cpu().numpy()
                    image = axes[method_idx + 3, col].imshow(rd_db.T, extent=[rd_x_values[0], rd_x_values[-1], rd_vertical_values[0], rd_vertical_values[-1]], origin='lower', aspect='auto', cmap='turbo', vmin=-40.0, vmax=0.0)
                    rd_title = 'RD-selected' if col < 2 else 'RD-selected zero-Doppler suppressed'
                    axes[method_idx + 3, col].set(title=f'{method.upper()} {rd_title} {"XY" if is_xy else "XZ"}', xlabel='Forward X (m)', ylabel='Lateral Y (m)' if is_xy else 'Height Z (m)')
                    for person_idx in np.flatnonzero(bbox_mask):
                        person_bbox, joints = bbox[person_idx], gt_pose[person_idx]
                        joint_valid = np.isfinite(joints).all(axis=1)
                        color = plt.get_cmap('tab10')(person_idx % 10)
                        xmin, ymin, zmin, xmax, ymax, zmax = person_bbox
                        lower, upper = (ymin, ymax) if is_xy else (zmin, zmax)
                        axes[method_idx + 3, col].add_patch(Rectangle((xmin, lower), xmax - xmin, upper - lower, fill=False, edgecolor=color, linewidth=2, linestyle='--'))
                        vertical_coord = joints[:, 1] if is_xy else joints[:, 2]
                        axes[method_idx + 3, col].scatter(joints[joint_valid, 0], vertical_coord[joint_valid], s=14, color=color, edgecolors='white', linewidths=0.4, zorder=3)
                        for joint_a, joint_b in COCO_SKELETON:
                            if joint_valid[joint_a] and joint_valid[joint_b]:
                                axes[method_idx + 3, col].plot([joints[joint_a, 0], joints[joint_b, 0]], [vertical_coord[joint_a], vertical_coord[joint_b]], color=color, linewidth=1.5, zorder=3)
                fig.colorbar(image, ax=axes[method_idx + 3, :].tolist(), fraction=0.015, pad=0.01).set_label('Relative power (dB)')

        fig.suptitle(f'Batch {batch_idx}, sample {b}, time {t}: radar + GT bbox')
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f'Saved (overwrite): {output_path}')
