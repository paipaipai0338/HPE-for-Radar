import random
import sys
import os
import torch
import copy
import numpy as np
from typing import *
from pathlib import Path
from torch import nn
from functools import partial
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader

from data2datasets.load_json import get_meta_info
from preprocess.radarprocess import get_bin_data, get_pc_data
from preprocess.lidarprocess import get_lidar_data
from preprocess.realsenseprocess import get_realsense_data
from preprocess.gtprocess import get_gt_boxes_list, get_gt_data
from data2datasets.utils import collate_pc_gt_fn


class HPE_Dataset(Dataset):
    
    def __init__(self, root_path='/mnt/huawei', sensor_config=None, mode='train', base_source='radar_high_bin', split_method='group', ratio=0.7, T=8, preload_cache=False):
        super(HPE_Dataset, self).__init__()
        assert mode in ['train', 'val'], 'mode disnmatched'
        split_method = split_method.lower()
        assert split_method in ['person', 'group', 'sequence'], 'split_method has unmatched method'

        self.root_path = Path(root_path)
        self.base_source = base_source
        self.ratio = ratio
        self.T = T
        self.preload_cache = preload_cache
        self.calib_cache = {}
        self.pointcloud_cache = {}
        self.gt_cache = {}
        self.skip_bad_samples = 0
        self.bad_files = set()
        self.npy_valid_cache = {}
        self.gt_valid_cache = {}

        # 加载元信息
        json_path = self.root_path / 'data description.json'
        self.meta_info = get_meta_info(json_path)
        
        # 定义传感器，是否选取该传感器
        self.sensor_config = {
            'lidar': False,
            'radar_low_bin': False,
            'radar_high_bin': True,
            'radar_low_pc': False,
            'radar_high_pc': True,
            'gt': True,
            'realsense': False,
        } if sensor_config is None else sensor_config

        assert self.sensor_config[base_source] is True, 'base_source in sensor_config is False'
        assert self.sensor_config.get('gt') is True, '单人数据集必须启用 gt'
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
            'lidar',
            'radar_low_pc',
            'radar_high_pc',
            'gt',
            'realsense',
        }
        # 更新 meta_info
        self._build_aligned_data()
        
        # 数据划分 TODO：sequence
        self.meta_info_splited = {'train': copy.deepcopy(self.meta_info), 'val': copy.deepcopy(self.meta_info)}
        if split_method == 'person':
            # 按人来划分
            train_person_ids = {'0', '1', '2', '3', '5'}
            val_person_ids = {'4'}

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
                    starts = list(range(0, frame - T + 1))
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
        if file_path in self.npy_valid_cache:
            return self.npy_valid_cache[file_path]

        try:
            array = np.load(
                file_path,
                mmap_mode="r",
            )

            # 雷达点云必须为非空的 [N, 6]。部分文件虽然能正常读取
            # npy header，但实际保存的是 shape=(0,) 的一维空数组，
            # 会在 __getitem__ 中执行 frame_pc[:, :3] 时崩溃。
            if sensor_name in {'radar_low_pc', 'radar_high_pc'}:
                valid = (
                    array.ndim == 2
                    and array.shape[1] == 6
                    and array.shape[0] > 0
                )
            else:
                valid = True

            if not valid:
                if (
                    file_path not in self.bad_files
                    and len(self.bad_files) < 10
                ):
                    print(
                        "跳过形状无效的文件: "
                        f"sensor={sensor_name}, "
                        f"path={file_path}, "
                        f"shape={array.shape}, "
                        f"dtype={array.dtype}"
                    )
                self.bad_files.add(file_path)

            del array

        except Exception as exc:
            valid = False

            if (
                file_path not in self.bad_files
                and len(self.bad_files) < 10
            ):
                print(
                    f"跳过损坏文件: "
                    f"sensor={sensor_name}, "
                    f"path={file_path}, "
                    f"error={type(exc).__name__}: {exc}"
                )

            self.bad_files.add(file_path)

        self.npy_valid_cache[file_path] = valid

        return valid

    def _is_valid_window(
        self,
        window_by_sensor: Dict[str, List[str]],
    ) -> bool:
        gt_files = window_by_sensor.get('gt')
        if gt_files is None:
            return False

        for file_path in gt_files:
            if file_path not in self.gt_valid_cache:
                try:
                    gt = get_gt_data(file_path)
                    valid = (
                        gt is not None
                        and gt.ndim == 3
                        and gt.shape[1:] == (17, 3)
                        and gt.shape[0] == 1
                        and np.isfinite(gt).all()
                    )
                except Exception as exc:
                    valid = False
                    if (
                        file_path not in self.bad_files
                        and len(self.bad_files) < 10
                    ):
                        print(
                            "跳过无效 GT 文件: "
                            f"path={file_path}, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                    self.bad_files.add(file_path)

                self.gt_valid_cache[file_path] = valid

            # 当前单人版本只接收恰好包含一个人的 GT。
            # 多人数据的拆分与选择逻辑留待后续实现。
            if not self.gt_valid_cache[file_path]:
                return False

        for sensor_name, window_files in window_by_sensor.items():
            if self.suffix_map[sensor_name] != ".npy":
                continue

            for file_path in window_files:
                if not self._is_valid_npy(
                    sensor_name,
                    file_path,
                ):
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
                                'group_name: aligned_frames
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
                    
                    if not aligned_frames:
                        print(f"{group_name} 对齐后没有数据")
                        continue
                    group_data_path[f'{group_name}'] = aligned_frames
                entry['group_data_path'] = group_data_path
    
    def _align_multi_sensor_files(
        self,
        sources: Dict[str, Optional[Path]],
        max_delta_sec: Optional[float] = 0.5,
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
                base = Path(file).stem
                sec_str, ns_str = base.split('_')
                sec = int(sec_str)
                ns = int(ns_str)
                unix_ts = sec + ns * 1e-9
                times.append(unix_to_datetime(unix_ts))
            return times
        
        def list_files(dir_path: Optional[str], suffix: str) -> Tuple[List[str], List[datetime]]:
            """列出目录中匹配后缀的文件并解析时间戳"""
            if not dir_path or not suffix:
                return [], []
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
            files, times = list_files(path, suffix)
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

                frame_paths[name] = os.path.join(
                    data['path'],
                    data['files'][sensor_idx],
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
        radar_high_bin_path = radar_high_path / 'Bin'
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
        get_data_function_dict = {
            'lidar': get_lidar_data,
            'radar_low_bin': get_bin_data,
            'radar_high_bin': get_bin_data,
            'radar_low_pc': get_pc_data,
            'radar_high_pc': get_pc_data,
            'gt': get_gt_data,
            'realsense': get_realsense_data,
        }

        return get_data_function_dict[sensor_name]

    def preload_data_cache(self) -> None:
        """
        在主进程中预热点云和 GT 缓存。

        Linux 默认 fork worker 时，这些只读缓存的数据 buffer 可以被子进程共享。
        """
        for sensor_name, path_windows in self.data_path_list.items():
            if sensor_name not in self.cached_sensor_names:
                continue

            load_fn = self._get_sensor_loader(sensor_name)
            seen_paths = set()
            for window_paths in path_windows:
                for path in window_paths:
                    path_key = str(path)
                    if path_key in seen_paths:
                        continue

                    self._cache_frame_data(
                        sensor_name=sensor_name,
                        path=path,
                        load_fn=load_fn,
                    )
                    seen_paths.add(path_key)

    def _get_sensor_data_from_path(self, sensor_name: str, sensor_path: List[Path|str]) -> List:
        if sensor_path is None:
            return None

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

    def clear_data_cache(self) -> None:
        """
        清空点云和 GT 的内存缓存；不会影响标定和 npy 有效性缓存。
        """
        self.pointcloud_cache.clear()
        self.gt_cache.clear()

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
            data = self._get_sensor_data_from_path(sensor_name, paths)
            samples[sensor_name] = data

            if sensor_name == 'gt':
                date = paths[0].split('/')[3]
        
        calib = self._load_calib_T(date)
        raw_gt = samples['gt']

        samples['gt_for_high'] = (
            self._transform_gt_sequence(
                gt_sequence=raw_gt,
                R=calib['gt_to_high']['R'],
                t=calib['gt_to_high']['t'],
            )
        )

        samples['gt_for_low'] = (
            self._transform_gt_sequence(
                gt_sequence=raw_gt,
                R=calib['gt_to_low']['R'],
                t=calib['gt_to_low']['t'],
            )
        )

        samples['high_to_low_R'] = [
            calib['high_to_low']['R'].copy()
            for _ in range(self.T)
        ]
        samples['high_to_low_t'] = [
            calib['high_to_low']['t'].copy()
            for _ in range(self.T)
        ]

        # temp：单人 GT 框选高位雷达点云，并以髋部中心进行位置归一化。
        gt_boxes = get_gt_boxes_list(
            samples['gt_for_high'],
            threshold=0.1,
        )

        for frame_idx in range(self.T):
            frame_gt = samples['gt_for_high'][frame_idx]
            frame_pc = samples['radar_high_pc'][frame_idx]
            num_people = frame_gt.shape[0]

            if num_people > 1:
                raise ValueError(
                    "当前版本只支持单人估计，"
                    f"frame_idx={frame_idx}, num_people={num_people}"
                )

            # 没有 GT 时无法确定人体点云区域，因此返回空点云。
            if num_people == 0:
                samples['radar_high_pc'][frame_idx] = frame_pc[:0].copy()
                continue

            bbox = gt_boxes[frame_idx][0]
            min_xyz = bbox[:3]
            max_xyz = bbox[3:]

            # 只保留落在该人 3D GT 包围盒内的雷达点。
            xyz = frame_pc[:, :3]
            inside = (
                (xyz >= min_xyz[None, :])
                & (xyz <= max_xyz[None, :])
            ).all(axis=1)
            selected_pc = frame_pc[inside].copy()

            # 使用该人的 11、12 号关节点中点作为坐标原点。
            offset = (
                frame_gt[0, 11, :] + frame_gt[0, 12, :]
            ) / 2.0

            samples['gt_for_high'][frame_idx] = (
                frame_gt - offset[None, None, :]
            )
            selected_pc[:, :3] -= offset[None, :]
            samples['radar_high_pc'][frame_idx] = selected_pc
              
        return samples



if __name__ == '__main__':
    from matplotlib import pyplot as plt

    root_path = '/mnt/huawei'
    T = 4
    batch_sample_idx = 0
    time_idx = 0
    pc_3d_x_limits = (-3.0, 3.0)
    pc_3d_y_limits = (-3.0, 3.0)
    pc_3d_z_limits = (-3.0, 3.0)

    dataset = HPE_Dataset(root_path, T=T)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=partial(
            collate_pc_gt_fn,
            max_points=300,
            max_people=1,
        ),
        shuffle=False,
        num_workers=4,
    )

    for batch_idx, samples in enumerate(dataloader):
        pc_mask = samples['radar_high_pc']['mask'][
            batch_sample_idx, time_idx
        ].bool()
        gt_mask = samples['gt_for_high']['mask'][
            batch_sample_idx, time_idx
        ].bool()

        point_cloud = samples['radar_high_pc']['padded'][
            batch_sample_idx, time_idx
        ][pc_mask, :3].cpu().numpy()
        gt = samples['gt_for_high']['padded'][
            batch_sample_idx, time_idx
        ][gt_mask].cpu().numpy()

        point_cloud = point_cloud[
            np.isfinite(point_cloud).all(axis=1)
        ]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        if point_cloud.shape[0] > 0:
            ax.scatter(
                point_cloud[:, 0],
                point_cloud[:, 1],
                point_cloud[:, 2],
                s=5,
                c='blue',
                alpha=0.6,
                label='Radar point cloud',
            )

        if gt.shape[0] > 0:
            joints = gt[0]
            joints = joints[np.isfinite(joints).all(axis=1)]
            ax.scatter(
                joints[:, 0],
                joints[:, 1],
                joints[:, 2],
                s=30,
                c='red',
                marker='x',
                linewidth=2,
                label='GT',
            )

        ax.set_xlim(pc_3d_x_limits)
        ax.set_ylim(pc_3d_y_limits)
        ax.set_zlim(pc_3d_z_limits)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(
            f'radar high pc and gt, batch {batch_idx}, '
            f'sample {batch_sample_idx}, time {time_idx}'
        )
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(fontsize=8)

        fig.tight_layout()
        save_path = '/home/pai/Huawei/temp.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'visualization saved to: {save_path}')
        break
