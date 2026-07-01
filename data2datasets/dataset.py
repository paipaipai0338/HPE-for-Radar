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
from preprocess.gtprocess import get_gt_data
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

        # 加载元信息
        json_path = self.root_path / 'data description.json'
        self.meta_info = get_meta_info(json_path)
        
        # 定义传感器，是否选取该传感器
        self.sensor_config = {
            'lidar': False,
            'radar_low_bin': True,
            'radar_high_bin': True,
            'radar_low_pc': True,
            'radar_high_pc': True,
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
            'lidar',
            'radar_low_pc',
            'radar_high_pc',
            'gt',
            'realsense',
        }
        # 更新 meta_info
        self._build_aligned_data()
        
        # 数据划分 TODO：person, sequence
        self.meta_info_splited = {'train': copy.deepcopy(self.meta_info), 'val': copy.deepcopy(self.meta_info)}
        if split_method == 'person':
            # 按人来划分
            pass
        elif split_method == 'group':
            # 按照组来划分
            for person, person_data in self.meta_info.items():
                for idx, entry in enumerate(person_data):
                    valid_group = entry['valid_group']

                    self.meta_info_splited['train'][person][idx]['valid_group'] = self.meta_info_splited['train'][person][idx]['valid_group'][:int(ratio*len(valid_group))]
                    self.meta_info_splited['train'][person][idx]['group_data_path'] = {k: self.meta_info_splited['train'][person][idx]['group_data_path'][k] for k in self.meta_info_splited['train'][person][idx]['valid_group']}
                    self.meta_info_splited['val'][person][idx]['valid_group'] = self.meta_info_splited['val'][person][idx]['valid_group'][int(ratio*len(valid_group)):]
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

            # 确保至少读取并解析 header
            _ = array.shape
            _ = array.dtype

            del array
            valid = True

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

        calib = {
            'gt_to_low': {
                'R': R_low,
                't': t_low,
            },
            'gt_to_high': {
                'R': R_high,
                't': t_high,
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
              
        return samples



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from preprocess.radarprocess import Radar_Config, get_radar_res
    from preprocess.radarprocess_RPM2 import SingleRadarProjectionConfig, range_cube_to_rpm_projection_maps, power_to_db

    root_path = '/mnt/huawei'
    T = 4
    b, t = 0, 1
    clutter_mode = 'frame_difference'
    xy_x_limits = (0.1, 5.0)
    xy_y_limits = (-2.0, 2.0)
    xz_x_limits = (0.1, 5.0)
    xz_z_limits = (-1.5, 1.5)
    range_plot_limits = (0.1, 5.0)
    pc_3d_x_limits = (0.0, 6.0)
    pc_3d_y_limits = (-3.0, 3.0)
    pc_3d_z_limits = (-3.0, 3.0)

    radar_config = Radar_Config()
    projection_config = SingleRadarProjectionConfig()

    dataset = HPE_Dataset(root_path, T=T)
    collate_fn = partial(
        collate_pc_gt_fn,
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
        for key, value in samples.items():
            if isinstance(value, dict) and 'padded' in value:
                print(f"{key}: padded {value['padded'].shape}, mask {value['mask'].shape}")
            else:
                print(f"{key}: {value.shape}")

        # =========================================================
        # 数据提取
        # =========================================================

        # Range FFT cube
        range_cube_low = samples['radar_low_bin']
        range_cube_high = samples['radar_high_bin']

        # 点云
        pc_low_mask = samples['radar_low_pc']['mask'][b, t].bool()
        pc_high_mask = samples['radar_high_pc']['mask'][b, t].bool()

        pc_low_valid = samples['radar_low_pc']['padded'][b, t][pc_low_mask]
        pc_high_valid = samples['radar_high_pc']['padded'][b, t][pc_high_mask]

        # GT
        gt_low_mask = samples['gt_for_low']['mask'][b, t].bool()
        gt_high_mask = samples['gt_for_high']['mask'][b, t].bool()

        gt_low = samples['gt_for_low']['padded'][b, t][gt_low_mask]
        gt_high = samples['gt_for_high']['padded'][b, t][gt_high_mask]

        # 转为 numpy
        pc_low_valid = pc_low_valid.detach().cpu().numpy()
        pc_high_valid = pc_high_valid.detach().cpu().numpy()
        gt_low = gt_low.detach().cpu().numpy()
        gt_high = gt_high.detach().cpu().numpy()

        # 只保留点云 xyz，并删除 NaN/Inf
        pc_low_valid_xyz = pc_low_valid[:, :3]
        pc_high_valid_xyz = pc_high_valid[:, :3]

        pc_low_valid_xyz = pc_low_valid_xyz[np.isfinite(pc_low_valid_xyz).all(axis=1)]
        pc_high_valid_xyz = pc_high_valid_xyz[np.isfinite(pc_high_valid_xyz).all(axis=1)]

        # =========================================================
        # 点云绘制范围过滤
        # =========================================================

        # 低位雷达水平 x-y 投影范围
        pc_low_xy_mask = (
            (pc_low_valid_xyz[:, 0] >= xy_x_limits[0])
            & (pc_low_valid_xyz[:, 0] <= xy_x_limits[1])
            & (pc_low_valid_xyz[:, 1] >= xy_y_limits[0])
            & (pc_low_valid_xyz[:, 1] <= xy_y_limits[1])
        )
        pc_low_xy = pc_low_valid_xyz[pc_low_xy_mask]

        # 低位雷达垂直 x-z 投影范围
        pc_low_xz_mask = (
            (pc_low_valid_xyz[:, 0] >= xz_x_limits[0])
            & (pc_low_valid_xyz[:, 0] <= xz_x_limits[1])
            & (pc_low_valid_xyz[:, 2] >= xz_z_limits[0])
            & (pc_low_valid_xyz[:, 2] <= xz_z_limits[1])
        )
        pc_low_xz = pc_low_valid_xyz[pc_low_xz_mask]

        # 低位雷达三维显示范围
        pc_low_3d_mask = (
            (pc_low_valid_xyz[:, 0] >= pc_3d_x_limits[0])
            & (pc_low_valid_xyz[:, 0] <= pc_3d_x_limits[1])
            & (pc_low_valid_xyz[:, 1] >= pc_3d_y_limits[0])
            & (pc_low_valid_xyz[:, 1] <= pc_3d_y_limits[1])
            & (pc_low_valid_xyz[:, 2] >= pc_3d_z_limits[0])
            & (pc_low_valid_xyz[:, 2] <= pc_3d_z_limits[1])
        )
        pc_low_3d = pc_low_valid_xyz[pc_low_3d_mask]

        # 高位雷达水平 x-y 投影范围
        pc_high_xy_mask = (
            (pc_high_valid_xyz[:, 0] >= xy_x_limits[0])
            & (pc_high_valid_xyz[:, 0] <= xy_x_limits[1])
            & (pc_high_valid_xyz[:, 1] >= xy_y_limits[0])
            & (pc_high_valid_xyz[:, 1] <= xy_y_limits[1])
        )
        pc_high_xy = pc_high_valid_xyz[pc_high_xy_mask]

        # 高位雷达垂直 x-z 投影范围
        pc_high_xz_mask = (
            (pc_high_valid_xyz[:, 0] >= xz_x_limits[0])
            & (pc_high_valid_xyz[:, 0] <= xz_x_limits[1])
            & (pc_high_valid_xyz[:, 2] >= xz_z_limits[0])
            & (pc_high_valid_xyz[:, 2] <= xz_z_limits[1])
        )
        pc_high_xz = pc_high_valid_xyz[pc_high_xz_mask]

        # 高位雷达三维显示范围
        pc_high_3d_mask = (
            (pc_high_valid_xyz[:, 0] >= pc_3d_x_limits[0])
            & (pc_high_valid_xyz[:, 0] <= pc_3d_x_limits[1])
            & (pc_high_valid_xyz[:, 1] >= pc_3d_y_limits[0])
            & (pc_high_valid_xyz[:, 1] <= pc_3d_y_limits[1])
            & (pc_high_valid_xyz[:, 2] >= pc_3d_z_limits[0])
            & (pc_high_valid_xyz[:, 2] <= pc_3d_z_limits[1])
        )
        pc_high_3d = pc_high_valid_xyz[pc_high_3d_mask]

        # =========================================================
        # Range axis
        # =========================================================
        _, _, R_low, _, _ = range_cube_low.shape
        _, _, R_high, _, _ = range_cube_high.shape

        range_res, _, _, _ = get_radar_res(radar_config)

        range_axis_low = torch.arange(
            R_low,
            device=range_cube_low.device,
            dtype=torch.float32,
        ) * range_res

        range_axis_high = torch.arange(
            R_high,
            device=range_cube_high.device,
            dtype=torch.float32,
        ) * range_res

        # =========================================================
        # RPM 投影
        # =========================================================
        projection_low = range_cube_to_rpm_projection_maps(
            range_cube=range_cube_low,
            range_axis=range_axis_low,
            wavelength=radar_config.lam,
            projection_config=projection_config,
            xy_limits=(xy_x_limits, xy_y_limits),
            xz_limits=(xz_x_limits, xz_z_limits),
            xy_size=(256, 256),
            xz_size=(256, 256),
            clutter_mode=clutter_mode,
        )

        projection_high = range_cube_to_rpm_projection_maps(
            range_cube=range_cube_high,
            range_axis=range_axis_high,
            wavelength=radar_config.lam,
            projection_config=projection_config,
            xy_limits=(xy_x_limits, xy_y_limits),
            xz_limits=(xz_x_limits, xz_z_limits),
            xy_size=(256, 256),
            xz_size=(256, 256),
            clutter_mode=clutter_mode,
        )

        # =========================================================
        # 投影时间索引
        # =========================================================
        projection_t_low = t - projection_low.time_start_index
        projection_t_high = t - projection_high.time_start_index

        if not 0 <= projection_t_low < projection_low.horizontal_xy_power.shape[1]:
            raise IndexError(
                f'低位雷达时间索引错误：raw_t={t}, '
                f'projection_t={projection_t_low}, '
                f'time_start_index={projection_low.time_start_index}'
            )

        if not 0 <= projection_t_high < projection_high.horizontal_xy_power.shape[1]:
            raise IndexError(
                f'高位雷达时间索引错误：raw_t={t}, '
                f'projection_t={projection_t_high}, '
                f'time_start_index={projection_high.time_start_index}'
            )

        # =========================================================
        # 投影功率转相对 dB
        # =========================================================
        horizontal_xy_db_low = power_to_db(projection_low.horizontal_xy_power[b, projection_t_low])
        vertical_xz_db_low = power_to_db(projection_low.vertical_xz_power[b, projection_t_low])
        range_azimuth_db_low = power_to_db(projection_low.range_azimuth_power[b, projection_t_low])
        range_elevation_db_low = power_to_db(projection_low.range_elevation_power[b, projection_t_low])

        horizontal_xy_db_high = power_to_db(projection_high.horizontal_xy_power[b, projection_t_high])
        vertical_xz_db_high = power_to_db(projection_high.vertical_xz_power[b, projection_t_high])
        range_azimuth_db_high = power_to_db(projection_high.range_azimuth_power[b, projection_t_high])
        range_elevation_db_high = power_to_db(projection_high.range_elevation_power[b, projection_t_high])

        # horizontal_xy_db_low = horizontal_xy_db_low - horizontal_xy_db_low.max()
        # vertical_xz_db_low = vertical_xz_db_low - vertical_xz_db_low.max()
        # range_azimuth_db_low = range_azimuth_db_low - range_azimuth_db_low.max()
        # range_elevation_db_low = range_elevation_db_low - range_elevation_db_low.max()

        # horizontal_xy_db_high = horizontal_xy_db_high - horizontal_xy_db_high.max()
        # vertical_xz_db_high = vertical_xz_db_high - vertical_xz_db_high.max()
        # range_azimuth_db_high = range_azimuth_db_high - range_azimuth_db_high.max()
        # range_elevation_db_high = range_elevation_db_high - range_elevation_db_high.max()

        horizontal_xy_db_low = horizontal_xy_db_low.detach().cpu().numpy()
        vertical_xz_db_low = vertical_xz_db_low.detach().cpu().numpy()
        range_azimuth_db_low = range_azimuth_db_low.detach().cpu().numpy()
        range_elevation_db_low = range_elevation_db_low.detach().cpu().numpy()

        horizontal_xy_db_high = horizontal_xy_db_high.detach().cpu().numpy()
        vertical_xz_db_high = vertical_xz_db_high.detach().cpu().numpy()
        range_azimuth_db_high = range_azimuth_db_high.detach().cpu().numpy()
        range_elevation_db_high = range_elevation_db_high.detach().cpu().numpy()

        # =========================================================
        # 投影坐标轴
        # =========================================================
        x_axis_xy_low = projection_low.horizontal_x_axis.detach().cpu().numpy()
        y_axis_low = projection_low.horizontal_y_axis.detach().cpu().numpy()
        x_axis_xz_low = projection_low.vertical_x_axis.detach().cpu().numpy()
        z_axis_low = projection_low.vertical_z_axis.detach().cpu().numpy()
        range_axis_low_np = projection_low.range_axis.detach().cpu().numpy()
        azimuth_axis_deg_low = np.rad2deg(projection_low.azimuth_axis_rad.detach().cpu().numpy())
        elevation_axis_deg_low = np.rad2deg(projection_low.elevation_axis_rad.detach().cpu().numpy())

        x_axis_xy_high = projection_high.horizontal_x_axis.detach().cpu().numpy()
        y_axis_high = projection_high.horizontal_y_axis.detach().cpu().numpy()
        x_axis_xz_high = projection_high.vertical_x_axis.detach().cpu().numpy()
        z_axis_high = projection_high.vertical_z_axis.detach().cpu().numpy()
        range_axis_high_np = projection_high.range_axis.detach().cpu().numpy()
        azimuth_axis_deg_high = np.rad2deg(projection_high.azimuth_axis_rad.detach().cpu().numpy())
        elevation_axis_deg_high = np.rad2deg(projection_high.elevation_axis_rad.detach().cpu().numpy())

        # =========================================================
        # 点云转换到 Range-Azimuth / Range-Elevation 坐标
        # =========================================================
        pc_low_range = np.linalg.norm(pc_low_valid_xyz, axis=1)
        pc_low_angle_mask = pc_low_range > 1e-6
        pc_low_angle_xyz = pc_low_valid_xyz[pc_low_angle_mask]
        pc_low_range = pc_low_range[pc_low_angle_mask]
        pc_low_azimuth_deg = np.rad2deg(np.arcsin(np.clip(pc_low_angle_xyz[:, 1] / pc_low_range, -1.0, 1.0)))
        pc_low_elevation_deg = np.rad2deg(np.arcsin(np.clip(pc_low_angle_xyz[:, 2] / pc_low_range, -1.0, 1.0)))

        pc_low_ra_mask = (
            (pc_low_range >= range_plot_limits[0])
            & (pc_low_range <= range_plot_limits[1])
            & (pc_low_azimuth_deg >= azimuth_axis_deg_low[0])
            & (pc_low_azimuth_deg <= azimuth_axis_deg_low[-1])
        )
        pc_low_re_mask = (
            (pc_low_range >= range_plot_limits[0])
            & (pc_low_range <= range_plot_limits[1])
            & (pc_low_elevation_deg >= elevation_axis_deg_low[0])
            & (pc_low_elevation_deg <= elevation_axis_deg_low[-1])
        )

        pc_high_range = np.linalg.norm(pc_high_valid_xyz, axis=1)
        pc_high_angle_mask = pc_high_range > 1e-6
        pc_high_angle_xyz = pc_high_valid_xyz[pc_high_angle_mask]
        pc_high_range = pc_high_range[pc_high_angle_mask]
        pc_high_azimuth_deg = np.rad2deg(np.arcsin(np.clip(pc_high_angle_xyz[:, 1] / pc_high_range, -1.0, 1.0)))
        pc_high_elevation_deg = np.rad2deg(np.arcsin(np.clip(pc_high_angle_xyz[:, 2] / pc_high_range, -1.0, 1.0)))

        pc_high_ra_mask = (
            (pc_high_range >= range_plot_limits[0])
            & (pc_high_range <= range_plot_limits[1])
            & (pc_high_azimuth_deg >= azimuth_axis_deg_high[0])
            & (pc_high_azimuth_deg <= azimuth_axis_deg_high[-1])
        )
        pc_high_re_mask = (
            (pc_high_range >= range_plot_limits[0])
            & (pc_high_range <= range_plot_limits[1])
            & (pc_high_elevation_deg >= elevation_axis_deg_high[0])
            & (pc_high_elevation_deg <= elevation_axis_deg_high[-1])
        )

        # =========================================================
        # 打印关键数据shape
        # =========================================================
        print("\n" + "="*50)
        print(f"Batch {batch_idx}, Sample {b}, Time {t}")
        print("="*50)
        
        # Range cube shape
        print(f"range_cube_low shape: {range_cube_low.shape}")  # [B, T, R, Az, El]
        print(f"range_cube_high shape: {range_cube_high.shape}")
        
        # Point cloud shape
        print(f"pc_low_valid shape: {pc_low_valid.shape}")  # [N, 4+]
        print(f"pc_high_valid shape: {pc_high_valid.shape}")
        print(f"pc_low_valid_xyz shape: {pc_low_valid_xyz.shape}")  # [N, 3]
        print(f"pc_high_valid_xyz shape: {pc_high_valid_xyz.shape}")
        
        # GT shape
        print(f"gt_low shape: {gt_low.shape}")  # [P, J, 3]
        print(f"gt_high shape: {gt_high.shape}")
        
        # 过滤后的点云shape
        print(f"pc_low_xy shape: {pc_low_xy.shape}")
        print(f"pc_low_xz shape: {pc_low_xz.shape}")
        print(f"pc_high_xy shape: {pc_high_xy.shape}")
        print(f"pc_high_xz shape: {pc_high_xz.shape}")
        
        # Projection shapes
        print(f"horizontal_xy_db_low shape: {horizontal_xy_db_low.shape}")  # [H, W]
        print(f"vertical_xz_db_low shape: {vertical_xz_db_low.shape}")
        print(f"range_azimuth_db_low shape: {range_azimuth_db_low.shape}")
        print(f"range_elevation_db_low shape: {range_elevation_db_low.shape}")
        print(f"horizontal_xy_db_high shape: {horizontal_xy_db_high.shape}")
        print(f"vertical_xz_db_high shape: {vertical_xz_db_high.shape}")
        print(f"range_azimuth_db_high shape: {range_azimuth_db_high.shape}")
        print(f"range_elevation_db_high shape: {range_elevation_db_high.shape}")
        
        # PC in range-azimuth/elevation space
        print(f"pc_low_range shape: {pc_low_range.shape}")
        print(f"pc_low_azimuth_deg shape: {pc_low_azimuth_deg.shape}")
        print(f"pc_low_elevation_deg shape: {pc_low_elevation_deg.shape}")
        print(f"pc_high_range shape: {pc_high_range.shape}")
        print(f"pc_high_azimuth_deg shape: {pc_high_azimuth_deg.shape}")
        print(f"pc_high_elevation_deg shape: {pc_high_elevation_deg.shape}")
        print("="*50 + "\n")

        # =========================================================
        # 创建画布
        # =========================================================
        fig = plt.figure(figsize=(40, 11))

        ax1 = fig.add_subplot(2, 5, 1)
        ax2 = fig.add_subplot(2, 5, 2)
        ax3 = fig.add_subplot(2, 5, 3)
        ax4 = fig.add_subplot(2, 5, 4)
        ax5 = fig.add_subplot(2, 5, 5, projection='3d')

        ax6 = fig.add_subplot(2, 5, 6)
        ax7 = fig.add_subplot(2, 5, 7)
        ax8 = fig.add_subplot(2, 5, 8)
        ax9 = fig.add_subplot(2, 5, 9)
        ax10 = fig.add_subplot(2, 5, 10, projection='3d')

        # =========================================================
        # 低位雷达：水平 x-y 投影（增强散点可见性）
        # =========================================================
        image1 = ax1.imshow(
            horizontal_xy_db_low,
            origin='lower',
            aspect='auto',
            extent=[float(y_axis_low[0]), float(y_axis_low[-1]), float(x_axis_xy_low[0]), float(x_axis_xy_low[-1])],
            cmap='viridis',
            vmin=-40,
            vmax=40,
        )

        if len(pc_low_xy) > 0:
            ax1.scatter(
                pc_low_xy[:, 1], 
                pc_low_xy[:, 0], 
                s=8,                    # 增大点尺寸
                c='yellow', 
                alpha=0.7,              # 提高不透明度
                edgecolors='white',     # 添加白色边缘
                linewidth=0.5,
                label='PC'
            )

        for person_idx, joints in enumerate(gt_low):
            joints = joints[np.isfinite(joints).all(axis=1)]
            joints_xy_mask = (
                (joints[:, 0] >= xy_x_limits[0])
                & (joints[:, 0] <= xy_x_limits[1])
                & (joints[:, 1] >= xy_y_limits[0])
                & (joints[:, 1] <= xy_y_limits[1])
            )
            joints_xy = joints[joints_xy_mask]

            if len(joints_xy) == 0:
                continue

            ax1.scatter(
                joints_xy[:, 1],
                joints_xy[:, 0],
                s=30,                   # GT点明显增大
                marker='x',
                c='red',               # 改为更亮的颜色
                alpha=0.9,
                linewidth=2,
                label='GT' if person_idx == 0 else None,
            )

        ax1.set_title('radar low horizontal x-y projection')
        ax1.set_xlabel('Y (m, left positive)')
        ax1.set_ylabel('X (m, forward)')
        ax1.set_xlim(xy_y_limits)
        ax1.set_ylim(xy_x_limits)
        ax1.invert_xaxis()
        ax1.legend(fontsize=8)          # 添加图例
        fig.colorbar(image1, ax=ax1, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # =========================================================
        # 低位雷达：垂直 x-z 投影（增强散点可见性）
        # =========================================================
        image2 = ax2.imshow(
            vertical_xz_db_low,
            origin='lower',
            aspect='auto',
            extent=[float(x_axis_xz_low[0]), float(x_axis_xz_low[-1]), float(z_axis_low[0]), float(z_axis_low[-1])],
            cmap='viridis',
            vmin=-40,
            vmax=40,
        )

        if len(pc_low_xz) > 0:
            ax2.scatter(
                pc_low_xz[:, 0], 
                pc_low_xz[:, 2], 
                s=8,                    # 增大点尺寸
                c='yellow', 
                alpha=0.7,              # 提高不透明度
                edgecolors='white',
                linewidth=0.5,
                label='PC'
            )

        for person_idx, joints in enumerate(gt_low):
            joints = joints[np.isfinite(joints).all(axis=1)]
            joints_xz_mask = (
                (joints[:, 0] >= xz_x_limits[0])
                & (joints[:, 0] <= xz_x_limits[1])
                & (joints[:, 2] >= xz_z_limits[0])
                & (joints[:, 2] <= xz_z_limits[1])
            )
            joints_xz = joints[joints_xz_mask]

            if len(joints_xz) == 0:
                continue

            ax2.scatter(
                joints_xz[:, 0],
                joints_xz[:, 2],
                s=30,                   # GT点明显增大
                marker='x',
                c='red',               # 改为更亮的颜色
                alpha=0.9,
                linewidth=2,
                label='GT' if person_idx == 0 else None,
            )

        ax2.set_title('radar low vertical x-z projection')
        ax2.set_xlabel('X (m, forward)')
        ax2.set_ylabel('Z (m, upward)')
        ax2.set_xlim(xz_x_limits)
        ax2.set_ylim(xz_z_limits)
        ax2.legend(fontsize=8)
        fig.colorbar(image2, ax=ax2, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # =========================================================
        # 低位雷达：Range-Azimuth（增强散点可见性）
        # =========================================================
        image3 = ax3.imshow(
            range_azimuth_db_low,
            origin='lower',
            aspect='auto',
            extent=[
                float(azimuth_axis_deg_low[0]),
                float(azimuth_axis_deg_low[-1]),
                float(range_axis_low_np[0]),
                float(range_axis_low_np[-1]),
            ],
            cmap='viridis',
            vmin=-40,
            vmax=40,
        )

        if pc_low_ra_mask.any():
            ax3.scatter(
                pc_low_azimuth_deg[pc_low_ra_mask],
                pc_low_range[pc_low_ra_mask],
                s=8,                    # 增大点尺寸
                c='yellow',
                alpha=0.7,              # 提高不透明度
                edgecolors='white',
                linewidth=0.5,
                label='PC',
            )

        for person_idx, joints in enumerate(gt_low):
            joints = joints[np.isfinite(joints).all(axis=1)]

            if len(joints) == 0:
                continue

            joints_range = np.linalg.norm(joints, axis=1)
            joints_valid = joints_range > 1e-6
            joints = joints[joints_valid]
            joints_range = joints_range[joints_valid]

            if len(joints) == 0:
                continue

            joints_azimuth_deg = np.rad2deg(np.arcsin(np.clip(joints[:, 1] / joints_range, -1.0, 1.0)))
            joints_ra_mask = (
                (joints_range >= range_plot_limits[0])
                & (joints_range <= range_plot_limits[1])
                & (joints_azimuth_deg >= azimuth_axis_deg_low[0])
                & (joints_azimuth_deg <= azimuth_axis_deg_low[-1])
            )

            if not joints_ra_mask.any():
                continue

            ax3.scatter(
                joints_azimuth_deg[joints_ra_mask],
                joints_range[joints_ra_mask],
                s=30,                   # GT点明显增大
                marker='x',
                c='red',               # 改为更亮的颜色
                alpha=0.9,
                linewidth=2,
                label='GT' if person_idx == 0 else None,
            )

        ax3.set_title('radar low range-azimuth')
        ax3.set_xlabel('Azimuth (deg, left positive)')
        ax3.set_ylabel('Range (m)')
        ax3.set_xlim(float(azimuth_axis_deg_low[0]), float(azimuth_axis_deg_low[-1]))
        ax3.set_ylim(range_plot_limits)
        ax3.legend(fontsize=8)
        fig.colorbar(image3, ax=ax3, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # =========================================================
        # 低位雷达：Range-Elevation（增强散点可见性）
        # =========================================================
        image4 = ax4.imshow(
            range_elevation_db_low,
            origin='lower',
            aspect='auto',
            extent=[
                float(elevation_axis_deg_low[0]),
                float(elevation_axis_deg_low[-1]),
                float(range_axis_low_np[0]),
                float(range_axis_low_np[-1]),
            ],
            cmap='viridis',
            vmin=-40,
            vmax=40,
        )

        if pc_low_re_mask.any():
            ax4.scatter(
                pc_low_elevation_deg[pc_low_re_mask],
                pc_low_range[pc_low_re_mask],
                s=8,                    # 增大点尺寸
                c='yellow',
                alpha=0.7,              # 提高不透明度
                edgecolors='white',
                linewidth=0.5,
                label='PC',
            )

        for person_idx, joints in enumerate(gt_low):
            joints = joints[np.isfinite(joints).all(axis=1)]

            if len(joints) == 0:
                continue

            joints_range = np.linalg.norm(joints, axis=1)
            joints_valid = joints_range > 1e-6
            joints = joints[joints_valid]
            joints_range = joints_range[joints_valid]

            if len(joints) == 0:
                continue

            joints_elevation_deg = np.rad2deg(np.arcsin(np.clip(joints[:, 2] / joints_range, -1.0, 1.0)))
            joints_re_mask = (
                (joints_range >= range_plot_limits[0])
                & (joints_range <= range_plot_limits[1])
                & (joints_elevation_deg >= elevation_axis_deg_low[0])
                & (joints_elevation_deg <= elevation_axis_deg_low[-1])
            )

            if not joints_re_mask.any():
                continue

            ax4.scatter(
                joints_elevation_deg[joints_re_mask],
                joints_range[joints_re_mask],
                s=30,                   # GT点明显增大
                marker='x',
                c='red',               # 改为更亮的颜色
                alpha=0.9,
                linewidth=2,
                label='GT' if person_idx == 0 else None,
            )

        ax4.set_title('radar low range-elevation')
        ax4.set_xlabel('Elevation (deg, upward positive)')
        ax4.set_ylabel('Range (m)')
        ax4.set_xlim(float(elevation_axis_deg_low[0]), float(elevation_axis_deg_low[-1]))
        ax4.set_ylim(range_plot_limits)
        ax4.legend(fontsize=8)
        fig.colorbar(image4, ax=ax4, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # =========================================================
        # 低位雷达：点云和 GT
        # =========================================================
        if len(pc_low_3d) > 0:
            ax5.scatter(
                pc_low_3d[:, 0],
                pc_low_3d[:, 1],
                pc_low_3d[:, 2],
                s=5,                    # 3D点稍微增大
                c='blue',
                alpha=0.6,
                label='Radar point cloud',
            )

        for person_idx, joints in enumerate(gt_low):
            joints = joints[np.isfinite(joints).all(axis=1)]

            if len(joints) == 0:
                continue

            ax5.scatter(
                joints[:, 0], 
                joints[:, 1], 
                joints[:, 2], 
                s=30,                   # 3D GT点增大
                c='red',
                marker='x',
                linewidth=2,
                label=f'GT {person_idx}'
            )

        ax5.set_xlim(pc_3d_x_limits)
        ax5.set_ylim(pc_3d_y_limits)
        ax5.set_zlim(pc_3d_z_limits)
        ax5.set_box_aspect((1, 1, 1))
        ax5.set_title('radar low pc and gt')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_zlabel('Z (m)')
        ax5.legend(fontsize=7)

        # =========================================================
        # 高位雷达：水平 x-y 投影（增强散点可见性）
        # =========================================================
        image6 = ax6.imshow(
            horizontal_xy_db_high,
            origin='lower',
            aspect='auto',
            extent=[float(y_axis_high[0]), float(y_axis_high[-1]), float(x_axis_xy_high[0]), float(x_axis_xy_high[-1])],
            cmap='viridis',
            vmin=-40,
            vmax=40,
        )

        if len(pc_high_xy) > 0:
            ax6.scatter(
                pc_high_xy[:, 1], 
                pc_high_xy[:, 0], 
                s=8,                    # 增大点尺寸
                c='yellow', 
                alpha=0.7,              # 提高不透明度
                edgecolors='white',
                linewidth=0.5,
                label='PC'
            )

        for person_idx, joints in enumerate(gt_high):
            joints = joints[np.isfinite(joints).all(axis=1)]
            joints_xy_mask = (
                (joints[:, 0] >= xy_x_limits[0])
                & (joints[:, 0] <= xy_x_limits[1])
                & (joints[:, 1] >= xy_y_limits[0])
                & (joints[:, 1] <= xy_y_limits[1])
            )
            joints_xy = joints[joints_xy_mask]

            if len(joints_xy) == 0:
                continue

            ax6.scatter(
                joints_xy[:, 1],
                joints_xy[:, 0],
                s=30,                   # GT点明显增大
                marker='x',
                c='red',               # 改为更亮的颜色
                alpha=0.9,
                linewidth=2,
                label='GT' if person_idx == 0 else None,
            )

        ax6.set_title('radar high horizontal x-y projection')
        ax6.set_xlabel('Y (m, left positive)')
        ax6.set_ylabel('X (m, forward)')
        ax6.set_xlim(xy_y_limits)
        ax6.set_ylim(xy_x_limits)
        ax6.invert_xaxis()
        ax6.legend(fontsize=8)
        fig.colorbar(image6, ax=ax6, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # =========================================================
        # 高位雷达：垂直 x-z 投影（增强散点可见性）
        # =========================================================
        image7 = ax7.imshow(
            vertical_xz_db_high,
            origin='lower',
            aspect='auto',
            extent=[float(x_axis_xz_high[0]), float(x_axis_xz_high[-1]), float(z_axis_high[0]), float(z_axis_high[-1])],
            cmap='viridis',
            vmin=-40,
            vmax=40,
        )

        if len(pc_high_xz) > 0:
            ax7.scatter(
                pc_high_xz[:, 0], 
                pc_high_xz[:, 2], 
                s=8,                    # 增大点尺寸
                c='yellow', 
                alpha=0.7,              # 提高不透明度
                edgecolors='white',
                linewidth=0.5,
                label='PC'
            )

        for person_idx, joints in enumerate(gt_high):
            joints = joints[np.isfinite(joints).all(axis=1)]
            joints_xz_mask = (
                (joints[:, 0] >= xz_x_limits[0])
                & (joints[:, 0] <= xz_x_limits[1])
                & (joints[:, 2] >= xz_z_limits[0])
                & (joints[:, 2] <= xz_z_limits[1])
            )
            joints_xz = joints[joints_xz_mask]

            if len(joints_xz) == 0:
                continue

            ax7.scatter(
                joints_xz[:, 0],
                joints_xz[:, 2],
                s=30,                   # GT点明显增大
                marker='x',
                c='red',               # 改为更亮的颜色
                alpha=0.9,
                linewidth=2,
                label='GT' if person_idx == 0 else None,
            )

        ax7.set_title('radar high vertical x-z projection')
        ax7.set_xlabel('X (m, forward)')
        ax7.set_ylabel('Z (m, upward)')
        ax7.set_xlim(xz_x_limits)
        ax7.set_ylim(xz_z_limits)
        ax7.legend(fontsize=8)
        fig.colorbar(image7, ax=ax7, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # =========================================================
        # 高位雷达：Range-Azimuth（增强散点可见性）
        # =========================================================
        image8 = ax8.imshow(
            range_azimuth_db_high,
            origin='lower',
            aspect='auto',
            extent=[
                float(azimuth_axis_deg_high[0]),
                float(azimuth_axis_deg_high[-1]),
                float(range_axis_high_np[0]),
                float(range_axis_high_np[-1]),
            ],
            cmap='viridis',
            vmin=-40,
            vmax=40,
        )

        if pc_high_ra_mask.any():
            ax8.scatter(
                pc_high_azimuth_deg[pc_high_ra_mask],
                pc_high_range[pc_high_ra_mask],
                s=8,                    # 增大点尺寸
                c='yellow',
                alpha=0.7,              # 提高不透明度
                edgecolors='white',
                linewidth=0.5,
                label='PC',
            )

        for person_idx, joints in enumerate(gt_high):
            joints = joints[np.isfinite(joints).all(axis=1)]

            if len(joints) == 0:
                continue

            joints_range = np.linalg.norm(joints, axis=1)
            joints_valid = joints_range > 1e-6
            joints = joints[joints_valid]
            joints_range = joints_range[joints_valid]

            if len(joints) == 0:
                continue

            joints_azimuth_deg = np.rad2deg(np.arcsin(np.clip(joints[:, 1] / joints_range, -1.0, 1.0)))
            joints_ra_mask = (
                (joints_range >= range_plot_limits[0])
                & (joints_range <= range_plot_limits[1])
                & (joints_azimuth_deg >= azimuth_axis_deg_high[0])
                & (joints_azimuth_deg <= azimuth_axis_deg_high[-1])
            )

            if not joints_ra_mask.any():
                continue

            ax8.scatter(
                joints_azimuth_deg[joints_ra_mask],
                joints_range[joints_ra_mask],
                s=30,                   # GT点明显增大
                marker='x',
                c='red',               # 改为更亮的颜色
                alpha=0.9,
                linewidth=2,
                label='GT' if person_idx == 0 else None,
            )

        ax8.set_title('radar high range-azimuth')
        ax8.set_xlabel('Azimuth (deg, left positive)')
        ax8.set_ylabel('Range (m)')
        ax8.set_xlim(float(azimuth_axis_deg_high[0]), float(azimuth_axis_deg_high[-1]))
        ax8.set_ylim(range_plot_limits)
        ax8.legend(fontsize=8)
        fig.colorbar(image8, ax=ax8, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # =========================================================
        # 高位雷达：Range-Elevation（增强散点可见性）
        # =========================================================
        image9 = ax9.imshow(
            range_elevation_db_high,
            origin='lower',
            aspect='auto',
            extent=[
                float(elevation_axis_deg_high[0]),
                float(elevation_axis_deg_high[-1]),
                float(range_axis_high_np[0]),
                float(range_axis_high_np[-1]),
            ],
            cmap='viridis',
            vmin=-40,
            vmax=40,
        )

        if pc_high_re_mask.any():
            ax9.scatter(
                pc_high_elevation_deg[pc_high_re_mask],
                pc_high_range[pc_high_re_mask],
                s=8,                    # 增大点尺寸
                c='yellow',
                alpha=0.7,              # 提高不透明度
                edgecolors='white',
                linewidth=0.5,
                label='PC',
            )

        for person_idx, joints in enumerate(gt_high):
            joints = joints[np.isfinite(joints).all(axis=1)]

            if len(joints) == 0:
                continue

            joints_range = np.linalg.norm(joints, axis=1)
            joints_valid = joints_range > 1e-6
            joints = joints[joints_valid]
            joints_range = joints_range[joints_valid]

            if len(joints) == 0:
                continue

            joints_elevation_deg = np.rad2deg(np.arcsin(np.clip(joints[:, 2] / joints_range, -1.0, 1.0)))
            joints_re_mask = (
                (joints_range >= range_plot_limits[0])
                & (joints_range <= range_plot_limits[1])
                & (joints_elevation_deg >= elevation_axis_deg_high[0])
                & (joints_elevation_deg <= elevation_axis_deg_high[-1])
            )

            if not joints_re_mask.any():
                continue

            ax9.scatter(
                joints_elevation_deg[joints_re_mask],
                joints_range[joints_re_mask],
                s=30,                   # GT点明显增大
                marker='x',
                c='red',               # 改为更亮的颜色
                alpha=0.9,
                linewidth=2,
                label='GT' if person_idx == 0 else None,
            )

        ax9.set_title('radar high range-elevation')
        ax9.set_xlabel('Elevation (deg, upward positive)')
        ax9.set_ylabel('Range (m)')
        ax9.set_xlim(float(elevation_axis_deg_high[0]), float(elevation_axis_deg_high[-1]))
        ax9.set_ylim(range_plot_limits)
        ax9.legend(fontsize=8)
        fig.colorbar(image9, ax=ax9, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # =========================================================
        # 高位雷达：点云和 GT
        # =========================================================
        if len(pc_high_3d) > 0:
            ax10.scatter(
                pc_high_3d[:, 0],
                pc_high_3d[:, 1],
                pc_high_3d[:, 2],
                s=5,                    # 3D点稍微增大
                c='blue',
                alpha=0.6,
                label='Radar point cloud',
            )

        for person_idx, joints in enumerate(gt_high):
            joints = joints[np.isfinite(joints).all(axis=1)]

            if len(joints) == 0:
                continue

            ax10.scatter(
                joints[:, 0], 
                joints[:, 1], 
                joints[:, 2], 
                s=30,                   # 3D GT点增大
                c='red',
                marker='x',
                linewidth=2,
                label=f'GT {person_idx}'
            )

        ax10.set_xlim(pc_3d_x_limits)
        ax10.set_ylim(pc_3d_y_limits)
        ax10.set_zlim(pc_3d_z_limits)
        ax10.set_box_aspect((1, 1, 1))
        ax10.set_title('radar high pc and gt')
        ax10.set_xlabel('X (m)')
        ax10.set_ylabel('Y (m)')
        ax10.set_zlabel('Z (m)')
        ax10.legend(fontsize=7)

        # =========================================================
        # 保存
        # =========================================================
        fig.suptitle(
            f'batch {batch_idx}, sample {b}, time {t}, clutter {clutter_mode}'
        )
        fig.tight_layout()

        save_path = 'temp.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f'visualization saved to: {save_path}')

        # b, t = 0, 0

        # if not torch.isfinite(samples['radar_low_bin']).all():
        #     print(f'batch {batch_idx} radar_low_bin 存在 NaN 或 Inf，跳过')
        #     continue
        # if not torch.isfinite(samples['radar_high_bin']).all():
        #     print(f'batch {batch_idx} radar_high_bin 存在 NaN 或 Inf，跳过')
        #     continue

        # fig = plt.figure(figsize=(20, 11))
        # ax1 = fig.add_subplot(2, 3, 1)
        # ax2 = fig.add_subplot(2, 3, 2)
        # ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        # ax4 = fig.add_subplot(2, 3, 4)
        # ax5 = fig.add_subplot(2, 3, 5)
        # ax6 = fig.add_subplot(2, 3, 6, projection='3d')

        # # =========================================================
        # # 低位雷达
        # # =========================================================
        # doppler_cube_low, doppler_cube_mean_low, r_axis_low, v_axis_low = range_cube_to_doppler_cube(
        #     samples['radar_low_bin'], radar_config
        # )

        # pc_low_valid = samples['radar_low_pc']['padded'][b, t][
        #     samples['radar_low_pc']['mask'][b, t].bool()
        # ]
        # gt_low = samples['gt_for_low']['padded'][b, t][
        #     samples['gt_for_low']['mask'][b, t].bool()
        # ]

        # doppler_cube_low = doppler_cube_low[b, t].detach().cpu().numpy()
        # doppler_cube_mean_low = doppler_cube_mean_low[b, t].detach().cpu().numpy()
        # r_axis_low = r_axis_low.detach().cpu().numpy()
        # v_axis_low = v_axis_low.detach().cpu().numpy()

        # power_low = np.mean(np.abs(doppler_cube_low) ** 2, axis=-1)
        # power_clean_low = np.mean(np.abs(doppler_cube_mean_low) ** 2, axis=-1)

        # RD_map_db_low = 10.0 * np.log10(power_low + 1e-12)
        # RD_map_clean_db_low = 10.0 * np.log10(power_clean_low + 1e-12)

        # # 每张 RD 图峰值归一化为 0 dB
        # RD_map_db_low -= np.max(RD_map_db_low)
        # RD_map_clean_db_low -= np.max(RD_map_clean_db_low)

        # pc_low_valid = pc_low_valid.detach().cpu().numpy()
        # gt_low = gt_low.detach().cpu().numpy()
        # pc_low_valid_xyz = pc_low_valid[:, :3]
        # pc_low_valid_xyz = pc_low_valid_xyz[np.isfinite(pc_low_valid_xyz).all(axis=1)]

        # image1 = ax1.imshow(
        #     RD_map_db_low, origin='lower', aspect='auto',
        #     extent=[float(v_axis_low[0]), float(v_axis_low[-1]),
        #             float(r_axis_low[0]), float(r_axis_low[-1])],
        #     cmap='viridis', vmin=-40, vmax=40
        # )
        # ax1.set_title('radar low bin RD map')
        # ax1.set_xlabel('Velocity (m/s)')
        # ax1.set_ylabel('Range (m)')
        # fig.colorbar(image1, ax=ax1, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # image2 = ax2.imshow(
        #     RD_map_clean_db_low, origin='lower', aspect='auto',
        #     extent=[float(v_axis_low[0]), float(v_axis_low[-1]),
        #             float(r_axis_low[0]), float(r_axis_low[-1])],
        #     cmap='viridis', vmin=-40, vmax=40
        # )
        # ax2.set_title('radar low bin RD map clean')
        # ax2.set_xlabel('Velocity (m/s)')
        # ax2.set_ylabel('Range (m)')
        # fig.colorbar(image2, ax=ax2, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # if len(pc_low_valid_xyz) > 0:
        #     ax3.scatter(
        #         pc_low_valid_xyz[:, 0], pc_low_valid_xyz[:, 1], pc_low_valid_xyz[:, 2],
        #         s=3, alpha=0.5, label='Radar point cloud'
        #     )

        # for person_idx in range(gt_low.shape[0]):
        #     joints = gt_low[person_idx]
        #     joints = joints[np.isfinite(joints).all(axis=1)]
        #     if len(joints) == 0:
        #         continue
        #     ax3.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=20, label=f'GT {person_idx}')

        # ax3.set_xlim(0, 6)
        # ax3.set_ylim(-3, 3)
        # ax3.set_zlim(-3, 3)
        # ax3.set_box_aspect((1, 1, 1))
        # ax3.set_title('radar low pc and gt')
        # ax3.set_xlabel('X (m)')
        # ax3.set_ylabel('Y (m)')
        # ax3.set_zlabel('Z (m)')
        # ax3.legend(fontsize=7)

        # # =========================================================
        # # 高位雷达
        # # =========================================================
        # doppler_cube_high, doppler_cube_clean_high, r_axis_high, v_axis_high = range_cube_to_doppler_cube(
        #     samples['radar_high_bin'], radar_config
        # )

        # pc_high_valid = samples['radar_high_pc']['padded'][b, t][
        #     samples['radar_high_pc']['mask'][b, t].bool()
        # ]
        # gt_high = samples['gt_for_high']['padded'][b, t][
        #     samples['gt_for_high']['mask'][b, t].bool()
        # ]

        # doppler_cube_high = doppler_cube_high[b, t].detach().cpu().numpy()
        # doppler_cube_clean_high = doppler_cube_clean_high[b, t].detach().cpu().numpy()
        # r_axis_high = r_axis_high.detach().cpu().numpy()
        # v_axis_high = v_axis_high.detach().cpu().numpy()

        # power_high = np.mean(np.abs(doppler_cube_high) ** 2, axis=-1)
        # power_clean_high = np.mean(np.abs(doppler_cube_clean_high) ** 2, axis=-1)

        # RD_map_db_high = 10.0 * np.log10(power_high + 1e-12)
        # RD_map_clean_db_high = 10.0 * np.log10(power_clean_high + 1e-12)

        # # 每张 RD 图峰值归一化为 0 dB
        # RD_map_db_high -= np.max(RD_map_db_high)
        # RD_map_clean_db_high -= np.max(RD_map_clean_db_high)

        # pc_high_valid = pc_high_valid.detach().cpu().numpy()
        # gt_high = gt_high.detach().cpu().numpy()
        # pc_high_valid_xyz = pc_high_valid[:, :3]
        # pc_high_valid_xyz = pc_high_valid_xyz[np.isfinite(pc_high_valid_xyz).all(axis=1)]

        # image4 = ax4.imshow(
        #     RD_map_db_high, origin='lower', aspect='auto',
        #     extent=[float(v_axis_high[0]), float(v_axis_high[-1]),
        #             float(r_axis_high[0]), float(r_axis_high[-1])],
        #     cmap='viridis', vmin=-40, vmax=40
        # )
        # ax4.set_title('radar high bin RD map')
        # ax4.set_xlabel('Velocity (m/s)')
        # ax4.set_ylabel('Range (m)')
        # fig.colorbar(image4, ax=ax4, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # image5 = ax5.imshow(
        #     RD_map_clean_db_high, origin='lower', aspect='auto',
        #     extent=[float(v_axis_high[0]), float(v_axis_high[-1]),
        #             float(r_axis_high[0]), float(r_axis_high[-1])],
        #     cmap='viridis', vmin=-40, vmax=40
        # )
        # ax5.set_title('radar high bin RD map clean')
        # ax5.set_xlabel('Velocity (m/s)')
        # ax5.set_ylabel('Range (m)')
        # fig.colorbar(image5, ax=ax5, fraction=0.046, pad=0.04, label='Normalized power (dB)')

        # if len(pc_high_valid_xyz) > 0:
        #     ax6.scatter(
        #         pc_high_valid_xyz[:, 0], pc_high_valid_xyz[:, 1], pc_high_valid_xyz[:, 2],
        #         s=3, alpha=0.5, label='Radar point cloud'
        #     )

        # for person_idx in range(gt_high.shape[0]):
        #     joints = gt_high[person_idx]
        #     joints = joints[np.isfinite(joints).all(axis=1)]
        #     if len(joints) == 0:
        #         continue
        #     ax6.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=20, label=f'GT {person_idx}')

        # ax6.set_xlim(0, 6)
        # ax6.set_ylim(-3, 3)
        # ax6.set_zlim(-3, 3)
        # ax6.set_box_aspect((1, 1, 1))
        # ax6.set_title('radar high pc and gt')
        # ax6.set_xlabel('X (m)')
        # ax6.set_ylabel('Y (m)')
        # ax6.set_zlabel('Z (m)')
        # ax6.legend(fontsize=7)

        # fig.suptitle(f'batch {batch_idx}, sample {b}, time {t}')
        # fig.tight_layout()
        # save_path = 'temp.png'
        # fig.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.close(fig)

        # print(f'visualization saved to: {save_path}')
            
