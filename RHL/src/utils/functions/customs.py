from pathlib import Path
import re
import numpy as np
from typing import Any, Optional, Dict
import pickle
import os
import yaml
import pycocotools.mask as mask_util
import json
# 找时间戳最近的文件
def findClosestTimeFile(
        input_file_path: str | Path, 
        folder_path: str | Path
    ) -> Path | None:
    """
    在指定文件夹中，找到与输入文件时间戳差值最小的文件。
    
    :param input_file_path: 输入文件的路径
    :param folder_path: 目标文件夹的路径
    :return: 差值最小的文件 Path 对象，若未找到匹配文件则返回 None
    """
    input_path = Path(input_file_path)
    target_folder = Path(folder_path)
    if str(target_folder) == "/mnt/huawei/20260730/data_collection/group_016/ camera results /smoothed 3D":
        a=1
    # 正则表达式：匹配纯数字_纯数字 的文件名结构（忽略文件后缀）
    timestamp_pattern = re.compile(r'^(\d+)_(\d+)$')
    
    def parse_to_nanoseconds(file_name: str) -> int | None:
        """将文件名解析为总纳秒数"""
        stem = Path(file_name).stem  # 获取不带后缀的文件名
        match = timestamp_pattern.match(stem)
        if match:
            seconds = int(match.group(1))
            nanoseconds = int(match.group(2))
            # 转换为统一的纳秒整数：秒 * 10^9 + 纳秒
            return seconds * 10**9 + nanoseconds
        return None

    # 1. 解析输入文件的时间戳
    target_time = parse_to_nanoseconds(input_path.name)
    if target_time is None:
        raise ValueError(f"输入文件名称格式不正确 (应为 '秒_纳秒'): {input_path.name}")

    closest_file = None
    min_diff = float('inf')

    # 2. 遍历文件夹中的文件
    for file in target_folder.iterdir():
        if file.is_file():
            # 如果输入文件本身就在该文件夹中，且需要排除它自己，可以取消注释下面这行：
            # if file.name == input_path.name: continue
            
            file_time = parse_to_nanoseconds(file.name)
            if file_time is None:
                continue  # 跳过命名格式不符的文件
            
            # 计算时间差的绝对值
            diff = abs(target_time - file_time)
            
            # 更新最小值
            if diff < min_diff:
                min_diff = diff
                closest_file = file

    return closest_file

# 从pkl文件中读取人体骨架位置
def readHumanPose_pkl(
        file_path: str, 
    ) -> Optional[Any]:
    """
    读取pkl文件并返回其中的数据
    
    参数:
        file_path (str): pkl文件的路径
        encoding (str): 编码方式，默认为'utf-8'，仅对某些特殊格式有效
    
    返回:
        data: np.arrar
            shape (n,17,3)
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # print(f"成功读取文件: {file_path}")
    
    return data

# 解析配置文件
def parseYAML(file_path):
    """
    解析 voxel.yaml 文件，提取其中的关键字段
    
    参数:
        file_path (str): YAML 文件的路径
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def parse_pipeline_config(config_path:str) -> Dict[str,Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 格式解析错误: {e}")
    return raw_cfg.get("pipeline",{})

# 解析标签 json 文件
def parseLabel(json_file_path: str) -> dict:
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    parsed_instances = []
    for inst in data.get("instances", []):
        grid_shape = inst["mask"]["size"] # [16, 24, 8]，用于最后 reshape
        total_voxels = np.prod(grid_shape) # 15360
        
        # 1. 解码 mask
        rle_mask = {
            "size": [total_voxels, 1],  # 【核心修正】必须与编码时的 (N, 1) 形状保持一致
            "counts": inst["mask"]["counts"]
        }
        if isinstance(rle_mask["counts"], str):
            rle_mask["counts"] = rle_mask["counts"].encode('utf-8')
            
        decoded_mask_flat = mask_util.decode(rle_mask)
        decoded_mask = decoded_mask_flat.reshape(grid_shape).astype(np.uint8)
            
        # 2. 解码 confidence
        rle_conf = {
            "size": [total_voxels, 1],  # 【核心修正】同上
            "counts": inst["confidence"]["counts"]
        }
        if isinstance(rle_conf["counts"], str):
            rle_conf["counts"] = rle_conf["counts"].encode('utf-8')
            
        conf_list = inst["confidence"]["counts"]
        decoded_conf = np.array(conf_list, dtype=np.float32).reshape(grid_shape)

        inst_info = {
            "id": inst["id"],
            "bbox": inst["bbox"],
            "mask": decoded_mask,       # shape: [16, 24, 8]
            "confidence": decoded_conf  # shape: [16, 24, 8]
        }
        parsed_instances.append(inst_info)
        
    result = {
        "timestamp": data.get("timestamp"),
        "voxel_config": data.get("voxel_config"),
        "instances": parsed_instances
    }
    return result

# 点云分割函数，按照掩码区域分割为位于 掩码区域中的点 与 不在掩码区域中的点
def splitPointsByMask(label_dict: dict, raw_point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    根据体素掩码将原始点云划分为两部分：
    1. 位于掩码值为 1 的体素区域中的点（Foreground points）
    2. 不在值为 1 的体素区域中的点（Background/Out-of-mask points）
    
    两部分互不相交，且并集等于原始点云。
    
    参数:
        label_dict (dict): 包含 voxel_config 和 instances 信息的字典
        raw_point_cloud (np.ndarray): 原始点云，形状为 [P, 4] 或 [P, >=3]，前三列为 [x, y, z]
        
    返回:
        tuple[np.ndarray, np.ndarray]: (inside_mask_points, outside_mask_points)
    """
    if len(raw_point_cloud) == 0:
        return np.zeros((0, raw_point_cloud.shape[1]), dtype=raw_point_cloud.dtype), \
               np.zeros((0, raw_point_cloud.shape[1]), dtype=raw_point_cloud.dtype)

    # 1. 解析体素配置
    voxel_config = label_dict["voxel_config"]
    xlim = voxel_config.get("xlim") or voxel_config["region"]["XLIM"]
    ylim = voxel_config.get("ylim") or voxel_config["region"]["YLIM"]
    zlim = voxel_config.get("zlim") or voxel_config["region"]["ZLIM"]
    voxel_size = voxel_config.get("voxel_size") or voxel_config["resolution"]

    # 2. 获取网格形状并合并所有实例的 mask（按位或）
    grid_shape = None
    for inst in label_dict.get("instances", []):
        if "mask" in inst:
            grid_shape = inst["mask"].shape
            break
            
    # 如果没有任何实例或没有 mask，则所有点全部分配到“不在掩码区域”
    if grid_shape is None or not label_dict.get("instances"):
        return np.zeros((0, raw_point_cloud.shape[1]), dtype=raw_point_cloud.dtype), raw_point_cloud

    combined_mask = np.zeros(grid_shape, dtype=bool)
    for inst in label_dict["instances"]:
        mask = inst["mask"]
        combined_mask = np.logical_or(combined_mask, mask.astype(bool))

    # 3. 计算点云中每个点在当前体素配置下的离散索引
    x = raw_point_cloud[:, 0]
    y = raw_point_cloud[:, 1]
    z = raw_point_cloud[:, 2]

    ix = np.floor((x - xlim[0]) / voxel_size[0]).astype(int)
    iy = np.floor((y - ylim[0]) / voxel_size[1]).astype(int)
    iz = np.floor((z - zlim[0]) / voxel_size[2]).astype(int)

    # 4. 判断每个点是否在合法边界内
    in_bounds = (
        (ix >= 0) & (ix < grid_shape[0]) &
        (iy >= 0) & (iy < grid_shape[1]) &
        (iz >= 0) & (iz < grid_shape[2])
    )

    # 初始默认所有点都不在掩码值为1的区域
    inside_mask_bool = np.zeros(len(raw_point_cloud), dtype=bool)
    
    # 仅对在边界内的点进行掩码状态查询（越界点自动视作不在掩码内）
    if np.any(in_bounds):
        valid_indices = np.where(in_bounds)[0]
        inside_mask_bool[valid_indices] = combined_mask[
            ix[valid_indices], 
            iy[valid_indices], 
            iz[valid_indices]
        ]

    # 5. 划分为两部分（互补且无交集）
    inside_points = raw_point_cloud[inside_mask_bool]
    outside_points = raw_point_cloud[~inside_mask_bool]

    return inside_points, outside_points



# 
