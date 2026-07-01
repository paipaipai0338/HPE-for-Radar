from pathlib import Path

import numpy as np
import open3d as o3d

def get_lidar_data(file_path: Path|str) -> np.ndarray:
    """
    获取激光雷达 EM4 数据
    输入:
      file_path: str 或 path-like，PCD 点云文件路径，shape 为标量路径。
    中间变量:
      pcd: open3d.geometry.PointCloud，Open3D 点云对象。
      pointcloud: np.ndarray，dtype=float64，shape=(N, 3)，每行为 x/y/z 坐标。
    输出:
      pointcloud: np.ndarray，dtype=float64，shape=(N, 3)。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    point_clouds = np.asarray(pcd.points)
    return point_clouds
