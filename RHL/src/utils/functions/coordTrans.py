import numpy as np
from typing import Dict
import io


# 从npz文件中加载外参
def loadExtrinsics(file_path):
    """
    从指定的 NumPy 压缩文件 (.npz) 中加载估计的相机/雷达外参。

    参数:
        file_path (str): 存储外参数据的文件路径 (通常为 .npz 格式)。

    返回:
        dict: 包含两个元素的元组:
            - R_est (np.ndarray): 估计的旋转矩阵 (Rotation Matrix)。
            - t_est (np.ndarray): 估计的平移向量 (Translation Vector)。
    """
    with open(file_path, "rb") as f:
        buffer = io.BytesIO(f.read())
    data = np.load(buffer, allow_pickle=True)
    result = {}
    for key in data.files:
        result[key] = data[key]
    # return result["R_est"],result["t_est"]
    return result


def coordTrans(
        coord: np.ndarray,
        ext_info: Dict
    ) -> np.ndarray:
    """
    将坐标从传感器坐标系转换到参考坐标系（或反之），使用外参矩阵进行旋转和平移变换
    
    该函数通过旋转矩阵 R_est 和平移向量 t_est 对输入坐标进行刚性变换：
    转换后的坐标 = R_est @ coord + t_est
    
    Parameters
    ----------
    coord : np.ndarray
        待转换的坐标数据，支持两种形状：
        - 形状为 (n, m, 3)：n个点云帧，每帧m个点，每个点3维坐标
        - 形状为 (m, 3)：m个点，每个点3维坐标
        坐标格式为 (x, y, z)
    
    ext_info : Dict
        外参字典，必须包含以下键：
        - "R_est" : np.ndarray
            旋转矩阵，形状为 (3, 3)
        - "t_est" : np.ndarray
            平移向量，形状为 (3,) 或 (3, 1)
    
    Returns
    -------
    np.ndarray
        转换后的坐标数据，形状与输入 coord 保持一致
    
    Raises
    ------
    ValueError
        当 coord 的最后一维不为3时抛出
    KeyError
        当 ext_info 中缺少 "R_est" 或 "t_est" 键时抛出
    ValueError
        当旋转矩阵或平移向量形状不正确时抛出
    
    Examples
    --------
    >>> import numpy as np
    >>> coord = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)
    >>> ext_info = {
    ...     "R_est": np.eye(3),
    ...     "t_est": np.array([1.0, 2.0, 3.0])
    ... }
    >>> transformed = coordTrans(coord, ext_info)
    >>> print(transformed)
    [[2. 4. 6.]
     [5. 7. 9.]]
    
    Notes
    -----
    转换公式：p' = R @ p + t
    其中 R 为旋转矩阵，t 为平移向量
    """
    # 验证输入维度
    if coord.shape[-1] != 3:
        raise ValueError(f"coord的最后一维必须为3，当前形状为 {coord.shape}")
    
    # 提取旋转矩阵和平移向量
    try:
        R = ext_info["R_est"]
        t = ext_info["t_est"]
    except KeyError as e:
        raise KeyError(f"ext_info中缺少必要的键: {e}")
    
    # 验证旋转矩阵形状
    if R.shape != (3, 3):
        raise ValueError(f"旋转矩阵 R_est 必须为 (3, 3) 形状，当前为 {R.shape}")
    
    # 验证平移向量形状并确保为 (3,)
    if t.shape == (3, 1):
        t = t.flatten()
    elif t.shape != (3,):
        raise ValueError(f"平移向量 t_est 必须为 (3,) 或 (3, 1) 形状，当前为 {t.shape}")
    
    # 保存原始形状用于恢复
    original_shape = coord.shape
    
    # 将输入reshape为 (num_points, 3) 以便统一处理
    coord_flat = coord.reshape(-1, 3)
    
    # 执行坐标变换: p' = R @ p + t
    # 注意：对于每个点，需要执行 R @ point + t
    transformed_flat = coord_flat @ R.T + t
    
    # 恢复原始形状
    transformed = transformed_flat.reshape(original_shape)
    
    return transformed