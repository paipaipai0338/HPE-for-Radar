import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from typing import List
from src.utils.tracking.core import *

def drawVoxel(
    label_dict: dict, 
    fig, 
    ax, 
    show_confidence: bool = False,
    voxel_color: list = [30 / 255.0, 144 / 255.0, 255 / 255.0, 0.4],
    edge_color: str = 'k',
    text_color: str = 'darkred'
):
    """
    利用 matplotlib 句柄在 3D 窗口上细粒度绘制人体体素结构及置信度。
    
    参数:
        label_dict (dict): 解析后的字典。
        fig: matplotlib 的 figure 句柄。
        ax: matplotlib 的 3D 坐标轴句柄 (需满足 projection='3d')。
        show_confidence (bool): 是否在体素中心标注置信度数值。
        voxel_color (list): 细粒度体素模式下的 RGBA 颜色。
        edge_color (str): 体素网格边界线条颜色。
        text_color (str): 置信度文本标注颜色。
        
    返回:
        tuple: (fig, ax) 绘制完成后的对象。
    """
    voxel_config = label_dict["voxel_config"]
    voxel_size = voxel_config["voxel_size"]  # [dx, dy, dz]
    xlim = voxel_config["xlim"]
    ylim = voxel_config["ylim"]
    zlim = voxel_config["zlim"]
    
    # 构造统一的物理空间边界网格坐标
    x_edges = np.arange(xlim[0], xlim[1] + voxel_size[0] * 0.5, voxel_size[0])
    y_edges = np.arange(ylim[0], ylim[1] + voxel_size[1] * 0.5, voxel_size[1])
    z_edges = np.arange(zlim[0], zlim[1] + voxel_size[2] * 0.5, voxel_size[2])
    X_edges, Y_edges, Z_edges = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

    # 遍历每个实例
    for inst in label_dict["instances"]:
        mask = inst["mask"]             # shape: (nX, nY, nZ)
        confidence = inst["confidence"] # shape: (nX, nY, nZ)
        
        if not np.any(mask):
            continue
            
        voxel_bools = (mask > 0)
        
        # 细粒度体素渲染模式
        colors = np.empty(voxel_bools.shape + (4,), dtype=np.float32)
        colors[voxel_bools] = voxel_color
        
        ax.voxels(
            X_edges, Y_edges, Z_edges,
            voxel_bools,
            facecolors=colors,
            edgecolor=edge_color,
            linewidth=0.3
        )
        
        # 标注置信度文本
        if show_confidence:
            valid_indices = np.where(voxel_bools)
            x_centers = xlim[0] + (valid_indices[0] + 0.5) * voxel_size[0]
            y_centers = ylim[0] + (valid_indices[1] + 0.5) * voxel_size[1]
            z_centers = zlim[0] + (valid_indices[2] + 0.5) * voxel_size[2]
            p_conf = confidence[valid_indices]
            
            for x, y, z, conf in zip(x_centers, y_centers, z_centers, p_conf):
                ax.text(
                    x, y, z,
                    f"{conf:.2f}",
                    fontsize=6,
                    color=text_color,
                    ha='center',
                    va='center'
                )

    return fig, ax

def drawVoxelBoundingCube(
    label_dict: dict, 
    fig, 
    ax,
    bbox_color: list = [255 / 255.0, 140 / 255.0, 0 / 255.0, 0.2],
    edge_color: str = 'darkorange',
    linewidth: float = 0.8
):
    voxel_config = label_dict["voxel_config"]
    voxel_size = voxel_config["voxel_size"]
    xlim = voxel_config["xlim"]
    ylim = voxel_config["ylim"]
    zlim = voxel_config["zlim"]
    
    for inst in label_dict["instances"]:
        mask = inst["mask"]
        if not np.any(mask):
            continue
            
        voxel_bools = (mask > 0)
        valid_x, valid_y, valid_z = np.where(voxel_bools)
        min_x, max_x = valid_x.min(), valid_x.max()
        min_y, max_y = valid_y.min(), valid_y.max()
        min_z, max_z = valid_z.min(), valid_z.max()
        
        # 将体素索引转换为真实物理坐标范围
        x_min = xlim[0] + min_x * voxel_size[0]
        x_max = xlim[0] + (max_x + 1) * voxel_size[0]
        y_min = ylim[0] + min_y * voxel_size[1]
        y_max = ylim[0] + (max_y + 1) * voxel_size[1]
        z_min = zlim[0] + min_z * voxel_size[2]
        z_max = zlim[0] + (max_z + 1) * voxel_size[2]
        
        # 构建 8 个顶点
        corners = np.array([
            [x_min, y_min, z_min],  # 0
            [x_max, y_min, z_min],  # 1
            [x_max, y_max, z_min],  # 2
            [x_min, y_max, z_min],  # 3
            [x_min, y_min, z_max],  # 4
            [x_max, y_min, z_max],  # 5
            [x_max, y_max, z_max],  # 6
            [x_min, y_max, z_max],  # 7
        ])
        
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[0], corners[3], corners[7], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]],
        ]
        
        poly = Poly3DCollection(
            faces,
            facecolors=bbox_color,
            edgecolors=edge_color,
            linewidths=linewidth,
            linestyles='-'
        )
        ax.add_collection3d(poly)

    return fig, ax

def drawBbox(
    label_dict: dict, 
    fig, 
    ax, 
    fill: bool = True,
    face_color: list = [1.0, 0.2, 0.2, 0.15],
    edge_color: str = 'red'
):
    """
    根据 label_dict 中的归一化 bbox 数据，在 3D 坐标系中绘制 3D Bounding Box。
    
    参数:
        label_dict (dict): 包含 voxel_config 与 instances 列表的字典。
        fig: matplotlib 的 figure 句柄。
        ax: matplotlib 的 3D 坐标轴句柄 (需满足 projection='3d')。
        fill (bool): 是否填充包围盒表面（默认为 True 半透明填充，False 为纯线框）。
        face_color (list): 包围盒表面的 RGBA 填充颜色，默认浅红色半透明。
        edge_color (str 或 list): 包围盒边界线条的颜色，默认红色。
        
    返回:
        tuple: (fig, ax) 绘制完成后的句柄对象。
    """
    voxel_config = label_dict["voxel_config"]
    xlim = voxel_config["xlim"]
    ylim = voxel_config["ylim"]
    zlim = voxel_config["zlim"]
    
    # 计算三个物理轴向的全长
    lx = xlim[1] - xlim[0]
    ly = ylim[1] - ylim[0]
    lz = zlim[1] - zlim[0]
    
    for inst in label_dict.get("instances", []):
        bbox_norm = inst.get("bbox")
        if bbox_norm is None:
            continue
            
        # 1. 提取归一化的中心坐标和尺寸
        cx_norm, cy_norm, cz_norm, sx_norm, sy_norm, sz_norm = bbox_norm
        
        # 2. 反归一化到真实物理坐标系
        cx = xlim[0] + cx_norm * lx
        cy = ylim[0] + cy_norm * ly
        cz = zlim[0] + cz_norm * lz
        
        sx = sx_norm * lx
        sy = sy_norm * ly
        sz = sz_norm * lz
        
        # 计算 8 个顶点的极值范围
        x_min, x_max = cx - sx / 2.0, cx + sx / 2.0
        y_min, y_max = cy - sy / 2.0, cy + sy / 2.0
        z_min, z_max = cz - sz / 2.0, cz + sz / 2.0
        
        # 3. 构建 8 个顶点的坐标
        corners = np.array([
            [x_min, y_min, z_min],  # 0
            [x_max, y_min, z_min],  # 1
            [x_max, y_max, z_min],  # 2
            [x_min, y_max, z_min],  # 3
            [x_min, y_min, z_max],  # 4
            [x_max, y_min, z_max],  # 5
            [x_max, y_max, z_max],  # 6
            [x_min, y_max, z_max],  # 7
        ])
        
        # 4. 组合 6 个面的顶点索引
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # 底面
            [corners[4], corners[5], corners[6], corners[7]],  # 顶面
            [corners[0], corners[1], corners[5], corners[4]],  # 前面
            [corners[2], corners[3], corners[7], corners[6]],  # 后面
            [corners[0], corners[3], corners[7], corners[4]],  # 左面
            [corners[1], corners[2], corners[6], corners[5]],  # 右面
        ]
        
        # 5. 渲染 3D 面与边界线
        current_face_color = face_color if fill else [0.0, 0.0, 0.0, 0.0]
        poly = Poly3DCollection(
            faces,
            facecolors=current_face_color,
            edgecolors=edge_color,
            linewidths=1.2,
            linestyles='-'
        )
        ax.add_collection3d(poly)
    
    return fig, ax

def drawHumanPose(
    human_pose: np.ndarray,
    keys_color: str = 'red',
    joint_lines_color: str = 'blue',
    hp_label: str = "human pose",
    fig = None,
    ax = None
):
    """
    绘制经过外参矩阵调整后的姿态骨骼图 (使用 Matplotlib)
    Param:
        human_pose: np.ndarray, shape: (n, 17, 3)
    """
    
    # 固定的 17 点关节点连接逻辑
    SKELETON_EDGES = [
        [0, 1], [0, 2], [1, 3], [2, 4],
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [13, 15], [12, 14], [14, 16]
    ]
    
    n_persons = human_pose.shape[0]
    
    for i in range(n_persons):
        # 1. 坐标外参变换
        # 原 pose 形状为 (17, 3)，通过乘以 R 的转置并加上平移向量，完成空间转换
        pose = human_pose[i]
        
        x = pose[:, 0]
        y = pose[:, 1]
        z = pose[:, 2]
        
        # 2. 绘制关键点 (仅对第0个人打上 label，避免 legend 重复)
        current_label = hp_label if i == 0 else None
        ax.scatter(x, y, z, c=keys_color, s=25, label=current_label, zorder=5)
        
        # 3. 绘制骨骼连接线
        for edge in SKELETON_EDGES:
            p1, p2 = edge
            line_x = [pose[p1, 0], pose[p2, 0]]
            line_y = [pose[p1, 1], pose[p2, 1]]
            line_z = [pose[p1, 2], pose[p2, 2]]
            
            ax.plot(line_x, line_y, line_z, c=joint_lines_color, linewidth=2, zorder=4)

    # 图例去重与展示
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    
    return fig, ax

def drawPointCloud(
        point_cloud: np.ndarray,
        pt_color: str = 'white',
        pt_label: str = "point cloud",
        alpha: float = 1.0,
        fig=None,
        ax=None
    ):
    """
    绘制点云结果图 (使用 Matplotlib)
    Param:
        point_cloud: np.array,
            shape:  n,3 with (x,y,z)
    """
    
    # 提取 X, Y, Z 坐标
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    
    # 绘制散点图
    ax.scatter(x, y, z, c=pt_color, s=5,alpha=alpha, label=pt_label)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    return fig, ax

def drawVoxel_from_post_process_results(
    outputs: dict, 
    voxel_config: dict, 
    fig, 
    ax, 
    show_confidence: bool = True,
    voxel_color: list = [30 / 255.0, 144 / 255.0, 255 / 255.0, 0.4],
    edge_color: str = 'k',
    text_color: str = 'darkred'
):
    """
    根据后处理的 outputs 字典和 voxel_config 配置，在 3D 窗口上细粒度绘制体素结构，
    并在其最小外接立方体的边角处标注置信度。
    """
    region = voxel_config["region"]
    xlim = region["XLIM"]
    ylim = region["YLIM"]
    zlim = region["ZLIM"]
    voxel_size = voxel_config["resolution"]  # [dx, dy, dz]
    
    # 提取 Tensor 并转为 numpy 数组
    scores = outputs["scores"]
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
        
    masks = outputs["masks"]
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    # 构造统一的物理空间边界网格坐标
    x_edges = np.arange(xlim[0], xlim[1] + voxel_size[0] * 0.5, voxel_size[0])
    y_edges = np.arange(ylim[0], ylim[1] + voxel_size[1] * 0.5, voxel_size[1])
    z_edges = np.arange(zlim[0], zlim[1] + voxel_size[2] * 0.5, voxel_size[2])
    X_edges, Y_edges, Z_edges = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

    # 遍历每个预测实例 (N)
    num_instances = masks.shape[0]
    for i in range(num_instances):
        mask = masks[i]                 # shape: (X, Y, Z)
        confidence = scores[i]          # 标量置信度
        
        if not np.any(mask):
            continue
            
        voxel_bools = (mask > 0)
        
        # 细粒度体素渲染模式
        colors = np.empty(voxel_bools.shape + (4,), dtype=np.float32)
        colors[voxel_bools] = voxel_color
        
        ax.voxels(
            X_edges, Y_edges, Z_edges,
            voxel_bools,
            facecolors=colors,
            edgecolor=edge_color,
            linewidth=0.3
        )
        
        # 标注置信度文本：计算最小外接立方体，并在其某个边角处（例如最大 z、最大 y、最小 x 的角）标注一次
        if show_confidence:
            valid_x, valid_y, valid_z = np.where(voxel_bools)
            min_x, max_x = valid_x.min(), valid_x.max()
            min_y, max_y = valid_y.min(), valid_y.max()
            min_z, max_z = valid_z.min(), valid_z.max()
            
            # 选择外接立方体的某个特征物理顶点（如：顶端边缘）
            corner_x = xlim[0] + (max_x + 0.5) * voxel_size[0]
            corner_y = ylim[0] + (max_y + 0.5) * voxel_size[1]
            corner_z = zlim[0] + (max_z + 0.5) * voxel_size[2]
            
            ax.text(
                corner_x, corner_y, corner_z,
                f"{confidence:.2f}",
                fontsize=7,
                color=text_color,
                ha='center',
                va='bottom'
            )

    return fig, ax

def drawBoundingCube(
    bounding_cubes: np.ndarray,  # shape: (N, 6)
    fig,
    ax,
    scores: np.ndarray = None,   # shape: (N,)
    bbox_color: list = [255 / 255.0, 140 / 255.0, 0 / 255.0, 0.2],
    edge_color: str = 'darkorange',
    linewidth: float = 0.8,
    text_color: str = 'darkred',
    show_confidence: bool = True
):
    """
    根据多个 bounding_cube 的 6 维坐标绘制 3D 边界盒，并可选择标注置信度。
    
    参数:
        bounding_cubes (np.ndarray): 形状为 (N, 6) 的数组，每行为 [x_min, y_min, z_min, x_max, y_max, z_max]
        fig: matplotlib figure 对象
        ax: matplotlib axes 对象
        scores (np.ndarray): 形状为 (N,) 的置信度分数数组
        bbox_color (list): 边界盒填充颜色 [R, G, B, Alpha]
        edge_color (str): 边界盒边框颜色
        linewidth (float): 边框线宽
        text_color (str): 置信度文本颜色
        show_confidence (bool): 是否显示置信度
    """
    # 确保是 numpy 数组
    if isinstance(bounding_cubes, torch.Tensor):
        bounding_cubes = bounding_cubes.detach().cpu().numpy()
    
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    # 确保是 2D 数组
    if bounding_cubes.ndim == 1:
        bounding_cubes = bounding_cubes.reshape(1, -1)
    
    num_cubes = bounding_cubes.shape[0]
    
    # 遍历每个 bounding cube
    for i in range(num_cubes):
        cube = bounding_cubes[i]  # shape: (6,)
        x_min, y_min, z_min, x_max, y_max, z_max = cube
        
        # 构建 8 个顶点的坐标
        corners = np.array([
            [x_min, y_min, z_min],  # 0
            [x_max, y_min, z_min],  # 1
            [x_max, y_max, z_min],  # 2
            [x_min, y_max, z_min],  # 3
            [x_min, y_min, z_max],  # 4
            [x_max, y_min, z_max],  # 5
            [x_max, y_max, z_max],  # 6
            [x_min, y_max, z_max],  # 7
        ])
        
        # 组合 6 个面的顶点索引
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # 底面
            [corners[4], corners[5], corners[6], corners[7]],  # 顶面
            [corners[0], corners[1], corners[5], corners[4]],  # 前面
            [corners[2], corners[3], corners[7], corners[6]],  # 后面
            [corners[0], corners[3], corners[7], corners[4]],  # 左面
            [corners[1], corners[2], corners[6], corners[5]],  # 右面
        ]
        
        # 渲染 3D 外接盒表面与边框线
        poly = Poly3DCollection(
            faces,
            facecolors=bbox_color,
            edgecolors=edge_color,
            linewidths=linewidth,
            linestyles='-'
        )
        ax.add_collection3d(poly)
        
        # 在外接立方体的某个特征边角处标注置信度
        if show_confidence and scores is not None and i < len(scores):
            confidence = scores[i]
            corner_x, corner_y, corner_z = corners[7]
            ax.text(
                corner_x, corner_y, corner_z,
                f"{confidence:.2f}",
                fontsize=7,
                color=text_color,
                ha='center',
                va='bottom'
            )

    return fig, ax


def drawVoxelBoundingCube_from_post_process_results(
    outputs: dict, 
    voxel_config: dict, 
    fig, 
    ax,
    show_confidence: bool = True,
    bbox_color: list = [255 / 255.0, 140 / 255.0, 0 / 255.0, 0.2],
    edge_color: str = 'darkorange',
    linewidth: float = 0.8,
    text_color: str = 'darkred'
):
    """
    根据后处理的 outputs 字典和 voxel_config 配置，绘制包含所有体素的干净平滑的最小外接 3D 边界盒，
    并在其边角处标注置信度。
    """
    region = voxel_config["region"]
    xlim = region["XLIM"]
    ylim = region["YLIM"]
    zlim = region["ZLIM"]
    voxel_size = voxel_config["resolution"]  # [dx, dy, dz]
    
    scores = outputs["scores"]
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
        
    masks = outputs["masks"]
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    # 预先计算所有 bounding cubes
    all_bounding_cubes = []
    all_scores = []
    
    # 遍历每个预测实例 (N)
    num_instances = masks.shape[0]
    for i in range(num_instances):
        mask = masks[i]                 # shape: (X, Y, Z)
        confidence = scores[i]          # 标量置信度
        
        if not np.any(mask):
            continue
            
        voxel_bools = (mask > 0)
        
        # 1. 计算所有有效体素在三轴上的最小/最大索引范围
        valid_x, valid_y, valid_z = np.where(voxel_bools)
        min_x, max_x = valid_x.min(), valid_x.max()
        min_y, max_y = valid_y.min(), valid_y.max()
        min_z, max_z = valid_z.min(), valid_z.max()
        
        # 2. 将体素索引转换为真实物理坐标范围
        x_min = xlim[0] + min_x * voxel_size[0]
        x_max = xlim[0] + (max_x + 1) * voxel_size[0]
        y_min = ylim[0] + min_y * voxel_size[1]
        y_max = ylim[0] + (max_y + 1) * voxel_size[1]
        z_min = zlim[0] + min_z * voxel_size[2]
        z_max = zlim[0] + (max_z + 1) * voxel_size[2]
        
        # 3. 构建 bounding_cube
        bounding_cube = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
        all_bounding_cubes.append(bounding_cube)
        all_scores.append(confidence)
    
    # 转换为 numpy 数组
    all_bounding_cubes = np.array(all_bounding_cubes) if all_bounding_cubes else np.empty((0, 6))
    all_scores = np.array(all_scores) if all_scores else np.empty((0,))
    
    # 调用通用绘制函数
    fig,ax = drawBoundingCube(
        bounding_cubes=all_bounding_cubes,
        fig=fig,
        ax=ax,
        scores=all_scores,
        bbox_color=bbox_color,
        edge_color=edge_color,
        linewidth=linewidth,
        text_color=text_color,
        show_confidence=show_confidence
    )

    return fig, ax


def drawBbox_from_post_process_results(
    outputs: dict, 
    voxel_config: dict, 
    fig, 
    ax, 
    fill: bool = True,
    face_color: list = [1.0, 0.2, 0.2, 0.15],
    edge_color: str = 'red',
    text_color: str = 'darkred'
):
    """
    根据 outputs 中的归一化的 boxes 数据与 voxel_config 配置，在 3D 坐标系中绘制 3D Bounding Box，
    并在其顶角处自动标注置信度。
    
    参数:
        outputs (dict): 包含以下字段的字典:
                        - scores: tensor (N,), 置信度
                        - masks: tensor (N, X, Y, Z)
                        - boxes: tensor (N, 6), 格式为 [cx, cy, cz, sx, sy, sz]，归一化数值
        voxel_config (dict): 包含以下字段的配置字典:
                             - region: 包含 XLIM, YLIM, ZLIM 的字典
                             - resolution: 空间网格分辨率
        fig: matplotlib 的 figure 句柄。
        ax: matplotlib 的 3D 坐标轴句柄 (需满足 projection='3d')。
        fill (bool): 是否填充包围盒表面（默认为 True 半透明填充，False 为纯线框）。
        face_color (list): 包围盒表面的 RGBA 填充颜色，默认浅红色半透明。
        edge_color (str 或 list): 包围盒边界线条的颜色，默认红色。
        text_color (str): 置信度文本标注颜色，默认深红色。
        
    返回:
        tuple: (fig, ax) 绘制完成后的句柄对象。
    """
    # 提取空间物理边界
    region = voxel_config["region"]
    xlim = region["XLIM"]
    ylim = region["YLIM"]
    zlim = region["ZLIM"]
    
    # 计算三个物理轴向的全长
    lx = xlim[1] - xlim[0]
    ly = ylim[1] - ylim[0]
    lz = zlim[1] - zlim[0]
    
    boxes = outputs.get("boxes")
    if boxes is None:
        return fig, ax
        
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
        
    # 提取 scores 用于文本标注
    scores = outputs.get("scores")
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
        
    for i in range(boxes.shape[0]):
        bbox_norm = boxes[i]
        if bbox_norm is None or len(bbox_norm) < 6:
            continue
            
        # 1. 提取归一化的中心坐标和尺寸
        cx_norm, cy_norm, cz_norm, sx_norm, sy_norm, sz_norm = bbox_norm
        
        # 2. 反归一化到真实物理坐标系
        cx = xlim[0] + cx_norm * lx
        cy = ylim[0] + cy_norm * ly
        cz = zlim[0] + cz_norm * lz
        
        sx = sx_norm * lx
        sy = sy_norm * ly
        sz = sz_norm * lz
        
        # 计算 8 个顶点的极值范围
        x_min, x_max = cx - sx / 2.0, cx + sx / 2.0
        y_min, y_max = cy - sy / 2.0, cy + sy / 2.0
        z_min, z_max = cz - sz / 2.0, cz + sz / 2.0
        
        # 3. 构建 8 个顶点的坐标
        corners = np.array([
            [x_min, y_min, z_min],  # 0
            [x_max, y_min, z_min],  # 1
            [x_max, y_max, z_min],  # 2
            [x_min, y_max, z_min],  # 3
            [x_min, y_min, z_max],  # 4
            [x_max, y_min, z_max],  # 5
            [x_max, y_max, z_max],  # 6
            [x_min, y_max, z_max],  # 7
        ])
        
        # 4. 组合 6 个面的顶点索引
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # 底面
            [corners[4], corners[5], corners[6], corners[7]],  # 顶面
            [corners[0], corners[1], corners[5], corners[4]],  # 前面
            [corners[2], corners[3], corners[7], corners[6]],  # 后面
            [corners[0], corners[3], corners[7], corners[4]],  # 左面
            [corners[1], corners[2], corners[6], corners[5]],  # 右面
        ]
        
        # 5. 渲染 3D 面与边界线
        current_face_color = face_color if fill else [0.0, 0.0, 0.0, 0.0]
        poly = Poly3DCollection(
            faces,
            facecolors=current_face_color,
            edgecolors=edge_color,
            linewidths=1.2,
            linestyles='-'
        )
        ax.add_collection3d(poly)
        
        # 6. 在包围盒的特定顶角处自动标注置信度文本
        if scores is not None and i < len(scores):
            confidence = scores[i]
            corner_x, corner_y, corner_z = corners[7]
            ax.text(
                corner_x, corner_y, corner_z,
                f"{confidence:.2f}",
                fontsize=7,
                color=text_color,
                ha='center',
                va='bottom'
            )
    
    return fig, ax

def draw_tracking_targets(
    targets: List[TargetType],
    fig=None, 
    ax=None, 
    detection_color = [1.0, 0.0, 0.0, 1.0],
    active_color = [0.0, 1.0, 0.0, 1.0],
    free_color = [1.0, 1.0, 0.0, 1.0],
):
    """
    可视化目标列表 (3D版本)
    
    Args:
        fig: matplotlib figure对象
        ax: matplotlib axes对象
        ctrl: ctrlShowTracking控件实例
        targets: TargetType类的列表对象
    
    Returns:
        fig, ax
    """
    for target in targets:
        # 根据状态判断是否绘制
        # 获取对应状态的颜色
        if target.state.value == TrackState.DETECTION.value:
            color = detection_color
        elif target.state.value == TrackState.ACTIVE.value:
            color = active_color
        elif target.state.value == TrackState.FREE.value:
            color = free_color
        
        # 从S_hat中提取3D位置信息
        x = target.S_hat[0]
        y = target.S_hat[1]
        z = target.S_hat[2]
        
        # 绘制目标点
        ax.scatter(x, y, z, color=color)
        
        # 标明uid
        ax.text(x, y, z, f'uid:{target.uid}')
    
    return fig, ax

#