
from pathlib import Path
from data.dataset import PocHuPsenSif_for_eval
from src.models.milihupsen import DetrMiliHuPsen, DetrMiliHuPsenConfig
from src.models.modules.processor import PocPaddingProcessor
from typing import Dict, List, Tuple
import matplotlib
# 启用 WebAgg 后端
matplotlib.use("WebAgg")
# 可选：固定本地 Web 服务的端口
matplotlib.rcParams['webagg.port'] = 8988
matplotlib.rcParams['webagg.open_in_browser'] = False

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
from src.utils.tracking.core import *
from src.utils.tracking.functions import *
from src.utils.functions.visualize import *
from src.utils.functions.coordTrans import *
from src.utils.colors import *
from src.utils.functions.customs import (
    findClosestTimeFile,
    parseYAML,
    readHumanPose_pkl,
)
import json

# region 全局参数
CKPT_PATH= Path(r"/home/pai/Huawei/RHL/ckpt/ckpt_exp5")
VOXEL_CONFIG_PATH = Path(r"/home/pai/Huawei/RHL/src/configs/voxel.yaml")
ROOT_DIR = "/mnt/huawei"
DATE_TAG = "20260811"
GROUP_TAG = "group_030"
GROUP_TAGS = [f"{GROUP_TAG}"]
HUMAN_POSE_FOLDER = Path(rf"{ROOT_DIR}/{DATE_TAG}/data_collection/{GROUP_TAG}/camera results/smoothed 3D")
IMG2RADAR_EXT_FILE = Path(rf"{ROOT_DIR}/{DATE_TAG}/calib/extrinsic_img_to_radar_high.npz")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# endregion

def update_config(config, update_dict):
    for key, value in update_dict.items():
        if not hasattr(config, key):
            continue
        
        attr = getattr(config, key)
        # 如果原属性是自定义对象/配置类，且新值是字典，则递归更新
        if isinstance(value, dict) and hasattr(attr, "__dict__"):
            update_config(attr, value)
        # 如果原属性是字典，按字典更新
        elif isinstance(value, dict) and isinstance(attr, dict):
            attr.update(value)
        # 基本类型直接赋值
        else:
            setattr(config, key, value)

def load_detr_model():
    config_path = CKPT_PATH.joinpath("config.json")
    pt_path = CKPT_PATH.joinpath("DetrMiliHuPsen.pt")
    model_config = DetrMiliHuPsenConfig()
    if config_path.exists():
        with open(config_path, "r") as f:
            json_dict = json.load(f)
            update_config(model_config,json_dict)
    model=DetrMiliHuPsen(model_config)
    model.load_pt(pt_path=pt_path)
    return model.to(DEVICE).eval()

def resolve_point_cloud_velocity(
    raw_point_cloud: np.ndarray, 
    eps: float = 1e-10
) -> np.ndarray:
    """
    将原始雷达点云从 [x, y, z, v, ...] 解算并投影为 [x, y, z, vx, vy, vz] 格式。

    参数:
        raw_point_cloud (np.ndarray): 形状为 (N, 4) 或 (N, >=4) 的点云数组，
                                      前4列对应 [x, y, z, v] (空间坐标与径向多普勒速度)。
        eps (float): 防止除零的微小常数，默认 1e-10。

    返回:
        np.ndarray: 形状为 (N, 6) 的数组，列对应 [x, y, z, vx, vy, vz]，类型为 float32。
    """
    if raw_point_cloud is None or len(raw_point_cloud) == 0:
        return np.empty((0, 6), dtype=np.float32)

    # 提取 x, y, z 与径向速度 v
    px = raw_point_cloud[:, 0]
    py = raw_point_cloud[:, 1]
    pz = raw_point_cloud[:, 2]
    velocity = raw_point_cloud[:, 3]

    # 计算水平距离与空间总距离
    level_range = np.sqrt(px**2 + py**2)
    total_range = np.sqrt(px**2 + py**2 + pz**2)

    # 计算方位角与俯仰角三角函数
    cos_azimuth = px / (level_range + eps)
    sin_azimuth = py / (level_range + eps)
    cos_elevation = level_range / (total_range + eps)
    sin_elevation = pz / (total_range + eps)

    # 解算三维速度分量
    pvx = velocity * cos_azimuth * cos_elevation
    pvy = velocity * sin_azimuth * cos_elevation
    pvz = velocity * sin_elevation

    # 组合为 (N, 6) 格式
    return np.column_stack([px, py, pz, pvx, pvy, pvz]).astype(np.float32)

def create_detection_targets_from_result(
    result: Dict[str, torch.Tensor],
    voxel_config: dict,
    fmt_points: np.ndarray,
) -> Tuple[List[DetectionTarget], np.ndarray]:
    """
    将单帧检测结果转换为 DetectionTarget 列表，并在类内部记录各目标分配到的点云 mask/indices。

    参数:
        result (dict): 单个样本输出，包含:
                       - "boxes": Tensor/ndarray (K, 6), 格式为 [cx, cy, cz, sx, sy, sz] 归一化值
                       - "scores": Tensor/ndarray (K,), 置信度
        voxel_config (dict): 包含 region (XLIM, YLIM, ZLIM)
        fmt_points (np.ndarray): 原始点云 (N, 6),[x, y, z, vx, vy, vz]
    返回:
        targets (List[DetectionTarget]): 目标列表（每个对象内已封装归属当前目标的 point_mask）
    """
    targets: List[DetectionTarget] = []
    n_points = fmt_points.shape[0]

    # 2. 提取检测框
    boxes = result.get("boxes")
    if boxes is None or len(boxes) == 0:
        return targets
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()

    # 3. 物理范围尺寸参数
    region = voxel_config["region"]
    xlim, ylim, zlim = region["XLIM"], region["YLIM"], region["ZLIM"]
    lx = xlim[1] - xlim[0]
    ly = ylim[1] - ylim[0]
    lz = zlim[1] - zlim[0]

    for k in range(boxes.shape[0]):
        bbox_norm = boxes[k]
        if len(bbox_norm) < 6:
            continue

        cx_norm, cy_norm, cz_norm, sx_norm, sy_norm, sz_norm = bbox_norm[:6]

        # 反归一化中心与尺寸
        cx = xlim[0] + cx_norm * lx
        cy = ylim[0] + cy_norm * ly
        cz = zlim[0] + cz_norm * lz

        sx = sx_norm * lx
        sy = sy_norm * ly
        sz = sz_norm * lz

        x_min, x_max = cx - sx / 2.0, cx + sx / 2.0
        y_min, y_max = cy - sy / 2.0, cy + sy / 2.0
        z_min, z_max = cz - sz / 2.0, cz + sz / 2.0

        bbox_6d = np.array([x_min, y_min, z_min, x_max, y_max, z_max], dtype=np.float32)
        bbox_center = np.array([cx, cy, cz], dtype=np.float32)

        target = DetectionTarget(
            bounding_box=bbox_6d,
            bbox_center=bbox_center
        )
        if n_points > 0:
            # 判定属于当前框的点（各目标完全独立判断，点可复用）
            target_mask = (
                (fmt_points[:, 0] >= x_min) & (fmt_points[:, 0] <= x_max) &
                (fmt_points[:, 1] >= y_min) & (fmt_points[:, 1] <= y_max) &
                (fmt_points[:, 2] >= z_min) & (fmt_points[:, 2] <= z_max)
            )
            target.point_mask = target_mask
            target.point_indices = np.flatnonzero(target_mask)
            pts_in_target = fmt_points[target_mask]
            target.association_points = pts_in_target
            if len(pts_in_target) > 0:
                target.center_state = np.mean(pts_in_target, axis=0).astype(np.float32)
            else:
                target.center_state = np.array([0, 0, 0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            target.point_mask = np.zeros(0, dtype=bool)
            target.point_indices = np.empty(0, dtype=np.int64)
            target.association_points = np.empty((0, 6), dtype=np.float32)
            target.center_state = np.array([0, 0, 0, 0.0, 0.0, 0.0], dtype=np.float32)

        targets.append(target)

    return targets

class DatasetVisualizer:

    def __init__(
        self,
        dataset: PocHuPsenSif_for_eval,
        voxel_config: Dict,
        model,
        processor,
        threshold: float = 0.40,
    ):
        self.dataset = dataset
        self.voxel_config = voxel_config
        self.current_index = 0
        self.model = model
        self.processor = processor
        self.threshold = threshold

        # 跟踪
        self.tracking_info = TrackingInfo()
        self.kalman_stats=KalmanStats()
        self.tracking_alg_param = TrackingAlgParam()

        self.fig = None
        self.ax = None
        self.ax_2 = None
        self.text_box_left = None

        self.windowInit()

    def windowInit(self):
        self.fig = plt.figure(figsize=(12, 7))
        self.ax = self.fig.add_subplot(121, projection="3d")  # 跟踪与检测结果
        self.ax_2 = self.fig.add_subplot(122, projection="3d")  # GT与真值姿态

        self.ax = self.setAx(self.ax)
        self.ax_2 = self.setAx(self.ax_2)

        self.text_box_left = self.fig.text(
            0.02, 0.02,  # x, y位置（归一化坐标，0-1）
            "",  # 初始文本
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                alpha=0.8
            ),
            transform=self.fig.transFigure  # 使用figure坐标系
        )
        self.text_box_right = self.fig.text(
            0.80, 0.02,  # x, y位置（归一化坐标，0-1）
            "",  # 初始文本
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                alpha=0.8
            ),
            transform=self.fig.transFigure  # 使用figure坐标系
        )

        self.text_box_left_up = self.fig.text(
            0.02, 0.20,  # x, y位置（归一化坐标，0-1）
            "",  # 初始文本
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                alpha=0.8
            ),
            transform=self.fig.transFigure  # 使用figure坐标系
        )


        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        if len(self.dataset) > 0:
            self.showFrame(self.current_index)

    def setAx(self, ax):
        xlim = self.voxel_config["region"]["XLIM"]
        ylim = self.voxel_config["region"]["YLIM"]
        zlim = self.voxel_config["region"]["ZLIM"]

        ax.set_xlabel("X (Radar Front) [m]")
        ax.set_ylabel("Y (Radar Left) [m]")
        ax.set_zlabel("Z (Radar Up) [m]")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        return ax

    def clearWindow(self):
        self.ax.clear()
        self.ax = self.setAx(ax=self.ax)
        self.ax_2.clear()
        self.ax_2 = self.setAx(ax=self.ax_2)

    def tracking(self, det_rsts, fmt_points):
        target_predict(
            tracking_info = self.tracking_info,
            kalman_stats = self.kalman_stats,
        )
        cal_tatget_associate_gate(
            tracking_info=self.tracking_info,
            tracking_alg_param=self.tracking_alg_param
        )
        set_target_apriori_associate_points(
            tracking_info=self.tracking_info,
            fmt_points=fmt_points,
            tracking_alg_param=self.tracking_alg_param
        )
        target_associate_greedy(
            tracking_info=self.tracking_info,
            det_rsts=det_rsts,
            tracking_alg_param=self.tracking_alg_param,
        )
        set_target(
            tracking_info=self.tracking_info,
            det_rsts=det_rsts,
            fmt_points=fmt_points,
            tracking_alg_param=self.tracking_alg_param
        )

        set_target_det_associate_points(
            tracking_info=self.tracking_info,
            det_rsts=det_rsts,
            fmt_points=fmt_points
        )
        
        target_update(
            tracking_info=self.tracking_info,
            det_rsts=det_rsts,
            tracking_alg_param=self.tracking_alg_param,
            kalman_stats=self.kalman_stats
        )
        set_target_size_associate_points(
            tracking_info=self.tracking_info,
            fmt_points=fmt_points
        )
        set_target_associate_points(tracking_info=self.tracking_info)
        
        temp_target_update(
            tracking_info=self.tracking_info,
            tracking_alg_param=self.tracking_alg_param
        )
        

    def showFrame(self, index: int):
        if len(self.dataset) == 0:
            print("数据集中没有找到可绘制的内容！")
            return

        sample_info = self.dataset[index]
        current_label_file = sample_info["file_path"]

        self.clearWindow()

        sample_data = self.dataset[index]
        file_path = sample_data["file_path"]
        raw_point_cloud = sample_data["raw_point_cloud"]
        fmt_points = resolve_point_cloud_velocity(raw_point_cloud)


        inputs = self.processor(raw_point_cloud = raw_point_cloud).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(inputs)

        results=self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.threshold,
        )

        det_rsts = create_detection_targets_from_result(
            result=results[0],
            voxel_config=self.voxel_config,
            fmt_points=fmt_points
        )
        self.tracking(det_rsts=det_rsts,fmt_points=fmt_points)
        draw_tracking_targets(
            targets=self.tracking_info.tracking_targets,
            fig = self.fig,
            ax = self.ax
        )
        pose_file = findClosestTimeFile(input_file_path=file_path, folder_path=HUMAN_POSE_FOLDER)
        human_pose_raw = readHumanPose_pkl(file_path=pose_file)
        ext_info = loadExtrinsics(IMG2RADAR_EXT_FILE)
        human_pose = coordTrans(human_pose_raw, ext_info)
        self.fig, self.ax = drawHumanPose(human_pose=human_pose, fig=self.fig, ax=self.ax)

        show_info_left = ""
        for tracking_target in self.tracking_info.tracking_targets:
            show_info_left += str(tracking_target)

        self.text_box_left.set_text(show_info_left)

        show_info_left_up = ""
        for temp_target in self.tracking_info.temp_targets:
            show_info_left_up += str(temp_target)
        self.text_box_left_up.set_text(show_info_left_up)

        show_info_right = ""
        for det_rst in det_rsts:
            show_info_right += str(det_rst)
        self.text_box_right.set_text(show_info_right)

        self.fig, self.ax_2 = drawBbox_from_post_process_results(
            outputs= results[0], voxel_config=self.voxel_config, 
            fig = self.fig, ax = self.ax_2
        )
        inside_points_mask = np.zeros(len(raw_point_cloud), dtype=bool)
        for tracking_target in self.tracking_info.tracking_targets:
            inside_points_mask = inside_points_mask | tracking_target.point_mask
        inside_points = raw_point_cloud[inside_points_mask]
        outside_points = raw_point_cloud[~inside_points_mask]
        self.fig, self.ax_2 = drawHumanPose(human_pose=human_pose, fig=self.fig, ax=self.ax_2)
        self.fig, self.ax_2 = drawPointCloud(
            point_cloud=inside_points, pt_color="blue",
            pt_label="valid points", fig=self.fig, ax=self.ax_2
        )
        self.fig, self.ax_2 = drawPointCloud(
            point_cloud=outside_points, pt_color="black",
            pt_label="invalid points", alpha=0.2,fig=self.fig, ax=self.ax_2
        )

        self.ax.set_title(f"Dataset Visualization: Frame [{index + 1}/{len(self.dataset)}]: {current_label_file.name}")

        # 5. 刷新画布显示
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """
        键盘事件响应函数：按下空格键切换至下一帧
        """
        if event.key == ' ': # 空格键
            if len(self.dataset) > 0:
                self.current_index = (self.current_index + 1) % len(self.dataset) # 循环播放
                self.showFrame(self.current_index)


def main():
    dataset = PocHuPsenSif_for_eval(
        root_dir=ROOT_DIR,
        date_tag=DATE_TAG,
        group_tags=GROUP_TAGS,
        consider_invalid=True,
    )
    print(f"成功加载数据集，样本总数: {len(dataset)}")

    voxel_config = parseYAML(file_path=VOXEL_CONFIG_PATH)
    processor = PocPaddingProcessor(voxel_config=voxel_config)

    app = DatasetVisualizer(
        dataset=dataset,
        voxel_config=voxel_config,
        model=load_detr_model(),
        processor=processor,
        threshold=0.40,
    )
    plt.show()


if __name__ == "__main__":
    main()