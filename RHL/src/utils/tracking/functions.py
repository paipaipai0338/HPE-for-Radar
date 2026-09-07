from src.utils.tracking.core import *
from typing import List
import numpy as np
import math
from typing import Optional, Union


# 建立新目标
def set_target(
    tracking_info: TrackingInfo,
    det_rsts: List[DetectionTarget],
    fmt_points: np.ndarray,
    tracking_alg_param: TrackingAlgParam
):
    global ID

    for det_rst in det_rsts:
        if det_rst.best_id != -1:
            continue
        # 判断目标当前检测位置是否位于 核心区域
        
        x = det_rst.bbox_center[0]
        y = det_rst.bbox_center[1]
        
        in_core_area = False
        if (
            tracking_alg_param.core_area.xback <= x <= tracking_alg_param.core_area.xfront and 
            tracking_alg_param.core_area.yright <= y <= tracking_alg_param.core_area.yleft
            ):
            in_core_area = True
        if not in_core_area:
            establish_target_points_thresh = 0.35 * tracking_alg_param.establish_target_points_thresh
        else:
            establish_target_points_thresh = tracking_alg_param.establish_target_points_thresh
        if len(det_rst.association_points) < establish_target_points_thresh:
            det_rst.discard = True
            continue
        isAdjacent = 0
        for target in tracking_info.tracking_targets:
            sub_x = target.S_apriori_hat[0] - det_rst.bbox_center[0]
            sub_y = target.S_apriori_hat[1] - det_rst.bbox_center[1]
            sub_z = target.S_apriori_hat[2] - det_rst.bbox_center[2]
            d = np.sqrt(sub_x * sub_x + sub_y * sub_y + sub_z * sub_z)
            if d < tracking_alg_param.adjDistThre:  
                isAdjacent = 1
                break
        if isAdjacent:
            continue
        det_rst.best_id = ID

        P_apriori_hat=np.zeros((6, 6), dtype=np.float32)
        P_apriori_hat[3, 3] = 0.5
        P_apriori_hat[4, 4] = 0.5
        P_apriori_hat[5, 5] = 0.5
        tracking_info.add_temp_target(
            TempTargetType(
                uid=ID,
                hit_count=0,
                size = np.array(
                    [
                        det_rst.bounding_box[3] - det_rst.bounding_box[0],
                        det_rst.bounding_box[4] - det_rst.bounding_box[1],
                        det_rst.bounding_box[5] - det_rst.bounding_box[2],
                    ], dtype=np.float32),
                apriori_associate_point_mask=np.zeros(len(fmt_points), dtype=bool),
                S_apriori_hat=np.concatenate([det_rst.bbox_center, det_rst.center_state[3:]]),
                S_hat=np.concatenate([det_rst.bbox_center, det_rst.center_state[3:]]),
                P_apriori_hat=P_apriori_hat,
                sFactor = 1.0,
                detect2freeCount = 0,
                detect2activeCount = 0,
                active2freeCount = 0,
                estNumOfPoints = 0.0,
                state = TrackState.DETECTION,
                isTargetStatic = 0,
                useStaticAssist = 0,
                delete_flag = 0,
                gC_det = 0.0,
                establish_target_points_thresh=establish_target_points_thresh,
            )
        )
        ID+=1
        ID=max(ID%80,1)


def cal_mahalanobis_partial_mat(
    v_diff: np.array,  # shape: (N, 6)
    conv_inv: np.array  # shape: (6, 6)
):
    """
    v_diff: shape (N, 6)
    conv_inv: shape (6, 6)
    return: shape (N,)
    """
    # 仅提取前 3 个维度的特征和对应的 3x3 协方差逆子矩阵
    v_sub = v_diff[:, :3]               # shape: (N, 3)
    cov_sub = conv_inv[:3, :3]          # shape: (3, 3)

    # 利用 einsum 批量计算 v_sub @ cov_sub @ v_sub.T 的对角线元素
    return np.einsum('ni,ij,nj->n', v_sub, cov_sub, v_sub)


# 更新目标关联点-主动
def update_tatget_associate_points_active(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    det_rst: DetectionTarget,
    fmt_points: np.ndarray,
    tracking_alg_param: TrackingAlgParam
):
    """
    主动更新目标关联点
    参数:
        fmt_points (np.ndarray): 原始点云 (N, 6),[x, y, z, vx, vy, vz]
    """
    # 基础掩码 (N,)
    base_point_mask = det_rst.point_mask

    # 1. 计算偏差 (N, 6)
    offset = fmt_points - tracking_target.S_apriori_hat

    # 2. 批量马氏距离判定 (N,)
    # 注意：cal_mahalanobis_partial_mat 计算出的是距离的平方 (D^2)
    # 如果 gain 是平方阈值直接比，如果是标准差阈值需取 np.sqrt(mp_distance)
    mp_distance = cal_mahalanobis_partial_mat(offset, tracking_target.gC_points_inv)
    mp_distance_threshold = tracking_alg_param.gain
    mp_mask = mp_distance < mp_distance_threshold

    # 3. 批量三维欧氏距离判定 (N,)
    # 直接计算平方和开方，或用 np.linalg.norm(offset[:, :3], axis=1)
    euclidean_distance = np.linalg.norm(offset[:, :3], axis=1)
    euclidean_distance_threshold = tracking_target.euclidean_distance_threshold
    euclidean_mask = euclidean_distance < euclidean_distance_threshold

    # 4. 马氏距离与欧式距离同时满足的掩码
    gating_mask = mp_mask & euclidean_mask

    # 5. 与基础掩码取并集 (OR 操作)
    updated_mask = base_point_mask | gating_mask
    tracking_target.point_mask = updated_mask
    tracking_target.point_indices = np.flatnonzero(updated_mask)

def get_target_apriori_associate_point_mask(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    fmt_points: np.ndarray,
    tracking_alg_param: TrackingAlgParam
):
    offset = fmt_points - tracking_target.S_apriori_hat
    
    # 2. 批量马氏距离判定 (N,)
    # 注意：cal_mahalanobis_partial_mat 计算出的是距离的平方 (D^2)
    # 如果 gain 是平方阈值直接比，如果是标准差阈值需取 np.sqrt(mp_distance)
    mp_distance = cal_mahalanobis_partial_mat(offset, tracking_target.gC_points_inv)
    mp_distance_threshold = tracking_alg_param.gain
    mp_mask = mp_distance < mp_distance_threshold
    
    # 3. 批量三维欧氏距离判定 (N,)
    # 直接计算平方和开方，或用 np.linalg.norm(offset[:, :3], axis=1)
    euclidean_distance = np.linalg.norm(offset[:, :3], axis=1)
    euclidean_distance_threshold = tracking_target.euclidean_distance_threshold
    euclidean_mask = euclidean_distance < euclidean_distance_threshold
    
    # 4. 马氏距离与欧式距离同时满足的掩码
    gating_mask = mp_mask & euclidean_mask

    tracking_target.apriori_associate_point_mask = gating_mask

# 计算目标先验位置关联点掩码
def set_target_apriori_associate_points(
    tracking_info: TrackingInfo,
    fmt_points: np.ndarray,
    tracking_alg_param: TrackingAlgParam
):
    for temp_target in tracking_info.temp_targets:
        get_target_apriori_associate_point_mask(
            tracking_target=temp_target,
            fmt_points=fmt_points,
            tracking_alg_param=tracking_alg_param,
        )
    for tracking_target in tracking_info.tracking_targets:
        get_target_apriori_associate_point_mask(
            tracking_target=tracking_target,
            fmt_points=fmt_points,
            tracking_alg_param=tracking_alg_param,
        )

# 计算目标检测位置关联点掩码
def set_target_det_associate_points(
    tracking_info: TrackingInfo,
    det_rsts: List[DetectionTarget],
    fmt_points: np.ndarray,
):
    n_points = len(fmt_points)
    for temp_target in tracking_info.temp_targets:
        is_associate = False
        for det_rst in det_rsts:
            if det_rst.best_id == temp_target.uid:
                is_associate = True
                temp_target.det_associate_point_mask = det_rst.point_mask
                break
        if not is_associate:
            temp_target.det_associate_point_mask = np.zeros(n_points, dtype=bool)
    # 设置正式对象检测关联点掩码
    for tracking_target in tracking_info.tracking_targets:
        is_associate = False
        for det_rst in det_rsts:
            if det_rst.best_id == tracking_target.uid:
                is_associate = True
                tracking_target.det_associate_point_mask = det_rst.point_mask
                break
        if not is_associate:
            tracking_target.det_associate_point_mask = np.zeros(n_points, dtype=bool)

def get_target_size_associate_points(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    fmt_points: np.ndarray,
):
    n_points = len(fmt_points)
    if n_points == 0:
        tracking_target.size_associate_point_mask = np.zeros(0,dtype=bool)
        return 
    x_min = tracking_target.bounding_box[0]
    y_min = tracking_target.bounding_box[1]
    z_min = tracking_target.bounding_box[2]
    x_max = tracking_target.bounding_box[3]
    y_max = tracking_target.bounding_box[4]
    z_max = tracking_target.bounding_box[5]
    target_mask = (
        (fmt_points[:, 0] >= x_min) & (fmt_points[:, 0] <= x_max) &
        (fmt_points[:, 1] >= y_min) & (fmt_points[:, 1] <= y_max) &
        (fmt_points[:, 2] >= z_min) & (fmt_points[:, 2] <= z_max)
    )
    tracking_target.size_associate_point_mask = target_mask

# 计算目标后验位置尺寸关联点掩码
def set_target_size_associate_points(
    tracking_info: TrackingInfo,
    fmt_points: np.ndarray,
):  
    for temp_target in tracking_info.temp_targets:
        get_target_size_associate_points(
            tracking_target=temp_target,
            fmt_points=fmt_points
        )
    for tracking_target in tracking_info.tracking_targets:
        get_target_size_associate_points(
            tracking_target=tracking_target,
            fmt_points=fmt_points
        )

def set_target_associate_points(
    tracking_info: TrackingInfo
):
    for temp_target in tracking_info.temp_targets:
        target_mask = (
            temp_target.det_associate_point_mask | 
            temp_target.apriori_associate_point_mask | 
            temp_target.size_associate_point_mask
        )
        temp_target.point_mask = target_mask
        temp_target.point_indices = np.flatnonzero(target_mask)
    for tracking_target in tracking_info.tracking_targets:
        target_mask = (
            tracking_target.det_associate_point_mask | 
            tracking_target.apriori_associate_point_mask | 
            tracking_target.size_associate_point_mask
        )
        tracking_target.point_mask = target_mask
        tracking_target.point_indices = np.flatnonzero(target_mask)

# 计算目标关联门限
def cal_target_associate_gate_unit(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    tracking_alg_param: TrackingAlgParam
):
    def adjust_gate_limits_for_fast_moving(
        tracking_target: Optional[Union[TargetType, TempTargetType]],
        adjusted_gate_limits: np.array,
        tracking_alg_param: TrackingAlgParam,
        velocity: float
    ):
        """
        Param:
            adjusted_gate_limits: 动态门限;
                shape: 4,(gate_x,gate_y,gate_vx,gate_vy)
            velocity: 当前跟踪目标的速度
        """
        vx = tracking_target.S_hat[3]
        vy = tracking_target.S_hat[4]
        vz = tracking_target.S_hat[5]
        theta_a = np.arctan2(vy, vx)
        theta_e = np.arctan2(vz, np.sqrt(vx**2 + vy**2))
        constant_speed = 1.0
        speed_factor = 1.2
        if velocity > constant_speed:
            speed_factor += constant_speed
        else:
            speed_factor += velocity
        cos_theta_a = np.cos(theta_a)
        sin_theta_a = np.sin(theta_a)
    
        cos_theta_e = np.cos(theta_e)
        sin_theta_e = np.sin(theta_e)
    
        longitudinal_scale = speed_factor   # 沿运动方向（纵向）
        lateral_scale = 1.0                 # 垂直运动方向（横向）
        local_x_limit = 1.0 * tracking_alg_param.gate_limits[0] * longitudinal_scale
        local_y_limit = 1.0 * tracking_alg_param.gate_limits[1] * lateral_scale
        local_z_limit = 1.0 * tracking_alg_param.gate_limits[2] * lateral_scale
        local_rl_limit = np.sqrt(local_x_limit**2 + local_y_limit**2)
    
        rotated_x_extent = abs(local_x_limit * cos_theta_a) + abs(local_y_limit * sin_theta_a)
        rotated_y_extent = abs(local_x_limit * sin_theta_a) + abs(local_y_limit * cos_theta_a)
        rotated_z_extent = abs(local_z_limit * sin_theta_e) + abs(local_rl_limit * cos_theta_e)
    
        adjusted_gate_limits[0] = rotated_x_extent
        adjusted_gate_limits[1] = rotated_y_extent
        adjusted_gate_limits[2] = rotated_z_extent
        return adjusted_gate_limits

    velocity_threshold = 0.2  # 速度阈值
    base_gate_limits = np.array([
        tracking_alg_param.gate_limits[0],    # x
        tracking_alg_param.gate_limits[1],    # y
        tracking_alg_param.gate_limits[2],    # z
        tracking_alg_param.gate_limits[3],    # vx
        tracking_alg_param.gate_limits[4],    # vy
        tracking_alg_param.gate_limits[5]     # vz
    ], dtype=np.float32)
    adjusted_gate_limits = base_gate_limits.copy()
    vx = tracking_target.S_hat[3]
    vy = tracking_target.S_hat[4]
    vz = tracking_target.S_hat[5]
    velocity = math.sqrt(vx * vx + vy * vy + vz * vz)

    scale_factor = 1.0
    if tracking_target.useStaticAssist == 1:
        scale_factor = 1.3
    elif int(tracking_target.state.value) == 1 and 0 < tracking_target.active2freeCount <= 5:
        scale_factor = 1.2
    
    if velocity > velocity_threshold:
        adjusted_gate_limits=adjust_gate_limits_for_fast_moving(
            tracking_target,adjusted_gate_limits,tracking_alg_param,velocity
        )
    else:
        adjusted_gate_limits[0] = base_gate_limits[0]
        adjusted_gate_limits[1] = base_gate_limits[1]
        adjusted_gate_limits[2] = base_gate_limits[2]

    adjusted_gate_limits[0] *= scale_factor
    adjusted_gate_limits[1] *= scale_factor
    adjusted_gate_limits[2] *= scale_factor

    if adjusted_gate_limits[0] > 2.8:
        adjusted_gate_limits[0] = 2.8
    if adjusted_gate_limits[1] > 2.8:
        adjusted_gate_limits[1] = 2.8
    if adjusted_gate_limits[2] > 2.8:
        adjusted_gate_limits[2] = 2.8

    euclidean_distance_threshold = 0.8
    if velocity > velocity_threshold:
        euclidean_distance_threshold += 1.3 * velocity
    
    euclidean_distance_threshold *= scale_factor
    if euclidean_distance_threshold > 1.6:
        euclidean_distance_threshold = 1.6


    tracking_target.gate_limits = adjusted_gate_limits
    tracking_target.euclidean_distance_threshold = euclidean_distance_threshold

def cal_tatget_associate_gate(
    tracking_info: TrackingInfo,
    tracking_alg_param: TrackingAlgParam
):
    for temp_target in tracking_info.temp_targets:
        cal_target_associate_gate_unit(temp_target, tracking_alg_param=tracking_alg_param)
    for tracking_target in tracking_info.tracking_targets:
        cal_target_associate_gate_unit(tracking_target, tracking_alg_param=tracking_alg_param)

# 新旧目标关联
def target_associate_unit(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    det_rsts: List[DetectionTarget],
    tracking_alg_param: TrackingAlgParam
):
    def cal_mahalanobis_partial(
        v_diff: np.array,
        conv_inv: np.array
    ):
        v_new = np.zeros(6, dtype=np.float32)
        v_new[0] = v_diff[0]
        v_new[1] = v_diff[1]
        v_new[2] = v_diff[2]
        return v_new @ conv_inv @ v_new

    adjusted_gate_limits = tracking_target.gate_limits
    euclidean_distance_threshold = tracking_target.euclidean_distance_threshold
    for det_rst in det_rsts:
        offset= tracking_target.S_apriori_hat[:3] - det_rst.bbox_center
        if ( 
            abs(offset[0]) > adjusted_gate_limits[0] or 
            abs(offset[1]) > adjusted_gate_limits[1] or
            abs(offset[2]) > adjusted_gate_limits[2]
        ):
            continue
        mp_distance=cal_mahalanobis_partial(offset,tracking_target.gC_inv)
        euclidean_distance = math.sqrt(
            offset[0] * offset[0] + 
            offset[1] * offset[1] + 
            offset[2] * offset[2]
        )
        if mp_distance < tracking_alg_param.gain and euclidean_distance < euclidean_distance_threshold:
            score = math.log(tracking_target.gC_det) + mp_distance
            if score < det_rst.best_score:
                det_rst.best_score = score
                det_rst.best_id = tracking_target.uid

def target_associate(
    tracking_info: TrackingInfo,
    det_rsts: List[DetectionTarget],
    tracking_alg_param: TrackingAlgParam
):
    for temp_target in tracking_info.temp_targets:
        target_associate_unit(
            tracking_target=temp_target,
            det_rsts=det_rsts,
            tracking_alg_param=tracking_alg_param,
        )
    for tracking_target in tracking_info.tracking_targets:
        target_associate_unit(
            tracking_target=tracking_target,
            det_rsts=det_rsts,
            tracking_alg_param=tracking_alg_param,
        )
    pass

def target_associate_greedy(
    tracking_info: TrackingInfo,
    det_rsts: List[DetectionTarget],
    tracking_alg_param: TrackingAlgParam
):
    candidates = []

    def cal_mahalanobis_partial(v_diff: np.ndarray, conv_inv: np.ndarray) -> float:
        v_new = np.zeros(6, dtype=np.float32)
        v_new[:3] = v_diff[:3]
        return float(v_new @ conv_inv @ v_new)

    # 1. 统合所有待匹配的航迹池 (成熟航迹 + 临时航迹)
    all_tracks = tracking_info.tracking_targets + tracking_info.temp_targets

    # 2. 收集所有有效候选对
    for trk in all_tracks:
        gate_limits = trk.gate_limits
        e_thresh = trk.euclidean_distance_threshold
        t_uid = str(trk.uid)

        for d_idx, det in enumerate(det_rsts):
            offset = trk.S_apriori_hat[:3] - det.bbox_center
            if (abs(offset[0]) > gate_limits[0] or 
                abs(offset[1]) > gate_limits[1] or 
                abs(offset[2]) > gate_limits[2]):
                continue

            mp_dist = cal_mahalanobis_partial(offset, trk.gC_inv)
            e_dist = math.sqrt(np.dot(offset, offset))
            det.mp_distance[t_uid] = mp_dist
            det.e_distance[t_uid] = e_dist

            if mp_dist < tracking_alg_param.gain and e_dist < e_thresh:
                score = math.log(max(trk.gC_det, 1e-6)) + mp_dist
                # 记录 (score, trk.uid, d_idx, trk)
                candidates.append((score, trk.uid, d_idx))

    # 3. 按得分升序排序（代价小的优先匹配）
    candidates.sort(key=lambda x: x[0])

    matched_track_uids = set()
    matched_dets = set()

    # 4. 贪心锁定匹配关系
    for score, t_uid, d_idx in candidates:
        if t_uid not in matched_track_uids and d_idx not in matched_dets:
            matched_track_uids.add(t_uid)
            matched_dets.add(d_idx)
            det_rsts[d_idx].best_score = score
            det_rsts[d_idx].best_id = t_uid

def track_event(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    tracking_alg_param: TrackingAlgParam,
    points_num: int,
):
    thre = 0
    match tracking_target.state.value:
        case TrackState.DETECTION.value:
            if points_num >= tracking_alg_param.pointsThre:
                if tracking_target.detect2freeCount > 0:
                    tracking_target.detect2freeCount -= 1
                
                tracking_target.detect2activeCount += 1
                
                # 连续命中达到激活阈值 -> 晋升为稳定航迹
                if tracking_target.detect2activeCount >= tracking_alg_param.det2actThre:
                    tracking_target.state = TrackState.ACTIVE
            else:
                tracking_target.detect2freeCount += 1
                if tracking_target.detect2activeCount > 0:
                    tracking_target.detect2activeCount -= 1
                
                # 特殊逻辑：如果目标速度为 0 (静止)，则加速消亡惩罚
                if (
                    tracking_target.S_hat[3] == 0.0 and 
                    tracking_target.S_hat[4] == 0.0 and
                    tracking_target.S_hat[5] == 0.0 
                    ):
                    tracking_target.detect2freeCount += 1
                    if tracking_target.detect2activeCount > 0:
                        tracking_target.detect2activeCount -= 1
                if tracking_target.detect2freeCount >= tracking_alg_param.det2freeThre:
                    tracking_target.state = TrackState.FREE
        case TrackState.ACTIVE.value:
            # 只要有变动点关联上（HIT）
            if points_num != 0:
                tracking_target.active2freeCount = 0
            
            # 漏检（MISS）
            else:
                tracking_target.active2freeCount += 1
                
                # 获取预测位置
                x = tracking_target.S_apriori_hat[0]
                y = tracking_target.S_apriori_hat[1]
                z = tracking_target.S_apriori_hat[2]
                
                # 判断目标当前预测位置是否依然在防区/场景（Box）内
                inScene = False
                if (
                    tracking_alg_param.box.xback <= x <= tracking_alg_param.box.xfront and 
                    tracking_alg_param.box.yright <= y <= tracking_alg_param.box.yleft and 
                    tracking_alg_param.box.zbottom <= z <= tracking_alg_param.box.ztop
                    ):
                    inScene = True
                
                # 根据是否在场景内以及边界条件，动态计算允许连续漏检的最大帧数门限 (thre)
                if inScene:
                    thre = 30
                else:
                    # 不在场景内（已出界）：默认生存阈值
                    thre = tracking_alg_param.active2freeThre  # 通常为 30
                    
                    # 容错：生存时间阈值不能高于航迹本身的寿命
                    if thre > tracking_target.heartBeatCount:
                        thre = tracking_target.heartBeatCount
                
                # 连续漏检超过计算出来的阈值 -> 释放航迹
                if tracking_target.active2freeCount > thre:
                    tracking_target.state = TrackState.FREE
        case _:
            pass

# 卡尔曼更新-主动
def kalman_update_active(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    det_rst: DetectionTarget,
    tracking_alg_param: TrackingAlgParam,
    kalman_stats: KalmanStats,
):
    def dispersion_cov(associate_points: np.array, centroid: np.array):
        """
        Param:
            centroid: shape (6,), x-mean, y-mean, z-mean, vx-mean, vy-mean, vz-mean
            associate_points: shape (N, 6), N个关联点
        Return:
            covariance: np.array (6, 6)，协方差矩阵
        """
        # 计算所有点到质心的残差: (N, 6)
        delta = associate_points - centroid  # 广播机制
        
        # 计算协方差矩阵: (6, 6)
        # delta.T @ delta 等价于 sum(outer(delta_i, delta_i))
        covariance = delta.T @ delta / len(associate_points)
        
        return covariance
    tracking_target.heartBeatCount += 1
    U_max = np.array([-1000.0, -1000.0, -1000.0], dtype=np.float32)
    U_min = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
    for association_point in det_rst.association_points:
        u = np.array(
            [association_point[3], association_point[4], association_point[5]] , dtype=np.float32
        )
        U_max = np.maximum(U_max, u)
        U_min = np.minimum(U_min, u)
    associate_points_num = len(det_rst.association_points)
    if associate_points_num > 0 :
        if associate_points_num > tracking_target.estNumOfPoints:
            tracking_target.estNumOfPoints = float(associate_points_num)     # 该航迹所关联的点的总数
        else:
            tracking_target.estNumOfPoints = ( 
                0.99 * tracking_target.estNumOfPoints + 
                0.01 * associate_points_num
            )
        if tracking_target.estNumOfPoints < tracking_alg_param.pointsThre:
            tracking_target.estNumOfPoints = float(tracking_alg_param.pointsThre)
        center_state = np.concatenate([det_rst.bbox_center, det_rst.center_state[3:]])
        spread_velocity = U_max - U_min
        spread_size = np.abs(det_rst.bounding_box[3:] - det_rst.bounding_box[:3])
        spread = np.concatenate([spread_size, spread_velocity])

        for m in range(6):
            if spread[m] < tracking_alg_param.spreadMin[m]:
                spread[m] = tracking_alg_param.spreadMin[m]
            if spread[m] > tracking_alg_param.gate_limits[m]:
                spread[m] = tracking_alg_param.gate_limits[m]
                
            if spread[m] > tracking_target.estSpread[m]:
                tracking_target.estSpread[m] = spread[m]
            else:
                tracking_target.estSpread[m] = (
                    (1 - tracking_alg_param.spreadAlpha) * tracking_target.estSpread[m] + 
                    tracking_alg_param.spreadAlpha * spread[m]
                )
        Rm = np.zeros((6, 6), dtype=np.float32)
        for m in range(3):
            sigma = (det_rst.bbox_center[m] - det_rst.center_state[m])
            Rm[m, m] = sigma * sigma
        for m in range(3):
            sigma = tracking_target.estSpread[3+m] * 0.5
            Rm[3+m, 3+m] = sigma * sigma
    else:
        # 将该目标速度维度置零
        center_state = tracking_target.S_apriori_hat.copy()
        tracking_target.S_apriori_hat[3] = 0.0
        tracking_target.S_apriori_hat[4] = 0.0
        tracking_target.S_apriori_hat[5] = 0.0

        Rm = np.zeros((6, 6), dtype=np.float32)
        for m in range(6):
            sigma = tracking_target.estSpread[m] * 0.5
            Rm[m, m] = sigma * sigma

    tracking_target.Center[:] = center_state
    
    if associate_points_num > tracking_alg_param.MinPointUpdateDispersion:
        # 离散度计算
        dispersion = dispersion_cov(det_rst.association_points, center_state)
        alpha = associate_points_num / tracking_target.estNumOfPoints
        tracking_target.gD = alpha * dispersion + (1 - alpha) * tracking_target.gD

    HPH = kalman_stats.H @ tracking_target.P_apriori_hat @ kalman_stats.H.T

    if associate_points_num >= tracking_alg_param.minpts:
        alpha = (tracking_target.estNumOfPoints - associate_points_num) / ((tracking_target.estNumOfPoints - 1) * associate_points_num)
        # 目标状态测量噪声协方差，并非单一点，而是所有点组成的整体目标
        Rc = (1.0 / associate_points_num) * Rm + alpha * tracking_target.gD
        
        Rc = Rc * tracking_alg_param.Rc_scale
        # 实际测量值-预测先验估计值（对于新的跟踪目标，该值为0）
        u_tilda = center_state - tracking_target.S_apriori_hat
        inv_Covar = np.linalg.inv(Rc + HPH)
        # 卡尔曼增益K
        K = tracking_target.P_apriori_hat @ kalman_stats.H.T @ inv_Covar

        # 更新状态估计值
        tracking_target.S_hat = tracking_target.S_apriori_hat + K @ u_tilda
        # 更新后验估计协方差矩阵
        tracking_target.P_hat = tracking_target.P_apriori_hat - K @ kalman_stats.H @ tracking_target.P_apriori_hat
    else:
        # 关联点不足，直接沿用先验预测值（观测不可靠不更新）
        tracking_target.S_hat[:] = tracking_target.S_apriori_hat
        tracking_target.P_hat[:, :] = tracking_target.P_apriori_hat
        
    if isinstance(tracking_target, TempTargetType):
        if associate_points_num > 0.8 * tracking_target.establish_target_points_thresh:
            tracking_target.hit_count += 1
        else:
            tracking_target.hit_count -= 1

    if tracking_target.isTargetStatic == 1:
        tracking_target.S_hat[0] = tracking_target.S_apriori_saved[0]
        tracking_target.S_hat[1] = tracking_target.S_apriori_saved[1]

    # 波门限制协方差矩阵
    # 它衡量的是 传感器的测量值 与基于 上一帧预测值 得到的 预期测量值 之间的 误差统计不确定性
    # 当前帧的检测点位 Y = HX + e_noise
    # X：目标的几何中心，e_noise：测量噪声
    # Y-HX_pre = H（X-X_pre）+ e_noise
    # cov(Y-HX_pre)=Hcov（X-X_pre）H.T+cov(e_noise)
    tracking_target.gC = HPH + Rm
    tracking_target.gC_inv = np.linalg.inv(tracking_target.gC)
    tracking_target.gC_det = float(np.linalg.det(tracking_target.gC))

    tracking_target.gC_points = tracking_target.gD + HPH + Rm
    tracking_target.gC_points_inv = np.linalg.inv(tracking_target.gC_points)
    tracking_target.gC_points_det = float(np.linalg.det(tracking_target.gC_points))

    # 目标 size 更新
    new_size = np.array(
        [
            det_rst.bounding_box[3] - det_rst.bounding_box[0],
            det_rst.bounding_box[4] - det_rst.bounding_box[1],
            det_rst.bounding_box[5] - det_rst.bounding_box[2],
        ], dtype=np.float32
    )
    tracking_target.size = tracking_alg_param.sizeSmoothFactor * tracking_target.size + (1 - tracking_alg_param.sizeSmoothFactor) * new_size
    # 计算目标后验位置尺寸
    x_min, x_max = tracking_target.S_hat[0] - tracking_target.size[0] / 2.0, tracking_target.S_hat[0] + tracking_target.size[0] / 2.0
    y_min, y_max = tracking_target.S_hat[1] - tracking_target.size[1] / 2.0, tracking_target.S_hat[1] + tracking_target.size[1] / 2.0
    z_min, z_max = tracking_target.S_hat[2] - tracking_target.size[2] / 2.0, tracking_target.S_hat[2] + tracking_target.size[2] / 2.0
    tracking_target.bounding_box = np.array([x_min, y_min, z_min, x_max, y_max, z_max], dtype=np.float32)

    track_event(
        tracking_target,
        tracking_alg_param,
        associate_points_num
    )

# 卡尔曼更新-被动
def kalman_update_passive(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    tracking_alg_param: TrackingAlgParam,
    kalman_stats: KalmanStats,
):
    tracking_target.heartBeatCount += 1

    uCentroid = tracking_target.S_apriori_hat.copy()
    tracking_target.S_apriori_hat[3] = 0.0
    tracking_target.S_apriori_hat[4] = 0.0
    tracking_target.S_apriori_hat[5] = 0.0
    tracking_target.Center[:] = uCentroid
    # 测量噪声的协方差矩阵
    Rm = np.zeros((6, 6), dtype=np.float32)
    for m in range(6):
        sigma = tracking_target.estSpread[m] * 0.5
        Rm[m, m] = sigma * sigma

    HPH = kalman_stats.H @ tracking_target.P_apriori_hat @ kalman_stats.H.T
    tracking_target.S_hat[:] = tracking_target.S_apriori_hat
    tracking_target.P_hat[:, :] = tracking_target.P_apriori_hat
    if tracking_target.isTargetStatic == 1:
        tracking_target.S_hat[0] = tracking_target.S_apriori_saved[0]
        tracking_target.S_hat[1] = tracking_target.S_apriori_saved[1]

    if isinstance(tracking_target, TempTargetType):
        tracking_target.remove = True
    # 波门限制协方差矩阵
    # 它衡量的是 传感器的测量值 与基于 上一帧预测值 得到的 预期测量值 之间的 误差统计不确定性
    # 当前帧的检测点位 Y = HX + e_noise
    # X：目标的几何中心，e_noise：测量噪声
    # Y-HX_pre = H（X-X_pre）+ e_noise
    # cov(Y-HX_pre)=Hcov（X-X_pre）H.T+cov(e_noise)
    tracking_target.gC = HPH + Rm
    tracking_target.gC_inv = np.linalg.inv(tracking_target.gC)
    tracking_target.gC_det = float(np.linalg.det(tracking_target.gC))


    tracking_target.gC_points = tracking_target.gD + HPH + Rm
    tracking_target.gC_points_inv = np.linalg.inv(tracking_target.gC_points)
    tracking_target.gC_points_det = float(np.linalg.det(tracking_target.gC_points))

    # 计算目标后验位置尺寸
    x_min, x_max = tracking_target.S_hat[0] - tracking_target.size[0] / 2.0, tracking_target.S_hat[0] + tracking_target.size[0] / 2.0
    y_min, y_max = tracking_target.S_hat[1] - tracking_target.size[1] / 2.0, tracking_target.S_hat[1] + tracking_target.size[1] / 2.0
    z_min, z_max = tracking_target.S_hat[2] - tracking_target.size[2] / 2.0, tracking_target.S_hat[2] + tracking_target.size[2] / 2.0
    tracking_target.bounding_box = np.array([x_min, y_min, z_min, x_max, y_max, z_max], dtype=np.float32)

    track_event(
        tracking_target,
        tracking_alg_param,
        points_num=0
    )

    pass

def target_update(
    tracking_info: TrackingInfo,
    det_rsts: List[DetectionTarget],
    tracking_alg_param: TrackingAlgParam,
    kalman_stats: KalmanStats,
):
    # 更新临时对象的 卡尔曼状态
    for temp_target in tracking_info.temp_targets:
        is_associate = False
        for det_rst in det_rsts:
            if det_rst.best_id == temp_target.uid:
                is_associate = True
                kalman_update_active(
                    tracking_target=temp_target,
                    det_rst=det_rst,
                    tracking_alg_param=tracking_alg_param,
                    kalman_stats=kalman_stats
                )
                break
        if not is_associate:
            kalman_update_passive(
                tracking_target=temp_target,
                tracking_alg_param=tracking_alg_param,
                kalman_stats=kalman_stats
            )
    # 更新正式跟踪对象的 卡尔曼状态
    for tracking_target in tracking_info.tracking_targets:
        is_associate = False
        for det_rst in det_rsts:
            if det_rst.best_id == tracking_target.uid:
                is_associate = True
                kalman_update_active(
                    tracking_target=tracking_target,
                    det_rst=det_rst,
                    tracking_alg_param=tracking_alg_param,
                    kalman_stats=kalman_stats
                )
                break
        if not is_associate:
            kalman_update_passive(
                tracking_target=tracking_target,
                tracking_alg_param=tracking_alg_param,
                kalman_stats=kalman_stats
            )

        if tracking_target.state.value == TrackState.FREE.value:
            tracking_target.delete_flag = 1

    tracking_info.clean_tracking_target()
    pass

def kalman_predict(
    tracking_target: Optional[Union[TargetType, TempTargetType]],
    kalman_stats: KalmanStats
):
    tracking_target.S_apriori_hat = np.dot(kalman_stats.F, tracking_target.S_hat).astype(np.float32)
    temp1 = np.dot(kalman_stats.F, np.dot(tracking_target.P_hat, kalman_stats.F.T)) + kalman_stats.Q
    tracking_target.P_apriori_hat = (0.5 * (temp1 + temp1.T)).astype(np.float32)

def target_predict(
    tracking_info: TrackingInfo,
    kalman_stats: KalmanStats
):
    for temp_target in tracking_info.temp_targets:
        kalman_predict(temp_target,kalman_stats)
    for tracking_target in tracking_info.tracking_targets:
        kalman_predict(tracking_target,kalman_stats)

def temp_target_update(
    tracking_info: TrackingInfo,
    tracking_alg_param: TrackingAlgParam,
):
    for temp_target in tracking_info.temp_targets:
        if temp_target.hit_count >= tracking_alg_param.temp2formatThre:
            tracking_info.add_tracking_target(
                TargetType(
                    uid=temp_target.uid,
                    point_mask=temp_target.point_mask,
                    point_indices=temp_target.point_indices,
                    det_associate_point_mask=temp_target.det_associate_point_mask,
                    apriori_associate_point_mask=temp_target.apriori_associate_point_mask,
                    size_associate_point_mask=temp_target.size_associate_point_mask,
                    bounding_box=temp_target.bounding_box,
                    size=temp_target.size,
                    gate_limits=temp_target.gate_limits,
                    euclidean_distance_threshold=temp_target.euclidean_distance_threshold,
                    heartBeatCount=temp_target.heartBeatCount,
                    allocationTime=temp_target.allocationTime,
                    Center=temp_target.Center,
                    S_hat=temp_target.S_hat,
                    S_apriori_hat=temp_target.S_apriori_hat,
                    P_hat=temp_target.P_hat,
                    P_apriori_hat=temp_target.P_apriori_hat,
                    S_apriori_saved=temp_target.S_apriori_saved,
                    gC_det=temp_target.gC_det,
                    gC_inv=temp_target.gC_inv,
                    gD=temp_target.gD,
                    gC=temp_target.gC,
                    gC_points_det=temp_target.gC_points_det,
                    gC_points_inv=temp_target.gC_points_inv,
                    gC_points=temp_target.gC_points,
                    estSpread=temp_target.estSpread,
                    sFactor=temp_target.sFactor,
                    detect2freeCount=temp_target.detect2freeCount,
                    detect2activeCount=temp_target.detect2activeCount,
                    active2freeCount=temp_target.active2freeCount,
                    estNumOfPoints=temp_target.estNumOfPoints,
                    state=temp_target.state,
                    delete_flag=temp_target.delete_flag,
                    isTargetStatic=temp_target.isTargetStatic,
                    useStaticAssist=temp_target.useStaticAssist
                )
            )
            temp_target.is2target = True
            temp_target.remove = True

    tracking_info.clean_temp_targets()

# 