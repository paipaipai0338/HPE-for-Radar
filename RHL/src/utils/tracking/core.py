import numpy as np
from dataclasses import dataclass,field
from enum import IntEnum

ID=1                    # 跟踪目标编号

temp_ID = 1                 # 临时目标编号

@dataclass
class TrackState(IntEnum):
    """目标跟踪状态 — TARGET_STATE"""
    DETECTION = 0x00
    ACTIVE = 0x01
    FREE = 0x02

@dataclass
class TargetType:
    """完整目标跟踪结构 — Target_Type (3D版本)"""
    uid: np.int16 = 0

    point_mask: np.ndarray = None               # 该目标在当前输入点云中的布尔掩码，形状: (N,), bool
    point_indices: np.ndarray = None            # 属于该目标的点在全局点云中的下标索引，形状: (M,), int64

    det_associate_point_mask: np.ndarray = None     # 模型检测出的关联点的布尔掩码，形状: (N,), bool
    apriori_associate_point_mask: np.ndarray = None     # 卡尔曼滤波的先验关联点的布尔掩码，形状: (N,), bool
    size_associate_point_mask: np.ndarray = None    # 后验位置估计结合尺寸信息得到的关联点的布尔掩码，形状: (N,), bool

    bounding_box: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))     # 3D包围框，[xyzxyz]
    size: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))     # 包围框尺寸，[sx,sy,sz]

    gate_limits: np.ndarray = None          # 自适应关联门限
    euclidean_distance_threshold: float = None      # 欧氏距离关联门限


    heartBeatCount: np.int32 = 0
    allocationTime: np.int32 = 0

    Center: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    correlationPoints: np.ndarray = field(default_factory=lambda: np.empty((0,6), dtype=np.float32)) 

    S_hat: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))            # 状态估计 【x,y,z,vx,vy,vz】
    S_apriori_hat: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))    # 状态预测  【x,y,z,vx,vy,vz】
    P_hat: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=np.float32))
    P_apriori_hat: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=np.float32))

    S_apriori_saved: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))

    gC_det: float = 0.0
    gC_inv: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=np.float32))
    gD: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=np.float32))
    gC: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=np.float32))

    gC_points_det: float = 0.0
    gC_points_inv: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=np.float32))
    gC_points: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=np.float32))

    estSpread: np.ndarray = field(default_factory=lambda: np.full(6, 0.2, dtype=np.float32))
    sFactor: float = 1.0

    detect2freeCount: np.int16 = 0      # 由'检测到'状态转换至'释放'状态的耐力值
    detect2activeCount: np.int16 = 0    # 由'检测到'状态转换至'激活'状态的耐力值
    active2freeCount: np.int16 = 0      # 由'激活'状态转换至'释放'状态的耐力值

    estNumOfPoints: float = 0.0
    state: TrackState = field(default_factory=lambda: TrackState.FREE)
    delete_flag: np.int8 = 0
    isTargetStatic: np.int8 = 0
    useStaticAssist: np.int8 = 0

    def __repr__(self):
        return (f"TargetType:\n"
            f"  uid: {self.uid}\n"
            f"  state: {self.state}\n"
            f"  estNumOfPoints: {int(self.estNumOfPoints)}\n"
            f"  heartBeatCount: {self.heartBeatCount}\n"
            f"  detect2freeCount: {self.detect2freeCount}\n"
            f"  detect2activeCount: {self.detect2activeCount}\n"
            f"  active2freeCount: {self.active2freeCount}\n")

@dataclass
class TempTargetType(TargetType):
    hit_count: int = 0
    is2target: bool = False     # 是否已成为正式 跟踪对象

    remove: bool = False        # 是否删除该临时对象
    
    establish_target_points_thresh: int = 1000        # 该临时目标建立的点数阈值

    def __repr__(self):
        return (f"TempTargetType:\n"
            f"  uid: {self.uid}\n"
            f"  hit_count: {self.hit_count}\n"
            f"  is2target: {self.is2target}\n"
            f"  remove:     {self.remove}\n")


@dataclass
class BoxType:
    """检测区域边界框 — Box_Type"""
    xfront: float = 6.0
    xback: float = 0.0
    yleft: float = 3.0
    yright: float = -3.0
    ztop: float = -2.0
    zbottom: float = 2.0

@dataclass
class CoreArea:
    """检测区域核心区域"""
    xfront: float = 5.0
    xback: float = 0.0
    yleft: float = 2.0
    yright: float = -2.0


@dataclass
class TrackingAlgParam:
    """跟踪参数集 — Parameter_Type"""

    gate_limits: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1000, 1000, 1000], dtype=np.float16))

    establish_target_points_thresh: int = 60        # 新目标建立所需最少点数
    pointsThre: np.int16 = 5       # detect→active 点数
    det2actThre: np.int16 = 7      # DETECTION→ACTIVE 帧数
    det2freeThre: np.int16 = 10    # DETECTION→FREE 帧数
    
    temp2formatThre: int = 5            # 临时对象 转换为 正式对象 所需至少连续命中次数

    static2freeThre: np.int16 = 50
    active2freeThre: np.int16 = 30
    active2outThre: np.int16 = 5

    minpts: np.int16 = 10           # 卡尔曼更新所需要最少的目标点数
    adjDistThre: float = 0.75        # 新建目标与已有目标邻接判断距离阈值（小于该阈值则不新建目标）

    gain: float = 15              # 马氏距离门限
    MinStaticVel: float = 0.15     # 静止速度阈值
    MinPointUpdateDispersion: float = 3.0
    spreadAlpha: float = 0.05
    spreadMin: np.ndarray = field(default_factory=lambda: np.full(6, 0.25, dtype=np.float32))

    box: BoxType = field(default_factory=BoxType)
    
    core_area: CoreArea = field(default_factory=CoreArea)

    sizeSmoothFactor: float = 0.2       # 最终 size = sizeSmoothFactor * 旧 size + (1 - sizeSmoothFactor) 新 size
    
    Rc_scale: float = 1.0   # 测量噪声协方差系数，小于 1.0 时，卡尔曼更新倾向于测量（模型预测），大于 1.0 时，卡尔曼更新倾向于卡尔曼预测的结果

@dataclass
class DetectionTarget:
    # 检测结果边界框
    bounding_box: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))     # 3D包围框，[xyzxyz]
    bbox_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))     # 包围框中心，[center_x, center_y, center_z]
    association_points: np.ndarray = None       # 检测结果包围点，形状:N,6；[x,y,z,vx,vy,vz]
    center_state: np.ndarray = None         # 关联点中心状态，形状:6；[mean_x,mean_y,mean_z,mean_vx,mean_vy,mean_vz]
    discard: bool = False           # 该目标该帧是否丢弃
    best_score: float = 100.0
    best_id: int = -1

    point_mask: np.ndarray = None               # 该目标在当前输入点云中的布尔掩码，形状: (N,), bool
    point_indices: np.ndarray = None            # 属于该目标的点在全局点云中的下标索引，形状: (M,), int64
    
    # debug
    mp_distance: dict = field(default_factory=dict)
    e_distance: dict = field(default_factory=dict)

    def __repr__(self):
        mp_dist_info=""
        for uid in self.mp_distance:
            mp_dist_info += f"uid:  {uid}\ndist:    {self.mp_distance[uid]:.2f}\n"
        e_dist_info=""
        for uid in self.e_distance:
            e_dist_info += f"uid:   {uid}\ndist:    {self.e_distance[uid]:.2f}\n"
        return (f"DetRes:\n"
            f"  best_id: {self.best_id}\n"
            f"  association_points_num: {len(self.association_points)}\n"
            f"  center_state:{np.round(self.center_state[:3], 2)}\n"
            f"  mp_distance: \n{mp_dist_info}"
            f"  e_distance: \n{e_dist_info}"
            f"  discard: {self.discard}\n")


class TrackingInfo:
    def __init__(self):
        self.tracking_targets=[]
        self.temp_targets=[]
        self.TargetOutNum=0         # 当前状态激活状态的目标数
    def add_tracking_target(self,tracking_target: TargetType):
        self.tracking_targets.append(tracking_target)

    def add_temp_target(self, temp_target: TempTargetType):
        self.temp_targets.append(temp_target)

    def clean_tracking_target(self):
        self.tracking_targets = [
            target for target in self.tracking_targets 
            if target.delete_flag != 1
        ]

    def clean_temp_targets(self):
        self.temp_targets = [
            target for target in self.temp_targets 
            if not target.remove
        ]


class KalmanStats:
    def __init__(self):
        DELTA_T = 0.2
        
        # 1. 状态转移矩阵 F (6x6)
        # 满足：x = x + vx*dt, y = y + vy*dt, z = z + vz*dt
        self.F = np.array([
            [1.0, 0.0, 0.0, DELTA_T, 0.0,     0.0    ],  # x
            [0.0, 1.0, 0.0, 0.0,     DELTA_T, 0.0    ],  # y
            [0.0, 0.0, 1.0, 0.0,     0.0,     DELTA_T],  # z
            [0.0, 0.0, 0.0, 1.0,     0.0,     0.0    ],  # vx
            [0.0, 0.0, 0.0, 0.0,     1.0,     0.0    ],  # vy
            [0.0, 0.0, 0.0, 0.0,     0.0,     1.0    ]   # vz
        ], dtype=np.float32)
        
        # 2. 过程噪声协方差矩阵 Q (6x6)
        # 延续你原本对位置噪声(0.05)和速度噪声(0.10)的设定，对应 [x, y, z, vx, vy, vz]
        self.Q = np.diag([0.0500, 0.0500, 0.0500, 0.1000, 0.1000, 0.1000]).astype(np.float32)
        
        # 3. 观测矩阵 H (6x6)
        # 延续原代码“全状态直接观测”的设定，调整为 6 维单位阵
        self.H = np.eye(6, dtype=np.float32)
# 