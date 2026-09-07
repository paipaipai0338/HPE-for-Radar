from src.models import(
    VFPS, VFES, HFES, TFS, SES, 
    SDS, MDS, LOSS_LIB, OPTIM,
    LR_SCHEDULER
)
from data import DATASETS, DATA_AUGMENTATIONS
from typing import Optional
import numpy as np
import random

# ==================== 构建体素特征预编码器 ====================
def build_voxel_feature_precoder(cfg):
    voxel_feature_precoder_type = cfg.type
    voxel_feature_precoder = VFPS.get(voxel_feature_precoder_type)
    
    return voxel_feature_precoder(cfg)


# ==================== 构建体素特征编码器 ====================
def build_voxel_feature_encoder(cfg):
    voxel_feature_encoder_type = cfg.type
    voxel_feature_encoder = VFES.get(voxel_feature_encoder_type)
    
    return voxel_feature_encoder(cfg)


# ==================== 构建时域特征编码器 ====================
def build_histInfo_feature_encoder(cfg):
    histInfo_feature_encoder_type = cfg.type
    histInfo_feature_encoder = HFES.get(histInfo_feature_encoder_type)
    
    return histInfo_feature_encoder(cfg)

# ==================== 构建时域特征融合编码器 ====================
def build_temp_feature_fusion_module(cfg):
    temp_feature_fusion_module_type = cfg.type
    temp_feature_fusion_module = TFS.get(temp_feature_fusion_module_type)
    
    return temp_feature_fusion_module(cfg)

# ==================== 构建状态编码器 ====================
def build_state_encoder(cfg):
    state_encoder_type = cfg.type
    state_encoder = SES.get(state_encoder_type)
    
    return state_encoder(cfg)


# ==================== 构建状态解码器 ====================
def build_state_decoder(cfg):
    state_decoder_type = cfg.type
    state_decoder = SDS.get(state_decoder_type)
    
    return state_decoder(cfg)


# ==================== 构建掩码解码器 ====================
def build_mask_decoder(cfg):
    mask_decoder_type = cfg.type
    mask_decoder = MDS.get(mask_decoder_type)
    
    return mask_decoder(cfg)


# ==================== 构建损失函数 ====================
def build_loss(cfg,total_steps):
    loss_type=cfg["type"]
    loss=LOSS_LIB.get(loss_type)
    params=cfg.get("params",None)
    if params is not None:
        max_steps_ratio=params.get("max_steps_ratio",None)
        if max_steps_ratio is not None:
            params["max_steps"]=min(int(total_steps*max_steps_ratio),total_steps)
        else:
            params["max_steps"]=total_steps
        return loss(**params)
    return loss


# ==================== 构建优化器 ====================
def build_optim(cfg,model):
    optim_type=cfg["type"]
    optim=OPTIM.get(optim_type)
    params=cfg.get("params",None)
    if params is not None:
        params["model"]=model
        return optim(**params)
    return optim()


# ==================== 构建学习率调度器 ====================
def build_lr_scheduler(cfg, optimizer, total_steps):
    from torch.optim.lr_scheduler import SequentialLR
    
    scheduler_types = cfg.get("type", [])
    params_list = cfg.get("params", [])
    
    schedulers = []
    milestones = []
    end_steps = 0
    
    for i, scheduler_type in enumerate(scheduler_types):
        # 1. 防御性获取与深拷贝 (修复字典就地修改污染配置的隐患)
        raw_params = params_list[i] if params_list and i < len(params_list) else {}
        cur_params = raw_params.copy() 
        
        # 2. 从注册表获取你封装好的调度器构建函数
        scheduler_builder = LR_SCHEDULER.get(scheduler_type)
        if scheduler_builder is None:
            raise KeyError(f"Scheduler '{scheduler_type}' not found in registry.")
            
        # 3. 处理动态步数计算
        steps_ratio = cur_params.pop("steps_ratio", None)
        if steps_ratio is not None:
            target_steps = int(total_steps * steps_ratio)
            cur_steps = min(target_steps - end_steps, total_steps - end_steps)
            cur_params["steps"] = max(1, cur_steps)  # 兜底：防止计算出 0 或负数导致报错
            
        if "steps" not in cur_params:
            raise ValueError(f"Scheduler {scheduler_type} requires 'steps' or 'steps_ratio'.")
            
        step_val = cur_params["steps"]
        end_steps += step_val
        milestones.append(end_steps)
        
        # 4. 实例化并追加 (核心修复：补全缺失的 optimizer 注入)
        # 此时 cur_params 中已经包含了统一定义的 'steps' 和其他特有参数
        schedulers.append(scheduler_builder(optimizer=optimizer, **cur_params))
        
    # 5. 生成串联调度器
    scheduler = SequentialLR(
        optimizer, 
        schedulers=schedulers, 
        milestones=milestones[:-1]
    )
    
    return scheduler



# ==================== 构建数据集 ====================
def build_dataset(cfg):
    if isinstance(cfg, dict):
        cfg = [cfg]
    datasets = []
    for ds_cfg in cfg:
        ds_type = ds_cfg["type"]
        ds_params = ds_cfg.get("params", {})
        dataset=DATASETS.get(ds_type)(**ds_params)
        datasets.append(dataset)
    return datasets


# ==================== 构建数据增强 ====================
class RandomApplyCompose:
    """
    随机子集数据增强组合器
    每次调用时，从 transforms 列表中随机选取 0 ~ max_num_transforms 个方法执行
    """

    def __init__(
        self,
        transforms: list,
        max_num_transforms: Optional[int] = None,
        shuffle_order: bool = False,
    ):
        """
        :param transforms: 数据增强算子列表
        :param max_num_transforms: 每次最多选取的增强方法数 X。默认为 None (即最大长度 len(transforms))
        :param shuffle_order: 选出增强方法后，是否随机打乱执行顺序 (默认 False，即保持原有配置列表的相对先后顺序)
        """
        self.transforms = transforms
        self.total_num = len(transforms)
        # 若未指定或指定值超出总数，则上限为总长度
        self.max_num = (
            self.total_num
            if (max_num_transforms is None or max_num_transforms > self.total_num)
            else max(0, max_num_transforms)
        )
        self.shuffle_order = shuffle_order

    def __call__(
        self,
        raw_point_cloud: np.ndarray,
        poses_radar: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if self.total_num == 0:
            return raw_point_cloud, poses_radar

        # 1. 随机确定本次调用的增强方法数量 k: [0, self.max_num]
        k = random.randint(0, self.max_num)
        if k == 0:
            return raw_point_cloud, poses_radar

        # 2. 随机无放回抽取 k 个增强算子
        if self.shuffle_order:
            # 随机选取并打乱顺序
            selected_transforms = random.sample(self.transforms, k)
        else:
            # 随机选取但保持 transforms 列表中的相对先后顺序
            selected_indices = sorted(random.sample(range(self.total_num), k))
            selected_transforms = [self.transforms[i] for i in selected_indices]

        # 3. 流式执行选中的增强算子
        for t in selected_transforms:
            raw_point_cloud, poses_radar = t(raw_point_cloud, poses_radar)

        return raw_point_cloud, poses_radar

    def __repr__(self):
        format_string = f"{self.__class__.__name__}(max_num={self.max_num}/{self.total_num}):"
        for t in self.transforms:
            format_string += f"\n    {t.__class__.__name__}"
        return format_string
    
def build_data_augmentation(cfg):
    pileline = cfg["pipeline"]
    if isinstance(pileline, dict):
        pileline = [pileline]
    augmentations = []
    for aug_cfg in pileline:
        aug_type = aug_cfg["type"]
        aug_params = aug_cfg.get("params", {})
        
        # 从注册表获取对应的增强类并实例化
        aug_cls = DATA_AUGMENTATIONS.get(aug_type)
        augmentation = aug_cls(**aug_params)
        augmentations.append(augmentation)

    return RandomApplyCompose(
        augmentations,
        max_num_transforms = cfg["max_num_transforms"],
        shuffle_order= cfg["shuffle_order"]
    )

# 