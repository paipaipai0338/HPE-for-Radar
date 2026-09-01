import torch 
from torch import nn

from models.RPM2.HRNet import HRNet18
from models.RPM2.ROIAlign import RoIAlign_Fun
from models.RPM2.transformer_helper import Attention, Transformer

class Feature_Extractor(nn.Module):
    def __init__(self, 
                inchannels,
                stage2_num_modules, stage2_num_branches, stage2_block, stage2_num_blocks, stage2_num_channels, stage2_fuse_method,
                stage3_num_modules, stage3_num_branches, stage3_block, stage3_num_blocks, stage3_num_channels, stage3_fuse_method,
                stage4_num_modules, stage4_num_branches, stage4_block, stage4_num_blocks, stage4_num_channels, stage4_fuse_method,
                ):
        super().__init__()
        self.backbone = HRNet18(
            inchannels=inchannels,
            stage2_num_modules=stage2_num_modules,
            stage2_num_branches=stage2_num_branches,
            stage2_block=stage2_block,
            stage2_num_blocks=stage2_num_blocks,
            stage2_num_channels=stage2_num_channels,
            stage2_fuse_method=stage2_fuse_method,
            stage3_num_modules=stage3_num_modules,
            stage3_num_branches=stage3_num_branches,
            stage3_block=stage3_block,
            stage3_num_blocks=stage3_num_blocks,
            stage3_num_channels=stage3_num_channels,
            stage3_fuse_method=stage3_fuse_method,
            stage4_num_modules=stage4_num_modules,
            stage4_num_branches=stage4_num_branches,
            stage4_block=stage4_block,
            stage4_num_blocks=stage4_num_blocks,
            stage4_num_channels=stage4_num_channels,
            stage4_fuse_method=stage4_fuse_method,
            )
        backbone_out_channels = stage4_num_channels[0]
        self.center_heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.center_offset_head = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1),
        )
        self.center_box_head = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1),
        )
        self.keypoint_heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
        )
    def forward(self, x):
        x = self.backbone(x)
        center_heatmap = self.center_heatmap_head(x)
        center_offset = self.center_offset_head(x)
        center_box = self.center_box_head(x)
        keypoint_heatmap = self.keypoint_heatmap_head(x)
        
        return center_heatmap, center_offset, center_box, keypoint_heatmap

class Multiview_Fusion_Network(nn.Module):
    def __init__(self, crop_size, num_joints):
        super().__init__()
        self.crop_size = crop_size
        self.num_joints = num_joints
        self.boxes_embedding = nn.Linear(4, 128)
        self.conv1d = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.conv2d_group = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=num_joints, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, keypoint_heatmap, boxes):
        feature_crops = RoIAlign_Fun(keypoint_heatmap, boxes, self.crop_size).reshape(*keypoint_heatmap.shape[:3], self.crop_size[0]*self.crop_size[1]) + self.boxes_embedding(boxes)[:, :, :, None]
        b, t, c, d = feature_crops.shape
        feature_out = self.conv1d(feature_crops.reshape(b*t, c, d)).reshape(b*t, 128, self.crop_size[0], self.crop_size[1])
        feature_out = self.conv2d_group(feature_out).reshape(b, t, self.num_joints, -1)
        return feature_out

class Spatial_Attention_Module(nn.Module):
    def __init__(self, crop_size, num_joints, drop_rate):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(num_joints, 2*crop_size[0] * crop_size[1]//16))    # J, 2*C//16
        self.drop_rate = drop_rate
        self.attention = Transformer(dim=2*crop_size[0]*crop_size[1]//16, depth=4, heads=4, dim_head=64, mlp_dim=512)
        
    def forward(self, feature_fusion):
        # feature_fusion: B, T, K, J, 2*C//16
        b, t, k, j, c = feature_fusion.shape
        feature_fusion = feature_fusion.reshape(b, t*k, j, c)
        if self.training:
            mask = torch.rand(b, t*k, j, 1, device=feature_fusion.device) > self.drop_rate
            feature_fusion = feature_fusion * mask.float()  # B, T*K, J, 2*C//16
        feature_fusion = feature_fusion + self.position_embedding[None, None, :, :]
        feature_fusion = feature_fusion.reshape(b*t*k, j, c)
        feature_fusion = self.attention(feature_fusion)
        feature_fusion = feature_fusion.reshape(b, t, k, j, c)
        return feature_fusion

class Temporal_Attention_Module(nn.Module):
    def __init__(self, crop_size, time_window, num_joints, drop_rate):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(time_window, num_joints*2*crop_size[0]*crop_size[1]//16))    # T, 2*C*J//16
        self.drop_rate = drop_rate
        self.attention = Transformer(dim=2*num_joints*crop_size[0]*crop_size[1]//16, depth=4, heads=4, dim_head=64, mlp_dim=512)
        
    def forward(self, feature_fusion):
        # feature_fusion: B, T, K, J, 2*C
        b, t, k, j, c = feature_fusion.shape
        feature_fusion = feature_fusion.permute(0, 2, 1, 3, 4)
        feature_fusion = feature_fusion.reshape(b, k, t, j*c)
        if self.training:
            mask = torch.rand(b, k, t, 1, device=feature_fusion.device) > self.drop_rate
            feature_fusion = feature_fusion * mask.float()  # B, K, T, 2*C*J//16
        feature_fusion = feature_fusion + self.position_embedding[None, None, :, :]
        feature_fusion = feature_fusion.reshape(b*k, t, c*j)
        feature_fusion = self.attention(feature_fusion)
        feature_fusion = feature_fusion.reshape(b, k, t, j, c).permute(0, 2, 1, 3, 4)                       # B, T, K, J, 2*C
        return feature_fusion

class RPM2(nn.Module):
    def __init__(self, 
                inchannels, crop_size, num_joints, drop_rate, time_window, map_size, xyz_limits,
                stage2_num_modules, stage2_num_branches, stage2_block, stage2_num_blocks, stage2_num_channels, stage2_fuse_method,
                stage3_num_modules, stage3_num_branches, stage3_block, stage3_num_blocks, stage3_num_channels, stage3_fuse_method,
                stage4_num_modules, stage4_num_branches, stage4_block, stage4_num_blocks, stage4_num_channels, stage4_fuse_method,
                ):
        super().__init__()
        self.map_size = map_size
        self.xyz_limits = xyz_limits

        self.feature_extractor_hor = Feature_Extractor(
                inchannels=inchannels,
                stage2_num_modules=stage2_num_modules,
                stage2_num_branches=stage2_num_branches,
                stage2_block=stage2_block,
                stage2_num_blocks=stage2_num_blocks,
                stage2_num_channels=stage2_num_channels,
                stage2_fuse_method=stage2_fuse_method,
                stage3_num_modules=stage3_num_modules,
                stage3_num_branches=stage3_num_branches,
                stage3_block=stage3_block,
                stage3_num_blocks=stage3_num_blocks,
                stage3_num_channels=stage3_num_channels,
                stage3_fuse_method=stage3_fuse_method,
                stage4_num_modules=stage4_num_modules,
                stage4_num_branches=stage4_num_branches,
                stage4_block=stage4_block,
                stage4_num_blocks=stage4_num_blocks,
                stage4_num_channels=stage4_num_channels,
                stage4_fuse_method=stage4_fuse_method,
                )
        self.feature_extractor_ver = Feature_Extractor(
                inchannels=inchannels,
                stage2_num_modules=stage2_num_modules,
                stage2_num_branches=stage2_num_branches,
                stage2_block=stage2_block,
                stage2_num_blocks=stage2_num_blocks,
                stage2_num_channels=stage2_num_channels,
                stage2_fuse_method=stage2_fuse_method,
                stage3_num_modules=stage3_num_modules,
                stage3_num_branches=stage3_num_branches,
                stage3_block=stage3_block,
                stage3_num_blocks=stage3_num_blocks,
                stage3_num_channels=stage3_num_channels,
                stage3_fuse_method=stage3_fuse_method,
                stage4_num_modules=stage4_num_modules,
                stage4_num_branches=stage4_num_branches,
                stage4_block=stage4_block,
                stage4_num_blocks=stage4_num_blocks,
                stage4_num_channels=stage4_num_channels,
                stage4_fuse_method=stage4_fuse_method,
                )

        self.mfn_hor = Multiview_Fusion_Network(crop_size=crop_size, num_joints=num_joints)
        self.mfn_ver = Multiview_Fusion_Network(crop_size=crop_size, num_joints=num_joints)
        self.sam = Spatial_Attention_Module(crop_size=crop_size, num_joints=num_joints, drop_rate=drop_rate)
        self.tam = Temporal_Attention_Module(crop_size=crop_size, time_window=time_window, num_joints=num_joints, drop_rate=drop_rate)
        self.feature_to_joint = nn.Linear(2*crop_size[0]*crop_size[1]//16, 3)

    def forward(self, model_input):
        hor = model_input['input'][:, :, 0:1, :, :]
        ver = model_input['input'][:, :, 1:2, :, :]
        gt = model_input['gt']
        B, T, C, H, W = hor.shape
        hor = hor.reshape(B*T, C, H, W)
        ver = ver.reshape(B*T, C, H, W)

        # 前端特征提取+目标检测定位
        center_heatmap_hor, center_offset_hor, center_box_hor, keypoint_heatmap_hor = self.feature_extractor_hor(hor)
        center_heatmap_ver, center_offset_ver, center_box_ver, keypoint_heatmap_ver = self.feature_extractor_ver(ver)

        center_heatmap_hor = center_heatmap_hor.reshape(B, T, *center_heatmap_hor.shape[1:])            # B, T, 1, 64, 64
        center_offset_hor = center_offset_hor.reshape(B, T, *center_offset_hor.shape[1:])               # B, T, 2, 64, 64
        center_box_hor = center_box_hor.reshape(B, T, *center_box_hor.shape[1:])                        # B, T, 4, 64, 64
        keypoint_heatmap_hor = keypoint_heatmap_hor.reshape(B, T, *keypoint_heatmap_hor.shape[1:])      # B, T, 128, 64, 64
        center_heatmap_ver = center_heatmap_ver.reshape(B, T, *center_heatmap_ver.shape[1:])
        center_offset_ver = center_offset_ver.reshape(B, T, *center_offset_ver.shape[1:])
        center_box_ver = center_box_ver.reshape(B, T, *center_box_ver.shape[1:])
        keypoint_heatmap_ver = keypoint_heatmap_ver.reshape(B, T, *keypoint_heatmap_ver.shape[1:])

        center_heatmap_pre = torch.stack([center_heatmap_hor, center_heatmap_ver], dim=2)                       # [B, T, 2, 1, 64, 64]
        center_offset_pre = torch.stack([center_offset_hor, center_offset_ver], dim=2)                          # [B, T, 2, 2, 64, 64]
        center_box_pre = torch.stack([center_box_hor, center_box_ver], dim=2)                                   # [B, T, 2, 4, 64, 64]

        # 依据检测结果估计人数 / 根据真值进行
        bbox = gt['bbox']               # B, T, K, 6
        bbox_valid = gt['mask']         # B, T, K
        K = bbox.shape[2]
        feat_H, feat_W = center_heatmap_hor.shape[-2:]
        feature_fusion_list = []
        center_heatmap, center_indices, center_offsets_sparse, center_boxes_sparse, inside, roi_boxes = bboxes_encoder(bbox, bbox_valid, feat_H, feat_W, self.xyz_limits)

        pose_valid = inside[:, :, 0, :] & inside[:, :, 1, :] & bbox_valid  # [B,T,K]

        for k in range(K):
            feature_cropped_hor = self.mfn_hor(keypoint_heatmap_hor, roi_boxes[:, :, 0, k, :])
            feature_cropped_ver = self.mfn_ver(keypoint_heatmap_ver, roi_boxes[:, :, 1, k, :])
            feature_fusion = torch.cat([feature_cropped_hor, feature_cropped_ver], dim=-1)

            valid_k = pose_valid[:, :, k]  # [B,T]
            feature_fusion = feature_fusion.masked_fill(~valid_k[:, :, None, None], 0.0)

            feature_fusion_list.append(feature_fusion)
        feature_fusion = torch.stack(feature_fusion_list, dim=2)        # B, T, K, J, 2*C//16
        feature_fusion = self.sam(feature_fusion)                       # B, T, K, J, 2*C//16
        feature_fusion = self.tam(feature_fusion)                       # B, T, K, J, 2*C//16

        pose = self.feature_to_joint(feature_fusion)                       # B, T, K, J, 3

        return_dict = {
            'center_heatmap': center_heatmap_pre,
            'center_offset_pre': center_offset_pre,
            'center_box_pre': center_box_pre,
            'pose': pose,
            "pose_valid": pose_valid,
            "target_inside": inside,
            "target_center_heatmap": center_heatmap,
            "target_center_indices": center_indices,
            "target_center_offsets_sparse": center_offsets_sparse,
            "target_center_boxes_sparse": center_boxes_sparse,
        }
        return return_dict

# 依据预测的heatmap和offset解码出bbox，用于推理
def bboxes_decoder(center_heatmap, center_offset, center_box, threshold=0.5):
    # 后续可基于需求新增
    pass

# 依据gt的bbox生成heatmap和offset，用于训练
def bboxes_encoder(bboxes, bboxes_valid, H, W,  xyz_limits, sigma=1):
    # bboxes: [B, T, K, 6] (x1, y1, z1, x2, y2, z2)
    # H, W: int
    # xyz_limits: List[Tuple[float, float]]
    def _coord_to_index(value, coord_min, coord_max, size):
        return (value - coord_min) / (coord_max - coord_min) * (size - 1)

    B, T, K, _ = bboxes.shape
    device = bboxes.device
    dtype = bboxes.dtype

    valid = bboxes_valid.to(device=device, dtype=torch.bool)

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = xyz_limits

    # ------------------------------------------------------------
    # 1. 
    # ------------------------------------------------------------
    x1, y1, z1 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2]
    x2, y2, z2 = bboxes[..., 3], bboxes[..., 4], bboxes[..., 5]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    cz = 0.5 * (z1 + z2)

    # ------------------------------------------------------------
    # 2. 物理坐标 -> 连续map索引
    #       H-axis = X
    #       W-axis = Y for horizontal
    #       W-axis = Z for vertical
    # ------------------------------------------------------------
    x1_idx = _coord_to_index(x1, x_min, x_max, H)
    x2_idx = _coord_to_index(x2, x_min, x_max, H)
    cx_idx = _coord_to_index(cx, x_min, x_max, H)

    y1_idx = _coord_to_index(y1, y_min, y_max, W)
    y2_idx = _coord_to_index(y2, y_min, y_max, W)
    cy_idx = _coord_to_index(cy, y_min, y_max, W)

    z1_idx = _coord_to_index(z1, z_min, z_max, W)
    z2_idx = _coord_to_index(z2, z_min, z_max, W)
    cz_idx = _coord_to_index(cz, z_min, z_max, W)

    x_low, x_high = torch.minimum(x1_idx, x2_idx), torch.maximum(x1_idx, x2_idx)
    y_low, y_high = torch.minimum(y1_idx, y2_idx), torch.maximum(y1_idx, y2_idx)
    z_low, z_high = torch.minimum(z1_idx, z2_idx), torch.maximum(z1_idx, z2_idx)

    # ------------------------------------------------------------
    # 3. 检查包围框是否合法：传入valid true，且中心在范围内；不要求包围框全部在区域内
    # ------------------------------------------------------------
    center_inside_hor = (
        valid
        & (cx_idx >= 0) & (cx_idx <= H - 1)
        & (cy_idx >= 0) & (cy_idx <= W - 1)
    )

    center_inside_ver = (
        valid
        & (cx_idx >= 0) & (cx_idx <= H - 1)
        & (cz_idx >= 0) & (cz_idx <= W - 1)
    )

    inside_hor = center_inside_hor
    inside_ver = center_inside_ver

    inside = torch.stack([inside_hor, inside_ver], dim=2)  # [B,T,2,K]

    # ------------------------------------------------------------
    # 4. 裁剪包围框，否则ROIAligned可能失败
    # ------------------------------------------------------------
    x_low_t = x_low.clamp(0, H - 1)
    x_high_t = x_high.clamp(0, H - 1)

    y_low_t = y_low.clamp(0, W - 1)
    y_high_t = y_high.clamp(0, W - 1)

    z_low_t = z_low.clamp(0, W - 1)
    z_high_t = z_high.clamp(0, W - 1)

    # ------------------------------------------------------------
    # 5. 获取整数索引以及小数偏移
    # ------------------------------------------------------------
    cx_int = cx_idx.floor().clamp(0, H - 1).long()
    cy_int = cy_idx.floor().clamp(0, W - 1).long()
    cz_int = cz_idx.floor().clamp(0, W - 1).long()

    dx = cx_idx - cx_int.to(dtype)
    dy = cy_idx - cy_int.to(dtype)
    dz = cz_idx - cz_int.to(dtype)

    center_indices_hor = torch.stack([cx_int, cy_int], dim=-1)                              # [B,T,K,2], [x_idx, y_idx]
    center_indices_ver = torch.stack([cx_int, cz_int], dim=-1)                              # [B,T,K,2], [x_idx, z_idx]
    center_indices = torch.stack([center_indices_hor, center_indices_ver], dim=2)           # [B,T,2,K,2]

    center_offsets_hor = torch.stack([dx, dy], dim=-1)                                      # [B,T,K,2], [dx, dy]
    center_offsets_ver = torch.stack([dx, dz], dim=-1)                                      # [B,T,K,2], [dx, dz]
    center_offsets_sparse = torch.stack([center_offsets_hor, center_offsets_ver], dim=2)    # [B,T,2,K,2]
    center_offsets_sparse = center_offsets_sparse.masked_fill(~inside[..., None], 0.0)

    # ------------------------------------------------------------
    # 6. Box targets in heatmap-index coordinates
    #       horizontal: [cx-x1, cy-y1, x2-cx, y2-cy]
    #       vertical:   [cx-x1, cz-z1, x2-cx, z2-cz]
    # ------------------------------------------------------------
    box_hor = torch.stack(
        [
            cx_idx - x_low_t,
            cy_idx - y_low_t,
            x_high_t - cx_idx,
            y_high_t - cy_idx,
        ],
        dim=-1,
    )  # [B,T,K,4]

    box_ver = torch.stack(
        [
            cx_idx - x_low_t,
            cz_idx - z_low_t,
            x_high_t - cx_idx,
            z_high_t - cz_idx,
        ],
        dim=-1,
    )  # [B,T,K,4]

    center_boxes_sparse = torch.stack([box_hor, box_ver], dim=2)  # [B,T,2,K,4]
    center_boxes_sparse = center_boxes_sparse.masked_fill(~inside[..., None], 0.0)

    # ------------------------------------------------------------
    # 7. Build dense heatmap target
    #    Direct Gaussian generation is cleaner than scatter + conv2d.
    # ------------------------------------------------------------
    h_axis = torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, 1, H, 1)
    w_axis = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, 1, 1, W)

    center_h = torch.stack([cx_int, cx_int], dim=2).to(dtype)  # [B,T,2,K]
    center_w = torch.stack([cy_int, cz_int], dim=2).to(dtype)  # [B,T,2,K]

    center_h = center_h.view(B, T, 2, K, 1, 1)
    center_w = center_w.view(B, T, 2, K, 1, 1)

    dist_sq = (h_axis - center_h).square() + (w_axis - center_w).square()

    heatmaps_per_box = torch.exp(-dist_sq / (sigma ** 2))
    heatmaps_per_box = heatmaps_per_box * inside[..., None, None].to(dtype)

    center_heatmap = heatmaps_per_box.sum(dim=3, keepdim=False)
    center_heatmap = center_heatmap.unsqueeze(3)  # [B,T,2,1,H,W]

    # ------------------------------------------------------------
    # 8. ROIAlign boxes
    #    RoIAlign_Fun expects [col1, row1, col2, row2].
    #
    #    Your map convention:
    #       horizontal map [H,W] = [X,Y]
    #       vertical map   [H,W] = [X,Z]
    #
    #    Therefore:
    #       horizontal ROI box = [y1, x1, y2, x2]
    #       vertical ROI box   = [z1, x1, z2, x2]
    # ------------------------------------------------------------
    roi_boxes_hor = torch.stack(
        [y_low_t, x_low_t, y_high_t, x_high_t],
        dim=-1,
    )  # [B,T,K,4], [col1,row1,col2,row2]

    roi_boxes_ver = torch.stack(
        [z_low_t, x_low_t, z_high_t, x_high_t],
        dim=-1,
    )  # [B,T,K,4], [col1,row1,col2,row2]

    roi_boxes = torch.stack([roi_boxes_hor, roi_boxes_ver], dim=2)  # [B,T,2,K,4]
    roi_boxes = roi_boxes.masked_fill(~inside[..., None], 0.0)
    roi_boxes[..., 2] = torch.maximum(roi_boxes[..., 2], roi_boxes[..., 0] + 1.0)
    roi_boxes[..., 3] = torch.maximum(roi_boxes[..., 3], roi_boxes[..., 1] + 1.0)

    return (
        center_heatmap,              # [B,T,2,1,H,W]
        center_indices,              # [B,T,2,K,2]
        center_offsets_sparse,       # [B,T,2,K,2]
        center_boxes_sparse,         # [B,T,2,K,4]
        inside,                      # [B,T,2,K]
        roi_boxes,                   # [B,T,2,K,4]
    )

if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    b, t, k, c, h, w = 64, 8, 4, 1, 256, 256

    device = set_device(1)
    model = build_model('RPM2').to(device)
    x = {
        'input': torch.zeros((b, t, 2 * c, h, w), device=device),
    }
    gt = {
        'mask': torch.ones((b, t, k), device=device, dtype=torch.bool),
        'bbox': torch.zeros((b, t, k, 6), device=device),
    }
    x['gt'] = gt
    profile_model("RPM2", model, x)
