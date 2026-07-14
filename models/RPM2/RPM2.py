import torch 
from torch import nn

from models.RPM2.HRNet import HRNet18
from models.RPM2.ROIAlign import RoIAlign_Fun

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

class MFN(nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size
        self.boxes_embedding = nn.Linear(4, crop_size[0]*crop_size[1])
        self.conv1d = nn.Conv1d(crop_size[0]*crop_size[1], crop_size[0]*crop_size[1] // 2)
    def forward(self, keypoint_heatmap, boxes):
        feature_crops = RoIAlign_Fun(keypoint_heatmap, boxes, self.crop_size) + boxes_embedding(boxes)[:, :, None, ...]
        feature_out = self.conv1d(feature_crops)

class RPM2(nn.Module):
    def __init__(self, 
                inchannels, crop_size,
                stage2_num_modules, stage2_num_branches, stage2_block, stage2_num_blocks, stage2_num_channels, stage2_fuse_method,
                stage3_num_modules, stage3_num_branches, stage3_block, stage3_num_blocks, stage3_num_channels, stage3_fuse_method,
                stage4_num_modules, stage4_num_branches, stage4_block, stage4_num_blocks, stage4_num_channels, stage4_fuse_method,
                ):
        super().__init__()

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

    def forward(self, model_input):
        hor = model_input['hor']
        ver = model_input['ver']
        B, T, C, H, W = hor.shape
        hor = hor.reshape(B*T, C, H, W)
        ver = ver.reshape(B*T, C, H, W)
        center_heatmap_hor, center_offset_hor, center_box_hor, keypoint_heatmap_hor = self.feature_extractor_hor(hor)
        center_heatmap_ver, center_offset_ver, center_box_ver, keypoint_heatmap_ver = self.feature_extractor_ver(ver)

        center_heatmap_hor = center_heatmap_hor.reshape(B, T, *center_heatmap_hor.shape[1:])
        center_offset_hor = center_offset_hor.reshape(B, T, *center_offset_hor.shape[1:])
        center_box_hor = center_box_hor.reshape(B, T, *center_box_hor.shape[1:])
        keypoint_heatmap_hor = keypoint_heatmap_hor.reshape(B, T, *keypoint_heatmap_hor.shape[1:])
        center_heatmap_ver = center_heatmap_ver.reshape(B, T, *center_heatmap_ver.shape[1:])
        center_offset_ver = center_offset_ver.reshape(B, T, *center_offset_ver.shape[1:])
        center_box_ver = center_box_ver.reshape(B, T, *center_box_ver.shape[1:])
        keypoint_heatmap_ver = keypoint_heatmap_ver.reshape(B, T, *keypoint_heatmap_ver.shape[1:])

        return_dict = {
            'center_heatmap_hor': center_heatmap_hor, 'center_heatmap_ver': center_heatmap_ver,
            'center_offset_hor': center_offset_hor, 'center_offset_ver': center_offset_ver,
            'center_box_hor': center_box_hor, 'center_box_ver': center_box_ver,
            'keypoint_heatmap_hor': keypoint_heatmap_hor, 'keypoint_heatmap_ver': keypoint_heatmap_ver,
        }
        return return_dict


def loss_function(model_output, gt, gt_mask):
    from preprocess.gtprocess import get_gt_boxes
    center_heatmap = model_output['center_heatmap_hor']  # [B, T, 1, H, W]

    
    return None  

if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model('RPM2').to(device)
    x = {
        'hor': torch.zeros((1, 3, 3, 256, 256), device=device),
        'ver': torch.zeros((1, 3, 3, 256, 256), device=device),
    }
    profile_model("RPM2", model, x)
    
