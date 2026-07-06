import torch 
from torch import nn

from models.RPM2.HRNet_helper import HRNet18

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



class RPM2(nn.Module):
    def __init__(self, 
                inchannels,
                stage2_num_modules, stage2_num_branches, stage2_block, stage2_num_blocks, stage2_num_channels, stage2_fuse_method,
                stage3_num_modules, stage3_num_branches, stage3_block, stage3_num_blocks, stage3_num_channels, stage3_fuse_method,
                stage4_num_modules, stage4_num_branches, stage4_block, stage4_num_blocks, stage4_num_channels, stage4_fuse_method,
                ):
        super().__init__()

        self.feature_extractor = Feature_Extractor(
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
        x = model_input['input']
        center_heatmap, center_offset, center_box, keypoint_heatmap = self.feature_extractor(x)
        return x

if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model('RPM2').to(device)
    x = {
        'input': torch.zeros((1, 3, 256, 256), device=device),
    }
    profile_model("RPM2", model, x)
