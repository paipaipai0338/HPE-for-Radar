import os
import torch
import torch.nn as nn
from typing import *

from models.VoxelNeXt.VoxelNeXt_helper import DynamicMeanVoxelizer, FixedPersonQueryHead, VoxelResBackBone8xVoxelNeXt



class VoxelNeXt(nn.Module):
    def __init__(
        self,
        num_point_features: int = 3,
        point_cloud_range: Sequence[float] = (0.0, -3.0, -2.0, 6.0, 3.0, 2.0),
        voxel_size: Sequence[float] = (0.025, 0.025, 0.1),
        append_time: bool = False,
        append_xyz: bool = False,
        backbone_channels: Sequence[int] = (16, 32, 64, 128, 128),
        spconv_kernel_sizes: Sequence[int] = (3, 3, 3, 3),
        backbone_out_channel: int = 128,
        head_shared_conv_channel: int = 128,
        query_num_heads: int = 4,
        query_num_layers: int = 2,
        query_ffn_dim: int = 256,
        query_dropout: float = 0.0,
        max_people: int = 100,
    ) -> None:
        """
        Args:
            num_point_features: Number of non-xyz channels in input C.
                For xyz-only input, set this to 0.
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
            voxel_size: [vx, vy, vz].
            append_time: Append the voxelizer's local frame index. Leave this
                disabled for frame-wise detection because each flattened
                detector sample contains exactly one frame.
            append_xyz: Append raw xyz as additional voxel features.
        """

        super().__init__()
        self.num_point_features = int(num_point_features)
        if len(backbone_channels) != 5:
            raise ValueError(
                "backbone_channels must contain 5 values, "
                f"got {backbone_channels}"
            )
        if len(spconv_kernel_sizes) != 4:
            raise ValueError(
                "spconv_kernel_sizes must contain 4 values, "
                f"got {spconv_kernel_sizes}"
            )
        if int(max_people) <= 0:
            raise ValueError(f"max_people must be positive, got {max_people}")

        self.voxelizer = DynamicMeanVoxelizer(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            append_time=append_time,
            append_xyz=append_xyz,
        )

        input_channels = max(self.num_point_features, 1) + self.voxelizer.num_extra_features
        grid_size = torch.tensor(tuple(self.voxelizer.grid_size), dtype=torch.long)
        self.backbone_3d = VoxelResBackBone8xVoxelNeXt(
            input_channels=input_channels,
            grid_size=grid_size,
            spconv_kernel_sizes=spconv_kernel_sizes,
            channels=backbone_channels,
            out_channel=backbone_out_channel,
        )
        self.query_head = FixedPersonQueryHead(
            input_channels=backbone_out_channel,
            hidden_dim=head_shared_conv_channel,
            num_heads=query_num_heads,
            num_layers=query_num_layers,
            ffn_dim=query_ffn_dim,
            dropout=query_dropout,
            max_people=max_people,
            point_cloud_range=point_cloud_range,
        )
        self.max_people = int(max_people)

    def forward(self, model_input: Dict[str, torch.Tensor]) -> Dict[str, object]:
        points = model_input["input"]
        mask = model_input["mask"]
        if not points.is_cuda:
            raise RuntimeError(
                "VoxelNeXt requires CUDA with the installed spconv build; "
                "move both the model and model_input tensors to a CUDA device."
            )
        actual_feature_dim = points.shape[-1] - 3
        if actual_feature_dim != self.num_point_features:
            raise ValueError(
                "Input feature mismatch: configured num_point_features="
                f"{self.num_point_features}, but input has {actual_feature_dim} "
                f"non-xyz channels (shape {tuple(points.shape)})."
            )
        if points.ndim != 4:
            raise ValueError(
                f"`input` must be [B, T, N, C], got {tuple(points.shape)}"
            )
        if mask.shape != points.shape[:3]:
            raise ValueError(
                f"`mask` must match [B, T, N]={tuple(points.shape[:3])}, "
                f"got {tuple(mask.shape)}"
            )

        batch_size, num_frames, num_points, channels = points.shape
        device = points.device

        # VoxelNeXt is a single-frame detector. Treat each frame as an
        # independent sparse-convolution sample, then restore B and T in the
        # variable-length output containers.
        frame_points = points.reshape(
            batch_size * num_frames, 1, num_points, channels
        )
        frame_mask = mask.reshape(batch_size * num_frames, 1, num_points)
        frame_batch_size = batch_size * num_frames

        batch_dict = self.voxelizer(frame_points, frame_mask)
        if batch_dict["voxel_features"].shape[0] == 0:
            encoded = None
        else:
            batch_dict = self.backbone_3d(batch_dict)
            encoded = batch_dict["encoded_spconv_tensor"]
            if os.environ.get("VOXELNEXT_DEBUG", "0") == "1":
                torch.cuda.synchronize(points.device)
                print("[VoxelNeXt] backbone complete", flush=True)

        bbox, objectness_logits = self.query_head(encoded, frame_batch_size)
        bbox = bbox.reshape(
            batch_size, num_frames, self.max_people, 6
        )
        objectness_logits = objectness_logits.reshape(
            batch_size, num_frames, self.max_people
        )

        return {
            "bbox": bbox,
            "objectness_logits": objectness_logits,
        }


if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model("VoxelNeXt").to(device)
    x = {
        "input": torch.zeros((10, 1, 300, 6), device=device),
        "mask": torch.ones((10, 1, 300), dtype=torch.bool, device=device),
    }
    profile_model("VoxelNeXt", model, x)
