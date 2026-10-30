import os
import torch
import torch.nn as nn
from typing import *

from models.VoxelNeXt.VoxelNeXt_helper import DynamicMeanVoxelizer, FixedJointQueryHead, VoxelResBackBone8xVoxelNeXt



class VoxelNeXt(nn.Module):
    """Sparse voxel pose regression: one cropped person, pose [B,T,1,J,3]."""

    def __init__(
        self,
        num_point_features: int = 2,
        xyz_limits: Sequence[Sequence[float]] = (
            (0.0, 6.0), (-3.0, 3.0), (-2.0, 2.0)
        ),
        voxel_size: Sequence[float] = (0.025, 0.025, 0.1),
        append_time: bool = False,
        append_xyz: bool = True,
        backbone_channels: Sequence[int] = (16, 32, 64, 128, 128),
        spconv_kernel_sizes: Sequence[int] = (3, 3, 3, 3),
        backbone_out_channel: int = 128,
        head_shared_conv_channel: int = 128,
        query_num_heads: int = 4,
        query_num_layers: int = 2,
        query_ffn_dim: int = 256,
        query_dropout: float = 0.0,
        num_joints: int = 17,
    ) -> None:
        """
        Args:
            num_point_features: Number of non-xyz feature channels to consume.
                Trailing metadata is ignored. For xyz-only input, set this to 0.
            xyz_limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
            voxel_size: [vx, vy, vz].
            append_time: Append the voxelizer's local frame index. Leave this
                disabled for frame-wise pose estimation because each flattened
                sample contains exactly one frame.
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
        if type(num_joints) is not int or num_joints <= 0:
            raise ValueError("num_joints must be a positive integer")
        if type(num_point_features) is not int or num_point_features < 0:
            raise ValueError("num_point_features must be a nonnegative integer")

        self.voxelizer = DynamicMeanVoxelizer(
            xyz_limits=xyz_limits,
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
        self.query_head = FixedJointQueryHead(
            input_channels=backbone_out_channel,
            hidden_dim=head_shared_conv_channel,
            num_heads=query_num_heads,
            num_layers=query_num_layers,
            ffn_dim=query_ffn_dim,
            dropout=query_dropout,
            num_joints=num_joints,
        )
        self.num_joints = num_joints

    def forward(self, model_input: Dict[str, torch.Tensor]) -> Dict[str, object]:
        points = model_input["input"]
        if not isinstance(points, torch.Tensor) or not points.is_floating_point():
            raise TypeError("input must be a floating-point tensor")
        if points.ndim != 4 or min(points.shape[:3]) < 1 or points.shape[-1] < 3 + self.num_point_features:
            raise ValueError("input must be nonempty [B,T,N,C], C >= 3 + num_point_features")
        mask = model_input.get("mask")
        if mask is None:
            mask = torch.ones(points.shape[:3], dtype=torch.bool, device=points.device)
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != points.shape[:3] or mask.device != points.device):
            raise ValueError("mask must be bool [B,T,N] on the input device")
        points = points[..., :3 + self.num_point_features].masked_fill(~mask[..., None], 0)
        if not torch.isfinite(points).all():
            raise ValueError("valid input points must be finite")
        if not points.is_cuda:
            raise RuntimeError("VoxelNeXt requires CUDA with the installed spconv build")

        batch_size, num_frames, num_points, channels = points.shape

        # Each input instance is one cropped person; frames are encoded independently.
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

        pose = self.query_head(encoded, frame_batch_size)
        frame_valid = torch.zeros(frame_batch_size, dtype=torch.bool, device=points.device)
        frame_valid[batch_dict["voxel_coords"][:, 0].long()] = True
        pose = pose.masked_fill(~frame_valid[:, None, None], 0)
        return {"pose": pose.reshape(batch_size, num_frames, 1, self.num_joints, 3)}



if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model("VoxelNeXt").to(device)
    x = {
        "input": torch.zeros((1, 1, 200, 5), device=device),
        "mask": torch.ones((1, 1, 200), dtype=torch.bool, device=device),
    }
    profile_model("VoxelNeXt", model, x)
