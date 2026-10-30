import torch
from torch import nn

from models.ConvNeXtV2.ConvNeXtV2_helper import Block, LayerNorm, trunc_normal_

class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        num_joints,
        in_chans,
        depths,
        dims,
        drop_path_rate,
        head_init_scale,
        xyz_limits,
        map_size,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must each contain 4 values")
        if in_chans != 3:
            raise ValueError("in_chans must be 3 for the three projected views")
        if len(xyz_limits) != 3 or any(
            len(axis) != 2 or axis[0] >= axis[1] for axis in xyz_limits
        ):
            raise ValueError("xyz_limits must contain three increasing [min, max] pairs")
        if len(map_size) != 2 or any(size <= 0 for size in map_size):
            raise ValueError("map_size must contain two positive values")

        self.depths = depths
        self.num_joints = num_joints
        self.map_size = tuple(int(size) for size in map_size)
        self.register_buffer(
            "xyz_limits",
            torch.as_tensor(xyz_limits, dtype=torch.float32),
            persistent=False,
        )
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_joints*3)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
    
    @torch.no_grad()
    def pc_to_3view_fixed(self, pc: torch.Tensor, point_mask: torch.Tensor):
        """
        pc: (B, N, 3), point_mask: (B, N)
        return: (B, 3, H, W), where (H, W) is map_size
          ch0: XOY (x-y)
          ch1: YOZ (y-z)
          ch2: XOZ (x-z)
        """
        if pc.ndim != 3 or pc.shape[-1] != 3:
            raise ValueError(f"pc must be [B,N,3], got {tuple(pc.shape)}")
        if point_mask.shape != pc.shape[:2]:
            raise ValueError(
                f"point_mask must be {tuple(pc.shape[:2])}, got {tuple(point_mask.shape)}"
            )

        batch_size = pc.shape[0]
        xyz_min = self.xyz_limits[:, 0].to(dtype=pc.dtype)
        xyz_max = self.xyz_limits[:, 1].to(dtype=pc.dtype)
        inside = point_mask.bool() & torch.isfinite(pc).all(dim=-1)
        inside &= ((pc >= xyz_min) & (pc < xyz_max)).all(dim=-1)
        safe_pc = torch.where(inside.unsqueeze(-1), pc, xyz_min)

        def coordinate_index(axis, size):
            index = torch.floor(
                (safe_pc[..., axis] - xyz_min[axis])
                / (xyz_max[axis] - xyz_min[axis])
                * size
            ).long()
            return index.clamp_(0, size - 1)

        def occupancy(u_idx, v_idx, height, width):
            pixels = u_idx * width + v_idx
            image = pc.new_zeros((batch_size, height * width))
            image.scatter_add_(1, pixels, inside.to(dtype=pc.dtype))
            return image.view(batch_size, height, width).gt_(0).to(pc.dtype)

        height, width = self.map_size
        x_height = coordinate_index(0, height)
        y_height = coordinate_index(1, height)
        y_width = coordinate_index(1, width)
        z_width = coordinate_index(2, width)
        image_xy = occupancy(x_height, y_width, height, width)
        image_yz = occupancy(y_height, z_width, height, width)
        image_xz = occupancy(x_height, z_width, height, width)
        return torch.stack((image_xy, image_yz, image_xz), dim=1)

    def forward(self, model_input):
        points = model_input["input"]
        point_mask = model_input["mask"]
        if points.ndim != 4:
            raise ValueError(f"input must be [B,T,N,D], got {tuple(points.shape)}")
        if point_mask.shape != points.shape[:3]:
            raise ValueError(
                f"mask must be {tuple(points.shape[:3])}, got {tuple(point_mask.shape)}"
            )

        batch_size, num_frames, num_points, _ = points.shape
        x = self.pc_to_3view_fixed(
            points[..., :3].reshape(batch_size * num_frames, num_points, 3),
            point_mask.reshape(batch_size * num_frames, num_points),
        )
        x = self.forward_features(x)
        x = self.head(x)
        pose = x.view(batch_size, num_frames, 1, self.num_joints, 3)
        return {"pose": pose}

if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model("ConvNeXtV2").to(device)
    x = {
        "input": torch.zeros((1, 16, 200, 5), device=device),
        "mask": torch.ones((1, 16, 200), dtype=torch.bool, device=device),
    }
    profile_model("ConvNeXtV2", model, x)
