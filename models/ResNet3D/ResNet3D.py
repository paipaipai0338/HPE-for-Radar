import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResNet3D(nn.Module):
    def __init__(
        self,
        num_joints,
        max_people,
        groups,
        inchannels,
        num_channels,
        kernel_sizes,
        strides,
        padding,
        if_sum,
    ):
        super().__init__()
        if not (
            len(num_channels)
            == len(kernel_sizes)
            == len(strides)
            == len(padding)
            == len(if_sum)
        ):
            raise ValueError("layer config lengths must match")

        self.num_joints = num_joints
        self.max_people = max_people
        self.inchannels = inchannels
        self.if_sum = [bool(value) for value in if_sum]

        in_channels = [inchannels, *num_channels[:-1]]
        self.layers = nn.ModuleList(
            ConvBlock(in_ch, out_ch, kernel, stride, pad, groups)
            for in_ch, out_ch, kernel, stride, pad in zip(
                in_channels,
                num_channels,
                kernel_sizes,
                strides,
                padding,
            )
        )

        shortcuts = []
        residual_channels = inchannels
        residual_stride = (1, 1, 1)
        for out_channels, stride, add_residual in zip(
            num_channels, strides, self.if_sum
        ):
            stride = (stride, stride, stride) if isinstance(stride, int) else stride
            residual_stride = tuple(
                current * value
                for current, value in zip(residual_stride, stride)
            )
            if add_residual and (
                residual_channels != out_channels
                or residual_stride != (1, 1, 1)
            ):
                shortcut = nn.Conv3d(
                    residual_channels,
                    out_channels,
                    kernel_size=1,
                    stride=residual_stride,
                    bias=False,
                )
            else:
                shortcut = nn.Identity()
            shortcuts.append(shortcut)
            if add_residual:
                residual_channels = out_channels
                residual_stride = (1, 1, 1)
        self.shortcuts = nn.ModuleList(shortcuts)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.pose_head = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_channels[-1] // 2, num_channels[-1] // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_channels[-1] // 2, max_people * num_joints * 4),
        )

    def forward(self, model_input):
        radar_cube = model_input["input"]
        if not isinstance(radar_cube, torch.Tensor):
            raise TypeError(
                f"model_input['input'] must be a torch.Tensor, got {type(radar_cube)}"
            )
        if radar_cube.is_complex() or not radar_cube.is_floating_point():
            raise TypeError(
                "ResNet3D expects a real floating-point Doppler-XYZ power cube, "
                f"got dtype={radar_cube.dtype}"
            )

        restore_time = radar_cube.ndim == 6
        if restore_time:
            batch_size, num_frames, channels, size_x, size_y, size_z = radar_cube.shape
            x = radar_cube.reshape(
                batch_size * num_frames,
                channels,
                size_x,
                size_y,
                size_z,
            )
        elif radar_cube.ndim == 5:
            batch_size, channels, _, _, _ = radar_cube.shape
            num_frames = None
            x = radar_cube
        else:
            raise ValueError(
                "model_input['input'] must be [B,T,D,X,Y,Z] or [B,D,X,Y,Z], "
                f"got shape={tuple(radar_cube.shape)}"
            )

        if channels != self.inchannels:
            raise ValueError(
                f"ResNet3D expects {self.inchannels} input channels, got {channels}"
            )

        residual = x
        for layer, shortcut, add_residual in zip(
            self.layers, self.shortcuts, self.if_sum
        ):
            x = layer(x)
            if add_residual:
                x = x + shortcut(residual)
                residual = x
        out = self.pose_head(self.pool(x).flatten(1)).view(x.size(0), self.max_people, self.num_joints, -1)
        pose = out[..., :3]
        confidence = torch.sigmoid(out[..., -1].mean(dim=-1))
        if restore_time:
            pose = pose.reshape(batch_size, num_frames, self.max_people, self.num_joints, 3)
            confidence = confidence.reshape(batch_size, num_frames, self.max_people)
        else:
            pose = pose.reshape(batch_size, self.max_people, self.num_joints, 3)
            confidence = confidence.reshape(batch_size, self.max_people)
        return {
            "pose": pose,
            "confidence" : confidence,
        }


if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    b, t, d, x_size, y_size, z_size = 1, 1, 64, 96, 96, 32

    device = set_device(0)
    model = build_model("ResNet3D").to(device)
    x = {
        "input": torch.zeros((b, t, d, x_size, y_size, z_size), device=device),
    }
    profile_model("ResNet3D", model, x)
