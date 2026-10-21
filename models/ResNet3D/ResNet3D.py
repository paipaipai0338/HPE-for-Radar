import torch
from torch import nn
from models.ResNet3D.roi_align import crop_spectra_to_gt, restore_spectra_people


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
        # self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # return self.act(self.bn(self.conv(x)))
        return self.act(self.conv(x))


class ResNet3D(nn.Module):
    """完整 DXYZ + 逐帧 3D 框 -> 内部 ROI 重采样 -> 单人特征 -> 雷达坐标姿态。"""

    def __init__(
        self,
        num_joints,
        groups,
        inchannels,
        num_channels,
        kernel_sizes,
        strides,
        padding,
        if_sum,
        xyz_limits,
        roi_cube_size,
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
        self.xyz_limits = xyz_limits
        self.roi_cube_size = roi_cube_size
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
            nn.Linear(num_channels[-1] // 2, num_joints * 3),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_channels[-1] // 2, num_channels[-1] // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_channels[-1] // 2, 1),
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

        if radar_cube.ndim != 6:
            raise ValueError('ResNet3D input must be full [B,T,D,X,Y,Z] spectra')
        if radar_cube.shape[2] != self.inchannels:
            raise ValueError(f'ResNet3D expects {self.inchannels} channels, got {radar_cube.shape[2]}')
        crops, context = crop_spectra_to_gt(
            radar_cube, {'bbox': model_input['bbox'], 'mask': model_input['bbox_mask']},
            self.xyz_limits, self.roi_cube_size)
        instances, frames, channels, size_x, size_y, size_z = crops['input'].shape
        if instances == 0:
            anchor = self.pose_head[-1].weight.sum()*0
            return restore_spectra_people({
                'pose': radar_cube.new_zeros(0, frames, 1, self.num_joints, 3)+anchor,
                'confidence': radar_cube.new_zeros(0, frames, 1)+anchor,
            }, context)
        x = crops['input'].reshape(instances*frames, channels, size_x, size_y, size_z)

        residual = x
        for layer, shortcut, add_residual in zip(
            self.layers, self.shortcuts, self.if_sum
        ):
            x = layer(x)
            if add_residual:
                x = x + shortcut(residual)
                residual = x
        pooled = self.pool(x).flatten(1)
        pose = self.pose_head(pooled).reshape(instances, frames, 1, self.num_joints, 3)
        confidence = torch.sigmoid(self.confidence_head(pooled)).reshape(instances, frames, 1)
        return restore_spectra_people({'pose': pose, 'confidence': confidence}, context)


if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    b, t, d, x_size, y_size, z_size = 1, 1, 70, 96, 96, 32

    device = set_device(0)
    model = build_model("ResNet3D").to(device)
    x = {
        "input": torch.zeros((b, t, d, x_size, y_size, z_size), device=device),
        "bbox": torch.tensor([0., -1., -1., 3., 1., 1.], device=device).reshape(1,1,1,6).expand(b,t,1,6),
        "bbox_mask": torch.ones(b,t,1,dtype=torch.bool,device=device),
    }
    profile_model("ResNet3D", model, x)
