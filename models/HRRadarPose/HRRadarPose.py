import torch
from torch import nn
import torch.nn.functional as F
from models.HRRadarPose.common import *

class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
        bn_type=None,
        bn_momentum=0.1,
        group=1,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.group = int(group)

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            bn_type=bn_type,
            bn_momentum=bn_momentum,
        )
        self.fuse_layers = self._make_fuse_layers(
            bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        stride=1,
        bn_type=None,
        bn_momentum=0.1,
    ):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.GroupNorm(8, self.num_inchannels[branch_index]),
                nn.Conv3d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    groups=self.group,
                    bias=False,
                ),
                # nn.BatchNorm3d(num_channels[branch_index] * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
                bn_type=bn_type,
                bn_momentum=bn_momentum,
                group=self.group,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    bn_type=bn_type,
                    bn_momentum=bn_momentum,
                    group=self.group,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self, num_branches, block, num_blocks, num_channels, bn_type, bn_momentum=0.1
    ):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    bn_type=bn_type,
                    bn_momentum=bn_momentum,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, bn_type, bn_momentum=0.1):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.GroupNorm(8, num_inchannels[j]),
                            nn.Conv3d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                groups=self.group,
                                bias=False,
                            ),
                            # nn.BatchNorm3d(num_inchannels[i]),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.GroupNorm(8, num_inchannels[j]),
                                    nn.Conv3d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        groups=self.group,
                                        bias=False,
                                    ),
                                    # nn.BatchNorm3d(num_outchannels_conv3x3),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.GroupNorm(8, num_inchannels[j]),
                                    nn.Conv3d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        groups=self.group,
                                        bias=False,
                                    ),
                                    # nn.BatchNorm3d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode="trilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HRRadarPose(nn.Module):
    def __init__(
        self,
        inchannels, group, num_joints,
        layer1_block,
        stage2_inplanes, stage2_num_modules, stage2_num_branches,
        stage2_block, stage2_num_blocks, stage2_num_channels, stage2_fuse_method,
        stage3_num_modules, stage3_num_branches, stage3_block,
        stage3_num_blocks, stage3_num_channels, stage3_fuse_method,
        stage4_num_modules, stage4_num_branches, stage4_block,
        stage4_num_blocks, stage4_num_channels, stage4_fuse_method,
    ):
        super().__init__()
        self.inchannels = int(inchannels)
        self.group = int(group)
        self.num_joints = num_joints

        cfg = {
            "LAYER1": {
                "INPLANES": self.inchannels,
                "BLOCK": layer1_block,
            },
            "STAGE2": {
                "INPLANES": int(stage2_inplanes),
                "NUM_MODULES": int(stage2_num_modules),
                "NUM_BRANCHES": int(stage2_num_branches),
                "NUM_BLOCKS": list(stage2_num_blocks),
                "NUM_CHANNELS": list(stage2_num_channels),
                "BLOCK": stage2_block,
                "FUSE_METHOD": stage2_fuse_method,
            },
            "STAGE3": {
                "NUM_MODULES": int(stage3_num_modules),
                "NUM_BRANCHES": int(stage3_num_branches),
                "NUM_BLOCKS": list(stage3_num_blocks),
                "NUM_CHANNELS": list(stage3_num_channels),
                "BLOCK": stage3_block,
                "FUSE_METHOD": stage3_fuse_method,
            },
            "STAGE4": {
                "NUM_MODULES": int(stage4_num_modules),
                "NUM_BRANCHES": int(stage4_num_branches),
                "NUM_BLOCKS": list(stage4_num_blocks),
                "NUM_CHANNELS": list(stage4_num_channels),
                "BLOCK": stage4_block,
                "FUSE_METHOD": stage4_fuse_method,
            },
        }
        configured_channels = {
            self.inchannels,
            int(stage2_inplanes),
            *map(int, stage2_num_channels),
            *map(int, stage3_num_channels),
            *map(int, stage4_num_channels),
        }
        invalid_channels = sorted(
            channel for channel in configured_channels
            if channel % self.group != 0
        )
        if invalid_channels:
            raise ValueError(
                f"All Conv3d channels must be divisible by group={self.group}; "
                f"invalid channels={invalid_channels}"
            )

        bn_type = None
        bn_momentum = None
        self.layer1_cfg = cfg["LAYER1"]
        self.stage2_cfg = cfg["STAGE2"]
        block = blocks_dict[self.layer1_cfg["BLOCK"]]
        self.layer1 = block(
            self.layer1_cfg['INPLANES'],
            self.stage2_cfg['INPLANES'],
            order='gcr',
            group=self.group,
        )
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [self.stage2_cfg["INPLANES"]], num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.stage4_cfg = cfg["STAGE4"] if "STAGE4" in cfg else None
        if not self.stage4_cfg is None:
            num_channels = self.stage4_cfg["NUM_CHANNELS"]
            block = blocks_dict[self.stage4_cfg["BLOCK"]]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))
            ]
            self.transition3 = self._make_transition_layer(
                pre_stage_channels, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
            )

            self.stage4, pre_stage_channels = self._make_stage(
                self.stage4_cfg,
                num_channels,
                multi_scale_output=True,
                bn_type=bn_type,
                bn_momentum=bn_momentum,
            )

        # Branch 0 is the highest-resolution output.  Use the channels returned
        # by _make_stage so block.expansion is accounted for automatically.
        self.backbone_out_channels = int(pre_stage_channels[0])
        self.body_center_head = nn.Sequential(
            nn.Conv3d(self.backbone_out_channels, 64, 3, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 8, 3, padding=1),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.keypoint_offset_head = nn.Sequential(
            nn.Conv3d(self.backbone_out_channels, 128, 3, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 3*self.num_joints, 3, padding=1),
        )

    def _make_transition_layer(
        self, num_channels_pre_layer, num_channels_cur_layer, bn_type, bn_momentum
    ):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.GroupNorm(8, num_channels_pre_layer[i]),
                            nn.Conv3d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                groups=self.group,
                                bias=False,
                            ),
                            # nn.BatchNorm3d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=False),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.GroupNorm(8, inchannels),
                            nn.Conv3d(
                                inchannels,
                                outchannels,
                                3,
                                2,
                                1,
                                groups=self.group,
                                bias=False,
                            ),
                            # nn.BatchNorm3d(outchannels),
                            nn.ReLU(inplace=False),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        layer_config,
        num_inchannels,
        multi_scale_output=True,
        bn_type=None,
        bn_momentum=0.1,
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    bn_type,
                    bn_momentum,
                    self.group,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _forward_features(self, x):
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        if not self.stage4_cfg is None:
            x_list = []
            for i in range(self.stage4_cfg["NUM_BRANCHES"]):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage4(x_list)

        return y_list

    def forward(self, model_input):
        radar_cube = model_input["input"]
        if not isinstance(radar_cube, torch.Tensor):
            raise TypeError(
                f"model_input['input'] must be a torch.Tensor, got {type(radar_cube)}"
            )
        if radar_cube.is_complex() or not radar_cube.is_floating_point():
            raise TypeError(
                "HRRadarPose expects a real floating-point Doppler-XYZ power cube, "
                f"got dtype={radar_cube.dtype}"
            )

        restore_time = radar_cube.ndim == 6
        if restore_time:
            batch_size, num_frames, channels, size_x, size_y, size_z = radar_cube.shape
            backbone_input = radar_cube.reshape(
                batch_size * num_frames,
                channels,
                size_x,
                size_y,
                size_z,
            )
        elif radar_cube.ndim == 5:
            batch_size, channels, size_x, size_y, size_z = radar_cube.shape
            num_frames = None
            backbone_input = radar_cube
        else:
            raise ValueError(
                "model_input['input'] must be [B,T,D,X,Y,Z] or [B,D,X,Y,Z], "
                f"got shape={tuple(radar_cube.shape)}"
            )

        if channels != self.inchannels:
            raise ValueError(
                f"HRRadarPose expects {self.inchannels} Doppler channels, "
                f"got {channels} in shape={tuple(radar_cube.shape)}"
            )

        features = self._forward_features(backbone_input.contiguous())
        features_high = features[0]
        body_center = self.body_center_head(features_high)
        keypoint_offset = self.keypoint_offset_head(features_high)
        if restore_time:
            body_center = body_center.reshape(batch_size, num_frames, 1, *body_center.shape[-3:])
            keypoint_offset = keypoint_offset.reshape(batch_size, num_frames, 3 * self.num_joints, *body_center.shape[-3:])
        else:
            body_center = body_center.reshape(batch_size, 1, *body_center.shape[-3:])
            keypoint_offset = keypoint_offset.reshape(batch_size, 3 * self.num_joints, *body_center.shape[-3:])
        return {
            'body_center': body_center,
            'keypoint_offset': keypoint_offset,
        }


    @torch.no_grad()
    def output_encoder(self, pose, pose_valid, xyz_limits, cube_size, gaussian_radius=1, center_joint_indices=(11, 12)):
        """
        Args:
            pose:               B, T, K, J, 3
            pose_valid:         B, T, K
            xyz_limits:         [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            cube_size:          X, Y, Z

        Returns:
            body_center:        B, T, 1, X, Y, Z
            center_indices: Flattened center indices ``x * Y * Z + y * Z + z``,
                shape ``[B, T, K]``. Invalid entries are zero.
            keypoint_offset: Joint offsets from the integer center voxel in
                feature-grid ``[x, y, z]`` units, shape ``[B, T, K, J, 3]``.
                Invalid entries are zero.
            target_valid: Final supervision mask after the spatial-range check,
                shape ``[B, T, K]``.
        """
        batch_size, num_frames, max_people, num_joints, _ = pose.shape
        size_x, size_y, size_z = (int(value) for value in cube_size)
        gaussian_radius = int(gaussian_radius)

        # [3, 2], coordinate order [x, y, z].
        xyz_limits_tensor = torch.as_tensor(xyz_limits,  dtype=pose.dtype, device=pose.device)
        xyz_min = xyz_limits_tensor[:, 0]  # [3], order [x, y, z].
        xyz_extent = xyz_limits_tensor[:, 1] - xyz_min  # [3], order [x, y, z].

        # All outputs are created on pose.device. Floating outputs inherit pose.dtype.
        body_center = pose.new_zeros((batch_size, num_frames, 1, size_x, size_y, size_z))  # [B, T, 1, X, Y, Z].
        center_indices = torch.zeros((batch_size, num_frames, max_people), dtype=torch.long,device=pose.device)  # [B, T, K].
        keypoint_offset = pose.new_zeros((batch_size, num_frames, max_people, num_joints, 3))  # [B, T, K, J, 3], last dimension is [x, y, z].

        input_valid = pose_valid.to(device=pose.device,dtype=torch.bool)  # [B, T, K].

        # COCO default: mean of joints 11 and 12. Shape [B, T, K, 3], order [x,y,z].
        center_world_xyz = pose[..., center_joint_indices, :].mean(dim=-2)

        # Number of voxels per world-space axis, shape [3], order [X, Y, Z].
        grid_size_xyz = pose.new_tensor((size_x, size_y, size_z))
        world_to_grid_scale = grid_size_xyz / xyz_extent  # [3], order [x, y, z].

        # Continuous grid coordinates, both with last dimension [x, y, z].
        pose_grid_xyz = ((pose - xyz_min) * world_to_grid_scale)  # [B, T, K, J, 3].
        center_grid_xyz = ((center_world_xyz - xyz_min) * world_to_grid_scale)  # [B, T, K, 3].

        # Do not clamp out-of-range people onto a heatmap boundary.
        center_inside = ((center_grid_xyz >= 0) & (center_grid_xyz < grid_size_xyz)).all(dim=-1)  # [B, T, K].
        target_valid = input_valid & center_inside  # [B, T, K].

        # floor matches the released HRRadarPose target encoder's integer conversion.
        safe_center_grid_xyz = torch.where(torch.isfinite(center_grid_xyz), center_grid_xyz, torch.zeros_like(center_grid_xyz))  # [B, T, K, 3].
        center_int_xyz = torch.floor(safe_center_grid_xyz).to(torch.long)
        # [B, T, K, 3], coordinate order [x, y, z].

        center_x = center_int_xyz[..., 0]  # [B, T, K].
        center_y = center_int_xyz[..., 1]  # [B, T, K].
        center_z = center_int_xyz[..., 2]  # [B, T, K].
        flat_center_indices = (
            center_x * (size_y * size_z) + center_y * size_z + center_z
        )  # [B, T, K].
        center_indices[target_valid] = flat_center_indices[target_valid]

        # Official target contract: joint grid coordinate minus integer center voxel.
        keypoint_offset = pose_grid_xyz - center_int_xyz.to(pose.dtype).unsqueeze(-2)
        # [B, T, K, J, 3], offset order [x, y, z].
        keypoint_offset = keypoint_offset.masked_fill(
            ~target_valid[..., None, None],
            0.0,
        )

        # Local isotropic 3D Gaussian kernel, shape [2R+1, 2R+1, 2R+1]
        # with tensor-axis order [x, y, z].
        diameter = 2 * gaussian_radius + 1
        gaussian_sigma = diameter / 6.0
        kernel_axis = torch.arange(
            -gaussian_radius,
            gaussian_radius + 1,
            dtype=pose.dtype,
            device=pose.device,
        )  # [2R+1].
        kernel_x, kernel_y, kernel_z = torch.meshgrid(
            kernel_axis,
            kernel_axis,
            kernel_axis,
            indexing="ij",
        )  # Each is [2R+1, 2R+1, 2R+1].
        # Keep the denominator used by the released HRRadarPose gaussian3D helper.
        gaussian_denominator = (2.0 * gaussian_sigma**2) ** 1.5
        gaussian_kernel = torch.exp(
            -(kernel_x.square() + kernel_y.square() + kernel_z.square())
            / gaussian_denominator
        )  # [2R+1, 2R+1, 2R+1], order [x, y, z].

        # K is small; sparse placement avoids allocating one dense map per person.
        valid_locations = target_valid.nonzero(as_tuple=False)
        # valid_locations: [N_valid, 3], row order [batch, time, person].
        for batch_idx, time_idx, person_idx in valid_locations.tolist():
            x_idx = int(center_x[batch_idx, time_idx, person_idx])
            y_idx = int(center_y[batch_idx, time_idx, person_idx])
            z_idx = int(center_z[batch_idx, time_idx, person_idx])

            x_start = max(0, x_idx - gaussian_radius)
            x_end = min(size_x, x_idx + gaussian_radius + 1)
            y_start = max(0, y_idx - gaussian_radius)
            y_end = min(size_y, y_idx + gaussian_radius + 1)
            z_start = max(0, z_idx - gaussian_radius)
            z_end = min(size_z, z_idx + gaussian_radius + 1)

            kernel_x_start = x_start - (x_idx - gaussian_radius)
            kernel_y_start = y_start - (y_idx - gaussian_radius)
            kernel_z_start = z_start - (z_idx - gaussian_radius)
            gaussian_patch = gaussian_kernel[
                kernel_x_start:kernel_x_start + (x_end - x_start),
                kernel_y_start:kernel_y_start + (y_end - y_start),
                kernel_z_start:kernel_z_start + (z_end - z_start),
            ]  # [local_X, local_Y, local_Z].

            heatmap_patch = body_center[batch_idx, time_idx, 0, x_start:x_end, y_start:y_end, z_start:z_end]  # [local_X, local_Y, local_Z].
            body_center[batch_idx, time_idx, 0, x_start:x_end, y_start:y_end, z_start:z_end] = torch.maximum(heatmap_patch, gaussian_patch)

        return body_center, center_indices, keypoint_offset, target_valid


if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    b, t, d, x_size, y_size, z_size = 1, 1, 64, 96, 96, 32

    device = set_device(0)
    model = build_model('HRRadarPose').to(device)
    x = {
        'input': torch.zeros((b, t, d, x_size, y_size, z_size), device=device),
    }
    profile_model("HRRadarPose", model, x)
