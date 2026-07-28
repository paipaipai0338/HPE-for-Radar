"""Self-contained helper components for the simplified VoxelNeXt wrapper.

The code in this file collects the inference-time pieces that were previously
spread across OpenPCDet/VoxelNeXt files:

    - padded point cloud -> sparse voxel conversion
    - spconv utility functions
    - VoxelNeXt sparse backbone
    - CenterNet-style sparse detection head
    - bbox decoding and a lightweight BEV NMS fallback

External runtime dependencies are intentionally kept small: torch,
torchvision, easydict and spconv. Boxes are axis-aligned, so BEV NMS operates
directly on their x-y min/max coordinates.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import copy
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms as torchvision_nms
from torch.nn.init import kaiming_normal_

try:
    from easydict import EasyDict
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install easydict: pip install easydict") from exc

def _configure_spconv_jit_environment() -> None:
    """Make the local editable CUDA build reproducible on normal invocations."""
    cuda_root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = cuda_root / "bin" / "nvcc"
    if not nvcc.is_file():
        return

    os.environ.setdefault("CUDA_HOME", str(cuda_root))
    cuda_bin = str(nvcc.parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if cuda_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([cuda_bin, *path_parts])

    # The installed editable packages are named for torch's CUDA runtime
    # (currently cumm-cu128/spconv-cu128).
    # Without this value, their JIT code generator silently switches to a
    # different CPU/default build graph on every import.
    if torch.version.cuda:
        os.environ.setdefault("CUMM_CUDA_VERSION", torch.version.cuda)


_configure_spconv_jit_environment()

try:
    import spconv.pytorch as spconv
except ImportError as exc:
    raise ImportError(
        "VoxelNeXt requires `spconv.pytorch`; install a CUDA-compatible "
        "spconv 2.x build."
    ) from exc


# The Native indice-pair path in the prebuilt spconv-cu120 wheel can SIGFPE
# before launching a CUDA kernel with newer PyTorch/CUDA stacks.  VoxelNeXt
# runs on CUDA, so explicitly use spconv's GPU implicit-GEMM implementation.
SPCONV_ALGO = spconv.ConvAlgo.MaskImplicitGemm


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        return out.replace_feature(new_features)
    out.features = new_features
    return out


def paired_axis_aligned_iou_3d(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """IoU for paired boxes in xyz-min/xyz-max format."""
    inter_min = torch.maximum(boxes_a[:, :3], boxes_b[:, :3])
    inter_max = torch.minimum(boxes_a[:, 3:], boxes_b[:, 3:])
    inter_size = (inter_max - inter_min).clamp_min(0)
    inter_volume = inter_size.prod(dim=-1)
    volume_a = (boxes_a[:, 3:] - boxes_a[:, :3]).clamp_min(0).prod(dim=-1)
    volume_b = (boxes_b[:, 3:] - boxes_b[:, :3]).clamp_min(0).prod(dim=-1)
    return inter_volume / (volume_a + volume_b - inter_volume).clamp_min(1e-6)


def pairwise_axis_aligned_iou_3d(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """All-pairs IoU matrix for xyz-min/xyz-max boxes."""
    inter_min = torch.maximum(boxes_a[:, None, :3], boxes_b[None, :, :3])
    inter_max = torch.minimum(boxes_a[:, None, 3:], boxes_b[None, :, 3:])
    inter_volume = (inter_max - inter_min).clamp_min(0).prod(dim=-1)
    volume_a = (
        (boxes_a[:, 3:] - boxes_a[:, :3]).clamp_min(0).prod(dim=-1)[:, None]
    )
    volume_b = (
        (boxes_b[:, 3:] - boxes_b[:, :3]).clamp_min(0).prod(dim=-1)[None, :]
    )
    return inter_volume / (
        volume_a + volume_b - inter_volume
    ).clamp_min(1e-6)


class FixedPersonQueryHead(nn.Module):
    """Fixed-cardinality set-prediction head for axis-aligned human boxes."""

    def __init__(
        self,
        input_channels,
        hidden_dim,
        num_heads,
        num_layers,
        ffn_dim,
        dropout,
        max_people,
        point_cloud_range,
    ):
        super().__init__()
        self.max_people = int(max_people)
        self.hidden_dim = int(hidden_dim)
        input_channels = int(input_channels)
        num_heads = int(num_heads)
        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"query hidden_dim={self.hidden_dim} must be divisible by "
                f"query_num_heads={num_heads}"
            )

        self.register_buffer(
            "point_cloud_range",
            torch.as_tensor(point_cloud_range, dtype=torch.float32),
            persistent=False,
        )
        self.input_projection = nn.Linear(input_channels, self.hidden_dim)
        self.position_embedding = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_embedding = nn.Embedding(self.max_people, self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(num_layers)
        )
        self.box_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 6),
        )
        self.objectness_head = nn.Linear(self.hidden_dim, 1)

    def _pad_sparse_memory(self, x, batch_size):
        features = self.input_projection(x.features)
        batch_indices = x.indices[:, 0].long()
        counts = torch.bincount(batch_indices, minlength=batch_size)
        max_tokens = max(int(counts.max().item()), 1)
        memory = features.new_zeros((batch_size, max_tokens, self.hidden_dim))
        padding_mask = torch.ones(
            (batch_size, max_tokens), dtype=torch.bool, device=features.device
        )

        spatial_shape = x.spatial_shape
        denom_y = max(int(spatial_shape[-2]) - 1, 1)
        denom_x = max(int(spatial_shape[-1]) - 1, 1)
        positions = torch.stack(
            [
                x.indices[:, -1].float() / denom_x,
                x.indices[:, -2].float() / denom_y,
            ],
            dim=-1,
        )
        features = features + self.position_embedding(positions)

        for batch_idx in range(batch_size):
            current = features[batch_indices == batch_idx]
            count = current.shape[0]
            if count:
                memory[batch_idx, :count] = current
                padding_mask[batch_idx, :count] = False
            else:
                # Keep one zero sentinel token so attention never receives an
                # all-masked memory row.
                padding_mask[batch_idx, 0] = False
        return memory, padding_mask

    def forward(self, x, batch_size):
        if x is None:
            memory = self.query_embedding.weight.new_zeros(
                (batch_size, 1, self.hidden_dim)
            )
            padding_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=memory.device
            )
        else:
            memory, padding_mask = self._pad_sparse_memory(x, batch_size)

        queries = self.query_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=padding_mask,
        )
        raw_boxes = self.box_head(decoded)
        objectness_logits = self.objectness_head(decoded).squeeze(-1)

        pc_min = self.point_cloud_range[:3]
        pc_extent = self.point_cloud_range[3:] - pc_min
        center = pc_min + raw_boxes[..., :3].sigmoid() * pc_extent
        size = F.softplus(raw_boxes[..., 3:]).clamp_max(pc_extent)
        boxes = torch.cat([center - size * 0.5, center + size * 0.5], dim=-1)
        return boxes, objectness_logits


class PersonSetCriterion(nn.Module):
    """Hungarian set loss for one-class, axis-aligned person detection."""

    def __init__(
        self,
        point_cloud_range,
        objectness_weight=1.0,
        bbox_l1_weight=5.0,
        bbox_iou_weight=2.0,
    ):
        super().__init__()
        self.register_buffer(
            "point_cloud_range",
            torch.as_tensor(point_cloud_range, dtype=torch.float32),
            persistent=False,
        )
        self.objectness_weight = float(objectness_weight)
        self.bbox_l1_weight = float(bbox_l1_weight)
        self.bbox_iou_weight = float(bbox_iou_weight)

    def _normalize_boxes(self, boxes):
        pc_min = self.point_cloud_range[:3]
        extent = (self.point_cloud_range[3:] - pc_min).clamp_min(1e-6)
        return torch.cat(
            [
                (boxes[..., :3] - pc_min) / extent,
                (boxes[..., 3:] - pc_min) / extent,
            ],
            dim=-1,
        )

    def forward(self, prediction, target):
        pred_boxes = prediction["bboxes"]
        pred_logits = prediction["objectness_logits"]
        gt_boxes = target.get("bbox", target.get("bboxes"))
        gt_valid = target.get("mask", target.get("valid"))
        if gt_boxes is None or gt_valid is None:
            raise KeyError(
                "Detection target must contain `bbox` and `mask` "
                "(or compatibility aliases `bboxes` and `valid`)."
            )
        gt_valid = gt_valid.bool()
        if pred_boxes.shape[:2] != gt_boxes.shape[:2]:
            raise ValueError(
                "Prediction and GT B,T dimensions differ: "
                f"{pred_boxes.shape[:2]} vs {gt_boxes.shape[:2]}"
            )

        flat_pred_boxes = pred_boxes.flatten(0, 1)
        flat_pred_logits = pred_logits.flatten(0, 1)
        flat_gt_boxes = gt_boxes.flatten(0, 1)
        flat_gt_valid = gt_valid.flatten(0, 1)
        objectness_target = torch.zeros_like(flat_pred_logits)
        matched_pred, matched_gt = [], []

        for frame_idx in range(flat_pred_boxes.shape[0]):
            current_gt = flat_gt_boxes[frame_idx][flat_gt_valid[frame_idx]]
            if current_gt.numel() == 0:
                continue
            if current_gt.shape[0] > flat_pred_boxes.shape[1]:
                raise ValueError(
                    f"Frame {frame_idx} contains {current_gt.shape[0]} people, "
                    f"but max_people={flat_pred_boxes.shape[1]}."
                )
            pred_norm = self._normalize_boxes(flat_pred_boxes[frame_idx])
            gt_norm = self._normalize_boxes(current_gt)
            l1_cost = torch.cdist(pred_norm, gt_norm, p=1)
            iou_cost = 1.0 - pairwise_axis_aligned_iou_3d(
                flat_pred_boxes[frame_idx], current_gt
            )
            matching_cost = (
                self.bbox_l1_weight * l1_cost
                + self.bbox_iou_weight * iou_cost
            )
            row_idx, col_idx = linear_sum_assignment(
                matching_cost.detach().cpu().numpy()
            )
            row_idx = torch.as_tensor(
                row_idx, dtype=torch.long, device=pred_boxes.device
            )
            col_idx = torch.as_tensor(
                col_idx, dtype=torch.long, device=pred_boxes.device
            )
            objectness_target[frame_idx, row_idx] = 1.0
            matched_pred.append(flat_pred_boxes[frame_idx, row_idx])
            matched_gt.append(current_gt[col_idx])

        loss_objectness = F.binary_cross_entropy_with_logits(
            flat_pred_logits, objectness_target
        )
        if matched_pred:
            matched_pred = torch.cat(matched_pred, dim=0)
            matched_gt = torch.cat(matched_gt, dim=0)
            loss_bbox_l1 = F.l1_loss(
                self._normalize_boxes(matched_pred),
                self._normalize_boxes(matched_gt),
            )
            loss_bbox_iou = (
                1.0 - paired_axis_aligned_iou_3d(matched_pred, matched_gt)
            ).mean()
        else:
            zero = pred_boxes.sum() * 0.0
            loss_bbox_l1 = zero
            loss_bbox_iou = zero

        total = (
            self.objectness_weight * loss_objectness
            + self.bbox_l1_weight * loss_bbox_l1
            + self.bbox_iou_weight * loss_bbox_iou
        )
        return {
            "loss": total,
            "loss_objectness": loss_objectness,
            "loss_bbox_l1": loss_bbox_l1,
            "loss_bbox_iou": loss_bbox_iou,
        }


class DynamicMeanVoxelizer(nn.Module):
    """Voxelize padded sparse point clouds and average features per voxel."""

    def __init__(
        self,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        append_time: bool = True,
        append_xyz: bool = False,
    ) -> None:
        super().__init__()
        self.point_cloud_range = tuple(float(x) for x in point_cloud_range)
        self.voxel_size = tuple(float(x) for x in voxel_size)
        self.append_time = append_time
        self.append_xyz = append_xyz
        self.grid_size = torch.Size(
            int(round((self.point_cloud_range[i + 3] - self.point_cloud_range[i]) / self.voxel_size[i]))
            for i in range(3)
        )

    @property
    def num_extra_features(self) -> int:
        return int(self.append_time) + (3 if self.append_xyz else 0)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if points.ndim != 4:
            raise ValueError(f"`input` must be [B, T, N, C], got {tuple(points.shape)}")
        if mask.ndim != 3:
            raise ValueError(f"`mask` must be [B, T, N], got {tuple(mask.shape)}")
        if points.shape[:3] != mask.shape:
            raise ValueError(f"`input` and `mask` shape mismatch: {points.shape[:3]} vs {mask.shape}")
        if points.shape[-1] < 3:
            raise ValueError("The last input channel dimension must contain at least xyz.")

        points = points.float().contiguous()
        valid_mask = mask.to(dtype=torch.bool)
        batch_size, num_frames, num_points, channels = points.shape
        device = points.device

        xyz = points[..., :3]
        pc_min = torch.as_tensor(self.point_cloud_range[:3], dtype=torch.float32, device=device)
        pc_max = torch.as_tensor(self.point_cloud_range[3:], dtype=torch.float32, device=device)
        voxel_size = torch.as_tensor(self.voxel_size, dtype=torch.float32, device=device)
        grid_size = torch.as_tensor(tuple(self.grid_size), dtype=torch.long, device=device)

        valid_mask = valid_mask & ((xyz >= pc_min) & (xyz < pc_max)).all(dim=-1)
        valid_mask = valid_mask & torch.isfinite(points).all(dim=-1)

        feature_dim = max(channels - 3, 1) + self.num_extra_features
        if not valid_mask.any():
            return {
                "batch_size": batch_size,
                "voxel_features": points.new_zeros((0, feature_dim)),
                "voxel_coords": torch.zeros((0, 4), dtype=torch.int32, device=device),
            }

        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(batch_size, num_frames, num_points)
        time_idx = torch.arange(num_frames, device=device).view(1, num_frames, 1).expand(batch_size, num_frames, num_points)

        xyz_valid = xyz[valid_mask]
        voxel_xyz = torch.floor((xyz_valid - pc_min) / voxel_size).long()
        voxel_xyz = torch.minimum(torch.maximum(voxel_xyz, torch.zeros_like(voxel_xyz)), grid_size - 1)
        voxel_coords = torch.cat([batch_idx[valid_mask].long().unsqueeze(-1), voxel_xyz[:, [2, 1, 0]]], dim=1)

        if channels > 3:
            point_features = points[..., 3:][valid_mask]
        else:
            point_features = points.new_ones((xyz_valid.shape[0], 1))

        parts = [point_features]
        if self.append_time:
            parts.append((time_idx[valid_mask].float() / max(num_frames - 1, 1)).unsqueeze(-1))
        if self.append_xyz:
            parts.append(xyz_valid)
        point_features = torch.cat(parts, dim=-1)

        unique_coords, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
        voxel_features = point_features.new_zeros((unique_coords.shape[0], point_features.shape[-1]))
        voxel_features.index_add_(0, inverse, point_features)
        counts = point_features.new_zeros((unique_coords.shape[0], 1))
        counts.index_add_(0, inverse, torch.ones((point_features.shape[0], 1), dtype=point_features.dtype, device=device))
        voxel_features = voxel_features / counts.clamp_min(1.0)

        return {
            "batch_size": batch_size,
            "voxel_features": voxel_features.contiguous(),
            "voxel_coords": unique_coords.int().contiguous(),
        }


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, conv_type="subm", norm_fn=None):
    if conv_type == "subm":
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, bias=False,
            indice_key=indice_key, algo=SPCONV_ALGO,
        )
    elif conv_type == "spconv":
        conv = spconv.SparseConv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=False, indice_key=indice_key,
            algo=SPCONV_ALGO,
        )
    elif conv_type == "inverseconv":
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError(conv_type)
    return spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super().__init__()
        assert norm_fn is not None
        # The following BatchNorm makes convolution bias redundant.  Keeping a
        # bias also selects a fused spconv path that is unsupported on CPU.
        bias = False
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1,
            bias=bias, indice_key=indice_key, algo=SPCONV_ALGO,
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=bias, indice_key=indice_key, algo=SPCONV_ALGO,
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = replace_feature(out, self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = replace_feature(out, self.relu(out.features + identity.features))
        return out


class VoxelResBackBone8xVoxelNeXt(nn.Module):
    def __init__(
        self,
        input_channels,
        grid_size,
        spconv_kernel_sizes,
        channels,
        out_channel,
        **kwargs,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        spconv_kernel_sizes = [int(value) for value in spconv_kernel_sizes]
        channels = [int(value) for value in channels]
        out_channel = int(out_channel)
        # spconv expects the spatial shape in z-y-x order.  The upstream
        # OpenPCDet implementation uses a NumPy array here, where
        # ``grid_size[::-1] + [1, 0, 0]`` is an element-wise addition.
        # A torch tensor does not support negative-step slicing, and converting
        # both operands to lists would accidentally create a six-dimensional
        # shape through list concatenation.
        self.sparse_shape = [
            int(grid_size[2].item()) + 1,
            int(grid_size[1].item()),
            int(grid_size[0].item()),
        ]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels, channels[0], 3, padding=1, bias=False,
                indice_key="subm1", algo=SPCONV_ALGO,
            ),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key="res1"),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key="res1"),
        )
        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=spconv_kernel_sizes[0] // 2, indice_key="spconv2", conv_type="spconv"),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key="res2"),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key="res2"),
        )
        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=spconv_kernel_sizes[1] // 2, indice_key="spconv3", conv_type="spconv"),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key="res3"),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key="res3"),
        )
        self.conv4 = spconv.SparseSequential(
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=spconv_kernel_sizes[2] // 2, indice_key="spconv4", conv_type="spconv"),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key="res4"),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key="res4"),
        )
        self.conv5 = spconv.SparseSequential(
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=spconv_kernel_sizes[3] // 2, indice_key="spconv5", conv_type="spconv"),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key="res5"),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key="res5"),
        )
        self.conv6 = spconv.SparseSequential(
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=spconv_kernel_sizes[3] // 2, indice_key="spconv6", conv_type="spconv"),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key="res6"),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key="res6"),
        )
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(
                channels[3], out_channel, 3, stride=1, padding=1,
                bias=False, indice_key="spconv_down2", algo=SPCONV_ALGO,
            ),
            norm_fn(out_channel),
            nn.ReLU(),
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(
                out_channel, out_channel, 3, stride=1, padding=1,
                bias=True, algo=SPCONV_ALGO,
            ),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )
        self.num_point_features = out_channel

    def bev_out(self, x_conv):
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        features_cat = x_conv.features
        indices_unique, inverse = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, inverse, features_cat)
        return spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x_conv.spatial_shape[1:],
            batch_size=x_conv.batch_size,
        )

    def forward(self, batch_dict):
        debug = os.environ.get("VOXELNEXT_DEBUG", "0") == "1"

        def debug_sparse(name, tensor):
            if not debug:
                return
            # Synchronizing after each stage makes asynchronous CUDA failures
            # appear at the operation that caused them.
            if tensor.features.is_cuda:
                torch.cuda.synchronize(tensor.features.device)
            print(
                f"[VoxelNeXt] {name}: features={tuple(tensor.features.shape)}, "
                f"indices={tuple(tensor.indices.shape)}, "
                f"spatial_shape={list(tensor.spatial_shape)}",
                flush=True,
            )

        input_sp_tensor = spconv.SparseConvTensor(
            features=batch_dict["voxel_features"],
            indices=batch_dict["voxel_coords"].int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict["batch_size"],
        )
        debug_sparse("input", input_sp_tensor)
        x = self.conv_input(input_sp_tensor)
        debug_sparse("conv_input", x)
        x_conv1 = self.conv1(x)
        debug_sparse("conv1", x_conv1)
        x_conv2 = self.conv2(x_conv1)
        debug_sparse("conv2", x_conv2)
        x_conv3 = self.conv3(x_conv2)
        debug_sparse("conv3", x_conv3)
        x_conv4 = self.conv4(x_conv3)
        debug_sparse("conv4", x_conv4)
        x_conv5 = self.conv5(x_conv4)
        debug_sparse("conv5", x_conv5)
        x_conv6 = self.conv6(x_conv5)
        debug_sparse("conv6", x_conv6)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])
        debug_sparse("multi_scale_fusion", x_conv4)

        bev = self.bev_out(x_conv4)
        debug_sparse("bev", bev)
        out = self.conv_out(bev)
        debug_sparse("conv_out", out)
        out = self.shared_conv(out)
        debug_sparse("shared_conv", out)
        batch_dict["encoded_spconv_tensor"] = out
        batch_dict["encoded_spconv_tensor_stride"] = 8
        return batch_dict


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]["out_channels"]
            num_conv = self.sep_head_dict[cur_name]["num_conv"]
            fc_list = []
            for _ in range(num_conv - 1):
                fc_list.append(
                    spconv.SparseSequential(
                        spconv.SubMConv2d(
                            input_channels, input_channels, kernel_size,
                            padding=kernel_size // 2, bias=use_bias,
                            indice_key=cur_name, algo=SPCONV_ALGO,
                        ),
                        nn.BatchNorm1d(input_channels),
                        nn.ReLU(),
                    )
                )
            fc_list.append(
                spconv.SubMConv2d(
                    input_channels, output_channels, 1, bias=True,
                    indice_key=cur_name + "out", algo=SPCONV_ALGO,
                )
            )
            fc = nn.Sequential(*fc_list)
            if "hm" in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if hasattr(spconv, "conv") and isinstance(m, spconv.conv.SparseConvolution):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            setattr(self, cur_name, fc)

    def forward(self, x):
        return {cur_name: getattr(self, cur_name)(x).features for cur_name in self.sep_head_dict}


def _topk_1d(obj: torch.Tensor, batch_size: int, batch_idx: torch.Tensor, K: int):
    topk_score_list, topk_inds_list, topk_classes_list = [], [], []
    num_classes = obj.shape[-1]
    for bs_idx in range(batch_size):
        batch_mask = batch_idx == bs_idx
        score = obj[batch_mask].permute(1, 0)
        if score.numel() == 0:
            topk_score_list.append(obj.new_zeros((K,)))
            topk_inds_list.append(torch.zeros((K,), dtype=torch.long, device=obj.device))
            topk_classes_list.append(torch.zeros((K,), dtype=torch.long, device=obj.device))
            continue
        local_k = min(K, score.numel())
        topk_scores, topk_inds = torch.topk(score.reshape(-1), local_k)
        if local_k < K:
            pad_n = K - local_k
            topk_scores = torch.cat([topk_scores, obj.new_zeros((pad_n,))])
            topk_inds = torch.cat([topk_inds, torch.zeros((pad_n,), dtype=torch.long, device=obj.device)])
        topk_score_list.append(topk_scores)
        topk_inds_list.append(topk_inds % score.shape[-1])
        topk_classes_list.append((topk_inds // score.shape[-1]).long().clamp(max=num_classes - 1))
    return torch.stack(topk_score_list), torch.stack(topk_inds_list), torch.stack(topk_classes_list)


def gather_feat_idx(feats: torch.Tensor, inds: torch.Tensor, batch_size: int, batch_idx: torch.Tensor):
    feats_list = []
    dim = feats.size(-1)
    expanded_inds = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), dim)
    for bs_idx in range(batch_size):
        batch_mask = batch_idx == bs_idx
        feat = feats[batch_mask]
        if feat.numel() == 0:
            feats_list.append(feats.new_zeros((inds.size(1), dim)))
        else:
            feats_list.append(feat.gather(0, expanded_inds[bs_idx].clamp(max=feat.shape[0] - 1)))
    return torch.stack(feats_list)


def decode_bbox_from_voxels(
    batch_size: int,
    indices: torch.Tensor,
    obj: torch.Tensor,
    center: torch.Tensor,
    center_z: torch.Tensor,
    dim: torch.Tensor,
    point_cloud_range: torch.Tensor,
    voxel_size: torch.Tensor,
    feature_map_stride: int,
    K: int,
    score_thresh: Optional[float],
    post_center_limit_range: torch.Tensor,
):
    batch_idx = indices[:, 0]
    spatial_indices = indices[:, 1:]
    scores, inds, class_ids = _topk_1d(obj, batch_size, batch_idx, K=K)

    center = gather_feat_idx(center, inds, batch_size, batch_idx)
    center_z = gather_feat_idx(center_z, inds, batch_size, batch_idx)
    dim = gather_feat_idx(dim, inds, batch_size, batch_idx)
    spatial_indices = gather_feat_idx(spatial_indices.float(), inds, batch_size, batch_idx)

    xs = (spatial_indices[:, :, -1:] + center[:, :, 0:1]) * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = (spatial_indices[:, :, -2:-1] + center[:, :, 1:2]) * feature_map_stride * voxel_size[1] + point_cloud_range[1]
    center_xyz = torch.cat([xs, ys, center_z], dim=-1)
    half_dim = dim * 0.5
    # Axis-aligned human box: [x_min, y_min, z_min, x_max, y_max, z_max].
    final_box_preds = torch.cat(
        [center_xyz - half_dim, center_xyz + half_dim], dim=-1
    )

    mask = (center_xyz >= post_center_limit_range[:3]).all(2)
    mask &= (center_xyz <= post_center_limit_range[3:]).all(2)
    if score_thresh is not None:
        mask &= scores > score_thresh

    ret = []
    for k in range(batch_size):
        cur_mask = mask[k]
        ret.append(
            {
                "pred_boxes": final_box_preds[k, cur_mask],
                "pred_scores": scores[k, cur_mask],
                "pred_labels": class_ids[k, cur_mask],
                "pred_ious": None,
            }
        )
    return ret


def class_agnostic_nms(box_scores: torch.Tensor, box_preds: torch.Tensor, nms_config, score_thresh: Optional[float] = None):
    if box_scores.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=box_scores.device), box_scores
    if score_thresh is not None:
        score_mask = box_scores >= score_thresh
        original_idxs = score_mask.nonzero(as_tuple=False).view(-1)
        box_scores = box_scores[score_mask]
        box_preds = box_preds[score_mask]
    else:
        original_idxs = torch.arange(box_scores.shape[0], device=box_scores.device)

    pre_max = min(int(nms_config.get("NMS_PRE_MAXSIZE", box_scores.shape[0])), box_scores.shape[0])
    post_max = int(nms_config.get("NMS_POST_MAXSIZE", pre_max))
    thresh = float(nms_config.get("NMS_THRESH", 0.1))
    order = torch.argsort(box_scores, descending=True)[:pre_max]
    candidate_boxes = box_preds[order]

    # torchvision's CUDA NMS expects BEV [x_min, y_min, x_max, y_max].
    bev_boxes = candidate_boxes[:, [0, 1, 3, 4]]
    keep = torchvision_nms(bev_boxes, box_scores[order], thresh)[:post_max]
    selected_in_filtered = order[keep]
    selected = original_idxs[selected_in_filtered]
    return selected, box_scores[selected_in_filtered]


class VoxelNeXtHead(nn.Module):
    """Inference-focused VoxelNeXt sparse CenterNet head."""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.register_buffer("point_cloud_range", torch.as_tensor(point_cloud_range, dtype=torch.float32), persistent=False)
        self.register_buffer("voxel_size", torch.as_tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get("FEATURE_MAP_STRIDE", None)
        self.class_names = list(class_names)
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            valid_names = [x for x in cur_class_names if x in self.class_names]
            self.class_names_each_head.append(valid_names)
            mapping = torch.as_tensor([self.class_names.index(x) for x in valid_names], dtype=torch.long)
            self.register_buffer(f"class_id_mapping_head_{len(self.class_id_mapping_each_head)}", mapping, persistent=False)
            self.class_id_mapping_each_head.append(mapping)

        assert sum(len(x) for x in self.class_names_each_head) == len(self.class_names)
        shared_channels = int(
            self.model_cfg.get("SHARED_CONV_CHANNEL", input_channels)
        )
        if shared_channels == input_channels:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = spconv.SparseSequential(
                spconv.SubMConv2d(
                    input_channels,
                    shared_channels,
                    1,
                    bias=False,
                    indice_key="head_input_projection",
                    algo=SPCONV_ALGO,
                ),
                nn.BatchNorm1d(shared_channels),
                nn.ReLU(),
            )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        kernel_size_head = self.model_cfg.get("KERNEL_SIZE_HEAD", 3)
        for cur_class_names in self.class_names_each_head:
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict["hm"] = EasyDict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=shared_channels,
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    use_bias=self.model_cfg.get("USE_BIAS_BEFORE_NORM", False),
                )
            )

    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        batch_index = voxel_indices[:, 0]
        return spatial_shape, batch_index, voxel_indices

    def generate_predicted_boxes(self, batch_size, pred_dicts, voxel_indices):
        post_cfg = self.model_cfg.POST_PROCESSING
        limit_range = torch.as_tensor(post_cfg.POST_CENTER_LIMIT_RANGE, dtype=torch.float32, device=voxel_indices.device)
        ret_dict = [{"pred_boxes": [], "pred_scores": [], "pred_labels": [], "pred_ious": []} for _ in range(batch_size)]

        for head_idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict["hm"].sigmoid()
            batch_center = pred_dict["center"]
            batch_center_z = pred_dict["center_z"]
            batch_dim = pred_dict["dim"].exp()
            final_pred_dicts = decode_bbox_from_voxels(
                batch_size=batch_size,
                indices=voxel_indices,
                obj=batch_hm,
                center=batch_center,
                center_z=batch_center_z,
                dim=batch_dim,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_cfg.SCORE_THRESH,
                post_center_limit_range=limit_range,
            )

            mapping = getattr(self, f"class_id_mapping_head_{head_idx}").to(voxel_indices.device)
            for batch_i, final_dict in enumerate(final_pred_dicts):
                final_dict["pred_labels"] = mapping[final_dict["pred_labels"].long()]
                selected, selected_scores = class_agnostic_nms(
                    box_scores=final_dict["pred_scores"],
                    box_preds=final_dict["pred_boxes"],
                    nms_config=post_cfg.NMS_CONFIG,
                    score_thresh=None,
                )
                ret_dict[batch_i]["pred_boxes"].append(final_dict["pred_boxes"][selected])
                ret_dict[batch_i]["pred_scores"].append(selected_scores)
                ret_dict[batch_i]["pred_labels"].append(final_dict["pred_labels"][selected])

        for batch_i in range(batch_size):
            device = voxel_indices.device
            if ret_dict[batch_i]["pred_boxes"]:
                ret_dict[batch_i]["pred_boxes"] = torch.cat(ret_dict[batch_i]["pred_boxes"], dim=0)
                ret_dict[batch_i]["pred_scores"] = torch.cat(ret_dict[batch_i]["pred_scores"], dim=0)
                ret_dict[batch_i]["pred_labels"] = torch.cat(ret_dict[batch_i]["pred_labels"], dim=0) + 1
            else:
                ret_dict[batch_i]["pred_boxes"] = torch.zeros((0, 6), dtype=torch.float32, device=device)
                ret_dict[batch_i]["pred_scores"] = torch.zeros((0,), dtype=torch.float32, device=device)
                ret_dict[batch_i]["pred_labels"] = torch.zeros((0,), dtype=torch.long, device=device)
        return ret_dict

    def forward(self, data_dict):
        x = self.input_projection(data_dict["encoded_spconv_tensor"])
        _, _, voxel_indices = self._get_voxel_infos(x)
        pred_dicts = [head(x) for head in self.heads_list]
        data_dict["final_box_dicts"] = self.generate_predicted_boxes(data_dict["batch_size"], pred_dicts, voxel_indices)
        return data_dict
