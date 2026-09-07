import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from src.models.module_outputs.processor_outputs import PocPaddingProcessorOutputs
from src.models.module_outputs.voxel_segmentation_outputs import VoxelSegmentationOutput


class PocPaddingProcessor(nn.Module):
    """
    毫米波雷达点云与标签处理器 (PocPaddingProcessor)
    """
    def __init__(self, voxel_config: dict):
        super().__init__()
        self.voxel_config = voxel_config
        
        # 解析体素配置
        self.xlim_np = np.array(voxel_config.get("xlim") or voxel_config["region"]["XLIM"], dtype=np.float32)
        self.ylim_np = np.array(voxel_config.get("ylim") or voxel_config["region"]["YLIM"], dtype=np.float32)
        self.zlim_np = np.array(voxel_config.get("zlim") or voxel_config["region"]["ZLIM"], dtype=np.float32)
        self.voxel_size_np = np.array(voxel_config.get("voxel_size") or voxel_config["resolution"], dtype=np.float32)
        
        # 计算网格形状 [X, Y, Z]
        x_edges = np.arange(self.xlim_np[0], self.xlim_np[1] + self.voxel_size_np[0] * 0.5, self.voxel_size_np[0])
        y_edges = np.arange(self.ylim_np[0], self.ylim_np[1] + self.voxel_size_np[1] * 0.5, self.voxel_size_np[1])
        z_edges = np.arange(self.zlim_np[0], self.zlim_np[1] + self.voxel_size_np[2] * 0.5, self.voxel_size_np[2])
        self.grid_shape_np = np.array([len(x_edges) - 1, len(y_edges) - 1, len(z_edges) - 1], dtype=np.int64)

    def forward(
        self,
        raw_point_cloud: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        gt_mask: Optional[List[List[Union[np.ndarray, torch.Tensor]]]] = None,
        gt_mask_conf: Optional[List[List[Union[np.ndarray, torch.Tensor]]]] = None,
        gt_bbox: Optional[List[List[Union[np.ndarray, torch.Tensor]]]] = None,
    ) -> Union[PocPaddingProcessorOutputs, Tuple[PocPaddingProcessorOutputs, Dict[str, torch.Tensor]]]:
        """
        前向处理接口
        若传入标签，返回 (outputs, gt_dict)；若未传标签（推理模式），仅返回 outputs。
        """
        # 1. 兼容性适配：将单样本输入包装为列表，统一按 Batch 处理
        if isinstance(raw_point_cloud, (np.ndarray, torch.Tensor)) and raw_point_cloud.ndim == 2:
            raw_point_clouds = [raw_point_cloud]
            is_single_sample = True
        else:
            raw_point_clouds = raw_point_cloud
            is_single_sample = False

        # 标签适配
        if gt_mask is not None and is_single_sample and not isinstance(gt_mask[0], (list, tuple)):
            gt_masks = [gt_mask]
            gt_mask_confs = [gt_mask_conf] if gt_mask_conf is not None else None
            gt_bboxes = [gt_bbox] if gt_bbox is not None else None
        else:
            gt_masks = gt_mask
            gt_mask_confs = gt_mask_conf
            gt_bboxes = gt_bbox

        batch_size = len(raw_point_clouds)
        
        # 2. 使用 pad_sequence 高效处理变长点云
        tensor_pts_list = []
        point_lengths = []
        for pc in raw_point_clouds:
            if isinstance(pc, np.ndarray):
                pc = torch.from_numpy(pc).float()
            else:
                pc = pc.float()
            tensor_pts_list.append(pc)
            point_lengths.append(pc.shape[0])

        padded_point_clouds = rnn_utils.pad_sequence(tensor_pts_list, batch_first=True, padding_value=0.0)
        
        max_p = padded_point_clouds.shape[1]
        device = padded_point_clouds.device
        range_tensor = torch.arange(max_p, device=device).unsqueeze(0).expand(batch_size, -1)
        lens_tensor = torch.tensor(point_lengths, device=device).unsqueeze(1)
        point_valid_mask = range_tensor < lens_tensor

        # 3. 将空间配置转换为 Tensor 存入输出类中，确保全员皆为 Tensor
        outputs = PocPaddingProcessorOutputs(
            raw_point_cloud=padded_point_clouds,
            point_valid_mask=point_valid_mask,
            xlim=torch.tensor(self.xlim_np, dtype=torch.float32, device=device),
            ylim=torch.tensor(self.ylim_np, dtype=torch.float32, device=device),
            zlim=torch.tensor(self.zlim_np, dtype=torch.float32, device=device),
            voxel_size=torch.tensor(self.voxel_size_np, dtype=torch.float32, device=device),
            grid_shape=torch.tensor(self.grid_shape_np, dtype=torch.long, device=device)
        )

        # 4. 标签清洗与对齐填充 (Training 模式)
        if gt_bboxes is not None:
            processed_bboxes = []
            processed_masks = []
            processed_confs = []
            
            for i in range(batch_size):
                b_boxes = gt_bboxes[i] if i < len(gt_bboxes) else []
                b_masks = gt_masks[i] if gt_masks is not None and i < len(gt_masks) else []
                b_confs = gt_mask_confs[i] if gt_mask_confs is not None and i < len(gt_mask_confs) else []
                
                valid_boxes, valid_masks, valid_confs = [], [], []
                
                for b, m, c in zip(b_boxes, b_masks, b_confs):
                    b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.array(b)
                    if np.isnan(b_np).any():
                        continue
                        
                    valid_boxes.append(torch.tensor(b_np, dtype=torch.float32) if not isinstance(b, torch.Tensor) else b)
                    valid_masks.append(torch.from_numpy(m) if isinstance(m, np.ndarray) else m)
                    valid_confs.append(torch.from_numpy(c) if isinstance(c, np.ndarray) else c)
                    
                processed_bboxes.append(valid_boxes)
                processed_masks.append(valid_masks)
                processed_confs.append(valid_confs)
                
            max_instances = max([len(b) for b in processed_bboxes]) if processed_bboxes else 0
            max_instances = max(max_instances, 1)
            
            X, Y, Z = self.grid_shape_np
            padded_bboxes = torch.zeros((batch_size, max_instances, 6), dtype=torch.float32, device=device)
            padded_masks = torch.zeros((batch_size, max_instances, X, Y, Z), dtype=torch.float32, device=device)
            padded_confs = torch.zeros((batch_size, max_instances, X, Y, Z), dtype=torch.float32, device=device)
            instance_valid_mask = torch.zeros((batch_size, max_instances), dtype=torch.bool, device=device)
            
            for i in range(batch_size):
                num_inst = len(processed_bboxes[i])
                if num_inst > 0:
                    padded_bboxes[i, :num_inst] = torch.stack(processed_bboxes[i]).to(device)
                    padded_masks[i, :num_inst] = torch.stack(processed_masks[i]).float().to(device)
                    padded_confs[i, :num_inst] = torch.stack(processed_confs[i]).float().to(device)
                    instance_valid_mask[i, :num_inst] = True
                    
            gt_dict = {
                "gt_bboxes": padded_bboxes,                 # [B, max_M, 6]
                "gt_masks": padded_masks,                   # [B, max_M, X, Y, Z]
                "gt_mask_confs": padded_confs,              # [B, max_M, X, Y, Z]
                "gt_instance_valid_mask": instance_valid_mask # [B, max_M]
            }
            return outputs, gt_dict
        else:
            return outputs

    def post_process_instance_segmentation(
        self,
        outputs: VoxelSegmentationOutput,
        threshold: float,
        mask_threshold: float,
    ):
        pred_logits = outputs.cls_logits[-1]  # (batch_size, num_queries)
        pred_boxes = outputs.pred_boxes[-1]  # (batch_size, num_queries, 4) in xyxy format
        pred_masks = outputs.pred_masks  # (batch_size, num_queries, height, width)
        presence_logits = outputs.presence_logits[-1]  # (batch_size, 1) or None

        batch_scores = pred_logits.sigmoid()

        batch_masks = pred_masks.sigmoid()

        batch_boxes = pred_boxes

        results = []

        for idx, (scores, boxes, masks) in enumerate(zip(batch_scores, batch_boxes, batch_masks)):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            masks = masks[keep]  # (num_keep, height, width)

            masks = (masks > mask_threshold).to(torch.long)

            results.append({"scores": scores, "boxes": boxes, "masks": masks})

        return results

    def post_process_object_detection(
        self,
        outputs: VoxelSegmentationOutput,
        threshold: float,
    ):
        pred_logits = outputs.cls_logits[-1]  # (batch_size, num_queries)
        pred_boxes = outputs.pred_boxes[-1]  # (batch_size, num_queries, 4) in xyxy format
        presence_logits = outputs.presence_logits[-1]  # (batch_size, 1) or None

        batch_scores = pred_logits.sigmoid()

        batch_boxes = pred_boxes

        results = []

        for idx, (scores, boxes) in enumerate(zip(batch_scores, batch_boxes)):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]

            results.append({"scores": scores, "boxes": boxes})

        return results
#