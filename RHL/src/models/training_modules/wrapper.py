
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Dict, List, Union, Any, Tuple
from src.models.module_outputs.processor_outputs import PocPaddingProcessorOutputs
from src.models.module_outputs.voxel_segmentation_outputs import (
    VoxelSegmentationOutput, DetrMiliHupSenOutput
)

class TrainingPipelineWrapper(nn.Module):
    def __init__(self, model, matcher, loss_calculators, **kwargs):
        super().__init__()
        self.model = model
        self.matcher = matcher
        self.loss_calculators = loss_calculators

    def forward(
        self,
        current_data: PocPaddingProcessorOutputs,
        history_points_list: Optional[List[PocPaddingProcessorOutputs]] = None,
        h_state: Optional[torch.Tensor] = None,
        gt_dict: Optional[Dict[str, torch.Tensor]] = None,
        global_step: int = 0,
        **kwargs: Any,
    ) -> Union[VoxelSegmentationOutput, Tuple[torch.Tensor, Dict[str, torch.Tensor], VoxelSegmentationOutput]]:
        
        # 1. 模型前向推理
        model_outputs: VoxelSegmentationOutput = self.model(
            current_data=current_data,
            history_points_list=history_points_list,
            h_state=h_state,
            **kwargs
        )

        # 2. 匈牙利二分图匹配
        pred_dict = {
            "cls_logits": model_outputs.cls_logits,
            "pred_boxes": model_outputs.pred_boxes,
            "pred_masks": model_outputs.pred_masks,
            "semantic_seg": model_outputs.semantic_seg
        }

        matched_indices = self.matcher(pred_dict=pred_dict, gt_dict=gt_dict)

        # 3. 打包所有上下文参数
        context_dict = {
            "cls_logits": model_outputs.cls_logits,
            "pred_boxes": model_outputs.pred_boxes,
            "pred_masks": model_outputs.pred_masks,
            "semantic_seg": model_outputs.semantic_seg,
            "gt_dict": gt_dict,
            "matched_indices": matched_indices,
            "return_dict": True,
            "global_step": global_step,
        }

        # 4. 计算并累加 Loss
        total_loss = 0.0
        loss_dict = {}

        for i, loss_calculator in enumerate(self.loss_calculators):
            if hasattr(loss_calculator, "step"):
                loss_calculator.step(global_step)
            component_loss, layer_losses_dict = loss_calculator(**context_dict)

            loss_name = getattr(loss_calculator, "name", f"loss_{i}")
            total_loss += component_loss

            # 记录到 loss_dict
            loss_dict[f"Loss_Components/{loss_name}_total"] = component_loss.detach()
            for lvl_name, lvl_loss in layer_losses_dict.items():
                loss_dict[f"Loss_Details/{loss_name}_{lvl_name}"] = lvl_loss.detach() if isinstance(lvl_loss, torch.Tensor) else lvl_loss

        loss_dict["Loss_Total/train_loss"] = total_loss.detach()

        return total_loss, loss_dict

class DetrTrainingPipelineWrapper(TrainingPipelineWrapper):
    def __init__(self, model, matcher, loss_calculators, **kwargs):
        super().__init__(model, matcher, loss_calculators, **kwargs)


    def forward(
        self,
        current_data: PocPaddingProcessorOutputs,
        history_points_list: Optional[List[PocPaddingProcessorOutputs]] = None,
        h_state: Optional[torch.Tensor] = None,
        gt_dict: Optional[Dict[str, torch.Tensor]] = None,
        global_step: int = 0,
        **kwargs: Any,
    ):
        # 1. 模型前向推理
        model_outputs: DetrMiliHupSenOutput = self.model(
            current_data=current_data,
            history_points_list=history_points_list,
            h_state=h_state,
            **kwargs
        )
        # 2. 匈牙利二分图匹配
        pred_dict = {
            "cls_logits": model_outputs.cls_logits,
            "pred_boxes": model_outputs.pred_boxes,
        }
        matched_indices = self.matcher(pred_dict=pred_dict, gt_dict=gt_dict)
        # 3. 打包所有上下文参数
        context_dict = {
            "cls_logits": model_outputs.cls_logits,
            "pred_boxes": model_outputs.pred_boxes,
            "gt_dict": gt_dict,
            "matched_indices": matched_indices,
            "return_dict": True,
            "global_step": global_step,
        }
        # 4. 计算并累加 Loss
        total_loss = 0.0
        loss_dict = {}
        
        for i, loss_calculator in enumerate(self.loss_calculators):
            if hasattr(loss_calculator, "step"):
                loss_calculator.step(global_step)
            component_loss, layer_losses_dict = loss_calculator(**context_dict)
        
            loss_name = getattr(loss_calculator, "name", f"loss_{i}")
            total_loss += component_loss
        
            # 记录到 loss_dict
            loss_dict[f"Loss_Components/{loss_name}_total"] = component_loss.detach()
            for lvl_name, lvl_loss in layer_losses_dict.items():
                loss_dict[f"Loss_Details/{loss_name}_{lvl_name}"] = lvl_loss.detach() if isinstance(lvl_loss, torch.Tensor) else lvl_loss
        
        loss_dict["Loss_Total/train_loss"] = total_loss.detach()
        
        return total_loss, loss_dict


#       