import torch
from transformers import Trainer
from typing import Dict, Any, Optional, Tuple, Union


class MiliHuPsenTrainer(Trainer):
    """
    针对 3D 毫米波人体感知任务 (MiliHuPsen) 定制的 Hugging Face Trainer。
    
    核心特性:
    1. 适配 TrainingPipelineWrapper 的前向输入输出协议；
    2. 传递 global_step 动态驱动多任务 Loss 调度器；
    3. 自定义滑动窗口多指标聚合器 (_custom_metrics_tracker)，将分项及层级损失平滑写入 TensorBoard / WandB；
    4. 适配评估/验证阶段的 prediction_step，支持纯 Loss 计算与模型预测输出。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_metrics_tracker: Dict[str, float] = {}
        self._tracking_steps: int = 0

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        统一计算当前 Batch 的总损失，并收集分项损失指标。
        """
        # 1. 前向传播：传入 state.global_step 驱动动态 Loss 调度
        # inputs 已经由 DataLoader 的 collate_fn 整理好并放置于对应计算设备
        total_loss, loss_dict = model(
            global_step=self.state.global_step,
            **inputs
        )

        # 2. 多 GPU / DDP 降维保护
        if isinstance(total_loss, torch.Tensor) and total_loss.dim() > 0:
            total_loss = total_loss.mean()

        # 3. 训练阶段指标累加（用于 logging 平滑平均）
        if self.args.should_log and model.training and isinstance(loss_dict, dict):
            for key, val in loss_dict.items():
                if isinstance(val, torch.Tensor):
                    val_mean = val.mean().item() if val.dim() > 0 else val.item()
                else:
                    val_mean = float(val)
                self._custom_metrics_tracker[key] = self._custom_metrics_tracker.get(key, 0.0) + val_mean
            self._tracking_steps += 1

        # 4. 按 HF Trainer 规范返回
        return (total_loss, None) if return_outputs else total_loss

    def prediction_step(
        self,
        model: Any,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[Any], Optional[Any]]:
        """
        评估/验证阶段的前向推理与指标收集。
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
            if loss is not None and loss.dim() > 0:
                loss = loss.mean()

        return (loss, None, None)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        重写日志输出函数，将积累的详细损失分项计算算术平均后合并到 logs 中，
        随后清空计数器，交由父类写盘（TensorBoard / WandB / 控制台）。
        """
        if self._tracking_steps > 0:
            for key, val in self._custom_metrics_tracker.items():
                logs[key] = round(val / self._tracking_steps, 4)

            # 重置追踪状态
            self._custom_metrics_tracker = {}
            self._tracking_steps = 0

        super().log(logs, *args, **kwargs)