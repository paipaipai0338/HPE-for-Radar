import math
import torch


def get_cosine_schedule_with_warmup(
    optimizer,
    num_epochs,
    warmup_epoch,
    min_lr=0.0,
):
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if warmup_epoch < 0:
        raise ValueError(f"warmup_epoch must be non-negative, got {warmup_epoch}")
    if warmup_epoch >= num_epochs:
        raise ValueError(
            f"warmup_epoch must be smaller than num_epochs, "
            f"got warmup_epoch={warmup_epoch}, num_epochs={num_epochs}"
        )

    base_lr = optimizer.param_groups[0]['lr']
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch):
        if warmup_epoch > 0 and epoch < warmup_epoch:
            return float(epoch + 1) / float(max(1, warmup_epoch))
        progress = (epoch - warmup_epoch) / float(max(1, num_epochs - warmup_epoch))
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
