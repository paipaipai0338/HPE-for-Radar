from pathlib import Path

import torch


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(save_path, epoch, model, optimizer, scheduler, metric, best_metric, scaler=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metric_state_dict": {
            "train": metric["train"].state_dict(),
            "val": metric["val"].state_dict(),
        },
        "best_metric": best_metric,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, save_path)


def load_training_checkpoint(checkpoint_path, model, optimizer, scheduler, metric, device, scaler=None):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    metric["train"].load_state_dict(checkpoint["metric_state_dict"]["train"])
    metric["val"].load_state_dict(checkpoint["metric_state_dict"]["val"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_metric = checkpoint["best_metric"]

    return start_epoch, best_metric


def load_model_checkpoint(checkpoint_path, model, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return checkpoint


def load_init_checkpoint(checkpoint_path, model):
    """仅在权重完全匹配时初始化模型；失败则保留原始权重。"""
    original_state = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state = checkpoint.get('model_state_dict', checkpoint)
        current = model.state_dict()
        if state.keys() != current.keys() or any(
            not torch.is_tensor(state[key])
            or state[key].shape != value.shape
            or state[key].dtype != value.dtype
            or state[key].layout != value.layout
            for key, value in current.items()
        ):
            print(f'跳过 init_checkpoint: {checkpoint_path}，模型权重不完全匹配')
            return False

        # strict=True 也可能先写入部分权重再报错，保留备份用于回退。
        original_state = {key: value.detach().cpu().clone() for key, value in current.items()}
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        if original_state is not None:
            model.load_state_dict(original_state, strict=True)
        print(f'跳过 init_checkpoint: {checkpoint_path}，保留原始初始化权重: {exc}')
        return False

    print(f'已加载 init_checkpoint 模型权重: {checkpoint_path}')
    return True
