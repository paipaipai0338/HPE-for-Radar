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
