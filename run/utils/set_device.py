import torch

def set_device(dev_id):
    torch.cuda.set_device(dev_id)
    device = torch.device(f"cuda:{dev_id}")
    return device