from torch.optim import AdamW
from src.models import OPTIM

@OPTIM.register("AdamW")
def optimizer_AdamW(
        model,lr=5e-4,weight_decay=0.05,
        betas=(0.9,0.999),eps=1e-8,groups=None
    ):
    if groups is None:
        groups={}
    param_buckets={}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. 初始赋予全局基础参数
        current_lr = lr
        current_wd = weight_decay

        # 2. 模块级覆盖 (根据 custom_rules 匹配名称)
        # 如果参数名包含设定的关键字，则覆盖掉基础参数
        for keyword, rule in groups.items():
            if keyword in name:
                current_lr = rule.get('lr', current_lr)
                current_wd = rule.get('weight_decay', current_wd)
        if len(param.shape) == 1 or name.endswith(".bias"):
            current_wd = 0.0  # 基础硬规则
            if "no_decay" in groups:
                current_lr = groups["no_decay"].get('lr', current_lr)
                current_wd = groups["no_decay"].get('weight_decay', 0.0)

        bucket_key = (current_lr, current_wd)
        if bucket_key not in param_buckets:
            param_buckets[bucket_key] = []
        param_buckets[bucket_key].append(param)

    param_groups = []
    for (lr, wd), params in param_buckets.items():
        param_groups.append({
            'params': params,
            'lr': lr,
            'weight_decay': wd
        })

    # 实例化并返回优化器
    optimizer = AdamW(param_groups, betas=betas, eps=eps)
    return optimizer