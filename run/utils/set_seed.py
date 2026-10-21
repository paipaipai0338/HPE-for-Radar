import os
import random
import numpy as np
import torch

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # True 表示强制 cuDNN 只用确定性算法，保证每次运行结果可复现（但可能变慢）；False 表示允许使用非确定性算法，速度更快但结果可能有细微差异、无法完全复现。
    torch.backends.cudnn.deterministic = False
    # 固定输入尺寸 + 追求速度 → 开 benchmark = True；输入尺寸多变或需要结果可复现 → 保持 False
    torch.backends.cudnn.benchmark = False