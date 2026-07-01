from pathlib import Path
import numpy as np
import pickle

def get_gt_data(path: Path|str) -> np.ndarray:
    with open(path, 'rb') as ff:
        gt = pickle.load(ff)
    has_nan = np.isnan(gt).any(axis=(1, 2))  # 形状 (a,)
    gt = gt[~has_nan]  # 形状 (a-1, b, c)
    return gt