from pathlib import Path
import numpy as np


def get_realsense_data(path: Path|str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    data = raw.reshape(-1, 3)
    xyz_new = np.zeros_like(data)
    xyz_new[:, 0] = data[:, 2]  # X ← Z (前)
    xyz_new[:, 1] = -data[:, 0]  # Y ← -X (左)
    xyz_new[:, 2] = -data[:, 1]  # Z ← -Y (上)
    return xyz_new