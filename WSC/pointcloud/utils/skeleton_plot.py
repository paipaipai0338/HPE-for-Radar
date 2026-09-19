from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def configure_skeleton_axes(
    ax: Axes,
    axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    title: str,
) -> None:
    ax.set(xlim=axis_limits[0], ylim=axis_limits[1], zlim=axis_limits[2], title=title)
    ax.set_xlabel("low radar x (m)")
    ax.set_ylabel("low radar y (m)")
    ax.set_zlabel("low radar z (m)")
    ax.view_init(elev=18, azim=-65)


def draw_skeleton(ax: Axes, pose: np.ndarray, skeleton: list[list[int]]) -> None:
    pose = np.asarray(pose, dtype=np.float64)
    finite = np.isfinite(pose).all(axis=1)
    if not finite.any():
        return
    ax.scatter(*pose[finite].T, c="#222222", s=14, depthshade=False)
    for start, end in skeleton:
        if finite[start] and finite[end]:
            ax.plot(*pose[[start, end]].T, c="#222222", linewidth=1.5)
