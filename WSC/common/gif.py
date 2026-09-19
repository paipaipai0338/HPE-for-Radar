from pathlib import Path
from typing import Callable

from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure
from tqdm import tqdm


def export_gif(
    figure: Figure,
    draw_frame: Callable[[int], None],
    frame_count: int,
    path: Path | str,
    fps: int,
    dpi: float | None = None,
) -> None:
    """Draw and save figure frames as a GIF at an optional output resolution."""
    if frame_count < 1 or fps < 1:
        raise ValueError("frame_count and fps must be positive")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(figure, str(output), dpi or figure.dpi):
        for index in tqdm(range(frame_count), desc="export gif", unit="frame"):
            draw_frame(index)
            writer.grab_frame()
