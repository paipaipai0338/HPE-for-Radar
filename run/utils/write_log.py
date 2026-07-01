from datetime import datetime
from pathlib import Path


def write_log(log_path, message, show=True):
    """
    将日志追加写入 log.txt，同时可选输出到终端。
    """
    log_path = Path(log_path)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    time_str = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    message = str(message)
    line = f"[{time_str}] {message}"

    if show:
        print(line)

    with log_path.open(
        mode="a",
        encoding="utf-8",
    ) as file:
        file.write(line + "\n")