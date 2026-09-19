"""Pack radar BIN, PC and GT files per group without changing their bytes."""

import argparse
import hashlib
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


PACK_NAMES = {".bin": "frames.binpack", ".npy": "frames.pcpack", ".pkl": "frames.gtpack"}
SENSOR_DIRS = {
    "dpct高位机/Bin": ".bin", 
    # "dpct低位机/Bin": ".bin",
    "dpct高位机/PC": ".npy", 
    # "dpct低位机/PC": ".npy",
    "camera results/smoothed 3D": ".pkl",
}
INDEX_NAME = "frames_index.npz"
CHUNK_SIZE = 16 * 1024 * 1024


def timestamp_ns(path: Path) -> int:
    second, nanosecond = path.stem.split("_")
    return int(second) * 1_000_000_000 + int(nanosecond)


def discover_data_dirs(root: Path, dates: list[str] | None) -> list[tuple[Path, str]]:
    requested = set(dates) if dates else None
    directories = []
    for date_dir in root.iterdir():
        if requested is not None and date_dir.name not in requested:
            continue
        collection_dir = date_dir / "data_collection"
        if not collection_dir.is_dir():
            continue
        for group_dir in collection_dir.iterdir():
            for relative_dir, suffix in SENSOR_DIRS.items():
                data_dir = group_dir / relative_dir
                if data_dir.is_dir():
                    directories.append((data_dir, suffix))
    return sorted(directories)


def source_frames(data_dir: Path, suffix: str) -> list[Path]:
    frames = []
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        try:
            timestamp_ns(path)
        except (ValueError, TypeError):
            continue
        frames.append(path)
    return sorted(frames)


def existing_pack_matches(
    pack_path: Path,
    index_path: Path,
    frames: list[Path],
) -> bool:
    if not pack_path.is_file() or not index_path.is_file():
        return False
    try:
        with np.load(index_path, allow_pickle=False) as index:
            names = np.asarray(index["frame_names"]).astype(str).reshape(-1)
            offsets = np.asarray(index["offsets"], dtype=np.int64).reshape(-1)
            lengths = np.asarray(index["lengths"], dtype=np.int64).reshape(-1)
            timestamps = np.asarray(index["timestamps_ns"], dtype=np.int64).reshape(-1)
            hashes = np.asarray(index["sha256"]).astype(str).reshape(-1)
        expected_names = [path.name for path in frames]
        expected_lengths = np.asarray([path.stat().st_size for path in frames])
        expected_timestamps = np.asarray([timestamp_ns(path) for path in frames])
        count = len(frames)
        return (
            len(names) == len(offsets) == len(lengths)
            == len(timestamps) == len(hashes) == count
            and names.tolist() == expected_names
            and np.array_equal(lengths, expected_lengths)
            and np.array_equal(timestamps, expected_timestamps)
            and (count == 0 or offsets[0] == 0)
            and (count < 2 or np.all(offsets[1:] == offsets[:-1] + lengths[:-1]))
            and (count == 0 or offsets[-1] + lengths[-1] == pack_path.stat().st_size)
        )
    except (OSError, ValueError, KeyError):
        return False


def pack_group(source_dir: Path, target_dir: Path, frames: list[Path], pack_name: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    pack_path = target_dir / pack_name
    index_path = target_dir / INDEX_NAME
    pack_tmp = target_dir / f".{pack_name}.tmp"
    index_tmp = target_dir / f".{INDEX_NAME}.tmp"

    offsets = []
    lengths = []
    hashes = []
    offset = 0
    with pack_tmp.open("wb") as output:
        for frame in frames:
            digest = hashlib.sha256()
            offsets.append(offset)
            length = 0
            with frame.open("rb") as source:
                while chunk := source.read(CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    length += len(chunk)
            lengths.append(length)
            hashes.append(digest.hexdigest())
            offset += length
        output.flush()
        os.fsync(output.fileno())

    packed = (np.memmap(pack_tmp, mode="r", dtype=np.uint8)
              if offset else np.empty(0, dtype=np.uint8))
    for frame_index, expected_hash in enumerate(hashes):
        start = offsets[frame_index]
        end = start + lengths[frame_index]
        actual_hash = hashlib.sha256(packed[start:end]).hexdigest()
        if actual_hash != expected_hash:
            raise IOError(f"打包后字节校验失败: {frames[frame_index]}")
    del packed

    with index_tmp.open("wb") as output:
        np.savez(
            output,
            frame_names=np.asarray([path.name for path in frames]),
            timestamps_ns=np.asarray([timestamp_ns(path) for path in frames], dtype=np.int64),
            offsets=np.asarray(offsets, dtype=np.int64),
            lengths=np.asarray(lengths, dtype=np.int64),
            sha256=np.asarray(hashes),
        )
        output.flush()
        os.fsync(output.fileno())

    os.replace(pack_tmp, pack_path)
    os.replace(index_tmp, index_path)


def process_directory(
    source_dir: Path, suffix: str, source_root: Path, target_root: Path,
    overwrite: bool = False,
) -> tuple[str, int]:
    pack_name = PACK_NAMES[suffix]
    target_dir = target_root / source_dir.relative_to(source_root)
    pack_path = target_dir / pack_name
    index_path = target_dir / INDEX_NAME
    if not overwrite:
        # Skip before listing/stat-ing individual NAS frames.
        if pack_path.is_file() and index_path.is_file():
            return "reused", 0
        if pack_path.exists() or index_path.exists():
            raise FileExistsError(
                f"目标打包文件不完整，保留现有文件: {target_dir}；"
                "确认需要重建时使用 --overwrite"
            )
    frames = source_frames(source_dir, suffix)
    if not frames:
        return "empty", 0
    pack_group(source_dir, target_dir, frames, pack_name)
    if not existing_pack_matches(pack_path, index_path, frames):
        raise RuntimeError(f"最终打包文件校验失败: {target_dir}")
    return "packed", len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/huawei"))
    parser.add_argument("--target-root", type=Path, default=Path("/mnt/ssd/Huawei"))
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--overwrite", action="store_true", help="显式重建已有打包数据，默认直接跳过")
    parser.add_argument("--workers", type=int, default=8, help="并行打包线程数，默认 1")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须大于 0")

    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    if source_root == target_root:
        parser.error("source-root 和 target-root 必须不同")
    directories = discover_data_dirs(source_root, args.dates)
    print(f"Discovered {len(directories)} BIN/PC/GT directories", flush=True)

    counts = {"packed": 0, "reused": 0, "empty": 0}
    total_frames = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_directory, source_dir, suffix,
                            source_root, target_root, args.overwrite): source_dir
            for source_dir, suffix in directories
        }
        for group_index, future in enumerate(as_completed(futures), 1):
            source_dir = futures[future]
            action, frame_count = future.result()
            counts[action] += 1
            total_frames += frame_count
            print(
                f"[{group_index}/{len(directories)}] {action} "
                f"new_frames={frame_count} {source_dir.relative_to(source_root)}",
                flush=True,
            )

    print(
        f"Finished: packed={counts['packed']}, reused={counts['reused']}, "
        f"empty={counts['empty']}, new_frames={total_frames}",
        flush=True,
    )


if __name__ == "__main__":
    main()
