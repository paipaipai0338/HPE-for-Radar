import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


DEFAULT_BAD_FRAMES_JSON = (
    Path(__file__).resolve().parents[1]
    / "data2datasets"
    / "bad_bin_frames.json"
)


def frame_identity(index_path: Path, frame_name: str) -> dict:
    parts = index_path.parts
    marker = parts.index("data_collection")
    return {
        "date": parts[marker - 1],
        "group": parts[marker + 1],
        "sensor": parts[marker + 2],
        "frame": frame_name,
    }


def decoded_abs_stats(raw: np.ndarray) -> dict:
    raw8 = raw.reshape(-1, 8)[:, ::-1].reshape(-1)
    packed = np.frombuffer(raw8.tobytes(), dtype="<u4")
    exponent = (packed >> 28).astype(np.int32)
    real = (packed & 0x3FFF).astype(np.int32)
    imag = ((packed >> 14) & 0x3FFF).astype(np.int32)
    real[real >= 1 << 13] -= 1 << 14
    imag[imag >= 1 << 13] -= 1 << 14
    scale = np.exp2(exponent - 13).astype(np.float32)
    magnitude = np.hypot(real.astype(np.float32), imag.astype(np.float32)) * scale
    return {
        "abs_mean": float(magnitude.mean()),
        "abs_p99": float(np.quantile(magnitude, 0.99)),
        "abs_max": float(magnitude.max()),
    }


def write_bad_frames_manifest(
    path: Path,
    source_root: Path,
    expected_frame_bytes: int,
    thresholds: dict,
    numeric_frames: list,
    length_frames: list,
) -> None:
    bad_frames = []
    for row in numeric_frames:
        bad_frames.append({
            "date": row["date"],
            "group": row["group"],
            "sensor": row["sensor"],
            "frame_index": row["frame_index"],
            "frame_name": row["frame"],
            "group_frame_count": row["group_frame_count"],
            "bad_type": "numeric",
            "details": {
                "max_exponent": row["max_exponent"],
                "fraction_exp_eq15": row["fraction_exp_eq15"],
                "abs_mean": row["abs_mean"],
                "abs_max": row["abs_max"],
            },
        })
    for row in length_frames:
        bad_frames.append({
            "date": row["date"],
            "group": row["group"],
            "sensor": row["sensor"],
            "frame_index": row["frame_index"],
            "frame_name": row["frame"],
            "group_frame_count": row["group_frame_count"],
            "bad_type": "length",
            "details": {
                "actual_bytes": row["actual_bytes"],
                "expected_bytes": row["expected_bytes"],
            },
        })
    bad_frames.sort(
        key=lambda row: (
            row["date"], row["group"], row["frame_index"], row["bad_type"]
        )
    )
    manifest = {
        "schema_version": 1,
        "source_root": str(source_root),
        "generated_by": "run/check_bin.py",
        "expected_frame_bytes": expected_frame_bytes,
        "numeric_thresholds": thresholds,
        "summary": {
            "numeric": len(numeric_frames),
            "length": len(length_frames),
            "total": len(bad_frames),
        },
        "bad_frames": bad_frames,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def scan_pack(index_path: Path, chunk_frames: int, args) -> tuple[list, dict]:
    pack_path = index_path.with_name("frames.binpack")
    with np.load(index_path, allow_pickle=False) as index:
        names = np.asarray(index["frame_names"]).astype(str)
        offsets = np.asarray(index["offsets"], dtype=np.int64)
        lengths = np.asarray(index["lengths"], dtype=np.int64)

    frame_count = len(names)
    if not frame_count or not pack_path.is_file():
        raise ValueError("empty index or missing frames.binpack")
    if not (len(offsets) == len(lengths) == frame_count):
        raise ValueError("index field lengths differ")
    if offsets[0] != 0 or np.any(offsets[1:] != offsets[:-1] + lengths[:-1]):
        raise ValueError("frame offsets are not contiguous")
    if offsets[-1] + lengths[-1] != pack_path.stat().st_size:
        raise ValueError("index does not cover frames.binpack exactly")

    frame_bytes = args.expected_frame_bytes
    if frame_bytes % 8:
        raise ValueError("expected frame length is not 8-byte aligned")
    valid_frame = lengths == frame_bytes
    valid_indices = np.flatnonzero(valid_frame)
    values_per_frame = frame_bytes // 4
    mean_exponent = np.zeros(frame_count, dtype=np.float64)
    count_ge4 = np.zeros(frame_count, dtype=np.int64)
    count_ge8 = np.zeros(frame_count, dtype=np.int64)
    count_eq15 = np.zeros(frame_count, dtype=np.int64)
    max_exponent = np.zeros(frame_count, dtype=np.uint8)
    packed = np.memmap(pack_path, mode="r", dtype=np.uint8)

    runs = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) != 1) + 1)
    for run in runs:
        for position in range(0, len(run), chunk_frames):
            indices = run[position:position + chunk_frames]
            raw = packed[offsets[indices[0]]: offsets[indices[-1]] + frame_bytes]
            blocks = raw.reshape(len(indices), frame_bytes // 8, 8)
            exponent_sum = np.zeros(len(indices), dtype=np.int64)
            ge4 = np.zeros(len(indices), dtype=np.int64)
            ge8 = np.zeros(len(indices), dtype=np.int64)
            eq15 = np.zeros(len(indices), dtype=np.int64)
            maximum = np.zeros(len(indices), dtype=np.uint8)
            for byte_column in (0, 4):
                exponent = blocks[:, :, byte_column] >> 4
                exponent_sum += exponent.sum(axis=1, dtype=np.int64)
                ge4 += np.count_nonzero(exponent >= 4, axis=1)
                ge8 += np.count_nonzero(exponent >= 8, axis=1)
                eq15 += np.count_nonzero(exponent == 15, axis=1)
                maximum = np.maximum(maximum, exponent.max(axis=1))
            mean_exponent[indices] = exponent_sum / values_per_frame
            count_ge4[indices] = ge4
            count_ge8[indices] = ge8
            count_eq15[indices] = eq15
            max_exponent[indices] = maximum

    group_median = float(np.median(mean_exponent[valid_frame]))
    group_mad = float(np.median(np.abs(mean_exponent[valid_frame] - group_median)))
    relative_cutoff = group_median + max(
        args.min_mean_exponent_jump,
        args.mad_multiplier * 1.4826 * group_mad,
    )
    candidate_mask = valid_frame & (
        (mean_exponent > relative_cutoff)
        | (count_ge8 / values_per_frame >= args.min_ge8_fraction)
        | (count_eq15 / values_per_frame >= args.min_eq15_fraction)
    )

    candidates = []
    for frame_index in np.flatnonzero(candidate_mask):
        start = int(offsets[frame_index])
        raw = np.asarray(packed[start:start + frame_bytes])
        row = frame_identity(index_path, names[frame_index])
        row.update({
            "frame_index": int(frame_index),
            "group_frame_count": frame_count,
            "mean_exponent": float(mean_exponent[frame_index]),
            "group_median_mean_exponent": group_median,
            "group_mad_mean_exponent": group_mad,
            "fraction_exp_ge4": float(count_ge4[frame_index] / values_per_frame),
            "fraction_exp_ge8": float(count_ge8[frame_index] / values_per_frame),
            "fraction_exp_eq15": float(count_eq15[frame_index] / values_per_frame),
            "max_exponent": int(max_exponent[frame_index]),
            "index_path": str(index_path),
            "pack_path": str(pack_path),
        })
        row.update(decoded_abs_stats(raw))
        candidates.append(row)

    malformed = []
    for frame_index in np.flatnonzero(~valid_frame):
        row = frame_identity(index_path, names[frame_index])
        row.update({
            "frame_index": int(frame_index),
            "group_frame_count": frame_count,
            "actual_bytes": int(lengths[frame_index]),
            "expected_bytes": frame_bytes,
            "index_path": str(index_path),
            "pack_path": str(pack_path),
        })
        malformed.append(row)

    summary = {
        "indexed_frames": frame_count,
        "scanned_frames": int(valid_frame.sum()),
        "bytes": int(lengths.sum()),
        "frame_bytes": frame_bytes,
        "malformed": malformed,
        "max_exponent_histogram": {
            str(value): int(np.count_nonzero(max_exponent[valid_frame] == value))
            for value in np.unique(max_exponent[valid_frame])
        },
    }
    return candidates, summary


def self_check() -> None:
    raw = np.zeros((2, 8), dtype=np.uint8)
    raw[:, 0] = [0x30, 0xF0]
    raw[:, 4] = [0x80, 0x10]
    assert (raw[:, 0] >> 4).tolist() == [3, 15]
    assert (raw[:, 4] >> 4).tolist() == [8, 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan packed radar BIN frames for numeric and length errors"
    )
    parser.add_argument("--root", type=Path, default=Path("/mnt/ssd/Huawei"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expected-frame-bytes", type=int, default=1_048_576)
    parser.add_argument(
        "--bad-frames-json",
        type=Path,
        default=DEFAULT_BAD_FRAMES_JSON,
    )
    parser.add_argument("--min-mean-exponent-jump", type=float, default=0.1)
    parser.add_argument("--mad-multiplier", type=float, default=20.0)
    parser.add_argument("--min-ge8-fraction", type=float, default=1e-3)
    parser.add_argument("--min-eq15-fraction", type=float, default=1e-4)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return

    index_paths = sorted(args.root.rglob("frames_index.npz"))
    candidates = []
    errors = []
    total_indexed_frames = total_scanned_frames = total_bytes = 0
    max_exponent_histogram = {}
    malformed = []
    def collect(index_path, result=None, error=None):
        nonlocal total_indexed_frames, total_scanned_frames, total_bytes
        if error is None:
            rows, summary = result
            candidates.extend(rows)
            malformed.extend(summary["malformed"])
            total_indexed_frames += summary["indexed_frames"]
            total_scanned_frames += summary["scanned_frames"]
            total_bytes += summary["bytes"]
            for exponent, count in summary["max_exponent_histogram"].items():
                max_exponent_histogram[exponent] = max_exponent_histogram.get(exponent, 0) + count
        else:
            errors.append({"index_path": str(index_path), "error": f"{type(error).__name__}: {error}"})

    if args.workers == 1:
        results = []
        for index_path in index_paths:
            try:
                results.append((index_path, scan_pack(index_path, args.chunk_frames, args), None))
            except Exception as error:
                results.append((index_path, None, error))
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        futures = {
            executor.submit(scan_pack, index_path, args.chunk_frames, args): index_path
            for index_path in index_paths
        }
        results = (
            (futures[future], future.result(), None)
            if future.exception() is None
            else (futures[future], None, future.exception())
            for future in as_completed(futures)
        )

    for group_index, (index_path, result, error) in enumerate(results, 1):
        collect(index_path, result, error)
        if group_index % 20 == 0 or group_index == len(index_paths):
            print(
                f"groups={group_index}/{len(index_paths)} frames={total_scanned_frames} "
                f"candidates={len(candidates)} errors={len(errors)}",
                flush=True,
            )
    if args.workers != 1:
        executor.shutdown()

    candidates.sort(key=lambda row: (row["date"], row["group"], row["frame_index"]))
    report = {
        "root": str(args.root),
        "groups": len(index_paths),
        "indexed_frames": total_indexed_frames,
        "scanned_frames": total_scanned_frames,
        "bytes": total_bytes,
        "thresholds": {
            "min_mean_exponent_jump": args.min_mean_exponent_jump,
            "mad_multiplier": args.mad_multiplier,
            "min_ge8_fraction": args.min_ge8_fraction,
            "min_eq15_fraction": args.min_eq15_fraction,
        },
        "max_exponent_histogram": dict(sorted(max_exponent_histogram.items(), key=lambda item: int(item[0]))),
        "candidate_groups": len({(row["date"], row["group"], row["sensor"]) for row in candidates}),
        "candidate_frames": len(candidates),
        "candidates": candidates,
        "malformed_frames": malformed,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    if candidates:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=candidates[0].keys())
            writer.writeheader()
            writer.writerows(candidates)
    malformed_csv_path = args.output.with_name(f"{args.output.stem}_malformed.csv")
    if malformed:
        with malformed_csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=malformed[0].keys())
            writer.writeheader()
            writer.writerows(malformed)
    if errors:
        raise RuntimeError(
            f"scan contains {len(errors)} group errors; "
            "bad frame manifest was not updated"
        )
    write_bad_frames_manifest(
        path=args.bad_frames_json,
        source_root=args.root,
        expected_frame_bytes=args.expected_frame_bytes,
        thresholds=report["thresholds"],
        numeric_frames=candidates,
        length_frames=malformed,
    )
    print(
        f"report={args.output}\ncsv={csv_path}\n"
        f"bad_frames_json={args.bad_frames_json}"
    )


if __name__ == "__main__":
    main()
