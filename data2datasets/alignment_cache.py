"""Persistent group alignment, compatible with the original path-based JSON."""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _source_signature(path, group_dir):
    try:
        stat = path.stat()
        signature = dict(relative_dir=os.path.relpath(path, group_dir), exists=True,
                         is_dir=path.is_dir(), mtime_ns=stat.st_mtime_ns)
        for filename in ('frames_index.npz', 'frames.binpack', 'frames.pcpack', 'frames.gtpack'):
            file = path / filename
            if file.is_file():
                stat = file.stat()
                signature[filename] = [stat.st_size, stat.st_mtime_ns]
        return signature
    except FileNotFoundError:
        return dict(relative_dir=os.path.relpath(path, group_dir), exists=False,
                    is_dir=False, mtime_ns=None)


def _timestamp_ns(name):
    seconds, nanos = Path(name).stem.split('_')
    return int(seconds) * 1_000_000_000 + int(nanos)


def _frame_index(directory, suffix, packed_frame_type):
    """Map logical frame names to current storage; never reuse cached pack offsets."""
    pack_name = {'.bin': 'frames.binpack', '.npy': 'frames.pcpack',
                 '.pkl': 'frames.gtpack'}.get(suffix)
    pack = directory / pack_name if pack_name else None
    index_path = directory / 'frames_index.npz'
    if pack is not None and pack.is_file() and index_path.is_file():
        with np.load(index_path, allow_pickle=False) as index:
            names, times = index['frame_names'], index['timestamps_ns']
            offsets, lengths = index['offsets'], index['lengths']
        count = len(names)
        pack_size = pack.stat().st_size
        if (names.shape != (count,) or any(a.shape != (count,) or a.dtype.kind not in 'iu'
                                          for a in (times, offsets, lengths))
                or np.any(offsets < 0) or np.any(lengths < 0)
                or np.any(offsets > pack_size) or np.any(lengths > pack_size - offsets)
                or (count and (offsets[0] != 0
                    or np.any(offsets[1:] != offsets[:-1] + lengths[:-1])
                    or int(offsets[-1]) + int(lengths[-1]) != pack_size))
                or (not count and pack_size)):
            raise ValueError(f'打包索引长度或偏移无效: {directory}')
        names = list(map(str, names))
        if any(_timestamp_ns(n) != int(t) for n, t in zip(names, times)):
            raise ValueError(f'打包时间戳与帧名不一致: {directory}')
        refs = [packed_frame_type(str(pack), n, int(t), int(o), int(l))
                for n, t, o, l in zip(names, times, offsets, lengths)]
    else:
        names = sorted(n for n in os.listdir(directory) if n.lower().endswith(suffix)) if directory.is_dir() else []
        refs = [str(directory / n) for n in names]
    if (any(Path(n).name != n or not n.lower().endswith(suffix) for n in names)
            or len(set(names)) != len(names)
            or len({_timestamp_ns(n) for n in names}) != len(names)):
        raise ValueError(f'帧名或时间戳重复/无效: {directory}')
    return dict(zip(names, refs))


def _timeline_digest(index):
    timeline = [(name, _timestamp_ns(name)) for name in sorted(index)]
    return hashlib.sha256(json.dumps(timeline, separators=(',', ':')).encode()).hexdigest()


def load_or_build_alignment(group_dir, sources, config, build, packed_frame_type, *, cache_only=False):
    group_dir = Path(group_dir)
    sources = {name: Path(path) for name, path in sources.items() if path is not None}
    if not sources:
        return None if cache_only or build is None else build()
    digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()[:16]
    cache_path = group_dir / 'cache' / f'alignment_{digest}.json'
    needs_upgrade = False
    indices = {}

    def signatures():
        return {name: _source_signature(path, group_dir) for name, path in sorted(sources.items())}

    def load(expected):
        nonlocal needs_upgrade, indices
        needs_upgrade = False
        indices = {}
        if not cache_path.is_file():
            return None
        try:
            payload = json.loads(cache_path.read_text())
            if (not isinstance(payload, dict) or payload.get('schema_version') != 1
                    or payload.get('config') != config):
                return None
            frames = payload['aligned_frames']
            if not isinstance(frames, dict) or set(frames) != set(sources):
                return None
            result = {}
            count = payload['frame_count']
            if type(count) is not int or count < 0:
                return None
            indices = {name: _frame_index(path, config['enabled_sensors'][name], packed_frame_type)
                       for name, path in sources.items()}
            timelines = {name: _timeline_digest(index) for name, index in indices.items()}
            if 'source_timelines' in payload:
                if payload['source_timelines'] != timelines:
                    return None
            else:
                # Legacy caches need the original full timeline once before becoming portable.
                for name, signature in payload['source_signatures'].items():
                    if signature == expected.get(name):
                        continue
                    old_path = group_dir / signature['relative_dir']
                    if _source_signature(old_path, group_dir) != signature:
                        return None
                    old_index = _frame_index(old_path, config['enabled_sensors'][name], packed_frame_type)
                    if _timeline_digest(old_index) != timelines[name]:
                        return None
                if set(payload['source_signatures']) != set(sources):
                    return None
                needs_upgrade = True
            for name, refs in frames.items():
                if not isinstance(refs, list) or len(refs) != count:
                    return None
                result[name] = []
                for ref in refs:
                    if isinstance(ref, str):
                        path = Path(ref)
                        if path.is_absolute() or '..' in path.parts:
                            return None
                        frame_name = path.name
                    elif isinstance(ref, dict):
                        frame_name = ref['frame_name']
                        if (not isinstance(frame_name, str) or Path(frame_name).name != frame_name
                                or type(ref['timestamp_ns']) is not int
                                or ref['timestamp_ns'] != _timestamp_ns(frame_name)):
                            return None
                    else:
                        return None
                    result[name].append(indices[name][frame_name])
            return result if signatures() == expected else None
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f'对齐缓存无法读取，将重新计算: {cache_path}, {exc}')
            return None

    expected = signatures()
    cached = load(expected)
    if cached is not None and (cache_only or not needs_upgrade):
        return cached
    if cache_only:
        return None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock = cache_path.with_suffix('.json.lock').open('a+')
    except OSError as exc:
        print(f'对齐缓存不可写，将直接计算: {cache_path}, {exc}')
        return cached if cached is not None else (build() if build is not None else None)
    with lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        expected = signatures()
        cached = load(expected)
        if cached is not None and not needs_upgrade:
            return cached
        if cached is None and build is None:
            return None
        frames = cached if cached is not None else build()
        if not indices:
            indices = {name: _frame_index(path, config['enabled_sensors'][name], packed_frame_type)
                       for name, path in sources.items()}
        # Do not cache a result if sources changed while alignment was running.
        if signatures() != expected:
            return frames
        relative = {
            name: [asdict(ref) if isinstance(ref, packed_frame_type)
                   else Path(ref).relative_to(group_dir).as_posix() for ref in refs]
            for name, refs in frames.items()
        }
        counts = {len(refs) for refs in relative.values()}
        if set(relative) != set(sources) or len(counts) != 1:
            raise ValueError(f'不能缓存结构无效的对齐结果: {group_dir}')
        payload = dict(schema_version=1, config=config, source_signatures=expected,
                       source_timelines={name: _timeline_digest(index) for name, index in indices.items()},
                       frame_count=counts.pop(), aligned_frames=relative)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cache_path.parent,
                                             prefix=f'.{cache_path.name}.', delete=False) as file:
                temp_path = Path(file.name)
                json.dump(payload, file, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, cache_path)
        except OSError as exc:
            print(f'对齐缓存写入失败，本次仍使用计算结果: {cache_path}, {exc}')
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return frames


def reuse_bin_alignment(group_dir, sources, config, packed_frame_type, *, bin_dir=None):
    """FFTGen keeps BIN timestamps; reuse matching only when both full timelines agree."""
    sources = {name: Path(path) for name, path in sources.items() if path is not None}
    if (set(sources) != {'radar_high_pc', 'gt'}
            or config['base_source'] != 'radar_high_pc'
            or config.get('time_offsets_sec')):
        return None
    pc_dir = sources['radar_high_pc']
    if bin_dir is None:
        bin_dir = pc_dir.parent.parent / 'dpct高位机/Bin'
        if not all((bin_dir / name).is_file() for name in ('frames.binpack', 'frames_index.npz')):
            bin_dir = sources['gt'].parent.parent / 'dpct高位机/Bin'
        if not all((bin_dir / name).is_file() for name in ('frames.binpack', 'frames_index.npz')):
            bin_dir = Path(group_dir) / 'dpct高位机/Bin'
    bin_dir = Path(bin_dir)
    bin_config = {**config, 'base_source': 'radar_high_bin',
                  'enabled_sensors': {'gt': '.pkl', 'radar_high_bin': '.bin'}}
    cached = load_or_build_alignment(
        group_dir, {'gt': sources['gt'], 'radar_high_bin': bin_dir}, bin_config,
        None, packed_frame_type, cache_only=True,
    )
    if cached is None:
        return None
    try:
        bins = _frame_index(bin_dir, '.bin', packed_frame_type)
        pcs = _frame_index(pc_dir, '.npy', packed_frame_type)
        refs = {Path(name).stem: ref for name, ref in pcs.items()}
        if set(refs) != {Path(name).stem for name in bins}:
            return None
        result = {'gt': cached['gt'], 'radar_high_pc': [
            refs[Path(ref.frame_name if isinstance(ref, packed_frame_type) else ref).stem]
            for ref in cached['radar_high_bin']
        ]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None
    print(f'复用 BIN→GT 对齐: {group_dir}，{len(result["gt"])} 帧', flush=True)
    return result
