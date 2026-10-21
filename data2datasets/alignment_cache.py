"""Persistent group alignment, compatible with the original path-based JSON."""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def load_or_build_alignment(group_dir, sources, config, build, packed_frame_type):
    group_dir = Path(group_dir)
    sources = {name: Path(path) for name, path in sources.items() if path is not None}
    if not sources:
        return build()
    digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()[:16]
    cache_path = group_dir / 'cache' / f'alignment_{digest}.json'

    def signatures():
        result = {}
        for name, path in sorted(sources.items()):
            try:
                stat = path.stat()
                signature = dict(relative_dir=os.path.relpath(path, group_dir), exists=True,
                                 is_dir=path.is_dir(), mtime_ns=stat.st_mtime_ns)
                # Packed timestamps and offsets can change without changing directory mtime.
                for filename in ('frames_index.npz', 'frames.binpack', 'frames.pcpack', 'frames.gtpack'):
                    file = path / filename
                    if file.is_file():
                        stat = file.stat()
                        signature[filename] = [stat.st_size, stat.st_mtime_ns]
                result[name] = signature
            except FileNotFoundError:
                result[name] = dict(relative_dir=os.path.relpath(path, group_dir), exists=False,
                                    is_dir=False, mtime_ns=None)
        return result

    def load(expected):
        if not cache_path.is_file():
            return None
        try:
            payload = json.loads(cache_path.read_text())
            if (not isinstance(payload, dict) or payload.get('schema_version') != 1
                    or payload.get('config') != config
                    or payload.get('source_signatures') != expected):
                return None
            frames = payload['aligned_frames']
            if not isinstance(frames, dict) or set(frames) != set(sources):
                return None
            result = {}
            count = payload['frame_count']
            if type(count) is not int or count < 0:
                return None
            for name, refs in frames.items():
                if not isinstance(refs, list) or len(refs) != count:
                    return None
                result[name] = []
                for ref in refs:
                    if isinstance(ref, str):
                        path = Path(ref)
                        if path.is_absolute() or '..' in path.parts:
                            return None
                        result[name].append(str(group_dir / path))
                    elif isinstance(ref, dict):
                        pack = Path(ref['pack_path'])
                        frame_name = Path(ref['frame_name'])
                        if (pack.parent != sources[name] or pack.name not in
                                {'frames.binpack', 'frames.pcpack', 'frames.gtpack'}
                                or frame_name.name != ref['frame_name']
                                or not ref['frame_name']
                                or any(type(ref[k]) is not int for k in ('timestamp_ns', 'offset', 'length'))
                                or ref['offset'] < 0 or ref['length'] < 0
                                or ref['offset'] + ref['length'] > expected[name].get(pack.name, [-1])[0]):
                            return None
                        result[name].append(packed_frame_type(**ref))
                    else:
                        return None
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'对齐缓存无法读取，将重新计算: {cache_path}, {exc}')
            return None

    expected = signatures()
    cached = load(expected)
    if cached is not None:
        return cached
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock = cache_path.with_suffix('.json.lock').open('a+')
    except OSError as exc:
        print(f'对齐缓存不可写，将直接计算: {cache_path}, {exc}')
        return build()
    with lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        expected = signatures()
        cached = load(expected)
        if cached is not None:
            return cached
        frames = build()
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
