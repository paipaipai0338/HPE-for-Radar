"""Online point preparation, compact trajectory indices and ordered P4 fine-tuning steps."""
import hashlib
import importlib.util
import inspect
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from run.utils.load_config import load_config
from run.utils.p4_sequence_data import HPE_Dataset, collate_fn, align_window, group_fingerprint
from run.utils.process_one_epoch import prepare_person_pointcloud

META = ('quality', 'rotation', 'translation', 'baseline', 'target', 'timestamp')


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2]/path


class OnlineHPE(HPE_Dataset):
    """Reuse overlapping raw frames without retaining the entire dataset in each worker."""
    def __init__(self, **kwargs):
        self.recent_frames = OrderedDict()
        super().__init__(**kwargs)

    def _get_cached_frame_data(self, sensor_name, path, load_fn):
        from data2datasets.dataset_for_all_task import PackedFrame
        key = (sensor_name, path)
        frame = self.recent_frames.pop(key, None)
        if frame is None:
            frame = self._load_packed_frame(path) if isinstance(path, PackedFrame) else load_fn(path)
        self.recent_frames[key] = frame
        if len(self.recent_frames) > 256:
            self.recent_frames.popitem(last=False)
        return self._copy_cached_data(frame)


def make_raw_dataset(training, mode):
    d = training['data']
    return OnlineHPE(root_path=d['root_path'], packed_data_root=d['packed_data_root'],
        sensor_config=d['sensor_config'], mode=mode, base_source=d['base_source'], split_method='person',
        ratio=d['ratio'], T=d['T'], acc_frame=d['acc_frame'], preload_cache=False, enable_action=False,
        pointcloud_source='fftgen', val_person_id=d.get('val_person_id', '4'), window_stride=1,
        train_groups=d.get('train_groups') if mode == 'train' else None)


def prepare_input_index(config):
    experiment = resolve(config['baseline_experiment'])
    training = load_config(experiment/'config/config.yaml')
    if (training['data']['pointcloud_source'] != 'fftgen' or training['data']['split_method'] != 'person'
            or training['task'].get('center_on_pointcloud') or training['task'].get('center_xy_in_low')):
        raise ValueError('Fine-tuning expects the existing uncentered FFTGen person split')
    source = json.loads((resolve(config['init_experiment'])/'evidence_manifest.json').read_text())
    architecture = load_config(experiment/'config/model_config.yaml')
    binding = hashlib.sha256(Path(inspect.getfile(HPE_Dataset)).read_bytes()).hexdigest()
    manifest, datasets = {}, {}
    for mode in ('train', 'val'):
        dataset = datasets[mode] = make_raw_dataset(training, mode)
        groups = defaultdict(dict)
        for index, group in enumerate(dataset.sample_groups):
            stamp = dataset._frame_timestamp_ns(dataset.data_path_list['radar_high_pc'][index][-1])*1e-9
            if stamp in groups[group]:
                raise ValueError('Duplicate window endpoint within a group')
            groups[group][stamp] = index
        entries = list(source[mode].items())
        if config.get('smoke'):
            entries = [next((g, v) for g, v in entries if v['frames'])]
        context = dict(training=training, mode=mode, feature_dim=architecture['feat_dim'],
            input_batch_size=config['input_batch_size'], tracking={k:config[k] for k in (
                'max_frame_gap_seconds', 'track_max_center_distance', 'track_max_joint_distance')})
        manifest[mode] = {}
        for group, info in tqdm(entries, desc=f'{mode} window indices'):
            evidence_path = Path(info['cache'])
            evidence = torch.load(evidence_path, map_location='cpu', weights_only=False)
            allowed = set(list(groups[group])[:128]) if config.get('smoke') else set(groups[group])
            records = []
            for stream_id, stream in enumerate(evidence):
                stamps = stream['timestamp'].tolist()
                if not config.get('smoke') and not set(stamps) <= allowed:
                    raise ValueError(f'Source frames disappeared from {group}')
                rows = [i for i, stamp in enumerate(stamps) if stamp in allowed]
                if rows:
                    records.append(dict(source_stream=stream_id, source_rows=torch.tensor(rows),
                                        window_indices=torch.tensor([groups[group][stamps[i]] for i in rows])))
            del evidence
            if not records:
                if info['frames']:
                    raise ValueError(f'No source frames found in {group}')
                continue
            indices = sorted({int(i) for r in records for i in r['window_indices']})
            stat = evidence_path.stat()
            fingerprint = group_fingerprint(dataset, indices, binding, dict(version=2,
                evidence=(str(evidence_path), stat.st_size, stat.st_mtime_ns), context=context))
            index_path = resolve(config['index_dir'])/mode/group/(fingerprint+'.pt')
            if not index_path.is_file():
                index_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(records, index_path.with_suffix('.tmp'))
                index_path.with_suffix('.tmp').replace(index_path)
            manifest[mode][group] = dict(index=str(index_path), cache=str(evidence_path), online=context,
                frames=sum(len(r['source_rows']) for r in records), streams=len(records))
        expected = sum(v['frames'] for v in source[mode].values())
        if not config.get('smoke') and sum(v['frames'] for v in manifest[mode].values()) != expected:
            raise ValueError('Prepared split does not cover the complete source evidence')
    if set(manifest['train']) & set(manifest['val']):
        raise ValueError('Training and validation groups overlap')
    return manifest, datasets


def load_streams(entries):
    streams = []
    for info in entries.values():
        source = torch.load(info['cache'], map_location='cpu', weights_only=False)
        for record in torch.load(info['index'], map_location='cpu', weights_only=False):
            original = source[record['source_stream']]
            rows = record['source_rows']
            streams.append(dict(group=original['group'], online=info['online'],
                window_indices=record['window_indices'], **{k:original[k][rows] for k in META}))
    return streams


class OnlinePoints:
    def __init__(self, context, dataset=None):
        self.context = context
        self.dataset = dataset if dataset is not None else make_raw_dataset(context['training'], context['mode'])
        d = context['training']['data']
        self.shape = (d['T'], d['person_max_points'], 3+context['feature_dim'])

    def read(self, stream, start, size):
        output = torch.empty(size, *self.shape)
        config = self.context['tracking']
        training = self.context['training']
        data = {**training['data'], 'augmentation': {}}
        for begin in range(0, size, self.context['input_batch_size']):
            end = min(size, begin+self.context['input_batch_size'])
            indices = stream['window_indices'][start+begin:start+end].tolist()
            samples = collate_fn([self.dataset[i] for i in indices], max_people=data['max_people'])
            samples = align_window(samples, config)
            inputs, gt, valid, _, _, _ = prepare_person_pointcloud(samples, training['task'], data, torch.device('cpu'), 'val')
            complete = inputs['mask'].any(-1).all(-1) & gt['mask'].permute(0, 2, 1)[valid].all(-1)
            slots = valid.nonzero()
            for b, index in enumerate(indices):
                position = start+begin+b
                refs = self.dataset.data_path_list['radar_high_pc'][index]
                times = [self.dataset._frame_timestamp_ns(ref)*1e-9 for ref in refs]
                if (self.dataset.sample_groups[index] != stream['group'] or times[-1] != float(stream['timestamp'][position])
                        or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > config['max_frame_gap_seconds'])):
                    raise ValueError('Online window no longer matches the pinned trajectory')
                rotation, translation = samples['high_to_low_R'][b, -1], samples['high_to_low_t'][b, -1]
                if not (torch.allclose(rotation, stream['rotation'][position], atol=1e-6, rtol=0)
                        and torch.allclose(translation, stream['translation'][position], atol=1e-6, rtol=0)):
                    raise ValueError('Calibration changed since the source experiment')
                candidates = []
                for row in ((slots[:, 0] == b) & complete).nonzero().flatten().tolist():
                    pose = gt['padded'][b, -1, int(slots[row, 1])] @ rotation.T+translation
                    if torch.allclose(pose, stream['target'][position], atol=2e-5, rtol=0):
                        candidates.append(row)
                if len(candidates) != 1:
                    raise ValueError('Missing or ambiguous online identity; refusing to mix trajectory states')
                row = candidates[0]
                if not inputs['mask'][row].all():
                    raise ValueError('Complete windows must have valid resampled points in every slot')
                output[begin+b] = inputs['input'][row, ..., :self.shape[-1]]
        return output


class PointStepSampler(Sampler):
    """Keep one chronological trajectory in each lane, without resetting at gradient boundaries."""
    def __init__(self, streams, length, batch_size, shuffle):
        self.lengths = [len(s['target']) for s in streams]
        self.length, self.batch_size, self.shuffle = length, batch_size, shuffle
        if not self.lengths or min(self.lengths+[length, batch_size]) < 1:
            raise ValueError('Require nonempty trajectories and positive chunk/batch sizes')

    def __iter__(self):
        order = torch.randperm(len(self.lengths)).tolist() if self.shuffle else range(len(self.lengths))
        pending = iter(order)
        lanes = [None]*self.batch_size
        while True:
            for lane, current in enumerate(lanes):
                if current is None:
                    index = next(pending, None)
                    lanes[lane] = (index, 0) if index is not None else None
            if all(v is None for v in lanes):
                return
            yield tuple(lanes)
            for lane, current in enumerate(lanes):
                if current is not None:
                    index, start = current
                    lanes[lane] = (index, start+self.length) if start+self.length < self.lengths[index] else None


class PointSteps(Dataset):
    def __init__(self, streams, length, raw_dataset=None):
        self.streams, self.length = streams, length
        self.online = OnlinePoints(streams[0]['online'], raw_dataset)
        self.point_shape = self.online.shape

    def __len__(self):
        return len(self.streams)

    def __getitem__(self, requests):
        b, length = len(requests), self.length
        result = {k:self.streams[0][k].new_zeros(b, length, *self.streams[0][k].shape[1:])
                  for k in META if k != 'timestamp'}
        result.update(points=torch.zeros(b, length, *self.point_shape), valid=torch.zeros(b, length, dtype=torch.bool),
                      reset=torch.ones(b, dtype=torch.bool), position=torch.zeros(b, length, dtype=torch.long), group=['']*b)
        for lane, request in enumerate(requests):
            if request is None:
                continue
            index, start = request
            stream = self.streams[index]
            size = min(length, len(stream['target'])-start)
            result['points'][lane, :size] = self.online.read(stream, start, size)
            for key in META:
                if key != 'timestamp':
                    result[key][lane, :size] = stream[key][start:start+size]
            result['valid'][lane, :size] = True
            result['reset'][lane] = start == 0
            result['position'][lane, :size] = torch.arange(start, start+size)
            result['group'][lane] = stream['group']
        return result


def make_loader(streams, config, device, shuffle, raw_dataset=None):
    sampler = PointStepSampler(streams, config['chunk_length'], config['batch_size'], shuffle)
    return DataLoader(PointSteps(streams, config['chunk_length'], raw_dataset), sampler=sampler, batch_size=None,
                      num_workers=config['num_workers'], pin_memory=device.type == 'cuda',
                      persistent_workers=config['num_workers'] > 0)


def load_saved_joint_model(experiment, saved, device):
    classes = []
    for filename, name in (('P4Transformer.py', 'P4Transformer'), ('P4JointMemory.py', 'P4JointMemory'),
                           ('P4JointFineTune.py', 'P4JointFineTune')):
        spec = importlib.util.spec_from_file_location('joint_snapshot_'+name, experiment/filename)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        classes.append(getattr(module, name))
    frontend = classes[0](**saved['encoder_architecture'])
    decoder = classes[1](**saved['architecture'])
    model = classes[2](frontend, decoder).to(device)
    model.encoder_batch_size = saved['config'].get('encoder_batch_size', 64)
    model.checkpoint_encoder = saved['config'].get('checkpoint_encoder', False)
    model.load_state_dict(saved['model_state_dict'])
    return model


@torch.inference_mode()
def predict_saved_joint(model, stream, device, length=16, online=None):
    device = torch.device(device)
    if online is None:
        online = OnlinePoints(stream['online'])
    state, output = None, []
    for start in range(0, len(stream['target']), length):
        size = min(length, len(stream['target'])-start)
        batch = {k:stream[k][None, start:start+size].to(device) for k in ('quality', 'rotation', 'translation')}
        batch['points'] = online.read(stream, start, size)[None].to(device)
        batch['valid'] = torch.ones(1, size, device=device, dtype=torch.bool)
        # pointnet2 kernels launch on the current CUDA device, not just the tensor's device.
        with torch.cuda.device_of(batch['points']), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            encoded = model.encode(batch)
        pose, state = model.decoder(encoded, state)
        output.append(pose[0].float().cpu())
    return torch.cat(output).numpy()
