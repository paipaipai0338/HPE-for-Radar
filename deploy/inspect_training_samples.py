"""遍历训练集并点击 MPJPE 曲线检查人帧；结果只保存在内存。

python -B deploy/inspect_training_samples.py --experiment experiments/P4Transformer/20261019_161657
可用 --dataset-cache 指定已有 train.pkl，避免重建索引；--self-test 无需 GPU。
左图缩放后点击选样本，左右键逐个查看，右图拖动旋转；Sample 输入框可直接跳转。
一个样本是一个窗口内的有效人帧；重叠窗口中的同一物理帧会分别评分。
右侧只画该时刻的单帧点云；模型预测仍使用训练配置中的累积点云。
默认浏览 MPJPE >= P95 的样本（误差最大的约 5%，边界同值全部包含）；界面可修改误差范围。
右侧 XYZ 范围固定读取 run/config.yaml 的 data.xyz_limits。
"""

import argparse
from functools import partial
import importlib.util
import itertools
import json
from pathlib import Path
import pickle
import re
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import data2datasets.dataset_for_all_task as dataset_module
from data2datasets.alignment_cache import load_or_build_alignment
from data2datasets.load_json import get_meta_info
from metrics.pose import get_mpjpe
from preprocess.actionprocess import LABEL_NAMES, classify_actions
from preprocess.radarprocess import Radar_Config
from run.utils.checkpoint import load_model_checkpoint
from run.utils.load_config import load_config
from run.utils.process_one_epoch import (
    adapt_coordinate_frame, prepare_person_pointcloud, prepare_pose_target,
    predict_valid_person_frames, validate_coordinate_frame, validate_center_xy_in_low,
)
from utils.skeleton import get_pose_layout


def load_module(path):
    spec = importlib.util.spec_from_file_location('_training_inspector_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_experiment(experiment, checkpoint, device):
    cfg = load_config(experiment / 'config/config.yaml')
    if cfg['task']['input'] != 'radar_high_pc' or cfg['task']['output'] != 'gt_for_high':
        raise ValueError('此查看器支持 radar_high_pc / gt_for_high 点云姿态模型')
    # Review the original observations, without random transforms or point dropout.
    cfg['data']['augmentation'] = {}
    validate_coordinate_frame(cfg['task'], cfg['data'])
    validate_center_xy_in_low(cfg['task'], cfg['data'])
    name = cfg['model']['name']
    source = experiment / f'config/{name}.py'
    if not source.is_file():
        source = ROOT / f'models/{name}/{name}.py'
        print(f'未找到保存的模型源码，使用当前源码：{source}')
    model = getattr(load_module(source), name)(**load_config(experiment / 'config/model_config.yaml'))
    saved = load_model_checkpoint(experiment / f'checkpoint/{checkpoint}.pth', model, 'cpu')
    print(f'模型：{name}；checkpoint={checkpoint}.pth；epoch={saved["epoch"] + 1}；FP32')
    return model.to(device).eval(), cfg


def readonly_alignment(group_dir, sources, config, build, frame_type):
    cached = load_or_build_alignment(group_dir, sources, config, build, frame_type, cache_only=True)
    return cached if cached is not None else build()


def load_training_dataset(experiment, cfg, cache=None):
    data = cfg['data']
    manifest = experiment / 'config/data_description.json'
    if not manifest.is_file():
        manifest = ROOT / 'data2datasets/data description.json'
        print(f'注意：实验未保存划分清单，使用当前训练清单：{manifest}')
    groups = data.get('train_groups')
    report_path = experiment / 'config/dataset_report.json'
    if 'class3' in cfg:
        groups = json.loads(report_path.read_text())['groups']['train']
    meta = get_meta_info(manifest, mode='train', train_groups=groups)
    allowed = {f"{entry['date']}/{g}" for entries in meta.values() for entry in entries for g in entry['valid_group']}
    if cache is not None:
        with cache.open('rb') as stream:
            dataset = pickle.load(stream)
        expected = dict(T=data['T'], acc_frame=data['acc_frame'],
                        pointcloud_source=data.get('pointcloud_source', 'files'),
                        base_source=data['base_source'])
        if dataset.mode != 'train' or any(getattr(dataset, k) != v for k, v in expected.items()):
            raise ValueError('train.pkl 的训练模式、T、累积帧数或数据源与模型配置不一致')
        if data.get('window_stride') is not None and dataset.window_stride != data['window_stride']:
            raise ValueError('train.pkl 的 window_stride 与实验配置不一致')
        dataset.filter_prone_windows(data.get('include_prone', True))
    else:
        radar = Radar_Config()
        for key, value in cfg.get('radar', {}).items():
            if not hasattr(radar, key):
                raise ValueError(f'未知雷达配置：{key}')
            setattr(radar, key, value)
        radar.__post_init__()
        keys = ('root_path', 'packed_data_root', 'sensor_config', 'base_source', 'T', 'acc_frame',
                'pointcloud_source', 'include_prone', 'window_stride')
        with patch.object(dataset_module, 'get_meta_info', partial(get_meta_info, manifest)), \
             patch.object(dataset_module, 'load_or_build_alignment', readonly_alignment):
            dataset = dataset_module.HPE_Dataset(
                **{k: data[k] for k in keys if k in data}, mode='train', train_groups=groups,
                preload_cache=False, enable_action=False, radar_config=radar)
    dataset.preload_cache = False
    dataset.clear_data_cache()
    if set(dataset.sample_groups) - allowed:
        raise ValueError('索引包含实验训练清单以外的组，拒绝混入验证数据')
    missing = allowed - set(dataset.sample_groups)
    if missing:
        print(f'注意：清单中有 {len(missing)} 个组未出现在索引中，例如：{sorted(missing)[:8]}')
    log = experiment / 'log/group_metrics.txt'
    if log.is_file() and set(dataset.sample_groups) & set(re.findall(r'Val group (\S+) \|', log.read_text())):
        raise ValueError('训练索引与该实验历史验证组重叠')
    indices = list(range(len(dataset)))
    if 'class3' in cfg:
        target = LABEL_NAMES.index(cfg['class3']['posture'])
        classifier_source = experiment / 'config/actionprocess.py'
        classifier = load_module(classifier_source).classify_actions if classifier_source.is_file() else classify_actions
        frame_ok = {}
        for index in tqdm(indices, desc='Checking posture windows'):
            cal = dataset._load_calib_T(dataset.sample_groups[index].split('/')[0])['gt_to_low']
            for ref in dataset.data_path_list['gt'][index]:
                if ref not in frame_ok:
                    pose = dataset._get_sensor_data_from_path('gt', [ref])[0]
                    frame_ok[ref] = len(pose) == 1 and int(classifier(pose @ cal['R'].T + cal['t']).labels[0].argmax()) == target
        indices = [i for i in indices if all(frame_ok[ref] for ref in dataset.data_path_list['gt'][i])]
        expected = json.loads(report_path.read_text())['splits']['train']['windows']
        if len(indices) != expected:
            raise ValueError(f'姿态训练窗口与保存报告不一致：当前 {len(indices)}，保存 {expected}')
    if not indices:
        raise ValueError('训练集没有可检查的窗口')
    print(f'训练索引：{len(indices):,} 个窗口；{len(set(dataset.sample_groups))} 个组；'
          f'T={dataset.T}；stride={dataset.window_stride}；acc_frame={dataset.acc_frame}')
    return dataset, indices


@torch.inference_mode()
def evaluate(model, dataset, indices, cfg, device, batch_size=32, workers=2):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == 'cuda',
                        collate_fn=partial(dataset_module.collate_fn, max_people=cfg['data']['max_people']))
    records, predictions = [], []
    offset = 0
    for samples in tqdm(loader, desc='Training-set MPJPE', mininterval=1.):
        size = len(samples['group_id'])
        samples = adapt_coordinate_frame(samples, cfg['task'], cfg['data'], 'val')
        inputs, gt, instances, _, _, center = prepare_person_pointcloud(
            samples, cfg['task'], cfg['data'], device, 'val')
        if instances.any():
            output = predict_valid_person_frames(model, inputs)['pose']
            b, k = instances.nonzero(as_tuple=True)
            gt = prepare_pose_target(gt, cfg['task'].get('pose_format', 'coco'))
            truth, valid = gt['padded'][b, :, k], gt['mask'][b, :, k]
            if output.shape != truth.unsqueeze(2).shape or not torch.isfinite(output).all():
                raise ValueError('模型输出 shape 错误或含 NaN/Inf')
            pose = output[:, :, 0].float()
            errors = get_mpjpe(pose, truth, cfg['task'].get('pose_format', 'coco'))
            person, t = valid.nonzero(as_tuple=True)
            batch_indices = b[person].cpu().numpy()
            chunk = np.empty(len(t), dtype=[('window', 'i8'), ('frame', 'i4'), ('person', 'i4'), ('mpjpe', 'f4')])
            chunk['window'] = np.asarray(indices[offset:offset + size])[batch_indices]
            chunk['frame'], chunk['person'] = t.cpu().numpy(), k[person].cpu().numpy()
            chunk['mpjpe'] = errors[person, t].cpu().numpy() * 100
            records.append(chunk)
            absolute_pose = pose[person, t] + center[b[person], t, k[person]][:, None]
            predictions.append(absolute_pose.cpu().numpy())
        offset += size
    if offset != len(indices) or not records:
        raise ValueError('未完成训练集遍历或没有满足 GT 裁剪点数 >=10 的人帧')
    records, predictions = np.concatenate(records), np.concatenate(predictions)
    print(f'有效人帧：{len(records):,}；MPJPE={records["mpjpe"].mean():.2f} cm；'
          f'P95={np.quantile(records["mpjpe"], .95):.2f} cm；最大={records["mpjpe"].max():.2f} cm')
    return records, predictions


def display_sample(dataset, cfg, record, prediction):
    index, t, k = (int(record[key]) for key in ('window', 'frame', 'person'))
    samples = dataset_module.collate_fn([dataset[index]], max_people=cfg['data']['max_people'])
    samples = adapt_coordinate_frame(samples, cfg['task'], cfg['data'], 'val')
    _, gt, _, _, _, center = prepare_person_pointcloud(samples, cfg['task'], cfg['data'], 'cpu', 'val')
    gt = prepare_pose_target(gt, cfg['task'].get('pose_format', 'coco'))
    if 'radar_high_pc' in dataset.data_path_list:
        ref = dataset.data_path_list['radar_high_pc'][index][t]
        points = dataset._get_sensor_data_from_path('radar_high_pc', [ref])[0][:, :3]
    else:
        # Online point generation still needs its warm-up history, without accumulating output frames.
        with patch.object(dataset, '_accumulate_pointcloud_sequence',
                          side_effect=lambda frames, history_count, acc_frame: frames[history_count:]):
            points = dataset[index]['radar_high_pc'][t][:, :3]
    if cfg['task'].get('coordinate', 'high') == 'low':
        rotation, translation = samples['high_to_low_R'][0, t].numpy(), samples['high_to_low_t'][0, t].numpy()
        points = points @ rotation.T + translation
    truth = (gt['padded'][0, t, k] + center[0, t, k]).numpy()
    bounds = gt['bbox'][0, t, k].numpy() + np.tile(center[0, t, k].numpy(), 2)
    inside = ((points >= bounds[:3]) & (points <= bounds[3:])).all(-1)
    corners = np.array(list(itertools.product(*zip(bounds[:3], bounds[3:]))))
    if cfg['task'].get('coordinate', 'high') == 'high':
        rotation, translation = samples['high_to_low_R'][0, t].numpy(), samples['high_to_low_t'][0, t].numpy()
        points, truth, prediction, corners = [a @ rotation.T + translation for a in (points, truth, prediction, corners)]
    ref = dataset.data_path_list[dataset.base_source][index][t]
    name = ref.frame_name if hasattr(ref, 'frame_name') else Path(ref).name
    return dict(points=points, inside=inside, truth=truth, prediction=prediction, corners=corners,
                group=dataset.sample_groups[index], frame_name=name)


class Viewer:
    def __init__(self, dataset, cfg, records, predictions, title):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox

        for action, key in (('back', 'left'), ('forward', 'right')):
            plt.rcParams[f'keymap.{action}'] = [k for k in plt.rcParams[f'keymap.{action}'] if k != key]
        self.dataset, self.cfg, self.records, self.predictions = dataset, cfg, records, predictions
        self.edges = get_pose_layout(cfg['task'].get('pose_format', 'coco'))['edges']
        self.xyz_limits = np.asarray(load_config(ROOT / 'run/config.yaml')['data']['xyz_limits'], dtype=float)
        if (self.xyz_limits.shape != (3, 2) or not np.isfinite(self.xyz_limits).all()
                or not (self.xyz_limits[:, 1] > self.xyz_limits[:, 0]).all()):
            raise ValueError('run/config.yaml 的 data.xyz_limits 必须为三个有限且递增的坐标范围')
        self.fig = plt.figure(figsize=(17, 8))
        grid = self.fig.add_gridspec(1, 2, width_ratios=[1.1, 1], left=.06, right=.98, bottom=.17, top=.88)
        self.curve = self.fig.add_subplot(grid[0])
        self.scene = self.fig.add_subplot(grid[1], projection='3d')
        self.curve.plot(records['mpjpe'], lw=.6, color='#aeb4bb', alpha=.6, label='All samples')
        self.filtered, = self.curve.plot([], [], '.', color='#e45c3a', ms=2, label='In error range')
        self.lower_line = self.curve.axhline(0, color='#e45c3a', ls='--', lw=.8)
        self.upper_line = self.curve.axhline(0, color='#e45c3a', ls='--', lw=.8)
        self.curve.set(xlabel='Sample index (window / frame / person)', ylabel='MPJPE (cm)',
                       title=f'{len(records):,} scored person-frames | mean {records["mpjpe"].mean():.2f} cm')
        self.curve.grid(alpha=.2)
        self.curve.legend(loc='upper right', fontsize=8)
        self.cursor = self.curve.axvline(0, color='#e45c3a', lw=1)
        self.marker, = self.curve.plot([], [], 'o', color='#e45c3a', ms=6)
        self.box = TextBox(self.fig.add_axes([.09, .06, .16, .045]), 'Sample ', initial='0')
        self.box.on_submit(self.submit)
        lower, upper = float(np.quantile(records['mpjpe'], .95)), float(records['mpjpe'].max())
        self.min_box = TextBox(self.fig.add_axes([.09, .12, .11, .035]), 'Min cm ', initial=str(lower))
        self.max_box = TextBox(self.fig.add_axes([.29, .12, .11, .035]), 'Max cm ', initial=str(upper))
        self.range_info = self.fig.text(.43, .13, '', fontsize=9)
        self.min_box.on_submit(self.set_range)
        self.max_box.on_submit(self.set_range)
        self.fig.suptitle(title, fontsize=12)
        self.info = self.fig.text(.29, .07, '', fontsize=9)
        self.fig.text(.06, .01, 'Enter Min/Max cm to filter. Click orange samples; Left/Right: previous/next eligible sample. Drag 3D to rotate.', fontsize=10)
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        self.index = -1
        self.set_range()
        print(f'默认可视化（误差最大的约 5%）：{self.range_description}', flush=True)

    def set_range(self, text=None):
        try:
            lower, upper = float(self.min_box.text), float(self.max_box.text)
            if not np.isfinite([lower, upper]).all() or not 0 <= lower <= upper:
                raise ValueError
        except ValueError:
            self.range_info.set_text('Invalid range: require finite 0 <= Min <= Max. Previous filter retained.')
            self.fig.canvas.draw_idle()
            return
        errors = self.records['mpjpe']
        allowed = np.flatnonzero((errors >= lower) & (errors <= upper))
        if not len(allowed):
            self.range_info.set_text('No samples in this range. Previous filter retained.')
            self.fig.canvas.draw_idle()
            return
        self.allowed = allowed
        self.range_description = (f'Range [{lower:.4f}, {upper:.4f}] cm | '
                                  f'{len(allowed):,} / {len(errors):,} samples ({len(allowed)/len(errors):.1%})')
        self.range_info.set_text(self.range_description)
        self.filtered.set_data(allowed, errors[allowed])
        self.lower_line.set_ydata([lower, lower])
        self.upper_line.set_ydata([upper, upper])
        position = np.searchsorted(allowed, self.index)
        if position == len(allowed) or allowed[position] != self.index:
            self.select(int(allowed[0]))
        self.fig.canvas.draw_idle()

    def select(self, index):
        if not 0 <= index < len(self.records):
            return
        position = np.searchsorted(self.allowed, index)
        if position == len(self.allowed) or self.allowed[position] != index:
            self.range_info.set_text(f'Sample {index} is outside the active error range.')
            self.fig.canvas.draw_idle()
            return
        row = self.records[index]
        sample = display_sample(self.dataset, self.cfg, row, self.predictions[index])
        self.index = index
        self.range_info.set_text(self.range_description)
        self.cursor.set_xdata([index, index])
        self.marker.set_data([index], [row['mpjpe']])
        self.box.eventson = False
        self.box.set_val(str(index))
        self.box.eventson = True
        axis = self.scene
        elevation, azimuth = axis.elev, axis.azim
        axis.clear()
        points, inside = sample['points'], sample['inside']
        axis.scatter(*points[~inside].T, c='#aeb4bb', s=3, alpha=.35, label='Outside GT crop', axlim_clip=True)
        axis.scatter(*points[inside].T, c='#1674df', s=8, alpha=.85, label='Inside GT crop', axlim_clip=True)
        for pose, color, label in [(sample['truth'], '#15956b', 'GT'), (sample['prediction'], '#e45c3a', 'Prediction')]:
            for j, (a, b) in enumerate(self.edges):
                axis.plot(*pose[[a, b]].T, color=color, lw=2, label=label if j == 0 else None, axlim_clip=True)
            axis.scatter(*pose.T, c=color, s=12, axlim_clip=True)
        corners = sample['corners']
        for a, b in itertools.combinations(range(8), 2):
            if (a ^ b) in (1, 2, 4):
                axis.plot(*corners[[a, b]].T, c='#1674df', lw=.7, alpha=.65, axlim_clip=True)
        axis.set(xlim=self.xyz_limits[0], ylim=self.xyz_limits[1], zlim=self.xyz_limits[2],
                 xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                 title=f'{sample["group"]} | MPJPE {row["mpjpe"]:.2f} cm\nLow-radar coordinates | {inside.sum()} / {len(points)} crop / scene points')
        axis.set_box_aspect(self.xyz_limits[:, 1] - self.xyz_limits[:, 0])
        axis.view_init(elev=elevation, azim=azimuth)
        axis.legend(loc='upper left', fontsize=8)
        self.info.set_text(f'Sample {index} | window {row["window"]}, frame {row["frame"]}, person {row["person"]}\n'
                           f'{sample["frame_name"]} | single-frame scene; prediction uses training accumulation')
        self.fig.canvas.draw_idle()

    def submit(self, text):
        try:
            index = int(text)
        except ValueError:
            return
        self.select(index)

    def click(self, event):
        if event.inaxes != self.curve or event.button != 1 or event.xdata is None:
            return
        if self.fig.canvas.toolbar is not None and self.fig.canvas.toolbar.mode:
            return
        half_width = max(1, abs(self.curve.get_xlim()[1] - self.curve.get_xlim()[0]) * 6 / self.curve.bbox.width)
        start, end = max(0, int(event.xdata-half_width)), min(len(self.records), int(event.xdata+half_width)+1)
        indices = self.allowed[np.searchsorted(self.allowed, start):np.searchsorted(self.allowed, end)]
        if len(indices):
            xy = self.curve.transData.transform(np.column_stack((indices, self.records['mpjpe'][indices])))
            distance = np.square(xy - [event.x, event.y]).sum(-1)
            nearest = distance.argmin()
            if distance[nearest] <= 12**2:
                self.select(int(indices[nearest]))

    def key(self, event):
        if event.inaxes in (self.box.ax, self.min_box.ax, self.max_box.ax):
            return
        if event.key in ('left', 'right'):
            position = np.searchsorted(self.allowed, self.index) + (1 if event.key == 'right' else -1)
            if 0 <= position < len(self.allowed):
                self.select(int(self.allowed[position]))


def self_test():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    from torch import nn

    class Dataset:
        base_source, T = 'radar_high_pc', 4
        sample_groups = ['test/group_001']
        data_path_list = {'radar_high_pc': [[Path(f'{t}.npy') for t in range(4)]]}

        def __len__(self):
            return 1

        def _get_sensor_data_from_path(self, sensor, refs):
            assert sensor == 'radar_high_pc'
            return [np.array([[2., 0., 0., 0., 0., 0.]] * 2 + [[6., 0., 0., 0., 0., 0.]], 'f4') for _ in refs]

        def __getitem__(self, index):
            points = [np.tile([2., 0., 0., 0., 0., 0.], (n, 1)).astype('f4') for n in (0, 9, 10, 12)]
            points = [np.concatenate((p, np.array([[6., 0., 0., 0., 0., 0.]], 'f4'))) for p in points]
            pose = np.tile([2., 0., 0.], (1, 17, 1)).astype('f4')
            return dict(group_id=self.sample_groups[index], radar_high_pc=points, gt_for_high=[pose.copy() for _ in points],
                        high_to_low_R=[np.eye(3, dtype='f4')] * 4, high_to_low_t=[np.array([1., 2., 1.], 'f4')] * 4)

    class Model(nn.Module):
        def forward(self, inputs):
            assert inputs['mask'].any(-1).all()
            pose = inputs['input'][..., :3].mean(-2)[:, :, None, None].expand(-1, -1, 1, 17, -1)
            return {'pose': pose + .1}

    dataset = Dataset()
    for fmt in ('coco', 'mocap'):
        for coordinate, centered in (('high', False), ('high', True), ('low', False)):
            cfg = dict(data=dict(person_max_points=20, max_people=4), task=dict(
                input='radar_high_pc', output='gt_for_high', coordinate=coordinate, pose_format=fmt,
                center_on_pointcloud=centered))
            class FormatModel(Model):
                def forward(self, inputs):
                    out = super().forward(inputs)
                    if fmt == 'mocap':
                        from utils.MOCAP21 import coco_to_mocap
                        out['pose'] = coco_to_mocap(out['pose'])
                    return out
            records, predictions = evaluate(FormatModel(), dataset, [0], cfg, torch.device('cpu'), workers=0)
            assert records['frame'].tolist() == [2, 3]
            np.testing.assert_allclose(records['mpjpe'], np.sqrt(3) * 10, atol=1e-4)
            view = Viewer(dataset, cfg, records, predictions, 'Self-test')
            view.select(1)
            assert view.index == 1
            np.testing.assert_allclose([view.scene.get_xlim(), view.scene.get_ylim(), view.scene.get_zlim()], view.xyz_limits)
            spans = np.diff(view.xyz_limits, axis=1).ravel()
            aspect = view.scene.get_box_aspect()
            np.testing.assert_allclose(aspect/aspect[0], spans/spans[0])
            sample = display_sample(dataset, cfg, records[0], predictions[0])
            assert sample['inside'].sum() == 2 and len(sample['points']) == 3
            np.testing.assert_allclose(sample['truth'][0], [3., 2., 1.], atol=1e-5)
            view.fig.canvas.draw()
            pixel = view.curve.transData.transform([0, records['mpjpe'][0]])
            view.click(SimpleNamespace(inaxes=view.curve, button=1, xdata=0., x=pixel[0], y=pixel[1]))
            assert view.index == 0
            view.key(SimpleNamespace(inaxes=view.curve, key='right'))
            assert view.index == 1
            view.submit('invalid')
            view.submit('-1')
            assert view.index == 1
            plt.close(view.fig)
            with patch.object(dataset, '_get_sensor_data_from_path', return_value=[np.empty((0, 6), 'f4')]):
                empty = display_sample(dataset, cfg, records[0], predictions[0])
                assert len(empty['points']) == 0 and not empty['inside'].any()
    ranged = np.repeat(records[:1], 20)
    ranged['mpjpe'] = np.tile([1., 2., 3.], 7)[:20]
    ranged['mpjpe'][[1, 3]] = [10., 20.]
    view = Viewer(dataset, cfg, ranged, np.repeat(predictions[:1], 20, axis=0), 'Range self-test')
    assert view.allowed.tolist() == [3] and view.index == 3
    assert float(view.min_box.text) == np.quantile(ranged['mpjpe'], .95)
    view.min_box.set_val('9')
    view.max_box.set_val('21')
    assert view.allowed.tolist() == [1, 3]
    view.key(SimpleNamespace(inaxes=view.curve, key='left'))
    assert view.index == 1
    view.key(SimpleNamespace(inaxes=view.curve, key='right'))
    assert view.index == 3
    view.submit('0')
    assert view.index == 3
    view.fig.canvas.draw()
    pixel = view.curve.transData.transform([1, ranged['mpjpe'][1]])
    view.click(SimpleNamespace(inaxes=view.curve, button=1, xdata=1., x=pixel[0], y=pixel[1]))
    assert view.index == 1
    pixel = view.curve.transData.transform([0, ranged['mpjpe'][0]])
    view.click(SimpleNamespace(inaxes=view.curve, button=1, xdata=0., x=pixel[0], y=pixel[1]))
    assert view.index == 1
    for bad in ('nan', '-1', '22', 'bad'):
        view.min_box.set_val(bad)
        assert view.allowed.tolist() == [1, 3] and view.index == 1
    view.min_box.set_val('0')
    view.max_box.set_val('0')
    assert view.allowed.tolist() == list(range(20))
    view.min_box.set_val('10')
    view.max_box.set_val('20')
    assert view.allowed.tolist() == [1, 3]
    view.key(SimpleNamespace(inaxes=view.min_box.ax, key='right'))
    assert view.index == 1
    plt.close(view.fig)
    print('Self-test passed: filtering, indexing, centering, COCO/MOCAP, range bounds, filtered navigation and rendering.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'experiments/P4Transformer/20261019_161657')
    parser.add_argument('--checkpoint', choices=('last', 'best'), default='last')
    parser.add_argument('--dataset-cache', type=Path, help='已有且与模型配置匹配的 train.pkl（只读）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--port', type=int, default=8990)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.self_test:
        self_test()
        return
    if args.batch_size <= 0 or args.workers < 0 or not 1 <= args.port <= 65535:
        parser.error('batch-size 必须为正数；workers 非负；port 必须在 1..65535')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model, cfg = load_experiment(args.experiment, args.checkpoint, device)
    dataset, indices = load_training_dataset(args.experiment, cfg, args.dataset_cache)
    records, predictions = evaluate(model, dataset, indices, cfg, device, args.batch_size, args.workers)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    import matplotlib
    matplotlib.use('WebAgg')
    matplotlib.rcParams.update({'webagg.port': args.port, 'webagg.address': '127.0.0.1', 'webagg.open_in_browser': True})
    import matplotlib.pyplot as plt
    viewer = Viewer(dataset, cfg, records, predictions, f'{args.experiment} | {args.checkpoint}.pth | no augmentation')
    plt.show()


if __name__ == '__main__':
    main()
