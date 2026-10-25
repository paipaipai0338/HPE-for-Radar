import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data2datasets.alignment_cache import load_or_build_alignment, reuse_bin_alignment
from data2datasets.dataset_for_all_task import HPE_Dataset, PackedFrame


class FFTGenAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.group = Path(self.tmp.name) / 'nas/20260722/data_collection/group_001'
        self.pc = self.group / 'fftGen/PC'
        self.bins = self.group / 'dpct高位机/Bin'
        self.gt = self.group / 'camera results/smoothed 3D'
        self.config = dict(algorithm_version='global_dp_v1', base_source='radar_high_pc',
                           max_delta_sec=0.3, one_to_one=True, time_offsets_sec={},
                           enabled_sensors={'gt': '.pkl', 'radar_high_pc': '.npy'})
        self.pc_refs = self.make_pack(self.pc, 'pcpack', '.npy', [3, 5, 7])
        self.bin_refs = self.make_pack(self.bins, 'binpack', '.bin', [1, 1, 1])
        self.gt_refs = self.make_pack(self.gt, 'gtpack', '.pkl', [2, 2, 2])
        self.sources = {'gt': self.gt, 'radar_high_pc': self.pc}
        bin_config = {**self.config, 'base_source': 'radar_high_bin',
                      'enabled_sensors': {'gt': '.pkl', 'radar_high_bin': '.bin'}}
        # Only two of the three BIN frames have matched GT.
        load_or_build_alignment(self.group, {'gt': self.gt, 'radar_high_bin': self.bins},
                                bin_config, lambda: {'gt': self.gt_refs[:2],
                                                     'radar_high_bin': self.bin_refs[:2]}, PackedFrame)

    def make_pack(self, folder, kind, suffix, lengths):
        folder.mkdir(parents=True)
        pack = folder / f'frames.{kind}'
        pack.write_bytes(b'x' * sum(lengths))
        names = [f'100_{i:09d}{suffix}' for i in range(len(lengths))]
        times = [100_000_000_000 + i for i in range(len(lengths))]
        offsets = np.cumsum([0] + lengths[:-1]).tolist()
        np.savez(folder / 'frames_index.npz', frame_names=names, timestamps_ns=times,
                 offsets=offsets, lengths=lengths)
        return [PackedFrame(str(pack), n, t, o, l)
                for n, t, o, l in zip(names, times, offsets, lengths)]

    def test_reuses_gt_matches_with_pc_offsets(self):
        result = reuse_bin_alignment(self.group, self.sources, self.config, PackedFrame)
        self.assertEqual(result, {'gt': self.gt_refs[:2], 'radar_high_pc': self.pc_refs[:2]})
        self.assertNotEqual(result['radar_high_pc'][1].offset, self.bin_refs[1].offset)

    def test_changed_gt_rejects_stale_bin_cache(self):
        with (self.gt / 'frames.gtpack').open('ab') as stream:
            stream.write(b'x')
        self.assertIsNone(reuse_bin_alignment(self.group, self.sources, self.config, PackedFrame))

    def test_reuses_raw_nas_bin_cache_without_bin_pack(self):
        (self.bins / 'frames.binpack').unlink()
        (self.bins / 'frames_index.npz').unlink()
        raw_bins = [self.bins / ref.frame_name for ref in self.bin_refs]
        for path in raw_bins:
            path.touch()
        bin_config = {**self.config, 'base_source': 'radar_high_bin',
                      'enabled_sensors': {'gt': '.pkl', 'radar_high_bin': '.bin'}}
        load_or_build_alignment(self.group, {'gt': self.gt, 'radar_high_bin': self.bins},
                                bin_config, lambda: {'gt': self.gt_refs[:2],
                                                     'radar_high_bin': raw_bins[:2]}, PackedFrame)
        self.assertEqual(reuse_bin_alignment(self.group, self.sources, self.config, PackedFrame),
                         {'gt': self.gt_refs[:2], 'radar_high_pc': self.pc_refs[:2]})

    def test_different_full_timeline_rejects_reuse(self):
        index_path = self.pc / 'frames_index.npz'
        with np.load(index_path) as index:
            data = {key: index[key].copy() for key in index.files}
        data['timestamps_ns'][-1] += 1
        np.savez(index_path, **data)
        self.assertIsNone(reuse_bin_alignment(self.group, self.sources, self.config, PackedFrame))

    def test_different_matching_configuration_rejects_reuse(self):
        self.assertIsNone(reuse_bin_alignment(self.group, self.sources,
                                              {**self.config, 'max_delta_sec': 0.5}, PackedFrame))

    def make_dataset(self):
        dataset = HPE_Dataset.__new__(HPE_Dataset)
        dataset.root_path = self.group.parents[2]
        dataset.packed_data_root = None
        dataset.pointcloud_source = 'fftgen'
        dataset.base_source = 'radar_high_pc'
        dataset.sensor_config = {'radar_high_pc': True, 'gt': True}
        dataset.suffix_map = {'radar_high_pc': '.npy', 'gt': '.pkl'}
        dataset.meta_info = {'1': [{'date': '20260722', 'valid_group': ['group_001']}]}
        return dataset

    def test_fftgen_bypasses_own_cache_and_never_creates_one(self):
        dataset = self.make_dataset()
        before = set((self.group / 'cache').iterdir())
        with patch('data2datasets.dataset_for_all_task.load_or_build_alignment',
                   side_effect=AssertionError('FFTGen must not use its own cache')), patch.object(
                       dataset, '_align_multi_sensor_files', side_effect=AssertionError('No FFTGen matching')):
            dataset._build_aligned_data()
        frames = dataset.meta_info['1'][0]['group_data_path']['group_001']
        self.assertEqual(frames, {'gt': self.gt_refs[:2], 'radar_high_pc': self.pc_refs[:2]})
        self.assertEqual(set((self.group / 'cache').iterdir()), before)

    def test_fftgen_fails_instead_of_realigning_when_bin_cache_invalid(self):
        with (self.gt / 'frames.gtpack').open('ab') as stream:
            stream.write(b'x')
        dataset = self.make_dataset()
        with patch.object(dataset, '_align_multi_sensor_files', side_effect=AssertionError('No matching')):
            with self.assertRaisesRegex(ValueError, 'BIN→GT'):
                dataset._build_aligned_data()


if __name__ == '__main__':
    unittest.main()
