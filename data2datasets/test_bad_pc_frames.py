import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data2datasets.dataset_for_all_task import HPE_Dataset, PackedFrame


class BadPCFramesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manifest_path = Path(self.directory.name) / 'bad_pc_frames.json'
        self.group = Path('/nas/20261010/data_collection/group_031')
        self.record = dict(date='20261010', group='group_031', sensor='dpct高位机',
                           frame_index=3, frame_name='1000_3.npy', bad_type='missing_arms')
        self.write_manifest([self.record])

    def write_manifest(self, records, version=1):
        self.manifest_path.write_text(json.dumps(dict(schema_version=version, bad_frames=records)))

    def dataset(self, packed=False, manifest=True):
        def ref(index):
            path = self.group / 'dpct高位机/PC' / f'1000_{index}.npy'
            return (PackedFrame(str(path.parent / 'frames.pcpack'), path.name,
                                1000 * 10**9 + index, index * 100, 100) if packed else str(path))
        pcs = [ref(i) for i in range(9)]
        gt = [str(self.group / f'camera results/smoothed 3D/1000_{i}.pkl') for i in range(9)]
        meta = {'person': [dict(date='20261010', valid_group=['group_031'],
                                group_data_path={'group_031': {'radar_high_pc': pcs, 'gt': gt}})]}
        with patch('data2datasets.dataset_for_all_task.get_meta_info', return_value=copy.deepcopy(meta)), \
             patch.object(HPE_Dataset, '_build_aligned_data'), \
             patch.object(HPE_Dataset, '_is_valid_npy', return_value=True), \
             redirect_stdout(io.StringIO()):
            ds = HPE_Dataset(root_path='/nas', sensor_config={'radar_high_pc': True, 'gt': True},
                             base_source='radar_high_pc', T=3, acc_frame=2,
                             preload_cache=False, bad_pc_frames_path=self.manifest_path if manifest else None)
        return ds, pcs

    def test_window_body_and_accumulation_history_are_both_excluded(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                ds, pcs = self.dataset(packed)
                self.assertEqual(ds.sky_start_indices, [0, 6])
                self.assertEqual(ds.skip_bad_samples, 5)
                self.assertEqual(ds.data_path_list['radar_high_pc'], [pcs[:3], pcs[6:]])
                self.assertEqual(ds.acc_path_list['radar_high_pc'], [[], pcs[4:6]])
                self.assertEqual(len(ds.sample_groups), len(ds.data_path_list['gt']))
                for body, history in zip(ds.data_path_list['radar_high_pc'], ds.acc_path_list['radar_high_pc']):
                    self.assertNotIn(pcs[3], body + history)

    def test_clean_windows_keep_the_original_time_axis(self):
        ds, pcs = self.dataset(manifest=False)
        self.assertEqual(ds.sky_start_indices, list(range(7)))
        self.assertEqual(ds.data_path_list['radar_high_pc'][2], pcs[2:5])

    def test_matching_is_independent_of_root_and_exact_for_group_sensor_and_frame(self):
        ds, pcs = self.dataset()
        with patch.object(ds, '_is_valid_npy', return_value=True):
            for bad in [pcs[3], pcs[3].replace('/nas/', '/ssd/')]:
                self.assertFalse(ds._is_valid_window({'radar_high_pc': [bad]}))
            for clean in [pcs[3].replace('group_031', 'group_030'),
                          pcs[3].replace('20261010', '20261009'),
                          pcs[3].replace('dpct高位机', 'dpct低位机'),
                          pcs[3].replace('1000_3.npy', '1000_4.npy')]:
                self.assertTrue(ds._is_valid_window({'radar_high_pc': [clean]}))

    def test_listed_frame_cannot_pass_low_point_window_allowance(self):
        ds, pcs = self.dataset()
        ds.low_point_files.add(str(pcs[3]))
        with patch.object(ds, '_is_valid_npy', return_value=False):
            self.assertFalse(ds._is_valid_window({'radar_high_pc': pcs}))

    def test_manifest_validation_and_existing_bin_manifest(self):
        bins = HPE_Dataset._load_bad_frames(HPE_Dataset.DEFAULT_BAD_BIN_FRAMES_PATH, {'numeric', 'length'})
        self.assertEqual(len(bins), 1033)
        for records, version in [([self.record, self.record], 1),
                                 ([dict(self.record, bad_type='numeric')], 1),
                                 ([dict(self.record, frame_index=-1)], 1),
                                 ([{'date': '20261010'}], 1), ([self.record], 2)]:
            self.write_manifest(records, version)
            with self.assertRaises(ValueError):
                HPE_Dataset._load_bad_frames(self.manifest_path, {'missing_arms'})
        self.assertEqual(HPE_Dataset._load_bad_frames(None, {'missing_arms'}), {})

    def test_existing_bin_blacklist_still_rejects_frames_with_correct_length(self):
        ds, _ = self.dataset()
        ds.radar_config = SimpleNamespace(num_samp=4, num_chirp=1, Tx=1, Rx=1)
        record = next(iter(ds.bad_bin_frames.values()))
        pack = (Path('/ssd') / record['date'] / 'data_collection' / record['group']
                / record['sensor'] / 'Bin/frames.binpack')
        bad = PackedFrame(str(pack), record['frame_name'], 0, 0, 8)
        clean = PackedFrame(str(pack), 'unlisted.bin', 0, 0, 8)
        with redirect_stdout(io.StringIO()):
            self.assertFalse(ds._is_valid_radar_bin('radar_high_bin', bad))
            self.assertTrue(ds._is_valid_radar_bin('radar_high_bin', clean))

    def test_production_manifest_matches_the_audited_frames_and_labels(self):
        manifest_path = HPE_Dataset.DEFAULT_BAD_PC_FRAMES_PATH
        manifest = json.loads(manifest_path.read_text())
        lookup = HPE_Dataset._load_bad_frames(manifest_path, {'missing_arms'})
        labels = json.loads(manifest_path.with_name('group_labels.json').read_text())['groups']
        self.assertEqual(len(lookup), manifest['summary']['total'])
        self.assertEqual(len(lookup), 755)
        for (date, group, sensor, frame), record in lookup.items():
            self.assertIn('开合跳', labels[f'{date}/{group}']['label'])
            self.assertEqual(sensor, 'dpct高位机')
            self.assertTrue(frame.endswith('.npy'))
            self.assertLess(record['frame_index'], record['group_frame_count'])
            self.assertTrue(record['details']['affected_people'])


if __name__ == '__main__':
    unittest.main()
