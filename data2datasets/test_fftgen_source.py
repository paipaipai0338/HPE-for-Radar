import tempfile
import unittest
from pathlib import Path

from data2datasets.dataset_for_all_task import HPE_Dataset


class FFTGenSourceTest(unittest.TestCase):
    def test_fftgen_never_falls_back_to_original_pointcloud_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = HPE_Dataset.__new__(HPE_Dataset)
            dataset.root_path = root / 'nas'
            dataset.packed_data_root = root / 'ssd'
            dataset.sensor_config = {'radar_high_pc': True, 'gt': True}
            group = Path('20260722/data_collection/group_001')
            original = dataset.packed_data_root / group / 'dpct高位机/PC'
            original.mkdir(parents=True)
            for name in ('frames.pcpack', 'frames_index.npz'):
                (original / name).touch()
            dataset.pointcloud_source = 'fftgen'
            paths = dataset._build_sensor_paths(dataset.root_path / group)
            self.assertEqual(paths['radar_high_pc'], dataset.root_path / group / 'fftGen/PC')
            self.assertEqual(paths['gt'], dataset.root_path / group / 'camera results/smoothed 3D')
            dataset.pointcloud_source = 'files'
            self.assertEqual(dataset._build_sensor_paths(dataset.root_path / group)['radar_high_pc'], original)


if __name__ == '__main__':
    unittest.main()
