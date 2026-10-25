from collections import OrderedDict
from unittest.mock import patch
import unittest

import numpy as np

from data2datasets.dataset_for_all_task import HPE_Dataset, PackedFrame


class DecodedBinCacheTest(unittest.TestCase):
    def test_overlapping_windows_reuse_decode_without_aliasing(self):
        dataset = HPE_Dataset.__new__(HPE_Dataset)
        dataset.decoded_bin_cache = OrderedDict()
        dataset.radar_config = None
        dataset._packed_bytes = lambda ref: np.array([ref.offset], dtype=np.complex64)
        refs = [PackedFrame('test', str(i), i, i, 1) for i in range(40)]
        with patch('data2datasets.dataset_for_all_task.bin_buffer_to_cube_range_fft',
                   side_effect=lambda raw, *args, **kwargs: raw.copy()) as decode:
            first = dataset._get_packed_bin_sequence(refs[:7])
            first[1][0] = -99
            second = dataset._get_packed_bin_sequence(refs[1:8])
            self.assertEqual(decode.call_count, 8)
            self.assertEqual(second[0][0], 1)
            dataset._get_packed_bin_sequence(refs[8:])
            self.assertEqual(len(dataset.decoded_bin_cache), 32)
            dataset._get_packed_bin_sequence(refs[:1])
            self.assertEqual(decode.call_count, 41)
        with self.assertRaisesRegex(ValueError, '不能跨越'):
            dataset._get_packed_bin_sequence([refs[0], PackedFrame('other', 'x', 0, 0, 1)])


if __name__ == '__main__':
    unittest.main()
