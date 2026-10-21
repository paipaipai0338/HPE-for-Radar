import json
import tempfile
import unittest
from pathlib import Path

from data2datasets.load_json import get_meta_info


class DataSplitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'data.json'
        self.data = {
            'subjects': [
                {'date': '20260101', 'person_id': '1', 'valid_group': ['a', 'bad']},
                {'date': '20260101', 'person_id': '4', 'valid_group': ['b']},
                {'date': '20260201', 'person_id': '1', 'valid_group': ['c', 'new', 'ignored']},
                {'date': '20260201', 'person_id': '4', 'valid_group': ['d']},
            ],
            'person_split': {
                'subjects_before': '20260201',
                'excluded_groups': {'20260101': ['bad']},
                'train_groups': {'20260201': ['c']},
                'val_groups': {'20260201': ['d']},
            },
        }

    def load(self, **kwargs):
        self.path.write_text(json.dumps(self.data))
        return get_meta_info(self.path, **kwargs)

    def test_config_controls_selection(self):
        self.assertEqual(self.load()['1'][0]['valid_group'], ['a', 'bad'])
        train = self.load(split_method='person')
        self.assertEqual(train['1'], [
            {'date': '20260101', 'valid_group': ['a']},
            {'date': '20260201', 'valid_group': ['c']},
        ])
        self.assertNotIn('4', train)
        self.assertNotIn('-1', train)
        val = self.load(split_method='person', mode='val')
        self.assertEqual(val['4'][0]['valid_group'], ['b'])
        self.assertEqual(val['4'][1]['valid_group'], ['d'])
        self.data['person_split']['train_groups']['20260201'] = ['new']
        self.assertEqual(self.load(split_method='person')['1'][1]['valid_group'], ['new'])

    def test_training_override_and_validation(self):
        override = {'20260201': ['c']}
        self.assertEqual(
            self.load(split_method='person', train_groups=override),
            {'1': [{'date': '20260201', 'valid_group': ['c']}]},
        )
        for override in ({'20260201': ['d']}, {'20260101': ['b']},
                         {'20260201': ['unknown']}, {}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.load(split_method='person', train_groups=override)
        with self.assertRaises(ValueError):
            self.load(split_method='group', train_groups={'20260201': ['c']})
        with self.assertRaises(ValueError):
            self.load(split_method='person', val_person_id='unknown')

    def test_config_overlap_and_invalid_groups(self):
        self.data['person_split']['val_groups']['20260201'] = ['c']
        with self.assertRaisesRegex(ValueError, '重叠'):
            self.load(split_method='person')
        self.data['person_split']['train_groups']['20260201'] = 'c'
        with self.assertRaisesRegex(ValueError, '映射'):
            self.load(split_method='person')

    def test_multi_always_trains_even_in_validation_lists(self):
        self.data['subjects'].extend([
            {'date': '20260101', 'person_id': '-1', 'valid_group': ['multi_old']},
            {'date': '20260201', 'person_id': '-1', 'valid_group': ['multi_new']},
        ])
        self.data['person_split']['val_groups']['20260201'].append('multi_new')
        self.data['person_split']['excluded_groups']['20260101'].append('multi_old')
        for person in ('1', '4'):
            train = self.load(split_method='person', val_person_id=person)
            val = self.load(split_method='person', mode='val', val_person_id=person)
            self.assertEqual([e['valid_group'] for e in train['-1']], [['multi_old'], ['multi_new']])
            self.assertNotIn('-1', val)
        override = self.load(split_method='person', train_groups={'20260201': ['multi_new']})
        self.assertEqual(override, {'-1': [{'date': '20260201', 'valid_group': ['multi_new']}]})

    def test_conflicting_person_ids_rejected(self):
        self.data['subjects'].append(
            {'date': '20260101', 'person_id': '-1', 'valid_group': ['b']},
        )
        with self.assertRaisesRegex(ValueError, '冲突'):
            self.load(split_method='person')

    def test_repository_multi_groups_are_train_only(self):
        train = get_meta_info(split_method='person')
        val = get_meta_info(split_method='person', mode='val')
        multi = {(e['date'], g) for e in train['-1'] for g in e['valid_group']}
        self.assertEqual(len(multi), 157)
        self.assertIn(('20260715', 'group_011'), multi)
        self.assertIn(('20261009', 'group_051'), multi)
        self.assertNotIn('-1', val)
        train_groups = {(e['date'], g) for entries in train.values() for e in entries for g in e['valid_group']}
        val_groups = {(e['date'], g) for entries in val.values() for e in entries for g in e['valid_group']}
        self.assertFalse(train_groups & val_groups)


if __name__ == '__main__':
    unittest.main()
