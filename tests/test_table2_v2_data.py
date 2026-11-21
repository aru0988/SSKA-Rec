import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sska_rec import core, table2_v2_data, tri_data


class Table2DataTest(unittest.TestCase):
    def test_census_train_fit_oov_and_fixed_label_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            census = root / 'Census-income'
            raw = census / 'raw'
            raw.mkdir(parents=True)
            train = self._census_rows(6)
            heldout = self._census_rows(8)
            heldout['class_worker'] = 'unseen'
            heldout['age'] = 48
            for frame, suffix, legacy in ((train, 'data', 'train'), (heldout, 'test', 'test')):
                frame.to_csv(raw / f'census-income.{suffix}', header=False, index=False)
                labels = pd.DataFrame({'label_income': (frame.income_50k == '50000+.').astype(int),
                                       'label_marital': (frame.marital_stat == 'Never married').astype(int)})
                labels.to_csv(census / f'{legacy}.gz', index=False)
            with patch.object(table2_v2_data, 'CENSUS_ROWS', (6, 4, 4)):
                spec = table2_v2_data.build_table2_spec('census', root, root / 'cache', root / 'meta', {})
                reloaded = table2_v2_data.build_table2_spec('census', root, root / 'cache', root / 'other-meta', {})
            rows = spec['sets']['test']
            self.assertEqual(spec['tasks'], ['income', 'marital'])
            self.assertEqual((len(rows.names), spec['xdim']), (40, 127))
            self.assertTrue({'education', 'sex', 'instance_weight', 'det_hh_fam_stat'} <= set(rows.names))
            self.assertTrue({'income_50k', 'marital_stat'}.isdisjoint(rows.names))
            np.testing.assert_array_equal(rows.x[:, rows.names.index('class_worker')], 0)
            self.assertGreater(rows.x[0, rows.names.index('age')], 1)
            self.assertEqual(spec['vocab']['class_worker'], 3)
            self.assertEqual(spec['data_signature']['preprocessing']['categorical_levels']['hisp_origin'],
                             ['NA', 'Other'])
            _, expected_test = core.make_split(8, 20260929)
            np.testing.assert_array_equal(rows.row_ids, 6 + expected_test)
            np.testing.assert_array_equal(rows.y[:, 0], (heldout.income_50k == '50000+.').to_numpy()[expected_test])
            self.assertEqual(spec['data_signature'], reloaded['data_signature'])

    def _census_rows(self, size):
        frame = pd.DataFrame({name: [3, 6] * (size // 2) for name in table2_v2_data.CENSUS_COLUMNS})
        frame['class_worker'] = ['z', 'a'] * (size // 2)
        frame['hisp_origin'] = ['NA', 'Other'] * (size // 2)
        frame['marital_stat'] = ['Never married', 'Married-civilian spouse present'] * (size // 2)
        frame['income_50k'] = ['- 50000.', '50000+.'] * (size // 2)
        return frame

    def test_kuairand_projects_columns_and_checks_historical_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['user_id', 'video_id', 'tab', 'hour', 'duration_ms']
            sets = {}
            for i, split in enumerate(tri_data.SPLITS):
                rows = core.Rows(np.zeros((2, 5), dtype=np.float32),
                                 np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.float32), names, [0, 1])
                rows.row_ids = np.arange(2 * i, 2 * i + 2)
                sets[split] = rows
            original = dict(sets=sets, tasks=['click', 'like', 'follow', 'forward'],
                            initial=['click', 'like'], dataset='kuairand', xdim=17,
                            rdim=128, widths=[1, 4, 4, 4, 4], vocab={})
            original['data_signature'] = tri_data.dataset_signature(original, [])
            historical = root / 'historical' / 'kuairand' / 'data' / 'data-signature.json'
            core.dump(historical, original['data_signature'])
            with patch.object(tri_data, 'build_spec', return_value=original):
                spec = table2_v2_data.build_table2_spec('kuairand', root, root / 'cache', root / 'meta', {},
                                                       root / 'historical')
                self.assertEqual(spec['tasks'], ['click', 'like'])
                self.assertEqual(spec['sets']['train'].y.shape, (2, 2))
                self.assertIs(spec['sets']['train'].x, sets['train'].x)
                self.assertEqual(original['sets']['train'].y.shape, (2, 4))
                self.assertEqual(set(spec['data_signature']['splits']['train']['labels']), {'click', 'like'})
                core.dump(historical, {'version': 'different'})
                with self.assertRaisesRegex(ValueError, 'Historical data signature differs'):
                    table2_v2_data.build_table2_spec('kuairand', root, root / 'cache', root / 'meta', {},
                                                       root / 'historical')
