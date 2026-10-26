"""Dependency-free provenance, serving selection and canonical-prior regressions."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

from src.Utils.blend import PRIOR_POLICY, load_local_prior, prior_is_complete
from src.Utils.model_metadata import (dataset_metadata, save_dataset_metadata,
                                      select_matching_model, write_model_metadata)


class ModelMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.provenance = dict(blend_k=10, blend_r=.75, prior_dates={'2026-27': '2026-06-21'},
                               prior_policy=PRIOR_POLICY)

    def model(self, name, k=10, r=.75, created='2026-10-26T12:00:00+00:00'):
        p = self.root / name
        p.touch()
        write_model_metadata(p, dict(self.provenance, blend_k=k, blend_r=r),
                             'dataset_2012-27_k10', 'SCHEMA', ['A'])
        sidecar = Path(str(p) + '.meta.json')
        metadata = json.loads(sidecar.read_text())
        metadata['created_at'] = created
        sidecar.write_text(json.dumps(metadata))
        return p

    def test_matching_settings_and_newest_not_accuracy(self):
        older = self.model('accuracy99.json')
        newer = self.model('accuracy50.json', created='2026-10-26T13:00:00+00:00')
        wrong_k = self.model('wrong_k.json', k=0)
        wrong_r = self.model('wrong_r.json', r=1)
        self.assertEqual(select_matching_model([older, newer, wrong_k, wrong_r], 10, .75, 'SCHEMA', ['A']), newer)
        with self.assertRaisesRegex(FileNotFoundError, 'No SCHEMA model matches'):
            select_matching_model([older, newer], 20, .75, 'SCHEMA', ['A'])
        with self.assertRaises(FileNotFoundError):
            select_matching_model([older], 10, .75, 'SCHEMA', ['A', 'B'])
        with self.assertRaises(FileNotFoundError):
            select_matching_model([older], 10, .75, 'OTHER_SCHEMA', ['A'])

    def test_prior_date_matches_when_training_includes_serving_season(self):
        model = self.model('prior.json')
        self.assertEqual(select_matching_model([model], 10, .75, 'SCHEMA', ['A'],
                                              '2026-27', '2026-06-21'), model)
        with self.assertRaises(FileNotFoundError):
            select_matching_model([model], 10, .75, 'SCHEMA', ['A'], '2026-27', '2026-06-20')

    def test_legacy_only_k_zero_and_corrupt_never_legacy(self):
        legacy = self.root / 'legacy.keras'
        legacy.touch()
        self.assertEqual(select_matching_model([legacy], 0, 1, 'SCHEMA', ['A']), legacy)
        with self.assertRaises(FileNotFoundError):
            select_matching_model([legacy], 10, .75, 'SCHEMA', ['A'])
        Path(str(legacy) + '.meta.json').write_text('{broken')
        with warnings.catch_warnings(record=True), self.assertRaises(FileNotFoundError):
            select_matching_model([legacy], 0, .75, 'SCHEMA', ['A'])

    def test_dataset_provenance_not_inferred_from_current_config(self):
        db = self.root / 'dataset.sqlite'
        with closing(sqlite3.connect(db)) as con:
            save_dataset_metadata(con, 'dataset_2012-27_k10', self.provenance)
        self.assertEqual(dataset_metadata(db, 'dataset_2012-27_k10'), self.provenance)
        with self.assertRaisesRegex(ValueError, 'rebuild'):
            dataset_metadata(db, 'dataset_2012-27_k20')
        with warnings.catch_warnings(record=True):
            self.assertEqual(dataset_metadata(db, 'dataset_2012-27')['blend_k'], 0)
        metadata = write_model_metadata(self.root / 'new.json', self.provenance,
                                        'dataset_2012-27_k10', 'SCHEMA', ['A'])
        self.assertEqual(metadata['prior_dates']['2026-27'], '2026-06-21')
        self.assertEqual(metadata['feature_count'], 1)

    def test_all_trainers_write_metadata_and_runners_receive_settings(self):
        import ast
        root = Path(__file__).resolve().parents[1]
        for family in ('NN', 'XGBoost'):
            for kind in ('ML', 'UO'):
                tree = ast.parse((root / f'src/Train-Models/{family}_Model_{kind}.py').read_text())
                names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
                self.assertTrue({'dataset_metadata', 'write_model_metadata', 'unique_model_path'} <= names)
        tree = ast.parse((root / 'main.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ('nn_runner', 'xgb_runner')]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all({'blend_k', 'blend_r'} <= {k.arg for k in n.keywords} for n in calls))


class FakeFrame:
    def __init__(self, records):
        self.records = records
        self.attrs = {}

    @classmethod
    def from_records(cls, records):
        return cls(records)

    def to_dict(self, mode):
        return self.records

    def to_sql(self, name, con, **kwargs):
        con.execute(f'CREATE TABLE "{name}" (TEAM_ID INTEGER, GP INTEGER)')
        con.executemany(f'INSERT INTO "{name}" VALUES (?, ?)', [(r['TEAM_ID'], r['GP']) for r in self.records])


class CanonicalPriorTests(unittest.TestCase):
    config = {'get-data': {'2025-26': {'start_date': '2025-10-21', 'end_date': '2026-6-21'},
                           '2026-27': {'start_date': '2026-10-20', 'end_date': '2027-6-20'}}}

    def records(self, gp=82):
        return [dict(TEAM_ID=i + 1, GP=gp) for i in range(30)]

    def test_shared_completeness_policy(self):
        self.assertTrue(prior_is_complete(self.records(58)))
        self.assertFalse(prior_is_complete(self.records(57)))
        self.assertFalse(prior_is_complete(self.records()[:29]))
        rows = self.records()
        rows[-1] = rows[0]
        self.assertFalse(prior_is_complete(rows))

    def test_local_prior_canonical_api_not_called(self):
        api = Mock(side_effect=AssertionError('must not fetch API for existing local prior'))
        with closing(sqlite3.connect(':memory:')) as con, patch.dict('sys.modules', {'pandas': SimpleNamespace(DataFrame=FakeFrame)}):
            FakeFrame(self.records()).to_sql('2026-06-21', con)
            local = load_local_prior(con, ['2026-06-21'], '2026-27', self.config, fill_missing=api)
            training = load_local_prior(con, ['2026-06-21'], '2026-27', self.config)
            self.assertEqual(local.records, training.records)
            self.assertEqual(local.attrs['prior_date'], '2026-06-21')
            api.assert_not_called()

    def test_existing_incomplete_disables_both_paths(self):
        api = Mock(side_effect=AssertionError('do not replace incomplete local data'))
        with closing(sqlite3.connect(':memory:')) as con, warnings.catch_warnings(record=True) as emitted:
            FakeFrame(self.records(20)).to_sql('2026-01-07', con)
            self.assertIsNone(load_local_prior(con, ['2026-01-07'], '2026-27', self.config))
            self.assertIsNone(load_local_prior(con, ['2026-01-07'], '2026-27', self.config, fill_missing=api))
            api.assert_not_called()
            self.assertTrue(any('effective blend_k=0' in str(w.message) for w in emitted))

    def test_api_fill_persisted_then_reused_by_training(self):
        api = Mock(return_value=FakeFrame(self.records(72)))
        with closing(sqlite3.connect(':memory:')) as con, patch.dict('sys.modules', {'pandas': SimpleNamespace(DataFrame=FakeFrame)}):
            serving = load_local_prior(con, [], '2026-27', self.config, fill_missing=api)
            training = load_local_prior(con, ['2026-06-21'], '2026-27', self.config)
            self.assertEqual(serving.records, training.records)
            self.assertEqual(serving.attrs, training.attrs)
            api.assert_called_once_with('2025-26')

    def test_incomplete_api_not_persisted(self):
        with closing(sqlite3.connect(':memory:')) as con, warnings.catch_warnings(record=True):
            result = load_local_prior(con, [], '2026-27', self.config,
                                      fill_missing=lambda season: FakeFrame(self.records(10)))
            self.assertIsNone(result)
            self.assertEqual(con.execute('SELECT count(*) FROM sqlite_master').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
