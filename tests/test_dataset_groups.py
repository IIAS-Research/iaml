"""Patient groups remain aligned after search sampling and nested validation."""
import unittest
import numpy as np
import pandas as pd

from iaml.dataset import Dataset
from iaml.splitters.kfold_splitter import kfold_splitter


class DatasetGroupTests(unittest.TestCase):
    def dataset(self):
        index = np.arange(120)*7 + 11
        X = pd.DataFrame({'marker': np.arange(120, dtype=float)}, index=index)
        groups = pd.DataFrame({'patient': np.repeat(np.arange(30), 4)}, index=index)
        return Dataset(X, np.arange(120)*.21, groups=groups)

    def check_alignment(self, dataset):
        self.assertEqual(len(dataset.X), len(dataset.groups))
        np.testing.assert_array_equal(dataset.groups.patient, dataset.X.marker.to_numpy().astype(int)//4)
        self.assertNotIn('patient', dataset.X)

    def test_search_sample_preserves_matching_group_rows(self):
        source = self.dataset()
        for count in [35, .5, 200]:
            sampled = source.sample(count)
            self.check_alignment(sampled)
            self.check_alignment(sampled.sample(12))

    def test_both_sides_of_grouped_fold_and_nested_sampling_remain_aligned(self):
        source = self.dataset().sample(90)
        for train, test in kfold_splitter(source, nb_folds=3):
            self.check_alignment(train)
            self.check_alignment(test)
            self.assertFalse(set(train.groups.patient) & set(test.groups.patient))
            self.check_alignment(train.sample(15))
            self.check_alignment(test.sample(10))

    def test_wrong_group_length_rejected_immediately(self):
        with self.assertRaisesRegex(ValueError, 'one row per feature row'):
            Dataset(pd.DataFrame({'a': range(5)}), range(5), groups=pd.DataFrame({'patient': [1, 2]}))


if __name__ == '__main__':
    unittest.main()
