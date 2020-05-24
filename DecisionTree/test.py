import unittest
import numpy as np
import tree


class TestTree(unittest.TestCase):
    def create_dataset(self):
        data = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
        target = np.array(['Yes', 'Yes', 'No', 'No', 'No'])
        labels = ['no surfacing', 'flippers']

        return data, target, labels

    def test_calculate_entropy(self):
        data, target, _ = self.create_dataset()
        entropy1 = tree.calculate_entropy(data, target)
        entropy2 = - 0.4 * np.log2(0.4) - 0.6 * np.log2(0.6)
        np.testing.assert_equal(entropy1, entropy2)

    def test_split_dataset(self):
        data, target, _ = self.create_dataset()

        data_split, target_split = tree.split_dataset(data, target, 0, 1)
        data_test = np.array([[1], [1], [0]])
        target_test = np.array(['Yes', 'Yes', 'No'])
        np.testing.assert_array_equal(data_split, data_test)
        np.testing.assert_array_equal(target_split, target_test)

        data_split, target_split = tree.split_dataset(data, target, 0, 0)
        data_test = np.array([[1], [1]])
        target_test = np.array(['No', 'No'])
        np.testing.assert_array_equal(data_split, data_test)
        np.testing.assert_array_equal(target_split, target_test)

    def test_choose_best_feature(self):
        data, target, _ = self.create_dataset()

        feature = tree.choose_best_feature(data, target)
        np.testing.assert_equal(feature, 0)

    def test_majority_class(self):
        data, target, _ = self.create_dataset()

        majority = tree.majority_class(target)
        self.assertEqual(majority, 'No')


if __name__ == '__main__':
    unittest.main()
