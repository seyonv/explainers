import unittest
from stats import mean


class Hidden(unittest.TestCase):
    def test_means(self):
        self.assertAlmostEqual(mean([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(mean([-1, -2]), -1.5)
        self.assertAlmostEqual(mean([7]), 7)
        self.assertAlmostEqual(mean([0.5, 0.25]), 0.375)

    def test_empty_still_raises(self):
        with self.assertRaises(ValueError):
            mean([])
