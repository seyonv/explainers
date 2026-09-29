import unittest
from intervals import merge


class Hidden(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(merge([]), [])
        self.assertEqual(merge([[5, 6], [1, 2], [2, 3]]), [[1, 3], [5, 6]])
        self.assertEqual(merge([[4, 9], [1, 10], [0, 0]]), [[0, 0], [1, 10]])
        self.assertEqual(merge([[3, 4], [1, 2]]), [[1, 2], [3, 4]])
