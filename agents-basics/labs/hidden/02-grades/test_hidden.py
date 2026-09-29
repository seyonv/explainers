import unittest
from grades import letter_grade


class Hidden(unittest.TestCase):
    def test_all_boundaries(self):
        for score, want in [(100, "A"), (90, "A"), (89.9, "B"), (80, "B"), (79.5, "C"),
                            (70, "C"), (60, "D"), (59.99, "F"), (0, "F")]:
            self.assertEqual(letter_grade(score), want, score)
