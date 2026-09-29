import unittest
from calendar_utils import days_in_month, is_leap


class Hidden(unittest.TestCase):
    def test_leap_rules(self):
        for year, want in [(1900, False), (2000, True), (2023, False), (2024, True),
                           (2100, False), (2400, True), (1600, True)]:
            self.assertEqual(is_leap(year), want, year)

    def test_february(self):
        self.assertEqual(days_in_month(2, 1900), 28)
        self.assertEqual(days_in_month(2, 2400), 29)
        self.assertEqual(days_in_month(12, 1900), 31)
