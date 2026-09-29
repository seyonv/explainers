import unittest
from duration import parse_duration


class Hidden(unittest.TestCase):
    def test_values(self):
        for text, want in [("45s", 45), ("2h", 7200), ("100s", 100), ("12h", 43200),
                           ("2h5m10s", 7510), ("1h1m1s", 3661), (" 90m ", 5400)]:
            self.assertEqual(parse_duration(text), want, text)

    def test_rejects_garbage(self):
        for text in ["", "abc", "5x"]:
            with self.assertRaises(ValueError):
                parse_duration(text)
