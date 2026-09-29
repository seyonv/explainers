import unittest
from slug import slugify


class Hidden(unittest.TestCase):
    def test_cases(self):
        for title, want in [("  Spaces   everywhere  ", "spaces-everywhere"),
                            ("C++ & Python 3", "c-python-3"),
                            ("already-a-slug", "already-a-slug"),
                            ("--Edge--", "edge"),
                            ("Top 10 Tips", "top-10-tips")]:
            self.assertEqual(slugify(title), want, title)
