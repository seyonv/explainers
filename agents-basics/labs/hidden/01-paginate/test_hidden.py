import unittest
from pages import get_page, total_pages


class Hidden(unittest.TestCase):
    def test_every_page(self):
        items = list("abcdefghij")
        self.assertEqual(get_page(items, 1, 4), list("abcd"))
        self.assertEqual(get_page(items, 3, 4), list("ij"))
        self.assertEqual(get_page(items, 4, 4), [])

    def test_pages_cover_everything_once(self):
        items = list(range(23))
        pages = [get_page(items, n, 5) for n in range(1, total_pages(items, 5) + 1)]
        self.assertEqual(sum(pages, []), items)
