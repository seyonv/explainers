"""Tests on synthetic discussion-shaped text (no downloads)."""
import tempfile, unittest
from pathlib import Path
from afd_corpus import clean, dedupe, split_products, write_prefix, train_dates, TEST_DATES, VAL_DATES

def raw(office, body, seq="999"):
    return f"\x01{seq}\nFXUS61 K{office} 011105\nAFD{office}\n\nArea Forecast Discussion\nNational Weather Service Testville {office}\n305 AM EST Fri Mar 1 2030\n\n{body}\n&&\n$$\n\nAB\n\x03"

LONG = ".SYNOPSIS...A made-up front drifts slowly across the made-up valley tonight."
OTHER = ".SHORT TERM...Synthetic showers taper off by a synthetic dawn, then sun returns."

class Clean(unittest.TestCase):
    def test_strips_framing_keeps_discussion(self):
        office, body = clean(split_products(raw("ABC", LONG).encode())[0])
        self.assertEqual(office, "ABC")
        self.assertTrue(body.startswith("Area Forecast Discussion"))
        self.assertNotIn("FXUS61", body); self.assertNotIn("$$", body); self.assertNotIn("&&", body); self.assertNotIn("\x03", body)

    def test_idempotent(self):
        _, body = clean(split_products(raw("ABC", LONG).encode())[0])
        self.assertEqual(clean(body)[1], body)

class Dedupe(unittest.TestCase):
    def test_repeated_paragraphs_dropped_per_office(self):
        a = clean(split_products(raw("ABC", LONG + "\n\n" + OTHER).encode())[0])
        b = clean(split_products(raw("ABC", LONG + "\n\n.UPDATE...A new synthetic paragraph that only the update has, long enough.").encode())[0])
        c = clean(split_products(raw("XYZ", LONG + "\n\n" + OTHER).encode())[0])
        out = dedupe([a, b, c])
        self.assertEqual(len(out), 3)
        self.assertIn(LONG, out[0][1]); self.assertNotIn(LONG, out[1][1]); self.assertIn(LONG, out[2][1])

    def test_exact_duplicate_product_dropped(self):
        a = clean(split_products(raw("ABC", LONG + "\n\n" + OTHER).encode())[0])
        self.assertEqual(len(dedupe([a, a])), 1)

class Splits(unittest.TestCase):
    def test_dates_never_overlap_and_test_is_after_2024(self):
        tr = set(train_dates())
        self.assertFalse(tr & set(TEST_DATES)); self.assertFalse(tr & set(VAL_DATES))
        self.assertTrue(all(d.year >= 2025 for d in TEST_DATES))
        self.assertTrue(all(d.year <= 2024 for d in tr))
        self.assertEqual(train_dates()[:2], train_dates()[:2])

    def test_prefixes_nested_and_bounded(self):
        prods = [f"product {i} " + "x" * 90 + "\n" for i in range(50)]
        with tempfile.TemporaryDirectory() as d:
            n1, _ = write_prefix(prods, Path(d) / "a.txt", 1000)
            n2, _ = write_prefix(prods, Path(d) / "b.txt", 3000)
            a, b = (Path(d) / "a.txt").read_text(), (Path(d) / "b.txt").read_text()
            self.assertTrue(b.startswith(a.rstrip("\n")))
            self.assertLessEqual(n1, 1000); self.assertGreater(n1, 1000 - 110)

if __name__ == "__main__":
    unittest.main()
