"""Sweep bookkeeping on synthetic results (made-up numbers, not measurements)."""
import unittest
from sweep import grid, crossover, done, command

class Sweep(unittest.TestCase):
    def test_grid_has_every_size_and_a_gpt2_per_corpus(self):
        g = grid()
        self.assertEqual(len(g), 25)
        self.assertEqual(sum(r["kind"] == "gpt2" for r in g), 5)
        mb1 = [r for r in g if r["mb"] == 1]
        self.assertEqual(max(r["flops"] for r in mb1 if r["kind"] == "scratch"), [r for r in mb1 if r["kind"] == "gpt2"][0]["flops"])

    def test_crossover_is_smallest_winning_size(self):
        fake = [
            {"model": "from scratch 1M", "size": "1M", "train_mb": 1, "val_bpb": 2.0, "test_bpb": 2.0}, {"model": "GPT-2 fine-tuned", "train_mb": 1, "val_bpb": 1.5, "test_bpb": 1.5},
            {"model": "from scratch 3M", "size": "3M", "train_mb": 10, "val_bpb": 1.2, "test_bpb": 1.2}, {"model": "GPT-2 fine-tuned", "train_mb": 10, "val_bpb": 1.4, "test_bpb": 1.4},
            {"model": "from scratch 3M", "size": "3M", "train_mb": 30, "val_bpb": 1.1, "test_bpb": 1.1}, {"model": "GPT-2 fine-tuned", "train_mb": 30, "val_bpb": 1.3, "test_bpb": 1.3},
        ]
        table, x = crossover(fake)
        self.assertEqual(x, 10); self.assertEqual(len(table), 3)
        self.assertIsNone(crossover(fake[:2])[1])

    def test_crossover_picks_on_validation_not_test(self):
        fake = [{"model": "from scratch 1M", "size": "1M", "train_mb": 1, "val_bpb": 1.9, "test_bpb": 1.8},
                {"model": "from scratch 3M", "size": "3M", "train_mb": 1, "val_bpb": 1.7, "test_bpb": 1.85},
                {"model": "GPT-2 fine-tuned", "train_mb": 1, "val_bpb": 1.5, "test_bpb": 1.5}]
        t = crossover(fake)[0][0]
        self.assertEqual(t["scratch_size"], "3M"); self.assertEqual(t["scratch_bpb"], 1.85)

    def test_done_and_commands(self):
        r = grid()[0]
        self.assertFalse(done([], r))
        self.assertFalse(done([{"model": "from scratch 1M", "size": "1M", "train_mb": 1}], r))      # a hand run doesn't count
        self.assertTrue(done([{"model": "from scratch 1M", "size": "1M", "train_mb": 1, "tokens_budget": r["tokens"]}], r))
        self.assertIn("tiny_gpt.py --size 1M --mb 1", command(r))

    def test_chart_writes_svg(self):
        import tempfile, os
        from sweep import chart
        fake = [{"model": "from scratch 1M", "size": "1M", "train_mb": 1, "val_bpb": 2.0, "test_bpb": 2.0}, {"model": "from scratch 1M", "size": "1M", "train_mb": 10, "val_bpb": 1.6, "test_bpb": 1.6},
                {"model": "GPT-2 fine-tuned", "train_mb": 1, "val_bpb": 1.5, "test_bpb": 1.5}, {"model": "GPT-2 fine-tuned", "train_mb": 10, "val_bpb": 1.4, "test_bpb": 1.4}]
        with tempfile.TemporaryDirectory() as d:
            n = chart(fake, os.path.join(d, "c.svg"))
            self.assertEqual(n, 2)
            self.assertIn("<polyline", open(os.path.join(d, "c.svg")).read())

if __name__ == "__main__":
    unittest.main()


class Quiz(unittest.TestCase):
    def test_excerpt_drops_header_and_year_lines(self):
        try:
            from sample import excerpt
        except ImportError:                       # sample.py needs torch; skip under plain python3
            self.skipTest("torch not installed")
        t = "Area Forecast Discussion\nNational Weather Service Testville XX\n305 AM EST Fri Mar 1 2030\n\n.SYNOPSIS...\nA made-up front.\nIssued at 300 AM EST Fri Mar 1 2030\nRain ends tonight.\nArea Forecast Discussion\nNational Weather Service Otherville XX\nSnow later.\n"
        e = excerpt(t, 600)
        self.assertNotIn("2030", e); self.assertNotIn("ville", e); self.assertIn("Rain ends tonight.", e); self.assertIn("Snow later.", e)
