import unittest
from safecracker import split, shortest_path, move_text
import frontier_eval as fe

class Episode(unittest.TestCase):
    def test_a_model_that_plays_the_shortest_route_wins(self):
        p = split("test", 3, "linear", 1)[0]
        plan = shortest_path(tuple(p["start"]), tuple(p["target"]), p["links"], 3)
        replies = iter(f"thinking...\n{move_text(i, d)}" for i, d in plan)
        chat = lambda model, messages: (next(replies), {"prompt_tokens": 100, "completion_tokens": 10})
        won, moves, usd, log = fe.episode(chat, "m", p, budget=2 * p["optimal"], prices={"m": (1.0, 2.0)})
        self.assertTrue(won); self.assertEqual(moves, len(plan))
        self.assertAlmostEqual(usd, len(plan) * (100 * 1.0 + 10 * 2.0) / 1e6)

    def test_an_invalid_reply_costs_a_move(self):
        p = split("test", 3, "linear", 1)[0]
        chat = lambda model, messages: ("I give up", {"prompt_tokens": 1, "completion_tokens": 1})
        won, moves, usd, log = fe.episode(chat, "m", p, budget=3, prices={})
        self.assertFalse(won); self.assertEqual(moves, 3); self.assertIsNone(log[0]["action"])

class Repeats(unittest.TestCase):
    def test_repeats_run_every_safe_again_and_pool(self):
        safes = split("test", 3, "linear", 4)
        chat = lambda model, messages: ("no idea", {"prompt_tokens": 1, "completion_tokens": 1})
        r = fe.run_model(chat, "m", safes, repeats=3, mult=1.0, prices={}, max_usd=10)
        self.assertEqual(r["episodes"], 12)
        self.assertEqual(len(r["per_repeat"]), 3)
        self.assertEqual(r["won"], 0)

if __name__ == "__main__":
    unittest.main()
