import unittest
from safecracker import gen, turn, bfs, canon, render, parse_action, actions

class Core(unittest.TestCase):
    def test_gear_moves_linked_dial(self):
        links = {0: [(1, "gear")], 1: [], 2: []}
        self.assertEqual(turn((0, 0, 0), links, 0, +1), (1, 1, 0))
        self.assertEqual(turn((0, 9, 0), links, 0, +1), (1, 0, 0))

    def test_pin_moves_only_on_wrap(self):
        links = {0: [(1, "pin")], 1: [], 2: []}
        self.assertEqual(turn((5, 5, 5), links, 0, +1), (6, 5, 5))   # no wrap
        self.assertEqual(turn((9, 5, 5), links, 0, +1), (0, 6, 5))   # 9 -> 0 wraps
        self.assertEqual(turn((0, 5, 5), links, 0, -1), (9, 4, 5))   # 0 -> 9 wraps

    def test_gen_is_deterministic_and_solvable(self):
        a, b = gen(7, 3, 2), gen(7, 3, 2)
        self.assertEqual(a, b)
        self.assertGreaterEqual(a["optimal"], 3)
        self.assertEqual(bfs(a["start"], a["target"], a["links"], 3), a["optimal"])

    def test_hard_tier_has_a_pin(self):
        p = gen(3, 4, 3, tier="hard")
        kinds = [k for v in p["links"].values() for _, k in v]
        self.assertIn("pin", kinds)

    def test_canon_ignores_dial_relabelling(self):
        l1 = {0: [(1, "gear")], 1: [], 2: []}
        l2 = {0: [], 1: [], 2: [(0, "gear")]}            # same shape, dials renamed
        self.assertEqual(canon(3, l1), canon(3, l2))
        l3 = {0: [(1, "pin")], 1: [], 2: []}
        self.assertNotEqual(canon(3, l1), canon(3, l3))

    def test_render_hides_wiring(self):
        p = gen(1, 3, 2)
        text = render({"code": p["start"], "target": p["target"], "moves_left": 12})
        self.assertIn("moves left: 12", text)
        self.assertNotIn("gear", text); self.assertNotIn("pin", text)

    def test_parse_action(self):
        self.assertEqual(parse_action("turn(2, +)", 3), (1, 1))      # dials are 1-based in text
        self.assertEqual(parse_action('{"dial": 1, "direction": "-"}', 3), (0, -1))
        self.assertIsNone(parse_action("turn(9, +)", 3))
        self.assertIsNone(parse_action("hello", 3))

from safecracker import split, play, random_policy, try_undo_policy, prober_policy
import random as _r

class Baselines(unittest.TestCase):
    def test_splits_never_share_a_wiring(self):
        test = split("test", 3, "linear", 40); train = split("train", 3, "linear", 80)
        tc = {canon(p["n"], p["links"]) for p in test}
        self.assertTrue(all(canon(p["n"], p["links"]) not in tc for p in train))

    def test_prober_beats_random_on_linear(self):
        ps = split("test", 3, "linear", 40)
        rnd = sum(play(random_policy(_r.Random(0)), p, 2 * p["optimal"])[0] for p in ps)
        prb = sum(play(prober_policy(), p, 2 * p["optimal"])[0] for p in ps)
        self.assertLessEqual(rnd, 2)
        self.assertGreaterEqual(prb, 30)            # research measured ~94% at 2x on n=3

    def test_play_respects_budget(self):
        p = split("test", 3, "linear", 1)[0]
        won, moves = play(random_policy(_r.Random(1)), p, 5)
        self.assertLessEqual(moves, 5)

from safecracker import traces

class Traces(unittest.TestCase):
    def test_traces_are_winning_chat_games_from_training_safes(self):
        ts = traces(3, n=3)
        self.assertEqual(len(ts), 3)
        for g in ts:
            roles = [m["role"] for m in g["messages"]]
            self.assertEqual(roles[0], "system"); self.assertEqual(roles[1], "user")
            self.assertTrue(all(m["role"] == "assistant" for m in g["messages"][2::2]))
            self.assertRegex(g["messages"][2]["content"], r"^turn\(\d, [+-]\)$")
            self.assertTrue(g["won"])

from safecracker import quick_table

class Quick(unittest.TestCase):
    def test_quick_table_only_measures_the_sizes_asked_for(self):
        rows = quick_table([3], 5)
        self.assertEqual({(r["tier"], r["n"]) for r in rows}, {("linear", 3), ("hard", 3)})
        self.assertTrue(all(0 <= r["prober"] <= 100 for r in rows))

if __name__ == "__main__":
    unittest.main()
