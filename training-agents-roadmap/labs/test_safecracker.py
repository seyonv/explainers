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

if __name__ == "__main__":
    unittest.main()
