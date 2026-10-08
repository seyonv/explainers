"""Tests on synthetic traces and made-up results (no downloads, no real data)."""
import json, tempfile, unittest
from pathlib import Path
from trace_lab import block_reuse, make_slice, slice_reuse, trace_reuse, kv_capacity, kendall_tau, rank, sharegpt_prompts, commands, BLOCK

def session(n_rounds, start=0.0, grow=320, out=50):
    rs, inp = [], 0
    for k in range(n_rounds):
        prefix = inp
        inp += grow
        rs.append({"round": k, "provider": "x", "input": inp, "prefix": prefix, "output": out, "start": start + 10 * k})
    return rs

class Reuse(unittest.TestCase):
    def test_block_reuse_counts_shared_prefixes_only(self):
        a = list(range(64)); b = list(range(32)) + [999] * 32
        self.assertAlmostEqual(block_reuse([a, b]), 2 / 8)        # b's first two blocks repeat a's
        self.assertEqual(block_reuse([[1] * 16, [2] * 16]), 0.0)

    def test_trace_reuse(self):
        t = trace_reuse({"s": session(3)})
        self.assertAlmostEqual(t["provider_cached_share"], (0 + 320 + 640) / (320 + 640 + 960))

    def test_slice_chains_reproduce_the_session_reuse(self):
        rows, info = make_slice({"s": session(5)}, minutes=10, scale=1, gap_mult=1)
        self.assertEqual(len(rows), 5)
        for prev, r in zip(rows, rows[1:]):
            self.assertEqual(r["hash_ids"][:len(prev["hash_ids"])], prev["hash_ids"])   # each round extends the last
        self.assertAlmostEqual(slice_reuse(rows), (0 + 20 + 40 + 60 + 80) / (20 + 40 + 60 + 80 + 100))

    def test_scaling_divides_lengths_and_drops_what_still_does_not_fit(self):
        big = session(3, grow=60000)
        rows, info = make_slice({"s": big}, minutes=10, scale=4, gap_mult=1)
        self.assertEqual(rows[0]["input_length"], 15000)
        self.assertEqual(info["dropped_too_long"], 1)            # 180000/4 = 45000 > 40960

class Sharegpt(unittest.TestCase):
    def test_first_turn_and_vllm_length_rules(self):
        convs = [{"conversations": [{"value": "a " * n}, {"value": "b " * o}]} for n, o in [(10, 10), (2, 10), (1100, 10), (900, 1200), (20, 3)]]
        convs.append({"conversations": [{"value": "only one turn"}]})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"; p.write_text(json.dumps(convs))
            got = sharegpt_prompts(p, 10, lambda s: s.split())
        self.assertEqual(sorted(len(x) for x in got), [10])     # only the (10, 10) pair passes every rule

class Kv(unittest.TestCase):
    def test_kv_capacity(self):
        cfg = {"num_hidden_layers": 2, "num_key_value_heads": 2, "head_dim": 4, "hidden_size": 16, "num_attention_heads": 4}
        k = kv_capacity(cfg, weight_bytes=100, gpu_bytes=1000, util=0.9)
        self.assertEqual(k["kv_bytes_per_token"], 2 * 2 * 2 * 4 * 2)
        self.assertEqual(k["kv_tokens"], (900 - 100) // 64)

class Ranking(unittest.TestCase):
    def test_tau(self):
        self.assertEqual(kendall_tau(list("abc"), list("abc")), 1.0)
        self.assertEqual(kendall_tau(list("abc"), list("cba")), -1.0)

    def test_rank_finds_the_chat_winner_on_agent_traffic(self):
        made_up = {"stock.sharegpt": 3.0, "fast.sharegpt": 5.0, "cache.sharegpt": 4.0, "stock.agent": 1.0, "fast.agent": 0.8, "cache.agent": 2.0}
        with tempfile.TemporaryDirectory() as d:
            for k, v in made_up.items():
                (Path(d) / f"{k}.json").write_text(json.dumps({"request_goodput": v}))
            r = rank(d)
        self.assertEqual(r["sharegpt_winner"], "fast"); self.assertEqual(r["sharegpt_winner_rank_on_agent"], 3)
        self.assertAlmostEqual(r["best_over_stock_on_agent"], 2.0)
        from trace_lab import slope_chart
        with tempfile.TemporaryDirectory() as d:
            slope_chart(r, Path(d) / "s.svg")
            self.assertEqual((Path(d) / "s.svg").read_text().count("<line"), 3)

    def test_rank_flags_a_chat_tie_and_reads_cache_counters(self):
        made_up = {"stock.sharegpt": 4.0, "fast.sharegpt": 3.95, "stock.agent": 1.0, "fast.agent": 1.5}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "results").mkdir(); (Path(d) / "logs").mkdir()
            for k, v in made_up.items():
                (Path(d) / "results" / f"{k}.json").write_text(json.dumps({"request_goodput": v}))
            (Path(d) / "logs" / "stock.before-agent.metrics.txt").write_text('vllm:num_preemptions_total{engine="0"} 2.0\nvllm:prefix_cache_queries_total{engine="0"} 100.0\nvllm:prefix_cache_hits_total{engine="0"} 10.0\n')
            (Path(d) / "logs" / "stock.after-agent.metrics.txt").write_text('vllm:num_preemptions_total{engine="0"} 7.0\nvllm:prefix_cache_queries_total{engine="0"} 1100.0\nvllm:prefix_cache_hits_total{engine="0"} 810.0\n')
            r = rank(Path(d) / "results")
        self.assertEqual(r["sharegpt_tied_with_winner"], ["fast"])
        self.assertEqual(r["agent_cache"]["stock"], {"preemptions": 5, "prefix_hit_rate": 0.8})

    def test_commands_pin_the_seed_and_slo(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            sl = Path(d) / "s.jsonl"; sl.write_text('{"timestamp": 0}\n' * 3)
            c = commands(4, slice_path=str(sl))
        self.assertIn("--num-prompts 3 ", c); self.assertIn("serve() {", c); self.assertIn("\nstop", c)
        self.assertIn("before-agent.metrics.txt", c); self.assertIn("after-agent.metrics.txt", c)
        self.assertIn("--goodput ttft:2000 tpot:50", c); self.assertIn("PYTHONHASHSEED=0", c); self.assertIn("--timed-trace-sec-multiplier 0.001", c)

if __name__ == "__main__":
    unittest.main()
