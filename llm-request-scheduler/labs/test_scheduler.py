"""python3 -m unittest test_scheduler.py"""
import unittest

from scheduler import (BlockManager, CostModel, Request, Scheduler, WaitQueue, clone, poisson_trace, simulate,
                       simulate_request_level, toy_trace)

cost = CostModel()


class BlockManagerTest(unittest.TestCase):
    def test_allocate_rounds_up_and_free_returns_blocks(self):
        bm = BlockManager(10, block_size=4)
        r = Request(0, 0, prompt_len=9, output_len=3)
        bm.allocate(r, 9)
        self.assertEqual(len(r.blocks), 3)          # ceil(9 / 4)
        self.assertEqual(bm.num_free(), 7)
        bm.free_request(r)
        self.assertEqual(bm.num_free(), 10)

    def test_prefix_hit_never_covers_last_prompt_token(self):
        bm = BlockManager(20, block_size=4, prefix_caching=True)
        toks = list(range(8))
        a = Request(0, 0, 8, 2, prompt_tokens=toks)
        bm.allocate(a, 8)
        a.num_computed = 8
        bm.cache_full_blocks(a)
        b = Request(1, 0, 8, 2, prompt_tokens=list(toks))
        bm.match_prefix(b)
        self.assertEqual(b.num_computed, 4)          # 2 full blocks cached, but the last is recomputed
        self.assertEqual(bm.ref[a.blocks[0]], 2)


class WaitQueueTest(unittest.TestCase):
    def test_orders(self):
        rs = [Request(i, t, p, o, priority=pr, tenant=tn, predicted_len=o)
              for i, (t, p, o, pr, tn) in enumerate([(0, 10, 500, 1, "A"), (1, 10, 5, 0, "A"), (2, 10, 50, 1, "B")])]
        for pol, want in [("fcfs", [0, 1, 2]), ("sjf", [1, 2, 0]), ("priority", [1, 0, 2])]:
            q = WaitQueue(pol)
            for r in rs:
                q.push(r)
            self.assertEqual([q.pop().rid for _ in rs], want, pol)

    def test_fair_prefers_least_served_tenant(self):
        q = WaitQueue("fair")
        a1, a2, b1 = Request(0, 0, 10, 5, tenant="A"), Request(1, 0, 10, 5, tenant="A"), Request(2, 1, 10, 5, tenant="B")
        for r in (a1, a2, b1):
            q.push(r)
        q.charge(a1, 100, 0)
        self.assertEqual(q.pop().tenant, "B")


class SchedulerTest(unittest.TestCase):
    def test_budget_and_max_seqs_respected(self):
        s = Scheduler(BlockManager(1000), max_num_batched_tokens=64, max_num_seqs=3)
        for i in range(10):
            s.add(Request(i, 0, 40, 5))
        for _ in range(50):
            if not s.has_work():
                break
            plan = s.step()
            self.assertLessEqual(plan.num_tokens, 64)
            self.assertLessEqual(len(s.running), 3)
            s.finish_step(plan, 0.0)

    def test_everything_finishes_with_exact_token_counts_under_preemption(self):
        tr = toy_trace()
        rep = simulate(tr, Scheduler(BlockManager(12, block_size=4), max_num_batched_tokens=16, max_num_seqs=4,
                                     watermark=0), cost)
        self.assertGreater(rep.preemptions, 0)
        for r in tr:
            self.assertEqual(r.num_generated, r.output_len)
            self.assertEqual(len(r.token_times), r.output_len)

    def test_blocks_all_returned(self):
        bm = BlockManager(3000)
        tr = poisson_trace(200, rate=30, seed=9)
        simulate(tr, Scheduler(bm), cost)
        self.assertEqual(bm.num_free(), 3000)

    def test_continuous_beats_static_on_throughput(self):
        base = poisson_trace(300, rate=40, seed=3)
        st = simulate_request_level(clone(base), cost, 32)
        ct = simulate(clone(base), Scheduler(BlockManager(cost.kv_blocks())), cost)
        self.assertGreater(ct.throughput, 3 * st.throughput)


if __name__ == "__main__":
    unittest.main()
