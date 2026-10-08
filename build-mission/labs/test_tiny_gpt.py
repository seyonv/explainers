"""The evaluation must score every byte exactly once, whatever the text length."""
import math, unittest
import torch
from tiny_gpt import TinyGPT, bpb

class Uniform(TinyGPT):
    def forward(self, idx):                       # predicts every byte with probability 1/256
        return torch.zeros(*idx.shape, 256)

class Eval(unittest.TestCase):
    def test_uniform_model_scores_8_bits_per_byte(self):
        m = Uniform(1, 16, 2, ctx=32)
        for n in (32, 33, 100, 1000, 1017):
            data = torch.randint(0, 256, (n,))
            self.assertAlmostEqual(bpb(m, data, "cpu", stride=8, batch=4), 8.0, places=5, msg=n)

    def test_param_counts_match_size_names(self):
        from tiny_gpt import SIZES, params
        for name, (L, d, h) in SIZES.items():
            p = params(TinyGPT(L, d, h)) / 1e6
            self.assertLess(abs(p - float(name[:-1])) / float(name[:-1]), 0.15, (name, p))

if __name__ == "__main__":
    unittest.main()
