# Card: layernorm-dropout-scale.html · Lecture 7 · layernorm / scaling up the model, adding dropout · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5571s
# Run from the course folder: python labs/layernorm-dropout-scale.py [--colab | --gpt [--quick]]   (default: demos + the 2.06 model, CPU)
import time
import torch
import torch.nn as nn
from common import Shakespeare, GPTLanguageModel, train, nparams, flag

ds = Shakespeare()
vocab_size = ds.vocab_size

def run(cfg, device, init_weights=True):
    torch.manual_seed(1337)
    model = GPTLanguageModel(vocab_size, cfg['n_embd'], cfg['n_head'], cfg['n_layer'], cfg['block_size'],
                             cfg['dropout'], init_weights=init_weights).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters', f'on {device}', flush=True)
    t0 = time.time()
    train(model, ds, cfg['batch_size'], cfg['block_size'], cfg['max_iters'], cfg['eval_interval'],
          cfg['learning_rate'], cfg['eval_iters'], device)
    dt = time.time() - t0
    print(f"trained {cfg['max_iters']} iters in {dt:.0f} s ({dt / cfg['max_iters']:.3f} s/iter incl. evals)")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(ds.decode(model.generate(context, max_new_tokens=300)[0].tolist()))

# the Colab's last cell (CPU-friendly) and gpt.py (the lecture's final model)
colab = dict(batch_size=16, block_size=32, max_iters=5000, eval_interval=500, learning_rate=1e-3, eval_iters=200,
             n_embd=64, n_head=4, n_layer=4, dropout=0.0)
gpt = dict(batch_size=64, block_size=256, max_iters=5000, eval_interval=500, learning_rate=3e-4, eval_iters=200,
           n_embd=384, n_head=6, n_layer=6, dropout=0.2)

if flag('--colab'):
    run(colab, 'cpu')
elif flag('--gpt'):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if flag('--quick'):   # 1/20 of the run, with cheaper evals: shows the loss falling and the s/iter
        gpt.update(max_iters=250, eval_interval=50, eval_iters=20)
    run(gpt, device)
else:
    # LayerNorm = the makemore BatchNorm1d with mean(0) -> mean(1): normalise each row (example), not each column
    class LayerNorm1d: # (used to be BatchNorm1d)

        def __init__(self, dim, eps=1e-5):
            self.eps = eps
            self.gamma = torch.ones(dim)
            self.beta = torch.zeros(dim)

        def __call__(self, x):
            xmean = x.mean(1, keepdim=True) # row mean
            xvar = x.var(1, keepdim=True) # row variance
            xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
            self.out = self.gamma * xhat + self.beta
            return self.out

    torch.manual_seed(1337)
    module = LayerNorm1d(100)
    x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
    x = module(x)
    print("x.shape:", x.shape)
    print(f"one feature across the batch  x[:,0]: mean {x[:,0].mean().item():.4f}, std {x[:,0].std().item():.4f}")
    print(f"one example's features        x[0,:]: mean {x[0,:].mean().item():.4e}, std {x[0,:].std().item():.4f}")
    xt = torch.randn(32, 100)
    diff = (LayerNorm1d(100)(xt) - nn.LayerNorm(100)(xt)).abs().max().item()
    print(f"vs nn.LayerNorm(100): max diff {diff:.4f} (x.var divides by n-1, nn.LayerNorm by n)")

    # dropout: zero a random 20% in training, scale the rest by 1/0.8, do nothing in eval
    torch.manual_seed(1337)
    drop = nn.Dropout(0.2)
    print("nn.Dropout(0.2) on ones(10), train:", drop(torch.ones(10)).tolist())
    print("fraction zeroed on 1M ones:", (drop(torch.ones(1_000_000)) == 0).float().mean().item())
    drop.eval()
    print("nn.Dropout(0.2) on ones(10), eval: ", drop(torch.ones(10)).tolist())

    for name, cfg in [("Colab's small config", colab), ("gpt.py", gpt)]:
        m = GPTLanguageModel(vocab_size, cfg['n_embd'], cfg['n_head'], cfg['n_layer'], cfg['block_size'], cfg['dropout'])
        print(f"{name}: n_embd {cfg['n_embd']}, {cfg['n_head']} heads, {cfg['n_layer']} layers, block {cfg['block_size']}:"
              f" {nparams(m):,} params")

    # the video's step before scaling up: 3 pre-norm Blocks (LayerNorm inside each + ln_f), n_embd 32, 4 heads
    print("== + LayerNorm: n_embd 32, 4 heads, 3 layers, block 8, batch 32, lr 1e-3, 5000 iters, CPU ==")
    small = dict(batch_size=32, block_size=8, max_iters=5000, eval_interval=500, learning_rate=1e-3, eval_iters=200,
                 n_embd=32, n_head=4, n_layer=3, dropout=0.0)
    run(small, 'cpu', init_weights=False)
