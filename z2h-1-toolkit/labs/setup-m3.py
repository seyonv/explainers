# Card: setup-m3 (check your M3 is ready for the course) · no lecture chapter (prerequisite)
# Run from z2h-1-toolkit/: python labs/setup-m3.py
import sys, platform, time
import torch, numpy
from common import data_path

print('python  :', sys.version.split()[0], '|', platform.machine())
print('torch   :', torch.__version__)
print('numpy   :', numpy.__version__)
print('mps built    :', torch.backends.mps.is_built())
print('mps available:', torch.backends.mps.is_available())
print('cuda available:', torch.cuda.is_available())

for name in ['names.txt', 'input.txt']:
    text = open(data_path(name), 'r').read()
    print(f'{name:9s}: {len(text):,} chars, {len(text.splitlines()):,} lines')

# a small matmul on cpu vs mps, just to see both devices work
for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
    a = torch.randn(2048, 2048, device=device)
    a @ a  # warm-up
    if device == 'mps': torch.mps.synchronize()
    t0 = time.time()
    for _ in range(10):
        b = a @ a
    if device == 'mps': torch.mps.synchronize()
    dt = (time.time() - t0) / 10
    print(f'{device}: 2048x2048 matmul {dt*1000:.1f} ms  ({2*2048**3/dt/1e9:.0f} GFLOP/s)')

# seeded draws are deterministic on your machine (but differ from the video's)
g = torch.Generator().manual_seed(2147483647)
print('torch.rand(3, generator=g):', torch.rand(3, generator=g))
