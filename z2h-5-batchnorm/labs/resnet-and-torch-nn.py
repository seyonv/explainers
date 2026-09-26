# Card: resnet-and-torch-nn (z2h-5-batchnorm) · Lecture 4 "real example: resnet50 walkthrough" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=3890s
# Run from the course folder: python labs/resnet-and-torch-nn.py   (the ResNet part is reading: torchvision/models/resnet.py, class Bottleneck)
import inspect, torch
import torch.nn as nn
from common import BatchNorm1d

torch.manual_seed(2147483647)

# 1. the defaults the lecture reads off the docs
print('nn.BatchNorm1d' + str(inspect.signature(nn.BatchNorm1d.__init__)).replace('self, ', ''))
print('nn.Linear     ' + str(inspect.signature(nn.Linear.__init__)).replace('self, ', ''))

# 2. parameters (trained by backprop) vs buffers (updated in forward) in PyTorch's BatchNorm1d
bn = nn.BatchNorm1d(200)
print('BatchNorm1d(200) parameters:', [(n, tuple(p.shape)) for n, p in bn.named_parameters()])
print('BatchNorm1d(200) buffers:   ', [(n, tuple(b.shape)) for n, b in bn.named_buffers()])

# 3. nn.Linear's default init: uniform in +-1/sqrt(fan_in)
lin = nn.Linear(30, 200)
w = lin.weight
print(f'nn.Linear(30, 200): max |w| {w.abs().max().item():.4f} vs 1/sqrt(30) = {30**-0.5:.4f}')
print(f'  std(w) {w.std().item():.4f} vs 1/sqrt(3*30) = {(3*30)**-0.5:.4f} (uniform) vs (5/3)/sqrt(30) = {(5/3)*30**-0.5:.4f} (Kaiming tanh)')

# 4. the motif, with the bias switched off because BatchNorm would cancel it
block = nn.Sequential(nn.Linear(30, 200, bias=False), nn.BatchNorm1d(200), nn.Tanh())
print(f'Linear(30,200,bias=False) -> BatchNorm1d(200) -> Tanh: {sum(p.numel() for p in block.parameters())} parameters '
      f'(with bias: {sum(p.numel() for p in nn.Linear(30, 200).parameters()) + 400})')

# 5. the lecture's BatchNorm1d class vs PyTorch's, same input, training mode
x = torch.randn(32, 200) * 3 + 1
ours = BatchNorm1d(200)
theirs = nn.BatchNorm1d(200, momentum=0.1)
a, b = ours(x), theirs(x)
print(f'our BN vs nn.BatchNorm1d output: max |diff| {(a - b).abs().max().item():.4f}  '
      f'(ours normalises with the unbiased var /(n-1), torch with the biased /n; ratio sqrt(32/31) = {(32/31)**0.5:.4f})')
print(f'running_var after one step: max |diff| {(ours.running_var - theirs.running_var).abs().max().item():.1e} (both use the unbiased var)')
