#!/usr/bin/env bash
# h100-ncu-roofline.sh: place a BF16 GEMM and a GEMV on the H100 roofline with Nsight Compute.
#
# Companion to perf-3-kernels/profiling.html. Not yet run by the author.
#
# Where:   any rented H100 box running Ubuntu with an NVIDIA driver and the CUDA toolkit
#          (Lambda, RunPod, Modal-style images). Needs nvidia-smi, python3, pip, and ncu
#          (ships with the CUDA toolkit; often /usr/local/cuda/bin/ncu).
# Runtime: about 10-15 minutes, mostly the torch install and ncu's replay passes.
# Cost:    about $0.50-1.00 at $2-3.50 per GPU-hour. The course's total GPU budget cap is $300-400,
#          so stop the instance when this finishes.
#
# What it does:
#   1. checks the GPU with nvidia-smi and finds ncu
#   2. pip installs torch
#   3. times a BF16 GEMM (4096 x 4096 x 4096) and a GEMV-shaped matmul
#      (batch 1 x 4096 -> 14336, the Llama-3.1-8B up-projection) with CUDA events,
#      and prints achieved TFLOPS and GB/s against 989 TFLOPS / 3.35 TB/s
#   4. profiles one launch of each with
#        ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart
#            --section SpeedOfLight_HierarchicalTensorRooflineChart
#      (section identifiers from the Nsight Compute Profiling Guide's section table)
#      and writes gemm.ncu-rep and gemv.ncu-rep
#
# Afterwards: download the two .ncu-rep files (scp) and open them in the Nsight Compute GUI
# (free desktop app). The Details page shows Speed of Light and the roofline charts; the
# achieved point for the GEMM should sit under the flat roof, the GEMV's under the slope.
#
# Usage: bash h100-ncu-roofline.sh

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/h100-ncu-roofline}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "== 1. GPU check =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: this box has no NVIDIA driver." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
  echo "Warning: this is not an H100; the 989 TF / 3.35 TB/s reference lines will be wrong." >&2
fi

NCU="$(command -v ncu || true)"
if [ -z "$NCU" ] && [ -x /usr/local/cuda/bin/ncu ]; then
  NCU=/usr/local/cuda/bin/ncu
fi
if [ -z "$NCU" ]; then
  echo "ncu not found. Install the CUDA toolkit's Nsight Compute (e.g. apt package cuda-nsight-compute-12-x)." >&2
  exit 1
fi
"$NCU" --version

echo "== 2. Python deps =="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet torch

cat > bench.py <<'PY'
import sys
import torch

PEAK_TFLOPS = 989.0   # H100 SXM dense BF16
PEAK_TBS = 3.35       # H100 SXM HBM3
RIDGE = PEAK_TFLOPS / PEAK_TBS

which = sys.argv[1]           # "gemm" or "gemv"
mode = sys.argv[2] if len(sys.argv) > 2 else "time"   # "time" or "profile"
torch.manual_seed(0)
dt = torch.bfloat16
if which == "gemm":
    M, K, N = 4096, 4096, 4096
else:
    M, K, N = 1, 4096, 14336      # x (1 x 4096) @ W (4096 x 14336)
# Uniform[-1, 1] fill, as CUTLASS recommends for realistic power
A = (torch.rand(M, K, device="cuda", dtype=dt) * 2 - 1)
B = (torch.rand(K, N, device="cuda", dtype=dt) * 2 - 1)
fn = lambda: A @ B

if mode == "profile":
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    sys.exit(0)

flops = 2 * M * K * N
nbytes = (M * K + K * N + M * N) * 2
# rotate copies of the big operand so the total is >= 2x the 50 MB L2 (CUTLASS buffer rotation)
copies = max(2, -(-100_000_000 // (K * N * 2)))
Bs = [B.clone() for _ in range(copies)]
for i in range(50):
    A @ Bs[i % copies]
torch.cuda.synchronize()
reps = 1000 if which == "gemm" else 2000
times = []
for i in range(reps):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    A @ Bs[i % copies]
    e.record()
    times.append((s, e))
torch.cuda.synchronize()
ms = sorted(s.elapsed_time(e) for s, e in times)
med = ms[len(ms) // 2]
p90 = ms[int(0.9 * len(ms)) - 1]
t = med * 1e-3
tflops = flops / t / 1e12
gbs = nbytes / t / 1e9
ai = flops / nbytes
print(f"{which}: {M}x{K}x{N} BF16, {copies} rotated copies of B")
print(f"  FLOPs {flops/1e9:.3f} G, bytes {nbytes/1e6:.2f} MB, arithmetic intensity {ai:.1f} FLOP/B (ridge {RIDGE:.0f})")
print(f"  median {med*1e3:.1f} us, p90 {p90*1e3:.1f} us")
print(f"  achieved {tflops:.1f} TFLOPS = {100*tflops/PEAK_TFLOPS:.1f}% of {PEAK_TFLOPS:.0f}")
print(f"  achieved {gbs:.0f} GB/s = {100*gbs/(PEAK_TBS*1e3):.1f}% of {PEAK_TBS} TB/s")
print("  bound:", "compute (right of the ridge)" if ai > RIDGE else "memory (left of the ridge)")
print(f"  floors: compute {flops/(PEAK_TFLOPS*1e12)*1e6:.1f} us, memory {nbytes/(PEAK_TBS*1e12)*1e6:.1f} us")
PY

echo "== 3. Timing with CUDA events (median and p90) =="
python3 bench.py gemm time
python3 bench.py gemv time

echo "== 4. Nsight Compute: Speed of Light + roofline sections =="
for w in gemm gemv; do
  "$NCU" --profile-from-start off \
         --section SpeedOfLight \
         --section SpeedOfLight_RooflineChart \
         --section SpeedOfLight_HierarchicalTensorRooflineChart \
         -o "$w" -f \
         python3 bench.py "$w" profile
  "$NCU" --import "$w.ncu-rep" --page details | head -60
done

echo
echo "Done. Reports: $WORKDIR/gemm.ncu-rep and $WORKDIR/gemv.ncu-rep"
echo "Download them, e.g.: scp <user>@<host>:$WORKDIR/*.ncu-rep ."
echo "Open them in the Nsight Compute GUI; compare ncu's Duration with the event medians above"
echo "(ncu locks clocks and flushes caches, so they will differ). Then stop the instance."
