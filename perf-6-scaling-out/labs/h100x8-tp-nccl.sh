#!/usr/bin/env bash
# h100x8-tp-nccl.sh — measure what course 6 only models, on one rented 8x H100 SXM node.
#
#   Part 1  nccl-tests all_reduce_perf, 8 B -> 8 GB on 8 GPUs: the busbw curve and the
#           small-message latency (the "illustrative 14 us" in labs/layouts.py and sizing.py).
#   Part 2  vLLM serving Llama-3.1-70B-Instruct three ways, each with `vllm bench serve`
#           at concurrency 1 and 64 (1,024 tokens in, 256 out):
#             TP2 FP8   (GPUs 0-1, online FP8 weights: --quantization fp8)
#             TP4 BF16  (GPUs 0-3)
#             TP8 BF16  (GPUs 0-7)
#           Compare output tok/s and TPOT with the ceilings in labs/sizing.py and layouts.py.
#
# Status:   NOT YET RUN BY THE AUTHOR. Expect to fix small things (driver/CUDA versions,
#           paths, vLLM flag renames). Flags checked against the vLLM docs on 2026-09-24:
#           docs.vllm.ai/en/latest/cli/bench/serve/, /configuration/engine_args/,
#           /serving/parallelism_scaling/, /features/quantization/llm_compressor/fp8/.
# Runtime:  ~60-90 min (weights download ~141 GB, three server start-ups, six benchmarks).
# Cost:     8x H100 on-demand is ~$16-32/hr depending on provider (Lambda $3.99/GPU-hr =
#           $31.92/hr, seen 2026-09-24) -> roughly $25-40, ~$48 worst case at Lambda.
#           Set yourself a budget cap ($300-400 for the whole course) and SHUT THE NODE DOWN
#           when this finishes.
# Needs:    Ubuntu with NVIDIA driver + CUDA toolkit (nvcc) + NCCL (Lambda Stack has them),
#           Python 3.10+, git, ~350 GB free disk.
#           HF_TOKEN: Llama-3.1-70B-Instruct is a gated model. Accept Meta's licence on
#           huggingface.co/meta-llama/Llama-3.1-70B-Instruct first, then
#           `export HF_TOKEN=hf_...` (a read token) before running.
# Run:      bash h100x8-tp-nccl.sh            results land in $OUT (default ~/h100x8-results)
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-70B-Instruct}
OUT=${OUT:-$HOME/h100x8-results}
WORK=${WORK:-$HOME/h100x8-work}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PORT=${PORT:-8000}
IN_LEN=1024
OUT_LEN=256
MAX_LEN=4096

: "${HF_TOKEN:?export HF_TOKEN=hf_... first (gated model; accept the licence on Hugging Face)}"
mkdir -p "$OUT" "$WORK"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "== node =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee "$OUT/gpus.csv"
nvidia-smi topo -m | tee "$OUT/topo.txt"          # expect NV18 between every pair (NVSwitch)
df -h "$HOME" | tail -1

# ---------------------------------------------------------------- Part 1: NCCL
echo "== part 1: nccl-tests all_reduce_perf, 8 GPUs, 8 B -> 8 GB =="
if [ ! -x "$WORK/nccl-tests/build/all_reduce_perf" ]; then
  git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git "$WORK/nccl-tests"
  if [ -n "${NCCL_HOME:-}" ]; then
    make -C "$WORK/nccl-tests" -j CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"
  else
    make -C "$WORK/nccl-tests" -j CUDA_HOME="$CUDA_HOME"
  fi
fi
# -b/-e: min/max bytes, -f 2: double each time, -g 8: 8 GPUs in one process,
# -w/-n: warm-up and timed iterations. Columns: size, time (us), algbw, busbw (GB/s).
"$WORK/nccl-tests/build/all_reduce_perf" -b 8 -e 8G -f 2 -g 8 -w 5 -n 20 | tee "$OUT/nccl_all_reduce.txt"
echo "Read: time (us) at 8 B-64 KB = the latency floor; busbw at 8 GB = the bandwidth ceiling"
echo "      (the course models 450 GB/s per direction; the Scaling Book reports ~370 GB/s busbw)."

# ---------------------------------------------------------------- Part 2: vLLM
echo "== part 2: vLLM $MODEL =="
if [ ! -d "$WORK/venv" ]; then
  python3 -m venv "$WORK/venv"
fi
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
pip install -q -U pip
pip install -q -U vllm
vllm --version | tee "$OUT/vllm_version.txt"

echo "downloading weights (safetensors only, ~141 GB)..."
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", ignore_patterns=["original/*"])
PY

SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

run_config() {   # name, visible GPUs, extra vllm serve args...
  local name=$1 gpus=$2; shift 2
  echo "-- $name on GPUs $gpus: vllm serve $* --"
  CUDA_VISIBLE_DEVICES=$gpus vllm serve "$MODEL" --port "$PORT" \
      --max-model-len "$MAX_LEN" "$@" > "$OUT/server_$name.log" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server $name died; see $OUT/server_$name.log"; tail -20 "$OUT/server_$name.log"; return 1
    fi
    sleep 10; waited=$((waited + 10))
    if [ "$waited" -ge 1800 ]; then echo "server $name not up after 30 min"; stop_server; return 1; fi
  done
  echo "up after ${waited}s"
  grep -iE "KV cache|GPU blocks|Maximum concurrency" "$OUT/server_$name.log" | tail -3 || true

  for conc in 1 64; do
    local n=$((conc == 1 ? 16 : 256))
    vllm bench serve --model "$MODEL" --port "$PORT" \
      --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
      --ignore-eos --num-prompts "$n" --max-concurrency "$conc" \
      --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 \
      --save-result --result-dir "$OUT" --result-filename "bench_${name}_c${conc}.json" \
      | tee "$OUT/bench_${name}_c${conc}.txt"
  done
  stop_server
}

run_config tp2-fp8  0,1             --tensor-parallel-size 2 --quantization fp8
run_config tp4-bf16 0,1,2,3         --tensor-parallel-size 4
run_config tp8-bf16 0,1,2,3,4,5,6,7 --tensor-parallel-size 8

# ---------------------------------------------------------------- summary
echo "== summary (compare with labs/sizing.py and the pipeline card's ceilings) =="
python - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
print(f"{'config':<10}{'conc':>5}{'out tok/s':>11}{'per GPU':>9}{'TPOT p50 ms':>13}{'TTFT p50 ms':>13}")
for name, gpus in (("tp2-fp8", 2), ("tp4-bf16", 4), ("tp8-bf16", 8)):
    for c in (1, 64):
        f = out / f"bench_{name}_c{c}.json"
        if not f.exists():
            print(f"{name:<10}{c:>5}  missing"); continue
        r = json.loads(f.read_text())
        tok = r.get("output_throughput", float("nan"))
        print(f"{name:<10}{c:>5}{tok:>11.0f}{tok / gpus:>9.0f}"
              f"{r.get('median_tpot_ms', float('nan')):>13.2f}{r.get('median_ttft_ms', float('nan')):>13.1f}")
PY
echo "done. Results in $OUT. Now shut the node down."
