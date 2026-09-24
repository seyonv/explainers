#!/usr/bin/env bash
# h100-vllm-bench.sh: benchmark a real inference server the way the benchmarking card says to.
# vLLM serving Llama-3.1-8B on one H100: the throughput-latency curve (output tok/s, TTFT, TPOT
# vs concurrency) in BF16 and FP8, open-loop Poisson vs bursty arrivals, and speculative decoding
# (n-gram and EAGLE-3) at concurrency 1 vs 64.
#
# Companion to perf-5-engines/benchmarking.html. Not yet run by the author.
#
# Where:   any rented single-H100 box (80 GB, SXM or PCIe) running Linux with an NVIDIA driver,
#          python3 >= 3.10 and pip. Provider-agnostic (Lambda, RunPod, Modal-style images).
# Runtime: about 45-60 minutes including downloads: the pip install, a ~16 GB model, the ShareGPT
#          file (~650 MB), a small EAGLE-3 draft, four vLLM start-ups (each profiles memory and
#          captures CUDA graphs) and ~20 benchmark runs.
# Cost:    about $2-4 at $2-3.50 per GPU-hour. The course's total GPU budget cap is $300-400,
#          so stop the instance as soon as this finishes.
#
# Model:   meta-llama/Llama-3.1-8B-Instruct is gated on Hugging Face. Accept the licence on its
#          model page, then:  export HF_TOKEN=hf_...
#          No token? Use an ungated model:  MODEL=Qwen/Qwen3-8B bash h100-vllm-bench.sh
#          (the EAGLE-3 step is then skipped, because its draft is trained for Llama-3.1-8B).
#
# What it runs (flags from the current vLLM docs; check them if a newer vLLM rejects one):
#   vllm bench serve   https://docs.vllm.ai/en/latest/cli/bench/serve/
#                      https://docs.vllm.ai/en/latest/benchmarking/cli/
#   FP8 (online, dynamic):  --quantization fp8   weights FP8 E4M3 per-tensor, activations scaled per
#                      forward pass, lm_head left in high precision; needs compute capability >= 8.9
#                      https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/
#   n-gram / EAGLE-3:  --speculative-config '{"method": "ngram" | "eagle3", ...}'
#                      https://docs.vllm.ai/en/latest/features/speculative_decoding/
#                      https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/
#                      (draft RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3, listed there)
#   1. BF16, random dataset, ISL 1,024 / OSL 256, --ignore-eos, --request-rate inf with
#      --max-concurrency 1, 8, 32, 64, 128 (closed loop: the docs' "maximum throughput" pattern)
#   2. BF16, same lengths, open loop: --request-rate 8 with --burstiness 1 (Poisson) and 0.25 (bursty)
#   3. BF16 on ShareGPT (real text, so drafts can match) at concurrency 1 and 64: the baseline
#   4. FP8, the same concurrency sweep as step 1
#   5-6. n-gram (4 draft tokens) and EAGLE-3 (2 draft tokens) on ShareGPT at 1 and 64
#   Every run uses a different --seed, so no run is served from the prefix cache of the one before
#   (the docs warn that repeated runs against one server inflate throughput). Each run starts with
#   --num-warmups warm-up requests that are not counted.
#
# Usage: bash h100-vllm-bench.sh          (optional env: MODEL, WORKDIR, PORT, LEVELS, ISL, OSL)

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EAGLE_DRAFT="${EAGLE_DRAFT:-RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3}"
WORKDIR="${WORKDIR:-$HOME/h100-vllm-bench}"
PORT="${PORT:-8000}"
LEVELS="${LEVELS:-1 8 32 64 128}"
ISL="${ISL:-1024}"
OSL="${OSL:-256}"
MAXLEN=4096
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "== 0. GPU check and install =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: this box has no NVIDIA driver." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
  echo "Warning: this is not an H100; numbers will not match the card's H100 predictions." >&2
fi

if [[ "$MODEL" == meta-llama/* ]] && [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set and $MODEL is gated." >&2
  echo "Either export HF_TOKEN=hf_... (after accepting the licence) or run MODEL=Qwen/Qwen3-8B bash $0" >&2
  exit 1
fi

python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet vllm
python3 -c "import vllm; print('vllm', vllm.__version__)"
command -v vllm >/dev/null 2>&1 || { echo "vllm CLI not on PATH (try ~/.local/bin)." >&2; exit 1; }

echo "== Downloading $MODEL (about 16 GB) and ShareGPT =="
python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
PY
[ -f ShareGPT_V3_unfiltered_cleaned_split.json ] || wget -q "$SHAREGPT_URL"

SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

# start_server <logfile> <extra vllm serve args...>
start_server() {
  local log="$1"; shift
  vllm serve "$MODEL" --port "$PORT" --max-model-len "$MAXLEN" --seed 0 "$@" >"$log" 2>&1 &
  SERVER_PID=$!
  echo "   vllm serve started (pid $SERVER_PID), log: $WORKDIR/$log"
  for _ in $(seq 1 180); do                       # up to 15 minutes
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "   server is up"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "   vllm serve exited early; last lines of $log:" >&2
      tail -40 "$log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "   server did not come up in 15 minutes; see $log" >&2
  exit 1
}

SEED=0
# bench <tag> <extra vllm bench serve args...>
bench() {
  local tag="$1"; shift
  SEED=$((SEED + 1))                              # new prompts every run: no prefix-cache carry-over
  vllm bench serve --backend vllm --model "$MODEL" --port "$PORT" --seed "$SEED" \
    --num-warmups 4 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-filename "bench-$tag.json" "$@" | tee "bench-$tag.txt"
}

random_args() { echo --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --ignore-eos; }
prompts_for() { local c="$1"; echo $(( c * 4 > 32 ? c * 4 : 32 )); }   # >= 4 waves of requests

echo "== 1. BF16: closed-loop concurrency sweep (ISL $ISL / OSL $OSL) =="
start_server serve-bf16.log --gpu-memory-utilization 0.9
for c in $LEVELS; do
  bench "bf16-c$c" $(random_args) --request-rate inf --max-concurrency "$c" --num-prompts "$(prompts_for "$c")"
done

echo "== 2. BF16: open-loop arrivals at 8 req/s, Poisson vs bursty =="
bench "bf16-rate8-b1"    $(random_args) --request-rate 8 --burstiness 1    --num-prompts 400
bench "bf16-rate8-b0.25" $(random_args) --request-rate 8 --burstiness 0.25 --num-prompts 400

echo "== 3. BF16 baseline on ShareGPT at concurrency 1 and 64 (for the speculative runs) =="
for c in 1 64; do
  bench "sharegpt-bf16-c$c" --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate inf --max-concurrency "$c" --num-prompts "$(prompts_for "$c")"
done
stop_server

echo "== 4. FP8 (online dynamic, --quantization fp8): the same concurrency sweep =="
start_server serve-fp8.log --gpu-memory-utilization 0.9 --quantization fp8
for c in $LEVELS; do
  bench "fp8-c$c" $(random_args) --request-rate inf --max-concurrency "$c" --num-prompts "$(prompts_for "$c")"
done
stop_server

spec_runs() {
  local name="$1"
  for c in 1 64; do
    bench "sharegpt-$name-c$c" --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
      --request-rate inf --max-concurrency "$c" --num-prompts "$(prompts_for "$c")"
  done
}

echo "== 5. Speculative decoding: n-gram (prompt lookup), 4 draft tokens =="
start_server serve-ngram.log --gpu-memory-utilization 0.9 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'
spec_runs ngram
stop_server

if [[ "$MODEL" == meta-llama/Llama-3.1-8B* ]]; then
  echo "== 6. Speculative decoding: EAGLE-3 draft $EAGLE_DRAFT, 2 draft tokens =="
  start_server serve-eagle3.log --gpu-memory-utilization 0.9 \
    --speculative-config "{\"method\": \"eagle3\", \"model\": \"$EAGLE_DRAFT\", \"num_speculative_tokens\": 2, \"draft_tensor_parallel_size\": 1}"
  spec_runs eagle3
  stop_server
else
  echo "== 6. skipped: the EAGLE-3 draft is trained for Llama-3.1-8B-Instruct, not $MODEL =="
fi

echo
echo "== Summary =="
python3 - "$ISL" "$OSL" <<'PY'
import json, os, sys
isl, osl = sys.argv[1], sys.argv[2]
def row(tag):
    f = f"bench-{tag}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d.get("output_throughput"), d.get("median_ttft_ms"), d.get("median_tpot_ms"), d.get("p99_ttft_ms"), d.get("completed")
def show(title, tags):
    print(title)
    print(f"  {'run':<28} {'output tok/s':>12} {'median TTFT':>12} {'median TPOT':>12} {'P99 TTFT':>10} {'done':>5}")
    for t in tags:
        r = row(t)
        if r is None:
            print(f"  {t:<28} (missing)"); continue
        f = lambda x, u="": "n/a" if x is None else f"{x:,.1f}{u}"
        print(f"  {t:<28} {f(r[0]):>12} {f(r[1], ' ms'):>12} {f(r[2], ' ms'):>12} {f(r[3], ' ms'):>10} {r[4] or 0:>5}")
levels = os.environ.get("LEVELS", "1 8 32 64 128").split()
show(f"Throughput-latency curve, random ISL {isl} / OSL {osl}, closed loop (max concurrency C):",
     [f"{p}-c{c}" for p in ("bf16", "fp8") for c in levels])
show("Open loop at 8 req/s (burstiness 1 = Poisson, 0.25 = bursty, CV = 1/sqrt(b) = 2):",
     ["bf16-rate8-b1", "bf16-rate8-b0.25"])
show("Speculative decoding on ShareGPT (BF16 target) at concurrency 1 and 64:",
     [f"sharegpt-{m}-c{c}" for m in ("bf16", "ngram", "eagle3") for c in (1, 64)])
print("\nCompare with the card: the batch-1 decode ceiling is ~209 tok/s (3.35 TB/s / 16.06 GB);")
print("speculation should help most at C = 1 and fade (or hurt) at C = 64.")
PY
echo "Logs, JSON results and text reports are in $WORKDIR. Stop the instance now."

# Try this:
#   LEVELS="1 2 4 8 16 32 64 128 256" bash h100-vllm-bench.sh   -> a finer curve; find the knee
#   add --goodput ttft:500 tpot:50 to bench()                     -> DistServe-style goodput per run
#   repeat one run with the SAME --seed against the same server   -> watch the prefix cache inflate it
