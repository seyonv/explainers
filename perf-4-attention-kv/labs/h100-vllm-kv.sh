#!/usr/bin/env bash
# h100-vllm-kv.sh: see vLLM's paged KV cache on a real H100. How many tokens and 8k
# sequences fit at two memory settings, and what a shared 2,000-token prefix does to TTFT.
#
# Companion to perf-4-attention-kv/paged-results.html. Not yet run by the author.
#
# Where:   any rented single-H100 box (80 GB, SXM or PCIe) running Linux with an NVIDIA
#          driver, python3 >= 3.10 and pip (Lambda, RunPod, Modal-style images).
# Runtime: about 20-30 minutes: the pip install, a ~16 GB model download, three vLLM
#          start-ups (each profiles memory and captures CUDA graphs) and two short benchmarks.
# Cost:    about $1-2 at $2-3.50 per GPU-hour. The course's total GPU budget cap is $300-400,
#          so stop the instance when this finishes.
#
# Model:   meta-llama/Llama-3.1-8B-Instruct is gated on Hugging Face. Accept the licence on
#          its model page, then:  export HF_TOKEN=hf_...
#          No token? Run with an ungated model instead:  MODEL=Qwen/Qwen3-8B bash h100-vllm-kv.sh
#          (the prediction then uses that model's own config.json and weight size).
#
# What it does:
#   1. checks the GPU and pip installs vllm (it brings its own torch)
#   2. starts `vllm serve` at --gpu-memory-utilization 0.9 and then 0.5 (max length 8,192),
#      greps the start-up log for the KV-cache lines, and prints them next to our prediction:
#        KV bytes  ~ util x total GPU memory - weights (- activations and CUDA graphs)
#        tokens    = KV bytes / KV bytes per token (Llama-3.1-8B: 131,072 B = 128 KiB)
#      Recent vLLM prints lines like "Available KV cache memory: ... GiB" and
#      "GPU KV cache size: N tokens, Maximum concurrency for 8,192 tokens per request: X.XXx";
#      older versions print "# GPU blocks: N". The exact wording changes between versions,
#      so the grep is loose; read the whole log (kept in $WORKDIR) if nothing matches.
#   3. runs `vllm bench serve` (the benchmark CLI, docs: https://docs.vllm.ai/en/latest/cli/bench/serve.html)
#      with the prefix_repetition dataset: one 2,000-token prefix shared by every request plus
#      a 100-token random suffix, first with --enable-prefix-caching, then with
#      --no-enable-prefix-caching (flags: https://docs.vllm.ai/en/latest/configuration/engine_args.html),
#      and prints the mean TTFT of both.
#
# Usage: bash h100-vllm-kv.sh          (optional env: MODEL, WORKDIR, PORT, NUM_PROMPTS, RATE)

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
WORKDIR="${WORKDIR:-$HOME/h100-vllm-kv}"
PORT="${PORT:-8000}"
MAXLEN=8192
PREFIX_LEN=2000
SUFFIX_LEN=100
OUT_LEN=64
NUM_PROMPTS="${NUM_PROMPTS:-200}"
RATE="${RATE:-4}"          # requests per second; low enough that TTFT is not all queueing
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "== 1. GPU check and install =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: this box has no NVIDIA driver." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
  echo "Warning: this is not an H100; the 80 GB prediction below will be off." >&2
fi
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"

if [[ "$MODEL" == meta-llama/* ]] && [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set and $MODEL is gated." >&2
  echo "Either export HF_TOKEN=hf_... (after accepting the licence) or run MODEL=Qwen/Qwen3-8B bash $0" >&2
  exit 1
fi

python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet vllm
python3 -c "import vllm; print('vllm', vllm.__version__)"
command -v vllm >/dev/null 2>&1 || { echo "vllm CLI not on PATH (try ~/.local/bin)." >&2; exit 1; }

echo "== Downloading $MODEL (about 16 GB) =="
python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
PY

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

kv_lines() {
  # Loose on purpose: vLLM's log wording changes between versions.
  grep -iE "model loading took|available kv cache memory|kv cache size|maximum concurrency|gpu blocks|num_gpu_blocks|cuda graph memory" "$1" \
    | sed -E 's/^.*\] //' | sort -u || echo "   (no KV lines matched; read $1)"
}

predict() {
  # predict <util> <logfile>: our back-of-envelope next to what vLLM reports
  python3 - "$MODEL" "$1" "$GPU_MIB" "$2" <<'PY'
import json, re, sys
from huggingface_hub import hf_hub_download
model, util, gpu_mib, log = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
cfg = json.load(open(hf_hub_download(model, "config.json")))
L = cfg["num_hidden_layers"]
K = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
H = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
per_tok = 2 * L * K * H * 2                                  # K and V, BF16
text = open(log, errors="ignore").read()
m = re.search(r"[Mm]odel loading took ([\d.]+) ?GiB", text)
weights = float(m.group(1)) * 2**30 if m else 16.06e9          # course fact for Llama-3.1-8B
for label, total in [("80 GB (course fact)", 80e9), (f"{gpu_mib:.0f} MiB (nvidia-smi)", gpu_mib * 2**20)]:
    kv = util * total - weights
    print(f"   predicted, total = {label}: util {util} x total - weights {weights/1e9:.2f} GB = "
          f"{kv/1e9:.2f} GB = {kv/2**30:.2f} GiB -> {kv/per_tok:,.0f} tokens "
          f"({per_tok:,} B/token), {kv/per_tok/16:,.0f} blocks of 16, {kv/per_tok/8192:.1f} x 8k sequences")
print("   (upper bounds: activations and CUDA graphs take a few GB more, so vLLM should report somewhat less)")
PY
}

run_bench() {
  local tag="$1"
  vllm bench serve --backend vllm --model "$MODEL" --port "$PORT" \
    --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len "$PREFIX_LEN" \
    --prefix-repetition-suffix-len "$SUFFIX_LEN" \
    --prefix-repetition-num-prefixes 1 \
    --prefix-repetition-output-len "$OUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$RATE" --seed 0 \
    --save-result --result-filename "bench-$tag.json" | tee "bench-$tag.txt"
}

echo "== 2a. vllm serve at gpu_memory_utilization 0.9 (prefix caching on) =="
start_server serve-0.9-prefix-on.log --gpu-memory-utilization 0.9 --enable-prefix-caching
echo "-- what vLLM reports:"; kv_lines serve-0.9-prefix-on.log
echo "-- our prediction:";    predict 0.9 serve-0.9-prefix-on.log

echo "== 3a. benchmark: shared ${PREFIX_LEN}-token prefix, prefix caching ON =="
run_bench prefix-on
stop_server

echo "== 3b. same benchmark, prefix caching OFF =="
start_server serve-0.9-prefix-off.log --gpu-memory-utilization 0.9 --no-enable-prefix-caching
run_bench prefix-off
stop_server

echo "== 2b. vllm serve at gpu_memory_utilization 0.5 =="
start_server serve-0.5.log --gpu-memory-utilization 0.5
echo "-- what vLLM reports:"; kv_lines serve-0.5.log
echo "-- our prediction:";    predict 0.5 serve-0.5.log
stop_server

echo
echo "== Summary =="
for f in serve-0.9-prefix-on.log serve-0.5.log; do
  echo "$f:"; kv_lines "$f" | grep -iE "kv cache size|concurrency|gpu blocks|available" || true
done
python3 - "$PREFIX_LEN" "$SUFFIX_LEN" <<'PY'
import json, sys
prefix, suffix = int(sys.argv[1]), int(sys.argv[2])
r = {t: json.load(open(f"bench-{t}.json")) for t in ("prefix-on", "prefix-off")}
for t, d in r.items():
    print(f"mean TTFT, prefix caching {t[7:]:>3}: {d['mean_ttft_ms']:8.1f} ms "
          f"(median {d['median_ttft_ms']:.1f} ms, {d['completed']} requests)")
print(f"ratio off/on: {r['prefix-off']['mean_ttft_ms'] / r['prefix-on']['mean_ttft_ms']:.1f}x")
floor = lambda n: 2 * 8.03e9 * n / 989e12 * 1e3
print(f"compare the prefill floors at 100% MFU (Llama-3.1-8B, 989 TFLOPS): "
      f"{prefix + suffix} tokens {floor(prefix + suffix):.1f} ms vs {suffix} tokens {floor(suffix):.1f} ms")
PY
echo "Logs and JSON results are in $WORKDIR. Stop the instance now."

# Try this:
#   RATE=inf NUM_PROMPTS=500 bash h100-vllm-kv.sh   -> TTFT becomes queueing; watch the ratio change
#   add --block-size 32 to both start_server calls  -> fewer, bigger blocks (paper's §7.2 ablation)
#   add --kv-cache-dtype fp8 to the 0.9 server       -> roughly twice the KV tokens (FP8 KV)
