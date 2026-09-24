#!/usr/bin/env bash
# h100-cost-sweep.sh — measure the batch dial from cost-per-token.html on one rented H100 SXM.
#
#   vLLM serves Llama-3.1-8B-Instruct twice:
#     bf16   the card's running example (16 GB of weights)
#     fp8    online FP8 weights (--quantization fp8); KV stays BF16 unless KV_FP8=1
#   Each is hit with `vllm bench serve` at concurrency 1, 8, 32, 64, 128, 256
#   (1,536 tokens in + 512 out, so every sequence ends at the card's 2k context).
#   The summary prints output tok/s, tok/s per user (1000 / median TPOT) and
#   $/M output tokens = PRICE / (output tok/s x 3600) x 1e6, next to the card's
#   ceiling ("our model", labs/cost-model.py) for the same batch.
#
# Read the result as: measured / ceiling. The ceiling assumes 100% of HBM bandwidth and
# counts no prefill; vLLM's output throughput includes the time spent on prefill, so the
# measured $/M is higher. Expect the gap to be largest at concurrency 1.
#
# Status:   NOT YET RUN BY THE AUTHOR. Expect to fix small things (driver/CUDA versions,
#           vLLM flag renames). Flags checked against the vLLM docs on 2026-09-24:
#           docs.vllm.ai/en/latest/cli/bench/serve/ and the result keys in
#           vllm/benchmarks/serve.py (output_throughput, median_tpot_ms, median_ttft_ms).
# Runtime:  ~30-45 min (vLLM install, ~16 GB weight download, two server start-ups,
#           twelve benchmarks of 1-2 min each).
# Cost:     one H100 SXM on-demand, $4.29/hr at Lambda 1x (seen 2026-09-24) -> ~$2-3.
#           SHUT THE INSTANCE DOWN when this finishes.
# Needs:    Ubuntu with an NVIDIA driver (Lambda Stack has it), Python 3.10+, ~60 GB disk.
#           HF_TOKEN: Llama-3.1-8B-Instruct is gated. Accept Meta's licence on
#           huggingface.co/meta-llama/Llama-3.1-8B-Instruct, then `export HF_TOKEN=hf_...`.
# Run:      PRICE=4.29 bash h100-cost-sweep.sh     (PRICE = what you pay per GPU-hour)
#           results land in $OUT (default ~/h100-cost-results)
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PRICE=${PRICE:-3.99}                  # $/GPU-hour; the course default is Lambda 8x $3.99
OUT=${OUT:-$HOME/h100-cost-results}
WORK=${WORK:-$HOME/h100-cost-work}
PORT=${PORT:-8000}
KV_FP8=${KV_FP8:-0}                   # 1 = also store the KV cache in FP8 for the fp8 run
IN_LEN=1536
OUT_LEN=512
MAX_LEN=4096
CONCURRENCY="1 8 32 64 128 256"

: "${HF_TOKEN:?export HF_TOKEN=hf_... first (gated model; accept the licence on Hugging Face)}"
mkdir -p "$OUT" "$WORK"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "== node =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee "$OUT/gpus.csv"
echo "price: \$$PRICE per GPU-hour"

if [ ! -d "$WORK/venv" ]; then
  python3 -m venv "$WORK/venv"
fi
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
pip install -q -U pip
pip install -q -U vllm
vllm --version | tee "$OUT/vllm_version.txt"

echo "downloading weights (safetensors only, ~16 GB)..."
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

run_config() {   # name, extra vllm serve args...
  local name=$1; shift
  echo "-- $name: vllm serve $* --"
  CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --port "$PORT" \
      --max-model-len "$MAX_LEN" --max-num-seqs 256 "$@" > "$OUT/server_$name.log" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server $name died; see $OUT/server_$name.log"; tail -20 "$OUT/server_$name.log"; return 1
    fi
    sleep 10; waited=$((waited + 10))
    if [ "$waited" -ge 1200 ]; then echo "server $name not up after 20 min"; stop_server; return 1; fi
  done
  echo "up after ${waited}s"
  # vLLM logs how many tokens of KV cache fit; compare with the card's 238 x 2k sequences.
  grep -iE "KV cache|GPU blocks|Maximum concurrency" "$OUT/server_$name.log" | tail -3 || true

  for conc in $CONCURRENCY; do
    local n=$((conc * 4 < 32 ? 32 : conc * 4))
    vllm bench serve --model "$MODEL" --port "$PORT" \
      --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
      --ignore-eos --num-prompts "$n" --max-concurrency "$conc" \
      --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 \
      --save-result --result-dir "$OUT" --result-filename "bench_${name}_c${conc}.json" \
      | tee "$OUT/bench_${name}_c${conc}.txt"
  done
  stop_server
}

run_config bf16
if [ "$KV_FP8" = "1" ]; then
  run_config fp8 --quantization fp8 --kv-cache-dtype fp8
else
  run_config fp8 --quantization fp8
fi

# ---------------------------------------------------------------- summary
echo "== summary: measured vs the card's ceiling (labs/cost-model.py, 2k context, BF16) =="
python - "$OUT" "$PRICE" "$CONCURRENCY" <<'PY'
import json, sys, pathlib
out, price, concs = pathlib.Path(sys.argv[1]), float(sys.argv[2]), [int(c) for c in sys.argv[3].split()]
N, BW, C, KV_SEQ, W = 8.03e9, 3.35e12, 989e12, 2048 * 128 * 1024, 16.06e9

def ceiling(b):   # course 2 step formula, BF16, 2k context; beyond B=238 the KV cache no longer fits
    b = min(b, 238)
    return b / (b * KV_SEQ / BW + max(2 * b * N / C, W / BW))

usd = lambda tps: price / (tps * 3600) * 1e6
print(f"{'config':<7}{'conc':>5}{'out tok/s':>11}{'per user':>10}{'TTFT p50':>10}{'$/M out':>9}"
      f"{'ceiling tok/s':>15}{'ceiling $/M':>13}{'measured/ceiling':>18}")
for name in ("bf16", "fp8"):
    for c in concs:
        f = out / f"bench_{name}_c{c}.json"
        if not f.exists():
            print(f"{name:<7}{c:>5}  missing"); continue
        r = json.loads(f.read_text())
        tok = r.get("output_throughput", float("nan"))
        tpot = r.get("median_tpot_ms", float("nan"))
        cap = ceiling(c)
        print(f"{name:<7}{c:>5}{tok:>11.0f}{1000 / tpot:>10.0f}{r.get('median_ttft_ms', float('nan')):>10.1f}"
              f"{usd(tok):>9.3f}{cap:>15.0f}{usd(cap):>13.3f}{tok / cap:>18.0%}")
print("ceiling = BF16 model; for the fp8 rows the FP8 ceiling is about 2x higher at small batch.")
PY
echo "done. Results in $OUT. Now shut the instance down."
