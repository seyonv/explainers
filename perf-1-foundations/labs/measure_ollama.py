# Prefill vs decode, measured on your Mac.
#
# Sends three prompts of growing length to a local Ollama model and reads the
# timings Ollama reports for each request:
#   prompt_eval_count / prompt_eval_duration  -> prefill (reading the prompt)
#   eval_count / eval_duration                -> decode (writing the answer)
#
# Needs Ollama running and a pulled model:
#   ollama pull qwen3.5:4b
#   python3 labs/measure_ollama.py qwen3.5:4b
#
# It is a benchmark: close other heavy apps first, and expect small run-to-run
# differences. The card's numbers come from one run on an Apple M3 (24 GB),
# Ollama 0.34.2, qwen3.5:4b Q4_K_M, on 2026-09-22.
import json, urllib.request, sys

WEIGHTS_GB = 3.4   # qwen3.5:4b Q4_K_M file size; approximate bytes read per decode step
PEAK_GBS = 100     # Apple M3 memory bandwidth

def gen(model, prompt, n=128):
    body = json.dumps({"model":model,"prompt":prompt,"stream":False,"think":False,
        "options":{"num_predict":n,"temperature":0,"seed":1}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type":"application/json"})))
    return r

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:4b"
gen(model, "hi", 4)  # warm up: load the model so load time doesn't pollute the first row

print(f"model: {model}   (128 generated tokens per row, temperature 0)")
print(f"{'prompt tok':>10} {'prefill s':>10} {'prefill tok/s':>14} {'decode tok/s':>13} "
      f"{'decode ms/tok':>14} {'prefill/decode':>15} {'implied GB/s':>13}")
for plen in [1, 8, 32]:
    p = ("The history of computing is long. " * plen) + "Write a short story about a GPU."
    r = gen(model, p, 128)
    n_in, t_in = r["prompt_eval_count"], r["prompt_eval_duration"]/1e9
    n_out, t_out = r["eval_count"], r["eval_duration"]/1e9
    prefill_tps, decode_tps = n_in/t_in, n_out/t_out
    print(f"{n_in:>10} {t_in:>10.2f} {prefill_tps:>14.1f} {decode_tps:>13.2f} "
          f"{1000/decode_tps:>14.1f} {prefill_tps/decode_tps:>14.1f}x {decode_tps*WEIGHTS_GB:>13.1f}")

print(f"\nimplied GB/s = decode tok/s x {WEIGHTS_GB} GB read per token (approximate: the file also")
print(f"holds a vision encoder that text decoding doesn't read). Peak for an M3 is ~{PEAK_GBS} GB/s.")

# Try this:
# 1. Make the prompt much longer (e.g. [1, 8, 32, 128, 512]). Prefill tok/s should keep
#    climbing until the chip's compute saturates, while decode tok/s drifts slowly down.
# 2. Change num_predict from 128 to 512: decode tok/s barely moves, so total time grows
#    almost linearly with answer length.
# 3. Try a smaller model (ollama pull qwen3:1.7b) and check that decode tok/s x file size
#    lands in the same ballpark of implied GB/s (use that model's file size).
