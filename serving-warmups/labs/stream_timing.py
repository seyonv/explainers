# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Rung A: time one streamed reply by hand.

Sends one chat request to a local OpenAI-compatible server (Ollama by default), records the clock
at every streamed chunk, and works out the numbers a load generator would report:
time to first token (TTFT), inter-token latency (ITL), output tokens per second, end-to-end time.

Setup (once):   ollama pull llama3.2:1b
Run:            uv run --python 3.12 stream_timing.py
Other server:   uv run --python 3.12 stream_timing.py --base-url http://localhost:8000/v1 --model NAME --api-key KEY
"""
import argparse, statistics, time
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--api-key", default="ollama")
ap.add_argument("--model", default="llama3.2:1b")
ap.add_argument("--max-tokens", type=int, default=120)
a = ap.parse_args()

client = OpenAI(base_url=a.base_url, api_key=a.api_key)
msgs = [{"role": "user", "content": "Explain in one paragraph why the sky is blue."}]

# Warm-up: the first request loads the model into memory. That wait is a cold start, not TTFT.
t = time.perf_counter()
client.chat.completions.create(model=a.model, messages=msgs, max_tokens=1)
print(f"warm-up request (includes loading the model): {time.perf_counter() - t:.2f} s\n")

t0 = time.perf_counter()
stamps, usage = [], None
stream = client.chat.completions.create(model=a.model, messages=msgs, max_tokens=a.max_tokens,
                                        temperature=0, stream=True, stream_options={"include_usage": True})
for chunk in stream:
    if chunk.usage:
        usage = chunk.usage
    if chunk.choices and chunk.choices[0].delta.content:
        stamps.append(time.perf_counter())
if len(stamps) < 2:
    raise SystemExit("The reply came back in fewer than two pieces, so there are no gaps to measure. Ask for a longer reply.")
t_end = stamps[-1]

n = len(stamps)
gaps = [b - x for x, b in zip(stamps, stamps[1:])]
ttft, e2e, itl = stamps[0] - t0, t_end - t0, statistics.mean(gaps)
print(f"chunks with text:        {n}")
if usage:
    print(f"server's token counts:   {usage.prompt_tokens} prompt, {usage.completion_tokens} output")
print(f"time to first token:     {1000 * ttft:.0f} ms")
print(f"inter-token latency:     mean {1000 * itl:.1f} ms, median {1000 * statistics.median(gaps):.1f} ms, "
      f"max {1000 * max(gaps):.1f} ms")
print(f"output speed:            {1 / itl:.1f} tokens/s   (1 / mean inter-token latency)")
print(f"end to end:              {e2e:.2f} s")
print(f"\ncheck: TTFT + (chunks - 1) x mean ITL = {ttft:.3f} + {n - 1} x {itl:.4f} = {ttft + (n - 1) * itl:.2f} s")
print(f"share of the wait that was TTFT: {100 * ttft / e2e:.0f}%")
