# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Rung B: the same 24 requests, sent one at a time and then several at once.

Part 1 is the mistake: a plain for-loop that you might call "24 users". Each request waits for the
one before it, so the server only ever sees one.
Part 2 keeps C requests in flight at once (a closed loop: a new one starts when one finishes),
for C = 1, 2, 4, 8, and checks Little's law: requests in flight = request rate x time per request.

Setup (once):   ollama pull llama3.2:1b
Run:            uv run --python 3.12 concurrency.py
A server that batches (llama.cpp's server with 8 slots, on its own port):
                llama-server -m MODEL.gguf -ngl 99 -np 8 -c 8192 --cont-batching --port 18777
                uv run --python 3.12 concurrency.py --base-url http://127.0.0.1:18777/v1
                (`ollama show --modelfile llama3.2:1b` prints the .gguf path on its FROM line)
Other server:   uv run --python 3.12 concurrency.py --base-url http://localhost:8000/v1 --model NAME --api-key KEY
"""
import argparse, asyncio, statistics, time
from openai import AsyncOpenAI, OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--api-key", default="ollama")
ap.add_argument("--model", default="llama3.2:1b")
ap.add_argument("--requests", type=int, default=24)
ap.add_argument("--max-tokens", type=int, default=32)
a = ap.parse_args()

def msgs(i):  # a different number in every prompt, so no request reuses another's cached prompt
    return [{"role": "user", "content": f"Ticket {i}: list ten reasons to test software before shipping it."}]

kw = dict(model=a.model, max_tokens=a.max_tokens, temperature=0)
sync, aclient = OpenAI(base_url=a.base_url, api_key=a.api_key), AsyncOpenAI(base_url=a.base_url, api_key=a.api_key)
sync.chat.completions.create(messages=msgs(0), **{**kw, "max_tokens": 1})          # warm-up, not timed

def row(label, wall, lat, toks):
    rate, mean = len(lat) / wall, statistics.mean(lat)
    print(f"{label:<22} {wall:>6.2f} s {sum(toks) / wall:>9.1f} {rate:>8.2f} {mean:>9.2f} s {max(lat):>8.2f} s {rate * mean:>10.2f}")

print(f"{a.requests} requests, up to {a.max_tokens} output tokens each, model {a.model}\n")
print(f"{'':<22} {'wall':>8} {'tok/s':>9} {'req/s':>8} {'mean lat':>11} {'max lat':>10} {'rate x lat':>10}")

# Part 1: the blocking loop.
lat, toks, t0 = [], [], time.perf_counter()
for i in range(a.requests):
    t = time.perf_counter()
    r = sync.chat.completions.create(messages=msgs(100 + i), **kw)
    lat.append(time.perf_counter() - t); toks.append(r.usage.completion_tokens)
row("blocking loop", time.perf_counter() - t0, lat, toks)

# Part 2: C requests in flight at once.
async def run(C):
    sem, lat, toks = asyncio.Semaphore(C), [], []
    async def one(i):
        async with sem:
            t = time.perf_counter()
            r = await aclient.chat.completions.create(messages=msgs(1000 * C + i), **kw)
            lat.append(time.perf_counter() - t); toks.append(r.usage.completion_tokens)
    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(a.requests)))
    row(f"async, C = {C} in flight", time.perf_counter() - t0, lat, toks)

async def main():
    for C in (1, 2, 4, 8):
        await run(C)
asyncio.run(main())
print("\nLittle's law: the last column (request rate x mean latency) is the average number of requests in flight.")
print("It sits a little under C because the run ends with fewer than C requests left.")
