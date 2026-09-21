# Shared facts: LLM speed and reliability

Every card writer read this before writing. The values below appear on the cards exactly as written here.

## The reader
- Setup: Apple M3, 24 GB, Ollama 0.34.2, qwen3.5:4b Q4_K_M. Also uses hosted APIs (Claude, open models via OpenRouter) and agent harnesses like Claude Code.
- Wants: an intuition for why a response or an agent task takes as long as it does, and what to change.

## The running example
"FizzBuzz in Python, code only" on the M3.
- Thinking off: 80–82 answer tokens, about 5.1–5.4 s, 15.7 tokens/s
- The longer "write FizzBuzz" answer: 653 tokens, 42.4 s, 15.4 tokens/s
- Thinking on: 1,764 tokens, 115.5 s, 15.3 tokens/s
- TTFT: 0.13 s warm with a 22-token prompt, 4.3 s cold (4.07 s load)

## Sourced numbers
| Fact | Value | Source | As of |
|---|---|---|---|
| gpt-oss-120b output speed by host | 48 (DeepInfra) to 1,745 (Cerebras) tokens/s, nine hosts | artificialanalysis.ai/models/gpt-oss-120b/providers | 2026-09-21 |
| gpt-oss-120b TTFT by host | 0.28–1.35 s (10k-token input) | same | 2026-09-21 |
| Claude Sonnet 5, non-reasoning | TTFT 1.39 s, 66 tokens/s | artificialanalysis.ai/leaderboards/models | 2026-09-21 |
| Claude Sonnet 5 effort ladder | 8.94 s none → 147.42 s max, end to end | same | 2026-09-21 |
| Sonnet 5 pricing | $2 / $10 per 1M input/output tokens, cache read $0.20 | OpenRouter endpoints API | 2026-09-21 |
| Host configs (precision, output cap, price) | per host, see _overview-hosts.html | openrouter.ai/api/v1/models/openai/gpt-oss-120b/endpoints | 2026-09-21 |
| Memory bandwidth | M3 100 GB/s, H100 3.35 TB/s, Cerebras WSE-3 21 PB/s | Apple, NVIDIA, The Register | — |

## Local measurements
| What | Command | Result | Date |
|---|---|---|---|
| Writing speed | `ollama run --verbose` / `/api/generate` eval rate | 15.4 tokens/s | 2026-09-21 |
| Prompt reading | 2,605-token prompt, `prompt_eval_duration` | 12.6 s fresh (33.9 s when the machine was busy), 0.41 s cached | 2026-09-21 |
| Network round trip | `curl -w "%{time_connect} %{time_appconnect}"`, 3 tries | localhost 0.3 ms · Anthropic 7–11 ms · DeepInfra 53–77 ms | 2026-09-21 |
| Latency spread | 20 identical 60-token requests | p50 4.66 s, p95 6.54 s, max 7.14 s | 2026-09-21 |

## Terms
time to first token (TTFT), output speed (tokens/s after the first token), output length, prefill, host, harness, time to a correct result.

## Colour meanings
- green `--accent`: the visible answer, the winner, the faster option
- grey `--faint`: hidden thinking, waiting, overhead
- red `--red`: ✗, cutoffs, waste, the slowest value

## Card list
Overviews: `_overview`, `_overview-hosts`, `_overview-agents` · Metrics: `ttft`, `output-speed`, `output-length` · Mechanisms: `memory-bandwidth`, `hardware`, `quantization`, `moe`, `speculative-decoding`, `prompt-caching` · Multipliers: `thinking-tokens`, `harness-calls`, `context-growth` · Agents: `tail-latency`, `retries-timeouts`, `compounding-errors`, `parallel-calls`, `model-routing`
