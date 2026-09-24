
## E1 · step time vs batch size (decode, 2,048-token context each)

| B | step ms | total tok/s | per-user tok/s |
|---|---|---|---|
| 1 | 4.9 | 205 | 205 |
| 8 | 5.4 | 1,472 | 184 |
| 32 | 7.4 | 4,349 | 136 |
| 64 | 9.9 | 6,450 | 101 |
| 128 | 15.1 | 8,505 | 66 |
| 238 | 23.9 | 9,973 | 42 |

## E2 · batching strategies (1,000 requests, prompt median 512, output median 200; SLO TTFT<=1 s, TPOT<=50 ms)


**2 req/s**

| strategy | out tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | in SLO | padding waste |
|---|---|---|---|---|---|---|
| static B=32 | 511 | 8,698.0 | 21,649.5 | 8.0 | 1% | 71% |
| dynamic B=32, 50 ms window | 518 | 1,957.1 | 8,064.8 | 5.4 | 27% | 58% |
| continuous (whole-prompt prefill) | 521 | 11.0 | 58.6 | 5.0 | 100% | 0% |
| continuous + chunked prefill (2,048) | 521 | 11.1 | 59.5 | 5.0 | 100% | 0% |

**20 req/s**

| strategy | out tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | in SLO | padding waste |
|---|---|---|---|---|---|---|
| static B=32 | 913 | 118,456.7 | 232,963.9 | 8.0 | 0% | 71% |
| dynamic B=32, 50 ms window | 930 | 112,814.0 | 226,609.2 | 8.0 | 0% | 70% |
| continuous (whole-prompt prefill) | 4,756 | 16.2 | 75.4 | 7.2 | 100% | 0% |
| continuous + chunked prefill (2,048) | 4,755 | 16.1 | 83.1 | 7.1 | 100% | 0% |

**60 req/s**

| strategy | out tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | in SLO | padding waste |
|---|---|---|---|---|---|---|
| static B=32 | 916 | 133,939.4 | 264,861.7 | 8.0 | 0% | 71% |
| dynamic B=32, 50 ms window | 892 | 144,444.8 | 273,920.9 | 8.3 | 0% | 71% |
| continuous (whole-prompt prefill) | 9,088 | 1,437.7 | 4,869.7 | 23.8 | 41% | 0% |
| continuous + chunked prefill (2,048) | 9,102 | 1,396.9 | 4,816.4 | 23.7 | 41% | 0% |

**capacity: highest rate (req/s) with >= 99% of requests inside the SLO, 3,000-request traces (long enough to reach steady state; 600-request traces overstate it)**

- static B=32 at 1 req/s: 0% in SLO
- dynamic B=32, 50 ms at 1 req/s: 54% in SLO
- continuous + chunked, course SLO (TTFT<=1 s, TPOT<=50 ms): 40 req/s
- continuous + chunked, tight SLO (TTFT<=350 ms, TPOT<=25 ms): 38 req/s

## E3 · chunked prefill vs whole-prompt prefill (long prompts: median 2,048, rate 8 req/s)

| config | out tok/s | TTFT p50 ms | TTFT p99 ms | ITL p99 ms | ITL max ms | TPOT p99 ms |
|---|---|---|---|---|---|---|
| whole prompt (no chunking) | 2,009 | 48.6 | 200.9 | 68.7 | 185.1 | 15.5 |
| chunked, budget 256 | 2,033 | 82.4 | 375.5 | 7.2 | 7.4 | 7.1 |
| chunked, budget 512 | 2,023 | 61.0 | 211.0 | 11.2 | 11.7 | 9.8 |
| chunked, budget 1,024 | 2,015 | 54.1 | 187.6 | 19.7 | 20.2 | 13.1 |
| chunked, budget 2,048 | 2,011 | 51.5 | 179.4 | 36.2 | 37.1 | 14.5 |
| chunked, budget 4,096 | 2,009 | 48.9 | 195.7 | 68.3 | 70.4 | 15.3 |

One 4,096-token prefill alone: 66.7 ms; joined to 64 decodes at 2k: 72.8 ms vs 9.9 ms for the decodes alone

## E4 · KV memory, admission and preemption (rate 30 req/s, 1,000 requests)

| KV blocks (tokens) | ~GB | out tok/s | TTFT p99 ms | TPOT p50 ms | preemptions | recomputed tokens |
|---|---|---|---|---|---|---|
| 30,469 (487,504) | 63.9 | 6,736 | 106.7 | 10.0 | 0 | 0 |
| 8,000 (128,000) | 16.8 | 6,736 | 106.7 | 10.0 | 0 | 0 |
| 4,000 (64,000) | 8.4 | 6,108 | 4,146.3 | 9.8 | 63 | 112,062 |
| 2,000 (32,000) | 4.2 | 4,063 | 25,883.0 | 7.5 | 270 | 625,438 |
| 1,000 (16,000) | 2.1 | 2,481 | 67,199.0 | 6.1 | 469 | 995,150 |

max_num_seqs sweep at full memory, rate 60 (overload):

| max_num_seqs | out tok/s | TPOT p50 ms | TTFT p50 ms |
|---|---|---|---|
| 16 | 2,694 | 5.8 | 38,875.0 |
| 64 | 6,407 | 9.1 | 9,654.4 |
| 128 | 8,056 | 14.0 | 4,709.9 |
| 256 | 8,784 | 24.7 | 1,862.3 |
| 512 | 8,980 | 37.2 | 519.7 |

## E5 · admission policy (rate 55 req/s = just past capacity, 1,000 requests, max_num_seqs 64)

| policy | TTFT p50 ms | TTFT p99 ms | E2E p50 s | E2E p99 s | worst E2E s | out tok/s |
|---|---|---|---|---|---|---|
| FCFS | 7,489.8 | 16,610.9 | 10.16 | 21.68 | 22.40 | 6,477 |
| SJF, perfect length oracle | 97.2 | 27,650.3 | 2.01 | 32.29 | 37.34 | 6,517 |
| SJF, predictor with 50% noise | 93.1 | 27,071.0 | 2.10 | 32.44 | 36.50 | 6,583 |
| SJF, predictor with 100% noise | 109.1 | 26,969.8 | 2.24 | 32.89 | 37.74 | 6,482 |

Priority classes (10% of requests priority 0 = interactive, 90% priority 1 = batch), same load:

| policy | class | TTFT p50 ms | TTFT p99 ms |
|---|---|---|---|
| fcfs | interactive | 7,395.3 | 16,449.3 |
| fcfs | batch | 7,500.4 | 16,610.9 |
| priority | interactive | 36.1 | 121.1 |
| priority | batch | 9,759.5 | 16,674.9 |

## E6 · fairness: tenant A sends 90% of traffic, B 10% (rate 60 req/s = overload, max_num_seqs 64)

| policy | tenant | requests | TTFT p50 ms | TTFT p99 ms | share of output tokens |
|---|---|---|---|---|---|
| fcfs | A | 889 | 9,278.5 | 20,831.4 | 88% |
| fcfs | B | 111 | 8,353.4 | 20,607.2 | 12% |
| fair | A | 889 | 10,974.0 | 20,858.3 | 79% |
| fair | B | 111 | 46.0 | 181.4 | 21% |

## E7 · prefix caching: 4 shared system prompts of 2,048 tokens + a unique suffix (median 128), rate 20 req/s

| config | prefix hit rate | TTFT p50 ms | TTFT p99 ms | out tok/s | prefill tokens computed |
|---|---|---|---|---|---|
| no prefix cache | 0.0% | 3,343.5 | 9,482.4 | 3,733 | 1,779,902 |
| prefix cache (block hashing) | 91.6% | 16.0 | 35.2 | 4,830 | 149,694 |

## E8 · the 8-request toy trace, step by step (budget 16 tokens, max 4 seqs, 12 blocks of 4 tokens)

Requests (id: arrival ms, prompt, output): 0: 0, 12, 4, 1: 0, 5, 9, 2: 0, 20, 2, 3: 1, 7, 6, 4: 2, 30, 3, 5: 4, 4, 8, 6: 6, 16, 5, 7: 10, 9, 7

| step | t start ms | scheduled (id:tokens, P=prefill D=decode) | preempted | running | waiting | free blocks |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 0:12P 1:4P | – | 2 | 1 | 8 |
| 1 | 4.79 | 0:1D 1:1P 2:14P | – | 3 | 3 | 2 |
| 2 | 9.59 | 0:1D 1:1D 2:6P | – | 3 | 4 | 1 |
| 3 | 14.39 | 0:1D 1:1D 2:1D | – | 3 | 5 | 0 |
| 4 | 19.18 | 1:1D 3:7P 4:8P | – | 3 | 3 | 6 |
| 5 | 23.98 | 1:1D 3:1D 4:14P | – | 3 | 3 | 1 |
| 6 | 28.77 | 1:1D 3:1D | 4 | 2 | 4 | 6 |
| 7 | 33.57 | 1:1D 3:1D 4:14P | – | 3 | 3 | 2 |
| 8 | 38.36 | 1:1D 3:1D | 4 | 2 | 4 | 6 |
| 9 | 43.16 | 1:1D 3:1D 4:14P | – | 3 | 3 | 1 |
| 10 | 47.95 | 4:16P | – | 1 | 3 | 4 |
| 11 | 52.75 | 4:1D 5:4P 6:11P | – | 3 | 1 | 0 |
| 12 | 57.54 | 4:1D 5:1D | 6 | 2 | 2 | 2 |
| 13 | 62.34 | 5:1D 6:15P | – | 2 | 1 | 6 |
| 14 | 67.13 | 5:1D 6:1P 7:9P | – | 3 | 0 | 3 |
| 15 | 71.93 | 5:1D 6:1D 7:1D | – | 3 | 0 | 2 |

Total steps 21, preemptions 3. Per request (TTFT ms, finish ms): 0: 4.8/19.2, 1: 9.6/48.0, 2: 14.4/19.2, 3: 23.0/48.0, 4: 50.7/62.3, 5: 53.5/91.1, 6: 65.9/91.1, 7: 61.9/100.7

## E9 · scheduler CPU cost per step, Python on the Apple M3 (single-threaded; the Mac was busy, so treat as ±50%)

Command: `python3 experiments.py e9` (run twice, 2026-09-24; both runs shown as a range)

| policy | waiting | running | µs per step() |
|---|---|---|---|
| fcfs | 100 | 100 | 510–598 |
| fcfs | 10,000 | 256 | 848–894 |
| sjf | 100 | 100 | 340–343 |
| sjf | 10,000 | 256 | 869–898 |
| fair | 100 | 100 | 360–376 |
| fair | 10,000 | 256 | 888–1,176 |

Cost scales with the number of *running* requests, not with the waiting queue (admission only peeks at its head).
