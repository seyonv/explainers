# Shared facts: Operating systems & concurrency (cs-7-operating-systems)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
Every heading of the cheat-sheet's Operating Systems page, which has headings but no content, filled in with measured Python experiments on an Apple M3: processes, threads and the GIL, locks to monitors, deadlock, context switches and scheduling.
- 13 concept cards plus `_overview.html` (written last by the main agent).
- **Running example / conventions for this course:** Every card uses the measured M3 numbers in the series table (fib(27), pipe ping-pong, lost updates, asyncio). The source file is headings only, so every card is supplemented: cite OSTEP (https://pages.cs.wisc.edu/~remzi/OSTEP/) chapters and the Python docs.

## Series facts (shared by every cs-* course)

### The source
- **ljeng/cheat-sheet**, commit `5cedb05` (2026-09-16), https://github.com/ljeng/cheat-sheet. Local copy (read your section in full before writing): `~/Desktop/repos/explainers/tasks/cs-kit/source/`.
- In the card footer, cite the exact source section, e.g. `Source: <a href="https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#quicksort">cheat-sheet › Sorting › Quicksort</a>`. For supplemented cards (`[S]`), say "The source heading is empty; this card is supplemented" and cite the references you used instead (CLRS, OSTEP, Hull, DDIA, the paper, the Python docs…).
- The source is terse: mostly problem statements plus one-line solutions, or dense paper notes. **Your job is to teach the idea**, not transcribe. Keep the source's own example or problem as the worked example wherever there is one, so the card is faithful to it.
- When the source is wrong, show the correct thing and add one muted line: "The source's version has X; corrected here." Never silently copy a bug.

### The series (link siblings only as `../<slug>/index.html#<card>` for cards that exist when you write; otherwise plain text)
cs-0-map · cs-1-data-structures · cs-2-algorithms · cs-3-graphs-dp-math · cs-4-design-foundations · cs-5-distributed-systems · cs-6-data-systems · cs-7-operating-systems · cs-8-probability-discrete · cs-9-probability-continuous · cs-10-financial-math

### The reader
- Preparing for software-engineering, systems-design and research-engineering (quant-adjacent) interviews. Comfortable programming; wants the **why** and a trace they can redo on paper, then a Python snippet they can run.
- Machine: Apple M3 (4 performance + 4 efficiency cores), 24 GB, macOS 15.3. Python **3.10.20** is `python3`; also installed: 3.14.7 (`python3.14`) and free-threaded 3.14.6 (`~/.local/share/uv/python/cpython-3.14.6+freethreaded-macos-aarch64-none/bin/python3.14t`). numpy 2.2.6 under 3.10. **No scipy.**

### Code: Python only
- **Every code example on every card is Python**, including where the source used Java or C++. Rewrite it idiomatically; don't transliterate. Use the standard library first (`heapq`, `collections`, `bisect`, `itertools`, `functools`, `math`, `random`, `threading`, `asyncio`). numpy is OK where it teaches something. No scipy (write `math.erf`-based `norm_cdf` when you need it).
- The snippet must **run under `python3` (3.10)**. Run it, put its real output on the card (a `# →` comment or a small output block), and save the exact snippet to `DIR/code/<card-slug>.py`. The saved file ends with a `if __name__ == "__main__":` demo that prints what the card shows. Link it in the footer: `Run it: <a href="https://github.com/seyonv/explainers/blob/main/<course>/code/<card-slug>.py">code/<card-slug>.py</a>`.
- Keep code short: aim for 8–30 lines on the card. Lines **≤ 72 characters**. Show the core, and leave the full version in `code/`.
- Code block markup (add this CSS to the card's `<style>`, verbatim, and use it for all code):
```css
.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;line-height:1.5;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:10px 0;white-space:pre;overflow-x:auto;tab-size:4;color:var(--text)}
.code .k{color:var(--accent);font-weight:600}.code .c{color:var(--muted)}.code .o{color:var(--muted)}
```
  `<pre class="code">` with HTML-escaped content. `.k` = keywords (def, return, for, while, if, class…), `.c` = comments, `.o` = printed output lines. No other highlighting.
- Complexity line on every algorithm or data-structure card: `Time O(…) · Space O(…)` plus one sentence on *why*.

### Colour meanings (same on every card)
- green `--accent`: the answer, the current or active element (pointer, pivot, frontier), the winner/faster option, what's kept
- grey `--faint` / `--surface2`: already processed, visited or discarded; overhead; waiting
- neutral `--text`/`--muted`: data at rest (array cells, nodes, rows)
- red `--red`: ✗, a bug, a violation (overflow, deadlock, lost update), the worst value, a cutoff

### Local measurements (Apple M3, 2026-09-23; `tasks/cs-kit/measure.py`, `race2.py`, `gil.py`; best of 3 unless noted)
Only measurements in this table may be shown as "measured on an M3". Label anything else you time yourself "one run on the author's M3" and keep it light (< 5 s of CPU, no parallel benchmarks).
| What | Result |
|---|---|
| `sorted()` 10⁶ random floats (Timsort), 3.10 | 0.130 s |
| `sorted()` on already-sorted 10⁶ | 0.157 s (includes the inner sort; Timsort on sorted input alone is O(n)) |
| `np.sort` 10⁶, kind="quicksort" (introsort) / "stable" (radix/timsort) | 0.031 s / 0.072 s |
| pure-Python insertion sort, n = 5,000 random | 0.828 s (vs `sorted()` 0.00026 s) |
| 1,000 membership tests, n = 10⁵: list / set / dict | 0.232 s / 24.6 µs / 26.5 µs (≈ 9,400× list→set) |
| `for i in range(len(L)): L[i] += 17`, 10⁷ items / numpy `a += 17` | 0.582 s / 0.00286 s (≈ 203×) (the source's 10⁸ numbers: 9.64 s vs 0.188 s) |
| `s = s + str(i) + " "` × 10⁵ / `" ".join(...)` | 0.468 s / 0.0073 s |
| 4 × fib(27), CPU-bound, 3.10: serial / 4 threads / 4 processes | 0.121 s / 0.121 s / 0.032 s |
| same, 3.14.7 (GIL) | 0.086 / 0.082 / 0.022 s |
| same, 3.14.6 free-threaded (no GIL) | 0.045 / 0.017 / 0.013 s |
| 8 × `sleep(0.1)`: serial / 8 threads | 0.835 s / 0.105 s |
| `multiprocessing.Process` start+join (spawn) | 72 ms (median of 5) |
| pipe ping-pong round trip between 2 processes | 13.6 µs |
| `threading.Event` ping-pong round trip between 2 threads | 8.2 µs |
| uncontended `Lock.acquire()+release()` / `with lock:` | 58 ns / 87 ns |
| lost updates, 4 threads × 10⁵ of `v=c; sleep(0); c=v+1`, no lock | 100,096 of 400,000 (3.10); with Lock: 400,000 |
| 4 threads × 10⁶ of `c["n"] += 1`: 3.10 / 3.14 GIL / 3.14t free-threaded | 4,000,000 / 4,000,000 / **1,082,391** |
| 10,000 concurrent `asyncio.sleep(0.1)` | 0.25 s total |
| `getrusage`: 200 × `sleep(1 ms)` voluntary switches / fib(30) (0.129 s) involuntary switches | 200 / 25 |
| `sys.getswitchinterval()` default | 0.005 s |

### Every card in the series (use these exact paths for Related links)
- **cs-1-data-structures**: big-o-amortized, python-toolkit, hash-buckets, sliding-window-counts, sliding-window-median, monotonic-deque, deque-rolling-dp, stack-parsing, monotonic-stack, linked-bucket-list, binary-tree-traversal, fenwick-tree, trie-segment-tree, heaps, fibonacci-heap, lfu-cache, random-pick-structures, stateful-api-read4, edge-case-parsing, testing-your-code
- **cs-2-algorithms**: insertion-sort, quicksort, quickselect, merge-sort, merge-sort-counting, heapsort, radix-sort, sort-lower-bound-stability, index-as-hash, binary-search-on-answer, binary-search-partition, lis-patience, prefix-sum-ordered-set, two-pointers-intervals, kmp, rolling-hash, manacher, suffix-automaton, divide-and-conquer, greedy-wildcard-justify, greedy-candy-patching
- **cs-3-graphs-dp-math**: graph-representations, dfs-flood-fill, bfs-layers, bidirectional-bfs, union-find, toposort, bipartite, eulerian-path, dijkstra, a-star-potentials, bellman-ford-floyd, mst-kruskal-prim, recursion-to-dp, dp-string-matching, dp-palindrome-cuts, dp-grid, dp-state-machine, dp-decode-ways, dp-path-reconstruction, backtracking, bit-arithmetic, newton-sqrt, gcd-number-theory, discrete-math-modular, fibonacci-counting, alias-method, factorial-number-system
- **cs-4-design-foundations**: requirements-scoping, tradeoffs, simplicity, robustness, api-layers, interfaces-vs-abstract, classes-encapsulation, inheritance-polymorphism, ood-practice, back-of-envelope, averages-and-expected-value, slo-arithmetic, algorithm-beats-hardware, compiler-optimizations, for-loop-vectorization, binary-tree-pattern
- **cs-5-distributed-systems**: correlated-failure, image-resize-sizing, image-resize-architecture, partial-failure-cap, backpressure-metrics, network-unreliability, end-to-end-argument, delay-timeouts-congestion, request-path, service-discovery, load-balancing-consumers, eventual-consistency-limits, i-confluence, exactly-once-transfer, quorums-staleness, compaction-xa, consensus-raft, consistent-hashing, sharding-hot-keys, idempotency-retries
- **cs-6-data-systems**: database-indexes, lsm-trees, bigtable-column-families, cache-conscious-structures, cache-patterns, in-memory-db-anticaching, log-structured-memory, mapreduce-model, mapreduce-execution, graph-navigation, query-complexity-classes, semi-naive-datalog, gpu-fair-share, backfill-gang-scheduling, scheduler-as-product, order-book-to-system, aad-tape
- **cs-7-operating-systems**: processes, threads, locks, mutexes, semaphores, monitors, deadlock, livelock, context-switch-how, context-switch-initiation, context-switch-hardware, scheduling, modern-concurrency
- **cs-8-probability-discrete**: counting-inclusion-exclusion, stars-and-bars, rook-placements-bridge, conditional-independence, bayes-table, convolution-discrete, first-step-analysis, markov-chains-martingales, mgf-identify, sigma-coverage-zoo, binomial-beta-binomial, geometric-memoryless, hypergeometric-capture-recapture, negative-binomial, poisson-sum-and-moments, uniform-discrete-continuous
- **cs-9-probability-continuous**: mixed-cdf, pareto-conditional, beta-quantiles, exponential-memoryless, weibull-gamma-function, normal-and-chebyshev, expected-revenue-overbooking, deductibles-and-limits, gini-lorenz, hierarchical-models, joint-table-conditional, order-statistics-min-max, sample-range-uniform, clt-sums, mgf-of-product, coupon-collector, normal-approx-binomial, conditional-on-sum, sample-size
- **cs-10-financial-math**: time-value-of-money, returns-log-returns, bonds-duration, portfolio-mean-variance, sharpe-and-capm, kelly-criterion, random-walks, brownian-motion-gbm, binomial-pricing, black-scholes-greeks, monte-carlo-pricing, ev-games-market-making


## Card list (cs-7-operating-systems)
| # | File | Title | Group | Source section | Brief |
|---|---|---|---|---|---|
| 1 | processes.html | Processes | Execution | [large-scale-design/operating-systems.md#processes](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#processes) | [S] Address space layout diagram (text, data, heap, stack), PCB contents, fork/exec, copy-on-write, IPC options. Measured: Process start+join 72 ms (spawn), 4 × fib(27) on 4 processes 0.032 s vs serial 0.121 s (3.8×). macOS default start method is spawn (since 3.8); fork vs spawn in the clarification. |
| 2 | threads.html | Threads and the GIL | Execution | [large-scale-design/operating-systems.md#threads](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#threads) | [S] Threads share heap, have own stack/registers; kernel vs user threads. The GIL: measured 4 × fib(27) serial/threads/processes on 3.10 (0.121/0.121/0.032), 3.14 (0.086/0.082/0.022) and free-threaded 3.14t (0.045/0.017/0.013); I/O-bound 8 × sleep(0.1) serial 0.835 s vs 8 threads 0.105 s. A bar chart of all of it. |
| 3 | locks.html | Locks and race conditions | Synchronization | [large-scale-design/operating-systems.md#locks](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#locks) | [S] Read-modify-write races. Measured: 4 threads × 10⁵ of read; sleep(0); write → 100,096 of 400,000 (75% lost), with Lock 400,000. Also: plain c += 1 lost nothing on 3.10 and 3.14-with-GIL but lost 73% on free-threaded 3.14t (1,082,391 of 4,000,000): **don't rely on the GIL**. Explain why += happened to survive with the GIL (switch checks happen between bytecodes, at specific points), without overclaiming. |
| 4 | mutexes.html | Mutexes | Synchronization | [large-scale-design/operating-systems.md#mutexes](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#mutexes) | [S] Mutex vs binary semaphore (ownership); Lock vs RLock (reentrant: show the deadlock of re-acquiring a Lock with acquire(timeout=0.1) returning False); spinlocks vs futex-style blocking; measured uncontended acquire+release 58 ns, `with lock` 87 ns; contention cost (lock convoys) in words. |
| 5 | semaphores.html | Semaphores | Synchronization | [large-scale-design/operating-systems.md#semaphores](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#semaphores) | [S] Dijkstra's P/V; counting vs binary. Python: Semaphore(3) around 10 fake downloads of 0.1 s each → total ≈ 0.4 s (4 waves; run and report); the in-flight count never exceeds 3 (record max). Bounded buffer with two semaphores. |
| 6 | monitors.html | Monitors and condition variables | Synchronization | [large-scale-design/operating-systems.md#monitors](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#monitors) | [S] Monitor = mutex + condition variables; Mesa vs Hoare (why `while` not `if`); Python threading.Condition producer/consumer with capacity 2 (print an interleaving); queue.Queue as a ready-made monitor; Java synchronized/wait/notify mentioned only as the origin of the term (no Java code). |
| 7 | deadlock.html | Deadlock | Concurrency issues | [large-scale-design/operating-systems.md#deadlock](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#deadlock) | [S] Two threads taking A→B and B→A; detect with acquire(timeout=1) (run it, show it times out); the four conditions and which one each fix breaks (global ordering breaks circular wait); wait-for graph cycle; dining philosophers with 5 forks in one diagram. |
| 8 | livelock.html | Livelock and starvation | Concurrency issues | [large-scale-design/operating-systems.md#livelock](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#livelock) | [S] Two 'polite' threads that release and retry; simulate deterministically (a discrete-time simulation is fine and honest: label it a simulation) with fixed vs jittered backoff: rounds until someone succeeds (compute average over 10,000 trials with a fixed seed). Starvation vs livelock vs deadlock table. Ethernet exponential backoff as the real-world fix. |
| 9 | context-switch-how.html | How a context switch works | Context switching | [large-scale-design/operating-systems.md#how-it-works](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#how-it-works) | [S] Registers, PC, SP, flags, FP/SIMD state saved to the kernel stack/PCB; switching page tables for process switches. Measured: pipe ping-pong between processes 13.6 µs per round trip (≥ 2 switches plus syscalls and Python overhead, so an upper bound on the raw switch; say so) vs thread Event round trip 8.2 µs. Literature raw switch cost ~1–5 µs incl. cache effects (cite a source, e.g. Tsuna's 2010 measurement or Li et al. 2007). |
| 10 | context-switch-initiation.html | What triggers a context switch | Context switching | [large-scale-design/operating-systems.md#how-its-initiated-by-the-operating-system](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#how-its-initiated-by-the-operating-system) | [S] Voluntary (blocking I/O, sleep, lock wait, yield) vs involuntary (time slice expiry via timer interrupt, higher-priority wakeup). Measured getrusage: 200 × sleep(1 ms) → 200 voluntary switches; fib(30) 0.129 s → 25 involuntary switches. CPython's own 5 ms GIL switch interval is a separate, user-level thing: say so. |
| 11 | context-switch-hardware.html | The hardware underneath | Context switching | [large-scale-design/operating-systems.md#underlying-hardware](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#underlying-hardware) | [S] User/kernel mode (rings; on the M3's ARM: EL0/EL1), the interrupt/exception vector, syscalls as traps, TLB and ASIDs (why tagged TLBs avoid flushes), cache pollution. Worked: a cold vs warm array sum in numpy (measure lightly: sum a 64 MB array twice, report both, label 'one run on the author's M3') to show cache/TLB warm-up. |
| 12 | scheduling.html | CPU scheduling | Scheduling and modern concurrency | [large-scale-design/operating-systems.md#scheduling](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#scheduling) | [S] Python simulator: jobs A=8, B=4, C=1 (all arrive at 0) under FCFS, SJF and RR(q=2): Gantt chart + average turnaround and response (compute). Convoy effect; MLFQ rules; CFS vruntime and red-black tree; macOS uses QoS classes on P/E cores (cite Apple docs). Priority inversion (Mars Pathfinder) in the clarification. |
| 13 | modern-concurrency.html | Modern concurrency constructs | Scheduling and modern concurrency | [large-scale-design/operating-systems.md#modern-concurrency-constructs](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/operating-systems.md#modern-concurrency-constructs) | [S] Measured: 10,000 asyncio.sleep(0.1) → 0.25 s total; compare to 10,000 threads (memory: default thread stack 512 KB on macOS: cite or measure lightly and label). Futures/executors (concurrent.futures); CSP channels (Go) vs actors (Erlang) in a table; compare-and-swap and lock-free counters; free-threaded 3.13+/3.14t (the fib numbers). When to pick which. |
