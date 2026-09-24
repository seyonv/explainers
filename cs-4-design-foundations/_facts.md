# Shared facts: Design foundations & object-oriented design (cs-4-design-foundations)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
The cheat-sheet's System Design framing pages: requirements, trade-offs, simplicity, object-oriented design with every Java example in Python, and the estimation and performance habits under all of it.
- 16 concept cards plus `_overview.html` (written last by the main agent).
- **Running example / conventions for this course:** Framing cards reuse the source's own TinyURL example: 100M new URLs/month, 100:1 read:write, 500-byte records, 5-year retention (these sizing inputs are illustrative; the source gives none): compute storage and QPS once in back-of-envelope and reuse.

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


## Card list (cs-4-design-foundations)
| # | File | Title | Group | Source section | Brief |
|---|---|---|---|---|---|
| 1 | requirements-scoping.html | Functional vs non-functional requirements | Framing a design | [large-scale-design/system-design/feature-sets.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/feature-sets.md) | Source: feature-sets.md. The source's questions (parking lot levels, vehicle types, push/pull, paid/free, tiers), TinyURL features, dashboard filters, the -ilities, 'store raw data, not summaries', the 3-question feature check. Make a two-column requirements table for TinyURL. |
| 2 | tradeoffs.html | Everything is a trade-off | Framing a design | [large-scale-design/system-design/tradeoffs.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/tradeoffs.md) | Source: tradeoffs.md. Put numbers on space vs time: bitmap of 4·10⁹ IPv4 addresses = 512 MB vs a set; two-pass half space double time. Rod-cutting DP in 10 lines. The secretary-problem 37% rule (compute 1/e). 32- vs 64-bit pointers. |
| 3 | simplicity.html | Simple is not easy | Framing a design | [large-scale-design/system-design/simplicity.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/simplicity.md) | Source: simplicity.md (Hickey, Moseley–Marks, the constructs table, the What/Who/How/When/Why checklist). Worked example: the same small feature written two ways in Python (stateful class vs pure function + explicit state) and count what each one couples. |
| 4 | robustness.html | Robustness | Framing a design | [large-scale-design/system-design/robustness.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/robustness.md) | Source: robustness.md (thin and scattered). Tic-tac-toe 3×3 → N×N win check in Python (O(1) per move with row/col/diag counters), substitution method for a recurrence, the '2.5× faster' robustness-vs-efficiency point. Saint-Exupéry quote. |
| 5 | api-layers.html | APIs as layers of abstraction | Framing a design | [large-scale-design/system-design/api-discussions.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/api-discussions.md) | Source: api-discussions.md (thin). Supplement: one record (a TinyURL link) shown at all four layers (Python dataclass → JSON → UTF-8 bytes with the real byte count → on disk). API concerns: versioning, degradation/load shedding, pagination, idempotency (link cs-5 idempotency-retries). |
| 6 | interfaces-vs-abstract.html | Interfaces vs abstract classes | Object-oriented design | [large-scale-design/system-design/interfaces.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/interfaces.md) | Source: interfaces.md: the 5-row comparison table, the StatusCallback example (the first block is labelled java but is C++), the Java interface version, XMLReader/XMLReaderImpl. **Rewrite all of it in Python**: abc.ABC with @abstractmethod vs typing.Protocol (structural), multiple inheritance as Python's mixin answer; show what happens when you instantiate an ABC with a missing method (the real TypeError text). Footnote [^11] is unreferenced in the source. |
| 7 | classes-encapsulation.html | Classes, objects and encapsulation | Object-oriented design | [large-scale-design/system-design/class-hierarchies.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/class-hierarchies.md) | Source: class-hierarchies.md › Point (**Java; rewrite in Python**): Point(5,10).relative_to(-5,5) → (0, 15). Python has no private: _name, __name mangling, @property, frozen dataclass as the idiomatic immutable Point. Show __repr__ output. |
| 8 | inheritance-polymorphism.html | Inheritance and polymorphism | Object-oriented design | [large-scale-design/system-design/class-hierarchies.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/class-hierarchies.md) | Source: Shape/Rectangle/Ellipse/paintShapes (**Java; rewrite in Python** with ABC). Compute bounds for Ellipse(center (0,0), a=3, b=2) → Rectangle w=6,h=4. Dynamic dispatch explained (MRO in Python). Composition over inheritance in the clarification. |
| 9 | ood-practice.html | Object-oriented design practice | Object-oriented design | [large-scale-design/system-design/class-hierarchies.md#the-parking-lot](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/class-hierarchies.md#the-parking-lot) | Source: Card/Deck mermaid, the Parking Lot object list (thin), call-centre escalation, online book reader ER, and 7 prompts with no content. Deliver a Python class sketch for the parking lot (Level, Spot, Vehicle, fits()) and a Fisher–Yates Deck.shuffle (the source only asks 'how does a Deck shuffle?'; answer it, with 52! ≈ 8.07e67). A table mapping each remaining prompt to its key classes. |
| 10 | back-of-envelope.html | Back-of-the-envelope estimation | Estimation and performance | [large-scale-design/system-design/resource-estimation-real-systems.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/resource-estimation-real-systems.md) | Source: resource-estimation (Mississippi 0.48 mi³/day; 2M 8-byte nodes costing 48 B each = 96 MB of 128 MB; n³ square roots at n=1000 = 200 s; Little's law 150 cases/25 per yr = 6 yr and the club example; Roebling's 6× derating). Add Python's own overhead: sys.getsizeof of an int/float/tuple/dict entry (measure). Then the TinyURL sizing from _facts. |
| 11 | averages-and-expected-value.html | Averages, expected value and chance | Estimation and performance | [large-scale-design/system-design/resource-estimation-real-systems.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/resource-estimation-real-systems.md) | Source: the statistics half of resource-estimation ($65K mean vs $25K median; pooled testing, prevalence 1/100, pools of 50 ≈ 21 tests expected: recompute; chuck-a-luck EV: **the source's formula is off; the true EV is −17/216 ≈ −$0.079**; Type I/II; secretary 37%). |
| 12 | slo-arithmetic.html | Availability and SLO arithmetic | Estimation and performance | [large-scale-design/system-design/distributed-systems.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/distributed-systems.md) | [S] The source gives 99.9% and 99.99% targets (distributed-systems.md image service) but never converts them. Table: 99%/99.9%/99.95%/99.99%/99.999% → downtime per 30-day month and year. Serial chain 0.9999^n for n=1,5,20; parallel redundancy 1−(1−p)^k and why correlated failure (cs-5) breaks it. Error budget: 43.2 min/month at 99.9%. |
| 13 | algorithm-beats-hardware.html | An algorithm beats a faster computer | Estimation and performance | [large-scale-design/system-design/compilers.md#algorithm-vs-hardware](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/compilers.md#algorithm-vs-hardware) | Source: compilers.md (Computer A 10¹⁰ instr/s, insertion 2n², n=10⁷ → 20,000 s ≈ 5.5 h; Computer B 10⁷ instr/s, merge 50 n lg n → 1,163 s ≈ 19 min). Recompute exactly. Proebsting's law (60%/yr hardware vs 4%/yr compilers). Tie to measured sorts. |
| 14 | compiler-optimizations.html | What compilers and interpreters do for you | Estimation and performance | [large-scale-design/system-design/compilers.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/compilers.md) | Source: compilers.md (the 'Tail-call Optimization' heading actually covers inlining in C++: note it). **The Java string-concatenation loop → Python**: measured s = s + str(i) 0.468 s vs join 0.0073 s for 10⁵. Show dis.dis of a tiny function; CPython does no TCO (recursion limit), constant folding exists (show dis of 2*3). PyPy/JIT in one line. |
| 15 | for-loop-vectorization.html | Why Python loops are slow | Estimation and performance | [large-scale-design/system-design/for-loop-problems.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/for-loop-problems.md) | Source: for-loop-problems.md (10⁸: 9.64 s vs 0.188 s; the five per-item CPython steps; np.abs(arr).mean() temporary; PyPy 25.9 → 0.48 s; Numba 4.85 → 0.064 s; SIMD). **The source says NumPy is 'implemented in Rust': it's C; correct it.** Use our measured 10⁷ numbers alongside the source's. |
| 16 | binary-tree-pattern.html | The binary tree as a design pattern | Estimation and performance | [large-scale-design/system-design/binary-trees.md](https://github.com/ljeng/cheat-sheet/blob/main/large-scale-design/system-design/binary-trees.md) | Source: binary-trees.md's four examples. Worked: Huffman for frequencies A:5,B:2,C:1,D:1 → A=0,B=10,C=110,D=111, bits = 5·1+2·2+1·3+1·3 = 15 vs 18 fixed-width (compute with heapq). BST degenerate O(n) vs balanced height log2(10⁶) ≈ 20. The credit decision tree. |
