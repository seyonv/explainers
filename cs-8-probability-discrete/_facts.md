# Shared facts: Probability I: counting, conditioning & discrete distributions (cs-8-probability-discrete)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
The cheat-sheet's General Probability page and the discrete half of Univariate Random Variables: every problem solved, the theory the source leaves out added, and the source's wrong answers corrected.
- 16 concept cards plus `_overview.html` (written last by the main agent).
- **Running example / conventions for this course:** The source is an exam-P style problem set with no theory and many unsolved problems. Each card: (1) the concept and its formulas, (2) the source's problem, solved step by step with every number computed in Python (fractions.Fraction for exact answers), (3) a simulation check where it's cheap (fixed seed, ≤ 10⁶ trials). Include a small reference strip (pmf, mean, variance, MGF) on every distribution card.

## Series facts (shared by every cs-* course)

### The source
- **ljeng/cheat-sheet**, commit `5cedb05` (2026-09-16), https://github.com/ljeng/cheat-sheet. Local copy (read your section in full before writing): `~/Desktop/repos/explainers/tasks/cs-kit/source/`.
- In the card footer, cite the exact source section, e.g. `Source: <a href="https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#quicksort">cheat-sheet › Sorting › Quicksort</a>`. For supplemented cards (`[S]`), say "The source heading is empty; this card is supplemented" and cite the references you used instead (CLRS, OSTEP, Hull, DDIA, the paper, the Python docs…).
- The source is terse: mostly problem statements plus one-line solutions, or dense paper notes. **Your job is to teach the idea**, not transcribe. Keep the source's own example or problem as the worked example wherever there is one, so the card is faithful to it.
- When the source is wrong, show the correct thing and add one muted line: "The source's version has X; corrected here." Never silently copy a bug.

### The series (link sibling cards as `../<course-slug>/<card>.html`, using the card list at the end of this section)
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
| `np.sort` 10⁶ float64, kind="quicksort" (introsort) / "stable" (timsort for floats; radix only for ≤16-bit ints) | 0.031 s / 0.072 s |
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


## Card list (cs-8-probability-discrete)
| # | File | Title | Group | Source section | Brief |
|---|---|---|---|---|---|
| 1 | counting-inclusion-exclusion.html | Inclusion–exclusion | Counting | [probability/general-probability.md#discrete-mathematics](https://github.com/ljeng/cheat-sheet/blob/main/probability/general-probability.md#discrete-mathematics) | Source: multiples of neither 6 nor 9 up to 1000 → 778 (correct in source); the three-way Venn (young, female, single = 880). Show the formula for 2 and 3 sets and verify both by brute force. |
| 2 | stars-and-bars.html | Stars and bars | Counting | [probability/general-probability.md#discrete-mathematics](https://github.com/ljeng/cheat-sheet/blob/main/probability/general-probability.md#discrete-mathematics) | Source: $20k in $1k units across 4 options with minimums 2/2/3/4 (stated only in the source: read the exact wording, solve it, and check by brute force). The C(n+k−1, k−1) derivation with a picture. |
| 3 | rook-placements-bridge.html | Choosing without conflicts, and bridge hands | Counting | [probability/general-probability.md](https://github.com/ljeng/cheat-sheet/blob/main/probability/general-probability.md) | Source: 6×5 grid, choose 3 cells with no shared row or column → C(6,3)·5·4·3 = 1,200; bridge hands (ace and king of at least one suit; all four of some denomination), stated only: solve both with inclusion–exclusion, exact fractions, and verify with a simulation of 10⁵ deals. |
| 4 | conditional-independence.html | Conditional independence | Conditioning | [probability/general-probability.md#basic-concepts-of-probability](https://github.com/ljeng/cheat-sheet/blob/main/probability/general-probability.md#basic-concepts-of-probability) | Source: 3-judge panel, P(guilty)=0.7, judges vote guilty w.p. 0.7 if guilty, 0.2 if innocent (stated only: read the exact question and solve it). Show P(J1 and J2) ≠ P(J1)P(J2) unconditionally but equal given G. |
| 5 | bayes-table.html | Bayes' theorem as a table | Conditioning | [probability/general-probability.md#basic-concepts-of-probability](https://github.com/ljeng/cheat-sheet/blob/main/probability/general-probability.md#basic-concepts-of-probability) | Source: smokers (≈0.42), car model year (0.008/0.036 ≈ 0.22), vaccine shipment with a binomial likelihood (n=25, 2 bad; stated only: solve). One prior×likelihood table layout used for all three. |
| 6 | convolution-discrete.html | Adding random variables: convolution | Conditioning | [probability/general-probability.md#basic-concepts-of-probability](https://github.com/ljeng/cheat-sheet/blob/main/probability/general-probability.md#basic-concepts-of-probability) | Source: P(N=n) = 1/2^(n+1), P(two-week total = 7) = 1/64 (correct). Show the convolution sum term by term, then the generating-function view. |
| 7 | first-step-analysis.html | First-step analysis | Conditioning | [probability/general-probability.md](https://github.com/ljeng/cheat-sheet/blob/main/probability/general-probability.md) | Source: craps (stated only: answer 244/495 ≈ 0.4929; derive it) and the Markov thief escape time 18 h (Applications in univariate-random-variables.md; correct). The equations-from-the-first-step method, solved with Fractions; a 10⁶-game craps simulation. |
| 8 | markov-chains-martingales.html | Markov chains and martingales | Conditioning | [probability/general-probability.md](https://github.com/ljeng/cheat-sheet/blob/main/probability/general-probability.md) | [S] Interview staples the source only touches via craps and the thief. Transition matrix of the thief problem, absorption probabilities via linear equations; gambler's ruin p=0.5 from 3 with target 10 → 0.3 and E[steps] = 3·7 = 21 via optional stopping on S_n and S_n² − n; fair vs unfair (p=0.49) ruin formula. Link cs-10 random-walks. |
| 9 | mgf-identify.html | Reading a distribution off its MGF | Discrete distributions | [probability/univariate-random-variables.md#discrete-univariate-distributions](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#discrete-univariate-distributions) | Source: the MGF → pmf Y ∈ {1,2,3} with 1/6, 2/6, 3/6; the Negative Binomial problem identified from its MGF as NB(5, 0.7) (answer only in source: show the matching). M'(0), M''(0) for the mean and variance. |
| 10 | sigma-coverage-zoo.html | How much lies within kσ? | Discrete distributions | [probability/univariate-random-variables.md#discrete-univariate-distributions](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#discrete-univariate-distributions) | Source: the coverage problem for Uniform(10), Bin(10, .6), Hypergeometric(5,10,30) (read the exact parameterisation), Poisson(4), Geometric(.25), NB(3,.5): stated only. Compute P(|X−μ| < kσ) for k=1,2,3 for all six exactly, table vs normal and Chebyshev's 1−1/k² bound. |
| 11 | binomial-beta-binomial.html | Binomial and beta-binomial | Discrete distributions | [probability/univariate-random-variables.md#binomial](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#binomial) | Source: Binomial › beta-binomial mixture (stated only: read and solve). Binomial reference strip; the mixture's inflated variance vs plain binomial with the same mean (compute). |
| 12 | geometric-memoryless.html | Geometric distribution | Discrete distributions | [probability/univariate-random-variables.md#geometric](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#geometric) | Source: p_{n+1} = p_n/5 → P(more than 1 claim) = 1/25 (correct). Reference strip; memoryless property proven in one line; the two conventions (trials vs failures). |
| 13 | hypergeometric-capture-recapture.html | Hypergeometric and capture–recapture | Discrete distributions | [probability/univariate-random-variables.md#hypergeometric](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#hypergeometric) | Source: tag 20, recapture 30 with 7 tagged (stated only: read the exact question; maximum-likelihood N ≈ 85; the Lincoln–Petersen estimate 20·30/7 = 85.7). Plot the likelihood over N and show where it peaks (compute). |
| 14 | negative-binomial.html | Negative binomial | Discrete distributions | [probability/univariate-random-variables.md#negative-binomial](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#negative-binomial) | Source: NB(5, 0.7) from an MGF (answer only). Reference strip; NB as a sum of r geometrics; its role as an over-dispersed Poisson for claim counts. |
| 15 | poisson-sum-and-moments.html | Poisson distribution | Discrete distributions | [probability/univariate-random-variables.md#poisson](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#poisson) | Source: sum of Poissons 0.3+0.5+0.7 = 1.5; P(X=2) = 4·P(X=3) → λ = 3/4, E[X²] = λ + λ² (stated only: finish it, = 21/16). Poisson as the limit of Bin(n, λ/n) (compute the difference for n=10,100,1000). |
| 16 | uniform-discrete-continuous.html | Uniform distributions | Discrete distributions | [probability/univariate-random-variables.md#uniform](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#uniform) | [S] for the discrete Uniform subsection (empty). Source continuous Uniform: real roots of a quadratic when Y ~ U(0,5): **the source says 2/5, which is wrong; the condition is Y ≥ 2 (read the quadratic to confirm) so the answer is 3/5**. Reference strips for both; mean and variance derivations. |
