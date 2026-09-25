# Shared facts: Probability II: continuous, multivariate & limit theorems (cs-9-probability-continuous)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
The continuous half of the source guide's Univariate Random Variables page, its applications, and the whole Multivariate page: every problem solved, empty sections filled, wrong answers corrected.
- 19 concept cards plus `_overview.html` (written last by the main agent).
- **Running example / conventions for this course:** Same card recipe as cs-8: concept + formulas, the source's problem solved step by step (exact where possible, else to 4 significant figures), and a cheap simulation check. There is no scipy: write norm_cdf with math.erf and invert it by bisection if needed. Integrate numerically with a simple Simpson's rule where needed.

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


## Card list (cs-9-probability-continuous)
| # | File | Title | Group | Source section | Brief |
|---|---|---|---|---|---|
| 1 | mixed-cdf.html | Mixed distributions | Continuous distributions | [probability/univariate-random-variables.md#continuous-univariate-distributions](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#continuous-univariate-distributions) | Source: two mixed-CDF problems. **The source's variance of 0 is wrong**: with a point mass of 0.5 at x=1 plus density x−1 on (1,2), Var = 5/36 ≈ 0.139 (re-derive from the source's exact CDF before trusting this). Solve the other (stated only) too. Plot the CDF with its jump. |
| 2 | pareto-conditional.html | Pareto distribution and conditional probability | Continuous distributions | [probability/univariate-random-variables.md#continuous-univariate-distributions](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#continuous-univariate-distributions) | Source: f(x) = 3x⁻⁴, conditional probability 0.578 (correct). Heavy tail vs exponential tail comparison (P(X > 10) for each with the same mean). |
| 3 | beta-quantiles.html | Beta distribution | Continuous distributions | [probability/univariate-random-variables.md#beta](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#beta) | Source: one line (stated only). Beta(2,5) pdf 30x(1−x)⁴, CDF in closed form, median and 95th percentile by bisection (compute), mean 2/7. Beta as a prior for a probability (link binomial-beta-binomial). |
| 4 | exponential-memoryless.html | Exponential distribution | Continuous distributions | [probability/univariate-random-variables.md#exponential](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#exponential) | [S] The heading is empty. Reference strip; memoryless proof; the pharmacy wait time from the multivariate page's Associated Applications (P(wait > 10) = e⁻¹) as the worked example; min of independent exponentials is exponential with summed rates. |
| 5 | weibull-gamma-function.html | Weibull distribution and the gamma function | Continuous distributions | [probability/univariate-random-variables.md#gamma](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#gamma) | Source: Gamma section, which is really a Weibull (rat survival): P(X > 100) = e^(−125/216) ≈ 0.561 (worked), E[X] = 120·Γ(4/3) ≈ 107.2 (not worked: derive). Say it's misfiled; add a Gamma reference strip too (Gamma(α, θ) as a sum of exponentials). |
| 6 | normal-and-chebyshev.html | The normal distribution and Chebyshev's bound | Continuous distributions | [probability/univariate-random-variables.md#normal](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#normal) | [S] Normal heading empty. Source Applications: blood pressure n=15: range rule s ≈ 16, x̄ = 136.07, s = 17.10, Chebyshev k=2 interval (102, 170) holds 14 of 15 (recompute all from the source's data). Normal strip with norm_cdf via math.erf. |
| 7 | expected-revenue-overbooking.html | Expected revenue: the overbooked bus | Applications | [probability/univariate-random-variables.md#applications](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#applications) | Source: 21 tickets, 20 seats. **The source says 996; the correct value is 1050 − 100·0.98²¹ = 984.57** (re-derive from the exact wording). Then optimise: expected revenue for selling 20..25 tickets (compute the best). |
| 8 | deductibles-and-limits.html | Deductibles and policy limits | Applications | [probability/univariate-random-variables.md#applications](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#applications) | Source: auto insurance with density printed as 0.5003e^(x/2) (**almost certainly e^(−x/2); say so**) and the reimbursement CDF G(115): **the source's 0.596 is wrong; P(X ≤ 150 | X > 20) = 1 − e^(−1.3) ≈ 0.727** (re-derive from the exact wording before trusting). Show the payment function graph. |
| 9 | gini-lorenz.html | Lorenz curve and Gini index | Applications | [probability/univariate-random-variables.md#applications](https://github.com/ljeng/cheat-sheet/blob/main/probability/univariate-random-variables.md#applications) | Source: f(x) = 3(1−x)² on [0,1] (stated only: compute the Lorenz curve and Gini exactly, and check numerically). Plot the Lorenz curve. |
| 10 | hierarchical-models.html | Hierarchical models | Multivariate | [probability/multivariate-random-variables.md#multivariate-distributions](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#multivariate-distributions) | Source: S ~ N(μ, σ²), R|S ~ N(s, 1) (stated only: E[R] = μ, Var(R) = σ² + 1, Cov(R, S) = σ²: derive via tower rule and total variance); the X-then-Y|X problem P(X + Y > 4) (stated only: solve). |
| 11 | joint-table-conditional.html | Joint and conditional distributions from a table | Multivariate | [probability/multivariate-random-variables.md#multivariate-distributions](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#multivariate-distributions) | Source: the joint table with one missing cell (= 0.23; verify it makes the table sum to 1), marginals, E[S | H = 73] and a conditional CV (stated only: compute all). Render the table with marginals. |
| 12 | order-statistics-min-max.html | Order statistics: min and max | Multivariate | [probability/multivariate-random-variables.md#order-statistics](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#order-statistics) | Source: min and max of 5 iid Exp(λ): 1 − e^(−5λa) and (1 − e^(−λa))⁵ (stated only: derive both, and E[max] = (1/λ)(1 + 1/2 + … + 1/5)). |
| 13 | sample-range-uniform.html | The range of a uniform sample | Multivariate | [probability/multivariate-random-variables.md#order-statistics](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#order-statistics) | Source: range of n uniforms (appears twice: duplicate): pdf n(n−1)r^(n−2)(1−r); E[R] = (n−1)/(n+1). Simulate n=5 and overlay (compute histogram vs pdf values). |
| 14 | clt-sums.html | Sums and the central limit theorem | Sums and limit theorems | [probability/multivariate-random-variables.md#linear-combinations](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#linear-combinations) | Source: job durations for 20 jobs each (P(A.J. < 900) ≈ 0.0127, P(M.J. < 900) ≈ 0.0184, P(A.J. finishes first) ≈ 0.690), charity total 90th percentile ≈ 6,342,543, Poisson trees in 100 acres P ≈ 0.646: all stated only: re-derive from the source's parameters. |
| 15 | mgf-of-product.html | The MGF of a product | Sums and limit theorems | [probability/multivariate-random-variables.md#linear-combinations](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#linear-combinations) | Source: stated only: M(t) = (1 − t²)^(−1/2), so E = 0 and Var = 1. Derive by conditioning on Y₂; check Var by simulation. |
| 16 | coupon-collector.html | Coupon collector | Sums and limit theorems | [probability/multivariate-random-variables.md#linear-combinations](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#linear-combinations) | Source: E = 14.7, Var = 38.99 (stated only: derive as a sum of geometrics). Generalise to n coupons: n·H_n (n=52 → 235.98). |
| 17 | normal-approx-binomial.html | Normal approximation to the binomial | Sums and limit theorems | [probability/multivariate-random-variables.md#associated-applications](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#associated-applications) | Source: P(at least 50 of 100 wait over 10 min) ≈ 0.004 with p = e⁻¹ (stated only: compute with and without continuity correction and exactly via math.comb). |
| 18 | conditional-on-sum.html | Conditioning on a sum | Sums and limit theorems | [probability/multivariate-random-variables.md#associated-applications](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#associated-applications) | Source: stated only: read the exact wording and solve; show the general trick (conditional distribution given the total). |
| 19 | sample-size.html | How big a sample? | Sums and limit theorems | [probability/multivariate-random-variables.md#associated-applications](https://github.com/ljeng/cheat-sheet/blob/main/probability/multivariate-random-variables.md#associated-applications) | Source: n = (1.96·3.3)² ≈ 41.8 → 42 (stated only: confirm the margin in the exact wording). Table of n for 90/95/99% and margins. |
