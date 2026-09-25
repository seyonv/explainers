# Shared facts: Algorithms, sorting & strings (cs-2-algorithms)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
The source guide's Sorting and Algorithms pages: every sort, binary search in its three disguises, the string-matching classics and greedy proofs, with hand traces and Python.
- 21 concept cards plus `_overview.html` (written last by the main agent).
- **Running example / conventions for this course:** Sorting cards all trace the same array: a = [38, 27, 43, 3, 9, 82, 10] (sorted: [3, 9, 10, 27, 38, 43, 82]). Use the measured sort timings from the table.

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


## Card list (cs-2-algorithms)
| # | File | Title | Group | Source section | Brief |
|---|---|---|---|---|---|
| 1 | insertion-sort.html | Insertion sort | Sorting | [coding-algorithms/sorting.md#insertion-sort](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#insertion-sort) | Source: C++ swap-based. Trace a, count swaps (= inversions). Measured: pure-Python insertion sort n=5,000 0.828 s vs sorted() 0.00026 s. Why Timsort uses it for small runs. |
| 2 | quicksort.html | Quicksort | Sorting | [coding-algorithms/sorting.md#quicksort](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#quicksort) | Source: iterative, explicit stack, Lomuto (C++). Trace the first two partitions of a. Worst case on sorted input with last-element pivot (count comparisons for n=8: 28). Random pivot / median-of-3 / introsort (np.sort kind='quicksort' is introsort: 0.031 s for 10⁶). |
| 3 | quickselect.html | Quickselect | Sorting | [coding-algorithms/sorting.md#quicksort](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#quicksort) | Source: Quicksort › Quickselect (Python 3-way list partition, pivot arr[0]). Trace k=3 on a → 10. Expected n + n/2 + … = 2n comparisons; worst case; median-of-medians in one line; heapq.nsmallest alternative. |
| 4 | merge-sort.html | Merge sort | Sorting | [coding-algorithms/sorting.md#merge-sort](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#merge-sort) | Source: bottom-up by width (C++). Trace widths 1,2,4 on a. Recurrence T(n)=2T(n/2)+n. Stability explained with a tuple example. np.sort kind='stable' 0.072 s. |
| 5 | merge-sort-counting.html | Counting during a merge | Sorting | [coding-algorithms/sorting.md#merge-sort](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#merge-sort) | Source: Reverse Pairs and Count Smaller (C++ merge versions + Python SortedList versions; SortedList is a third-party package; use bisect.insort, note its O(n) insert, or a Fenwick tree). Trace Count Smaller [5,2,6,1] → [2,1,1,0] through the merges. Reverse Pairs [1,3,2,3,1] → 2. |
| 6 | heapsort.html | Heapsort | Sorting | [coding-algorithms/sorting.md#heapsort](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#heapsort) | Source: in-place sift-down (C++). Trace heapify of a (show the array and tree). Why build-heap is O(n) (sum of heights). Not stable; poor cache behaviour vs quicksort. |
| 7 | radix-sort.html | Radix sort | Sorting | [coding-algorithms/sorting.md#radix-sort](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md#radix-sort) | Source: Python, **broken**: references an undefined `arr`, and its negative split puts 0 with the negatives and floors negatives wrongly. Write a correct LSD radix sort (base 10 for the trace, and handle negatives by offsetting or splitting properly), trace [170,45,75,-90,802,24,2,66]. When radix beats comparison sorts. |
| 8 | sort-lower-bound-stability.html | Why n log n, and which sort to use | Sorting | [coding-algorithms/sorting.md](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md) | [S] The source gives no complexities or trade-offs. Decision-tree argument: log2(n!) comparisons (n=7 → 12.3 → at least 13; n=10⁶ ≈ 1.85e7). A table of all 7 sorts: best/avg/worst, space, stable, in-place, when to use. Timsort as Python's choice (sorted 10⁶: 0.130 s). |
| 9 | index-as-hash.html | The array as its own hash table | Sorting | [coding-algorithms/sorting.md](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/sorting.md) | Source: top of sorting.md (C++ cyclic placement). Trace [3,4,-1,1] → 2 swap by swap. Why each swap places one value permanently → O(n). |
| 10 | binary-search-on-answer.html | Binary search on the answer | Searching | [coding-algorithms/algorithms.md#binary-search](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#binary-search) | Source: Binary Search (C++). Teach lo/hi invariants, then 'search the answer space with a monotone predicate'. Trace kth pair distance nums=[1,3,1], k=1 → 0 and a bigger one [1,6,1] k=3 → 5. Rotated min [2,2,2,0,1] and why duplicates make worst case O(n). |
| 11 | binary-search-partition.html | Median of two sorted arrays | Searching | [coding-algorithms/algorithms.md#binary-search](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#binary-search) | Source: Python. Trace A=[1,3,8,9,15], B=[7,11,18,19,21,25] → 11 with each partition's four boundary values. |
| 12 | lis-patience.html | Longest increasing subsequence | Searching | [coding-algorithms/algorithms.md#binary-search](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#binary-search) | Source: Russian Doll (Python, sort by (w, -h) then LIS). Trace LIS on [10,9,2,5,3,7,101,18] → 4 showing the tails array; then envelopes [[5,4],[6,4],[6,7],[2,3]] → 3 and why h is sorted descending. |
| 13 | prefix-sum-ordered-set.html | Prefix sums with an ordered set | Searching | [coding-algorithms/algorithms.md#binary-search](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#binary-search) | Source: C++ set::lower_bound. Python: bisect.insort into a list (O(n) insert, fine for interviews; say so). Trace the 1D subproblem [2,2,-1], k=3 → 3, then the column-pair reduction for matrix [[1,0,1],[0,-2,3]], k=2 → 2. |
| 14 | two-pointers-intervals.html | Two pointers and intervals | Searching | [coding-algorithms/algorithms.md](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md) | [S] Not a heading in the source, but its solutions use it everywhere (min window, kth pair distance, skyline). Teach opposite-end pointers (two-sum on sorted), same-direction pointers, and intervals (merge [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]], meeting rooms count via sweep). |
| 15 | kmp.html | Knuth–Morris–Pratt | Strings | [coding-algorithms/algorithms.md#searching](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#searching) | Source: KMP (C++, also code/knuth-morris-pratt.cpp) and Shortest Palindrome (lps on s + sep + reverse(s)). Build lps for 'ABABCABAB' step by step. Shortest palindrome of 'aacecaaa' → 'aaacecaaa'. |
| 16 | rolling-hash.html | Rolling hash (Rabin–Karp) | Strings | [coding-algorithms/algorithms.md#searching](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#searching) | [S] Companion the source lacks. Polynomial hash mod a prime, the slide formula with numbers (base 256, mod 101 on 'abr'→'bra'), expected O(n+m), collision probability ≈ 1/mod per comparison, double hashing. Longest Duplicate Substring ('banana' → 'ana') via binary search + rolling hash, as the standard alternative to the source's suffix automaton. |
| 17 | manacher.html | Manacher's algorithm | Strings | [coding-algorithms/algorithms.md#searching](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#searching) | Source: Longest Palindromic Substring (C++, '#' interleaving) and code/manacher.cpp (radius-array variant). Trace 'babad' → '#b#a#b#a#d#' radii and the mirror step. Compare with expand-around-centre O(n²). |
| 18 | suffix-automaton.html | Suffix automaton | Strings | [coding-algorithms/data-structures.md#graphs](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/data-structures.md#graphs) | Source: data-structures.md › Graphs (Python), where it's misfiled. Build the automaton for 'banana' (states, links, len), show how the longest repeated substring 'ana' falls out. Compare with suffix array and rolling-hash approaches. |
| 19 | divide-and-conquer.html | Divide and conquer and the master theorem | Paradigms | [coding-algorithms/algorithms.md#divide-and-conquer](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#divide-and-conquer) | [S] The heading is empty. Master theorem with three worked recurrences (merge sort, binary search, Karatsuba 3T(n/2)+n → n^1.585, show 3^10 vs 4^10 multiplications). Tie back to merge sort, quickselect, median of two arrays and closest pair in one line. |
| 20 | greedy-wildcard-justify.html | Greedy I: wildcard matching and text justification | Paradigms | [coding-algorithms/algorithms.md#greediness](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#greediness) | Source: Greediness (Wildcard C++, Text Justification Python). Trace s='adceb', p='*a*b' → True with the star/match pointers. Justify ['This','is','an','example','of','text','justification.'], width 16. Create Maximum Number, Strong Password (dup) and Find the Closest Palindrome are empty stubs; one line each. |
| 21 | greedy-candy-patching.html | Greedy II: candy and patching an array | Paradigms | [coding-algorithms/algorithms.md#greediness](https://github.com/ljeng/cheat-sheet/blob/main/coding-algorithms/algorithms.md#greediness) | Source: Candy and Patching Array (C++). Candy ratings [1,0,2] → 5 and [1,2,87,87,87,2,1] → 13 via two passes (the source's up/down-run counting is equivalent; explain). Patching nums=[1,5,10], n=20 → 2 patches, tracing reach. Why greedy is provably optimal here (exchange argument). |
