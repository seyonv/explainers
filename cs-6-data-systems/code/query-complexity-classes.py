"""How hard is a query? Data complexity vs expression complexity.

A relational-calculus (CALC) query with k variables can be evaluated
with k nested loops, one pointer per variable, as in the source's
LOGSPACE argument. So the work is n**k assignments over a domain of
n values, and the working memory is k pointers of ceil(log2 n) bits.

Fix the query (k) and grow the data (n): polynomial, degree k.
Fix the data (n) and grow the query (k): exponential in k.

The query here is "k people who all know each other" (a k-clique,
a conjunctive query) on a ring where i knows everyone within 2 seats.
Runs under python3 (3.10), standard library only.
"""
from itertools import product
from math import ceil, log2


def ring(n, d=2):
    """i knows j if they sit within d seats on a ring of n."""
    return {(i, j) for i in range(n) for j in range(n)
            if i != j and min((i - j) % n, (j - i) % n) <= d}


def clique_query(E, n, k):
    """Evaluate q(x1..xk) = AND of E(xi, xj) for all i < j.
    Builds the whole answer; returns (answers, assignments tried).
    E stands in for the read-only input tape."""
    tried = found = 0
    for xs in product(range(n), repeat=k):     # k nested loops
        tried += 1
        if all((xs[i], xs[j]) in E
               for i in range(k) for j in range(i + 1, k)):
            found += 1
    return found, tried


def pointer_bits(n, k):
    return k * ceil(log2(n))    # k pointers, log n bits each


if __name__ == "__main__":
    print("Data complexity: fixed query k = 3, grow n")
    prev = None
    for n in (10, 20, 40, 80):
        found, tried = clique_query(ring(n), n, 3)
        grow = f"x{tried // prev}" if prev else ""
        print(f"  n={n:<3} tried {tried:>7,} {grow:>3}  "
              f"answers {found:>3}  memory {pointer_bits(n, 3)} bits")
        prev = tried
    print("Expression complexity: fixed n = 10, grow k")
    prev = None
    for k in range(1, 7):
        found, tried = clique_query(ring(10), 10, k)
        grow = f"x{tried // prev}" if prev else ""
        print(f"  k={k}   tried {tried:>9,} {grow:>3}  "
              f"answers {found:>3}  memory {pointer_bits(10, k)} bits")
        prev = tried
