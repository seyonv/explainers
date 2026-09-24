"""Modular arithmetic essentials: fast power, Fermat inverse,
n-choose-k mod p from factorial tables, and the pigeonhole principle.

Run: python3 discrete-math-modular.py   (Python 3.10+)
"""
import math

P = 1_000_000_007           # prime; (P - 1)**2 < 2**63


def pow_mod(a, e, m, trace=None):
    """Right-to-left square-and-multiply: one pass over e's bits."""
    result, base = 1, a % m
    while e:
        if e & 1:
            result = result * base % m      # bit is 1: multiply in
        if trace is not None:
            trace.append((e & 1, base, result))
        base = base * base % m              # square for next bit
        e >>= 1
    return result


def inverse(a, p=P):
    """Fermat: a**(p-1) == 1 (mod p), so a**(p-2) is 1/a."""
    return pow(a, p - 2, p)


def comb_table(n, p=P):
    """Factorials and inverse factorials up to n, O(n) + one pow."""
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % p
    inv = [1] * (n + 1)
    inv[n] = pow(fact[n], p - 2, p)         # the only pow call
    for i in range(n, 0, -1):
        inv[i - 1] = inv[i] * i % p         # 1/(i-1)! = i/i!
    return fact, inv


def comb_mod(n, k, fact, inv, p=P):
    if not 0 <= k <= n:
        return 0
    return fact[n] * inv[k] % p * inv[n - k] % p


def zero_sum_window(nums):
    """Pigeonhole: len(nums)+1 prefix sums, len(nums) residues,
    so two prefixes share a residue; the slice between them sums
    to a multiple of len(nums)."""
    n = len(nums)
    seen, s = {0: 0}, 0
    for j, x in enumerate(nums, 1):
        s = (s + x) % n
        if s in seen:
            return seen[s], j               # nums[i:j]
        seen[s] = j


if __name__ == "__main__":
    tr = []
    r = pow_mod(3, 200, P, tr)
    print(f"200 = 0b{200:b}: {200 .bit_length()} bits, "
          f"{bin(200).count('1')} ones")
    print(f"{'bit':>3} {'base = 3^(2^i)':>14} {'result':>11}")
    for bit, base, res in tr:
        print(f"{bit:>3} {base:>14} {res:>11}")
    mults = len(tr) + bin(200).count("1")
    print(f"pow_mod(3, 200, P) = {r}  "
          f"({len(tr)} squarings + {bin(200).count('1')} "
          f"multiplies = {mults}; naive: 199)")
    assert r == pow(3, 200, P)
    print(f"3**200 has {len(str(3**200))} digits unreduced")

    print("\ninverse(3) =", inverse(3),
          " check 3 * it % P =", 3 * inverse(3) % P)
    print("mod 7: inverse(3, 7) =", inverse(3, 7),
          " pow(3, -1, 7) =", pow(3, -1, 7))
    print("10 / 2 mod 7 via inverse:", 10 % 7 * inverse(2, 7) % 7,
          " true 10 // 2 % 7 =", 10 // 2 % 7)
    print("Python -3 % 7 =", -3 % 7, "(C++/Java give -3)")

    fact, inv = comb_table(100_000)
    c = comb_mod(100, 50, fact, inv)
    print("\nC(100, 50) mod P =", c,
          " math.comb check:", math.comb(100, 50) % P)
    print("C(100, 50) exact has", len(str(math.comb(100, 50))),
          "digits")
    big = comb_mod(100_000, 50_000, fact, inv)
    exact = math.comb(100_000, 50_000)
    print("C(100000, 50000) mod P =", big,
          " math.comb check:", exact % P)
    print("C(100000, 50000) exact is", exact.bit_length(), "bits,",
          int(exact.bit_length() * math.log10(2)) + 1, "digits approx")

    nums = [3, 1, 4, 1, 5]
    pre = [0]
    for x in nums:
        pre.append(pre[-1] + x)
    print("\nprefix sums", pre, "mod 5:", [s % 5 for s in pre])
    i, j = zero_sum_window(nums)
    print(f"nums[{i}:{j}] = {nums[i:j]}, sum {sum(nums[i:j])}")
