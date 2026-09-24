"""Queues as rolling state: two deque problems from the cheat-sheet.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Queues > Number of People Aware of a Secret and Integer to English
Words. Same algorithms, rewritten for readability.
"""
from collections import deque


def people_aware(n, delay, forget):
    new = deque([0] * (forget - 1) + [1])  # learned on days 2-f .. 1
    share = 0                              # people sharing today
    for day in range(2, n + 1):
        share += new[-delay]     # learned on day - delay: starts now
        share -= new.popleft()   # learned on day - forget: forgets
        new.append(share)        # each sharer tells one new person
    return sum(new)              # learned in the last `forget` days


def trace_secret(n, delay, forget):
    new = deque([0] * (forget - 1) + [1])
    share = 0
    print(f"day 1  window {list(new)}  share 0  aware 1")
    for day in range(2, n + 1):
        joins = new[-delay]
        leaves = new.popleft()
        share += joins - leaves
        new.append(share)
        print(f"day {day}  +{joins} -{leaves}  share {share}"
              f"  window {list(new)}  aware {sum(new)}")
    return sum(new)


def brute_secret(n, delay, forget):
    learned = [1]                # the day each person learned it
    for day in range(2, n + 1):
        k = sum(d + delay <= day < d + forget for d in learned)
        learned += [day] * k
    return sum(n < d + forget for d in learned)


ONES = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
        "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
        "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen",
        "Nineteen"]
TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty",
        "Seventy", "Eighty", "Ninety"]
SCALES = ["", "Thousand", "Million", "Billion"]


def below_1000(n):
    words = []
    hundreds, n = divmod(n, 100)
    if hundreds:
        words += [ONES[hundreds], "Hundred"]
    if n >= 20:
        tens, n = divmod(n, 10)
        words.append(TENS[tens])
    if n:
        words.append(ONES[n])
    return words


def number_to_words(num):
    if num == 0:
        return "Zero"
    parts = deque()
    for scale in SCALES:         # lowest chunk comes out first ...
        num, chunk = divmod(num, 1000)
        if chunk:
            words = below_1000(chunk) + ([scale] if scale else [])
            parts.appendleft(" ".join(words))  # ... but reads last
    return " ".join(parts)


def source_number_to_words(num):
    """The cheat-sheet's version, kept to cross-check (tables above)."""
    one = ONES
    ten = ["", "Ten"] + TENS[2:]
    superwords = deque()
    i = 0
    while num:
        num, mod = divmod(num, 1000)
        if mod:
            subwords = deque()
            if mod >= 100:
                div, mod = divmod(mod, 100)
                subwords.extendleft([one[div], "Hundred"])
            if mod >= 20:
                div, mod = divmod(mod, 10)
                subwords.appendleft(ten[div])
            subwords.extendleft([one[mod], SCALES[i]])
            superwords.extendleft(subwords)
        i += 1
    if superwords:
        return " ".join(w for w in superwords if w)
    return "Zero"


def trace_words(num):
    parts = deque()
    for scale in SCALES:
        if not num:
            break
        num, chunk = divmod(num, 1000)
        if chunk:
            words = below_1000(chunk) + ([scale] if scale else [])
            parts.appendleft(" ".join(words))
        print(f"chunk {chunk:>3} ({scale or 'units'}): num left {num}"
              f"  deque {list(parts)}")


if __name__ == "__main__":
    import random

    print(people_aware(6, 2, 4))              # 5
    trace_secret(6, 2, 4)
    cases = [(n, d, f) for n in range(2, 15)
             for f in range(2, n + 1) for d in range(1, f)]
    bad = sum(people_aware(*c) != brute_secret(*c) for c in cases)
    print(f"secret: {len(cases)} cases with n < 15, {bad} mismatches")
    n, d, f = 1000, 1, 1000      # scale used in the card's grid
    print("answer digits:", len(str(people_aware(n, d, f))))
    print("re-sum adds:", sum(len(range(max(1, t - f + 1), t - d + 1))
                              for t in range(2, n + 1)))
    print("pop(0) moves:", (n - 1) * (f - 1),
          " deque steps:", 3 * (n - 1))

    print(number_to_words(1234567))
    trace_words(1234567)
    d = deque()
    d.extendleft(["One", "Hundred"])
    print("extendleft reverses:", list(d))  # ['Hundred', 'One']
    rng = random.Random(0)
    nums = list(range(0, 2000)) + [rng.randrange(2**31)
                                   for _ in range(100_000)]
    assert all(number_to_words(x) == source_number_to_words(x)
               for x in nums)
    print(len(nums), "numbers match the source's version")
