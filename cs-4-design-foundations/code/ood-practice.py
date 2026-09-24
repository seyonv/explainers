"""Object-oriented design practice: a parking lot and a card deck.

Python 3.10, standard library only. Run: python3 ood-practice.py
"""
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import ClassVar


# ---------- the parking lot ----------

class Size(Enum):
    SMALL = 1; COMPACT = 2; LARGE = 3


@dataclass
class Vehicle:
    plate: str
    permit: bool = False           # accessible-parking permit
    size: ClassVar[Size]


class Motorcycle(Vehicle): size = Size.SMALL
class Car(Vehicle): size = Size.COMPACT
class Bus(Vehicle): size = Size.LARGE


@dataclass
class Spot:
    name: str
    size: Size
    accessible: bool = False
    vehicle: Vehicle | None = None

    def fits(self, v):                      # the spot owns the rule
        return (self.vehicle is None
                and v.size.value <= self.size.value
                and (v.permit or not self.accessible))


@dataclass
class Level:
    number: int
    spots: list[Spot]

    def free(self, v):
        return [s for s in self.spots if s.fits(v)]


def first_fit(spots): return spots[0]      # whatever comes first
def best_fit(spots):                       # smallest spot that fits
    return min(spots, key=lambda s: (s.size.value, s.accessible))


@dataclass
class ParkingLot:
    levels: list[Level]
    choose: callable = best_fit             # policy is swappable

    def park(self, v):
        spots = [s for lv in self.levels for s in lv.free(v)]
        if not spots:
            return None
        spot = self.choose(spots)
        spot.vehicle = v
        return spot


def build_lot(policy):
    S, C, L = Size.SMALL, Size.COMPACT, Size.LARGE
    return ParkingLot([
        Level(1, [Spot("L1", L), Spot("C1", C), Spot("S1", S)]),
        Level(2, [Spot("L2", L), Spot("C2", C),
                  Spot("H1", C, accessible=True)]),
    ], policy)


ARRIVALS = [Car("car a"), Car("car b"), Motorcycle("moto"),
            Bus("bus x"), Car("car p", permit=True), Bus("bus y")]


def run_lot(policy):
    lot = build_lot(policy)
    out = []
    for v in ARRIVALS:
        spot = lot.park(v)
        out.append((v.plate, spot.name if spot else None))
    return out


# ---------- the deck ----------

class Suit(Enum):
    SPADES = "♠"
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"


class Rank(Enum):
    ACE, TWO, THREE, FOUR, FIVE, SIX, SEVEN = range(1, 8)
    EIGHT, NINE, TEN, JACK, QUEEN, KING = range(8, 14)


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self):
        face = {1: "A", 11: "J", 12: "Q", 13: "K"}
        return face.get(self.rank.value, str(self.rank.value)) \
            + self.suit.value


@dataclass
class Deck:
    cards: list[Card] = field(default_factory=lambda: [
        Card(r, s) for s in Suit for r in Rank])

    def shuffle(self, rng=random, trace=None):
        a = self.cards                     # Fisher-Yates, in place
        for i in range(len(a) - 1, 0, -1):
            j = rng.randrange(i + 1)       # 0 <= j <= i, uniform
            a[i], a[j] = a[j], a[i]        # a[i] is now final
            if trace is not None:
                trace.append((i, j, [str(c) for c in a]))

    def deal(self) -> Card:
        return self.cards.pop()

    def __len__(self):
        return len(self.cards)


def naive_shuffle_counts(n):
    """Swap each i with ANY index: n**n equally likely paths."""
    counts = Counter()
    for js in product(range(n), repeat=n):
        a = list(range(n))
        for i, j in enumerate(js):
            a[i], a[j] = a[j], a[i]
        counts[tuple(a)] += 1
    return counts


def fisher_yates_counts(n):
    counts = Counter()
    choices = [range(i + 1) for i in range(n - 1, 0, -1)]
    for js in product(*choices):
        a = list(range(n))
        for i, j in zip(range(n - 1, 0, -1), js):
            a[i], a[j] = a[j], a[i]
        counts[tuple(a)] += 1
    return counts


# ---------- the call centre (source prompt 2) ----------

@dataclass
class Employee:
    name: str
    rank: int                   # 0 fresher, 1 team lead, 2 manager
    busy: bool = False


def get_call_handler(staff, level=0):
    """Lowest-ranked free employee at or above the call's level."""
    for rank in range(level, 3):
        for e in staff:
            if e.rank == rank and not e.busy:
                return e
    return None                 # everyone busy: queue the call


if __name__ == "__main__":
    print("parking lot, same arrivals, two policies")
    for name, policy in [("first_fit", first_fit),
                         ("best_fit", best_fit)]:
        res = run_lot(policy)
        parked = sum(s is not None for _, s in res)
        print(f"  {name:9}", ", ".join(f"{p}->{s or 'REJECTED'}"
                                       for p, s in res))
        print(f"  {'':9} parked {parked} of {len(res)}")

    print("\nFisher-Yates on 5 cards, seed 7")
    d = Deck([Card(Rank(r), Suit.SPADES) for r in range(1, 6)])
    print("  start", " ".join(str(c) for c in d.cards))
    tr = []
    d.shuffle(random.Random(7), tr)
    for i, j, a in tr:
        print(f"  i={i} j={j}  {' '.join(a)}")
    ref = [Card(Rank(r), Suit.SPADES) for r in range(1, 6)]
    random.Random(7).shuffle(ref)
    print("  same as random.shuffle(seed 7):", ref == d.cards)

    full = Deck()
    full.shuffle(random.Random(1))
    print(f"\n52-card deck: {len(full)} cards, top card",
          full.deal(), f"then {len(full)} left")

    f52 = math.factorial(52)
    print(f"\n52! = {f52}")
    print(f"    ~ {f52:.2e} orderings ~ 2^{math.log2(f52):.1f}")
    print(f"    a 32-bit seed reaches 2^32 = {2**32:,} of them")

    print("\nn = 3: how often each ordering comes out")
    fy = fisher_yates_counts(3)
    nv = naive_shuffle_counts(3)
    for perm in sorted(fy):
        print(f"  {perm}  fisher-yates {fy[perm]}/6"
              f"   naive {nv[perm]}/27")
    print(f"  27 paths over 6 orderings: 27/6 = {27/6}, not whole")

    print("\ncall centre")
    staff = [Employee("fresher 1", 0, busy=True),
             Employee("fresher 2", 0, busy=True),
             Employee("lead 1", 1), Employee("pm", 2)]
    print("  new call ->", get_call_handler(staff).name)
    print("  escalated to manager ->",
          get_call_handler(staff, level=2).name)
