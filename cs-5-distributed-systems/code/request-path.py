"""The path of a request: DNS cache, longest-prefix match, stateless
vs stateful firewall, and three load-balancing policies.

Run: python3 request-path.py   (Python 3.10, standard library only)
Addresses are from the documentation ranges (RFC 5737); illustrative.
"""
import hashlib
import ipaddress as ip

# 1. DNS: a resolver cache that honours the TTL ----------------------
TTL = 60              # github.com A-record TTL (dig, 2026-09-24)
cache = {}            # name -> (address, expires_at)


def resolve(name, now, walk):
    hit = cache.get(name)
    if hit and now < hit[1]:
        return hit[0], "cache hit"
    addr = walk(name)         # root -> .com TLD -> authoritative
    cache[name] = (addr, now + TTL)
    return addr, "miss: walked root, TLD, authoritative"


# 2. Routing: longest-prefix match ----------------------------------
ROUTES = [                    # (prefix, next hop)
    ("0.0.0.0/0", "upstream ISP"),
    ("203.0.113.0/24", "region edge"),
    ("203.0.113.128/25", "load-balancer pool"),
]


def next_hop(dst):                  # ROUTES: [(prefix, hop)]
    a = ip.ip_address(dst)
    hits = [(ip.ip_network(p), h) for p, h in ROUTES
            if a in ip.ip_network(p)]
    net, hop = max(hits, key=lambda t: t[0].prefixlen)
    return str(net), hop


# 3. Firewalls: first matching rule wins ----------------------------
SERVER = "203.0.113.200"


def stateless(pkt):
    src, sport, dst, dport = pkt
    if dst == SERVER and dport == 443:
        return "allow r1"
    if sport == 443:     # needed so replies to our own
        return "allow r2"    # outbound calls get in
    return "deny r3"


def stateful(pkt, conns):          # pkt = (src, sport, dst, dport)
    src, sport, dst, dport = pkt
    if (dst, dport, src, sport) in conns:   # a reply we expect
        return "allow tracked"
    if dst == SERVER and dport == 443:
        conns.add(pkt)                # remember the new connection
        return "allow r1"
    return "deny"


# 4. Load balancing: round robin, least outstanding, hash -----------
SERVERS = ["s1", "s2", "s3"]


def round_robin(i, key, busy):
    return SERVERS[i % len(SERVERS)]


def least_outstanding(i, key, busy):
    return min(SERVERS, key=lambda s: busy[s])


def rendezvous(i, key, busy):     # same key, same server
    h = lambda s: hashlib.sha256(f"{s}|{key}".encode()).digest()
    return max(SERVERS, key=h)


def simulate(pick, reqs):
    """Each server runs one resize at a time, FIFO. Returns
    (server, latency) per request; reqs = (arrive_ms, key, cost_ms)."""
    free_at = {s: 0 for s in SERVERS}
    ends = {s: [] for s in SERVERS}
    out = []
    for i, (t, key, cost) in enumerate(reqs):
        busy = {s: sum(e > t for e in ends[s]) for s in SERVERS}
        s = pick(i, key, busy)
        start = max(t, free_at[s])
        free_at[s] = start + cost
        ends[s].append(start + cost)
        out.append((s, start + cost - t))
    return out


REQS = [(0, "big.jpg", 1000), (100, "a.jpg", 100),
        (200, "b.jpg", 100), (300, "huge.jpg", 1000),
        (400, "a.jpg", 100), (500, "b.jpg", 100)]


if __name__ == "__main__":
    print("DNS, TTL", TTL, "s")
    walk = lambda name: "140.82.112.3"
    for now in (0, 30, 61):
        print(f"  t={now:>2}s", *resolve("github.com", now, walk))

    print("Longest-prefix match")
    for dst in ("203.0.113.200", "203.0.113.7", "198.51.100.9"):
        print(f"  {dst:<14} ->", *next_hop(dst))

    print("Firewall in front of", SERVER)
    conns = {(SERVER, 40000, "192.0.2.10", 443)}  # we called out
    pkts = [("198.51.100.9", 51000, SERVER, 443),  # client request
            ("192.0.2.10", 443, SERVER, 40000),    # reply to us
            ("192.0.2.66", 443, SERVER, 40001)]    # forged reply
    for p in pkts:
        print(f"  {p[0]}:{p[1]} -> :{p[3]:<5}",
              f"stateless {stateless(p):<8}  stateful",
              stateful(p, conns))

    print("Load balancing, 3 servers, one resize at a time each")
    for name, pick in [("round robin", round_robin),
                       ("least out", least_outstanding),
                       ("hash", rendezvous)]:
        res = simulate(pick, REQS)
        cells = " ".join(f"{s}:{lat:<4}" for s, lat in res)
        worst = max(lat for _, lat in res)
        print(f"  {name:<11} {cells} max {worst}")
