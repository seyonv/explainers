"""Service discovery: a toy ZooKeeper-style registry.

Servers register ephemeral nodes tied to a session. A session
expires when the registry hears no heartbeat for TIMEOUT seconds;
its nodes are deleted and every watcher is told. Time is simulated
in whole seconds, so the trace is exact and repeatable.
"""

TIMEOUT = 20            # source: session timeout "typically 20 s"


class Registry:
    def __init__(self):
        self.nodes = {}          # path -> owning session
        self.last_beat = {}      # session -> last heartbeat time
        self.watchers = []       # callbacks, fired on any change

    def register(self, path, session, now):
        self.nodes[path] = session           # ephemeral node
        self.last_beat[session] = now
        self._notify(now, f"+ {path}")

    def heartbeat(self, session, now):
        if session in self.last_beat:        # expired: too late
            self.last_beat[session] = now
            return True
        return False

    def tick(self, now):
        for s, t in list(self.last_beat.items()):
            if now - t >= TIMEOUT:           # silent too long
                del self.last_beat[s]
                gone = [p for p, o in self.nodes.items() if o == s]
                for p in gone:
                    del self.nodes[p]
                    self._notify(now, f"- {p}")

    def children(self):
        return sorted(self.nodes)

    def _notify(self, now, what):
        for w in self.watchers:
            w(now, what, self.children())


def quorum(n):
    """Votes needed and crashes tolerated for an n-node ensemble."""
    need = n // 2 + 1
    return need, n - need


def simulate(until=70):
    reg, log = Registry(), []
    reg.watchers.append(lambda t, w, c: log.append((t, w, c)))
    for i in (1, 2, 3):
        reg.register(f"/resize/srv-{i}", f"s{i}", 0)
    s3 = "s3"                                # srv-3's session id
    for now in range(1, until):
        reg.heartbeat("s1", now)
        if now <= 5:                         # srv-2 crashes after 5
            reg.heartbeat("s2", now)
        if not 30 <= now < 55:               # srv-3: GC pause 30-54
            if not reg.heartbeat(s3, now):   # session expired
                s3 += "'"
                reg.register("/resize/srv-3", s3, now)
        reg.tick(now)
    return log


if __name__ == "__main__":
    print("watch events (t in s):")
    for t, what, kids in simulate():
        names = ", ".join(k.split("/")[-1] for k in kids)
        print(f"  t={t:>2}  {what:<16} -> [{names}]")

    rps = 10_000 // 40                       # per server, image resize
    ttl = 300                                # illustrative DNS TTL
    print(f"\n{rps} req/s per server (10,000 req/s / 40 servers)")
    print(f"registry: {TIMEOUT} s x {rps} = {TIMEOUT * rps:,} requests")
    print(f"DNS TTL:  {ttl} s x {rps} = {ttl * rps:,} requests")

    print("\n n  votes  survives")
    for n in range(1, 8):
        need, f = quorum(n)
        print(f" {n}  {need:>5}  {f:>8}")
