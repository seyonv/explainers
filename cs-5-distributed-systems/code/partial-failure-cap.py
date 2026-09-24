"""Partial failure and CAP: two replicas, one network partition.

Run: python3 partial-failure-cap.py   (Python 3.10, stdlib only)
"""
import math


# 1. Partial failure: three different faults, one identical symptom.
def call(server, key, value, fault):
    """Send a write; return what the caller sees."""
    if fault == "request lost":
        return "timeout"              # server never saw it
    if fault == "server crashed":
        return "timeout"              # maybe applied, maybe not
    server[key] = value               # the write happens here
    if fault == "reply lost":
        return "timeout"              # applied, but no one knows
    return "ok"


# 2. CAP: two replicas A and B hold the same key.
class TwoReplicas:
    def __init__(self, mode):
        self.mode = mode              # "CP" or "AP"
        self.data = {"A": {}, "B": {}}
        self.partitioned = False

    def write(self, node, key, value):
        if not self.partitioned:
            for d in self.data.values():
                d[key] = value        # ack only once both have it
            return "ok"
        if self.mode == "CP":
            return "error: can't reach peer"
        self.data[node][key] = value  # AP: accept locally
        return "ok"

    def read(self, node, key):
        if self.partitioned and self.mode == "CP":
            return "error: can't reach peer"
        return self.data[node][key]

    def heal(self):
        self.partitioned = False
        if self.mode == "AP":         # toy merge: A's copy wins;
            self.data["B"].update(self.data["A"])  # real: LWW, CRDTs


def scenario(mode):
    s = TwoReplicas(mode)
    steps = [("write A plan=free", s.write("A", "plan", "free"))]
    s.partitioned = True
    steps.append(("partition", "A and B can't talk"))
    steps.append(("write A plan=pro", s.write("A", "plan", "pro")))
    steps.append(("read  B plan", s.read("B", "plan")))
    s.heal()
    steps.append(("heal; read B plan", s.read("B", "plan")))
    return steps


# 3. The speed-of-light floor on one coordinated round trip.
FIBRE_KM_S = 200_000                  # rule of thumb, ~2/3 of c


def rtt_floor_ms(km):
    return 2 * km / FIBRE_KM_S * 1000


def great_circle_km(lat1, lon1, lat2, lon2, r=6371.0):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


if __name__ == "__main__":
    print("partial failure: the same write, three faults")
    for fault in ["none", "request lost", "server crashed",
                  "reply lost"]:
        server = {}
        seen = call(server, "plan", "pro", fault)
        if fault == "server crashed":
            state = "unknown"
        else:
            state = server.get("plan", "-")
        print(f"  {fault:15} caller sees {seen:8} server has {state}")

    for mode in ["CP", "AP"]:
        print(f"\n{mode}:")
        for step, result in scenario(mode):
            print(f"  {step:18} -> {result}")

    print("\nRTT floor in fibre (2 d / 200,000 km/s):")
    dc_ams = great_circle_km(38.9072, -77.0369, 52.3676, 4.9041)
    for label, km in [("AZs 100 km apart", 100),
                      ("Washington-Amsterdam", dc_ams)]:
        print(f"  {label:21} {km:7,.0f} km -> "
              f"{rtt_floor_ms(km):5.1f} ms")

    print("\nshare of a 500 ms p99 budget for 1 coordinated RTT:")
    azure = [("East US -> East US 2", 8),
             ("East US -> West US", 69),
             ("East US -> West Europe", 83),
             ("East US -> Japan East", 162),
             ("West Europe -> Australia East", 265)]
    for label, ms in azure:
        print(f"  {label:30} {ms:4} ms  {ms / 500:6.1%}"
              f"  {1000 / ms:6.1f} writes/s")
    print(f"  measured / floor, East US -> West Europe: "
          f"{83 / rtt_floor_ms(dc_ams):.2f}x")
