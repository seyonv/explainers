"""The end-to-end argument: careful file transfer, in Python.

1. Arithmetic: 10 GB through 10 hops that each let 1 byte in 10^9
   through corrupted (illustrative rate) -> expected bad bytes.
2. Chunked end-to-end retries: low-level reliability as performance.
3. A hop-by-hop simulation: every link CRC passes, yet the file
   arrives wrong; only the end-to-end hash notices.
4. Real sockets: send, shutdown(SHUT_WR), read until b"", and wait
   for B's own digest (the source's C snippet, rewritten).
5. A crash signal: kill the peer process, time the FIN/RST.
"""
import hashlib
import math
import random
import socket
import subprocess
import sys
import threading
import time
import zlib

GB, MB = 10**9, 10**6


def expected_bad(size, per_byte, hops):
    """Expected corrupted bytes and P(file arrives clean)."""
    lam = size * hops * per_byte
    p_clean = math.exp(size * hops * math.log1p(-per_byte))
    return lam, p_clean


def chunk_table(size, per_byte, hops, chunks):
    for c in chunks:
        lam, p = expected_bad(c, per_byte, hops)
        tries = 1 / p
        print(f"  {c / MB:>8,.0f} MB  lam={lam:<7g} "
              f"P(clean)={p:.3g}  tries={tries:.4g}  "
              f"extra={(tries - 1) * 100:.3g}%")


# --- 3. hop-by-hop simulation -------------------------------------
def router_copy(buf, rng, per_byte):
    """A router copies the packet in memory; rarely swaps 2 bytes."""
    b = bytearray(buf)
    for i in range(len(b) - 1):
        if rng.random() < per_byte:
            b[i], b[i + 1] = b[i + 1], b[i]
    return bytes(b)


def send_over_hops(data, hops, rng, per_byte):
    """Each link checks its own CRC; the bug is inside the routers."""
    link_failures = 0
    for _ in range(hops):
        crc = zlib.crc32(data)            # computed on the way out
        if zlib.crc32(data) != crc:       # checked on the way in
            link_failures += 1            # (the wire itself is fine)
        data = router_copy(data, rng, per_byte)
    return data, link_failures


def careful_transfer(data, chunk, hops, per_byte, seed=1):
    rng = random.Random(seed)
    out, sent, link_fail = [], 0, 0
    for i in range(0, len(data), chunk):
        piece = data[i:i + chunk]
        want = hashlib.sha256(piece).digest()   # end-to-end check
        while True:
            sent += len(piece)
            got, lf = send_over_hops(piece, hops, rng, per_byte)
            link_fail += lf
            if hashlib.sha256(got).digest() == want:
                break                           # else retry chunk
        out.append(got)
    return b"".join(out), sent, link_fail


# --- 4. real sockets: end-to-end acknowledgement ------------------
def receiver(srv):
    conn, _ = srv.accept()
    h = hashlib.sha256()
    while chunk := conn.recv(4000):   # b"" = A sent FIN, all read
        h.update(chunk)
    conn.sendall(h.digest())          # the app's own answer
    conn.close()                      # our FIN back to A


def send_file(addr, data):
    s = socket.create_connection(addr)
    s.sendall(data)                   # in the kernel, not yet at B
    s.shutdown(socket.SHUT_WR)        # FIN: "no more data from me"
    reply = b""
    while chunk := s.recv(4000):      # 0 bytes = B closed too
        reply += chunk
    s.close()
    return reply == hashlib.sha256(data).digest()


# --- 5. crash signal vs timeout ------------------------------------
PEER = ("import socket,sys,time\n"
        "s=socket.create_server(('127.0.0.1',0))\n"
        "print(s.getsockname()[1],flush=True)\n"
        "c,_=s.accept()\n"
        "time.sleep(60)\n")


def crash_signal_ms():
    p = subprocess.Popen([sys.executable, "-c", PEER],
                         stdout=subprocess.PIPE, text=True)
    port = int(p.stdout.readline())
    s = socket.create_connection(("127.0.0.1", port))
    s.settimeout(20)                  # ZooKeeper-style session timeout
    t0 = time.perf_counter()
    p.kill()                          # process dies, OS survives
    try:
        got = "FIN" if s.recv(1) == b"" else "data"
    except ConnectionResetError:
        got = "RST"
    ms = (time.perf_counter() - t0) * 1000
    s.close()
    p.wait()
    return got, ms


if __name__ == "__main__":
    size, per_byte, hops = 10 * GB, 1e-9, 10
    lam, p = expected_bad(size, per_byte, hops)
    print("1. 10 GB, 10 hops, 1e-9 bad bytes per byte per hop")
    print(f"   expected bad bytes = {size:.0e} x {hops} x "
          f"{per_byte:g} = {lam:g}")
    print(f"   P(no bad byte) = {p:.3g}")
    lam2, p2 = expected_bad(size, 1e-11, hops)
    print(f"   hops 100x better (1e-11): lam={lam2:g}, "
          f"P(clean)={p2:.3g}, tries={1 / p2:.3g}")

    print("2. retry per chunk on an end-to-end hash mismatch")
    chunk_table(size, per_byte, hops,
                [10 * GB, 1 * GB, 100 * MB, 10 * MB, 1 * MB])

    print("3. simulation: 1 MB, 64 KB chunks, 3 hops,"
          " 1e-6 swaps per byte")
    data = random.Random(0).randbytes(1 * MB)
    got, sent, lf = careful_transfer(data, 64 * 1024, 3, 1e-6)
    _, whole, _ = careful_transfer(data, 1 * MB, 3, 1e-6)
    rng = random.Random(1)
    once, lf1 = send_over_hops(data, 3, rng, 1e-6)
    print(f"   one pass, no end-to-end check: link CRC failures="
          f"{lf1}, file correct={once == data}")
    print(f"   with end-to-end hash: chunks={-(-MB // 65536)}, "
          f"bytes sent={sent:,}, "
          f"link CRC failures={lf}, file correct={got == data}")
    print(f"   same, one 1 MB chunk: bytes sent={whole:,}")

    print("4. sockets: sendall, shutdown(SHUT_WR), read to b''")
    srv = socket.create_server(("127.0.0.1", 0))
    t = threading.Thread(target=receiver, args=(srv,))
    t.start()
    ok = send_file(srv.getsockname(), data)
    t.join()
    srv.close()
    print(f"   B's digest matches A's: {ok}")

    print("5. peer process killed; how soon does A know?")
    got, ms = crash_signal_ms()
    print(f"   signal={got} after {ms:.2f} ms "
          f"(vs 20,000 ms session timeout)")
