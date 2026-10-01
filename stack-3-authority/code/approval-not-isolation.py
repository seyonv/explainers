"""A toy harness: an approval gate guards only the paths that call it.

Not Hermes' code. Two tools reach the same sibling directory. One
asks for approval (unattended, so it denies); the other never asks.
cwd is set to the workspace, which is not an access-control boundary.
"""
import subprocess
import tempfile
from pathlib import Path


def approve(action):
    # Unattended approval configured to deny, as in Hermes Part 5.
    print(f"  approval asked: {action} -> deny")
    return False


def write_file(path, text):
    # Tool path that calls the approval checkpoint.
    if not approve(f"write {path.name}"):
        return "blocked"
    path.write_text(text)
    return "ok"


def shell(cmd, cwd):
    # Tool path that never calls it. Runs as you, with your reach.
    r = subprocess.run(cmd, shell=True, cwd=cwd,
                       capture_output=True, text=True)
    return f"exit {r.returncode}", r.stdout.strip()


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp())
    ws, out = root / "workspace", root / "outside"
    ws.mkdir()
    out.mkdir()
    (out / "outside.txt").write_text("CANARY-OUTSIDE\n")

    print("write_file ../outside/a.txt:",
          write_file(out / "a.txt", "x"))
    print("  a.txt exists:", (out / "a.txt").exists())
    print("shell echo > ../outside/b.txt:",
          shell("echo x > ../outside/b.txt", ws)[0])
    print("  b.txt exists:", (out / "b.txt").exists())
    print("shell cat ../outside/outside.txt:",
          *shell("cat ../outside/outside.txt", ws))
