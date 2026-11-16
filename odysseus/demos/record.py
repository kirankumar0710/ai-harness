"""Run a demo and save its complete output to the demo's output/ folder.

    python3 demos/record.py skills --offline   # skills/output/offline.txt
    python3 demos/record.py memory             # memory/output/live.txt
    python3 demos/record.py gate --ask         # gate/output/live-ask.txt
    python3 demos/record.py context --label live-gemini
    python3 demos/record.py dice --out /some/folder/dice-live.txt

Everything after the demo name is passed to the demo. The output is shown
as it arrives, and prompts such as --ask's y/N work as usual (the answer
you type is not in the saved file; the result of each call is).

The walkthroughs quote parts of these files; the files are the complete
record. Before saving, personal paths are replaced: your home folder
becomes ~ and the system temp folder becomes <tmp>.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile

DEMOS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DEMOS_DIR)


def sanitise(text: str) -> str:
    """Replace the home folder and temp folder paths."""
    temp = os.path.realpath(tempfile.gettempdir())
    for path in sorted({temp, temp.removeprefix("/private"),
                        tempfile.gettempdir().rstrip("/")}, key=len,
                       reverse=True):
        text = text.replace(path, "<tmp>")
    return text.replace(os.path.expanduser("~"), "~")


def default_label(demo_args: list[str]) -> str:
    label = "offline" if "--offline" in demo_args else "live"
    if "--ask" in demo_args:
        label += "-ask"
    return label


def run_and_capture(command: list[str]) -> tuple[str, int]:
    """Run ``command``, echo its output as it arrives, return all of it."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    process = subprocess.Popen(command, cwd=PROJECT_DIR, env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    chunks = []
    assert process.stdout is not None
    while True:
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", "replace"), process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("demo", help="demo folder name, e.g. skills")
    parser.add_argument("--label", help="file name without .txt "
                        "(default: offline, live, live-ask)")
    parser.add_argument("--out", help="write to this file instead of "
                        "<demo>/output/<label>.txt (used by run.sh)")
    args, demo_args = parser.parse_known_args(argv)

    script = os.path.join(DEMOS_DIR, args.demo, f"{args.demo}.py")
    if not os.path.isfile(script):
        print(f"error: no demo at {script}", file=sys.stderr)
        return 1
    shown = " ".join(["python3", f"demos/{args.demo}/{args.demo}.py",
                      *demo_args])
    output, code = run_and_capture([sys.executable, script, *demo_args])

    provider = re.search(r"^provider: (.*)$", output, re.M)
    header = (f"# {args.demo} demo, recorded "
              f"{datetime.date.today().isoformat()}, "
              f"provider: {provider.group(1) if provider else 'unknown'}"
              f", exit code {code}\n\n")
    label = args.label or default_label(demo_args)
    path = os.path.abspath(args.out) if args.out else os.path.join(
        DEMOS_DIR, args.demo, "output", f"{label}.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + f"$ {shown}\n" + sanitise(output))
    shown_path = (os.path.relpath(path, PROJECT_DIR)
                  if path.startswith(PROJECT_DIR) else path)
    print(f"\nsaved: {shown_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
