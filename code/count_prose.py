#!/usr/bin/env python3
"""Strip HTML comments, then count the A-2 meta-language metric.

Provenance comments (<!-- src: ... -->) are metadata: they record where each number
came from and must NOT be compressed, so counting them in the A-2 metric would create
pressure to delete traceability. This reports both numbers so the gate can be defined
on prose without silently ignoring the raw figure.
"""
import re
import sys
from pathlib import Path

PAT = re.compile(r"pre-registered|registered|frozen", re.I)
COMMENT = re.compile(r"<!--.*?-->", re.S)


def main():
    tot_raw = tot_prose = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.is_file():
            print(f"missing: {arg}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        raw = len(PAT.findall(text))
        prose = len(PAT.findall(COMMENT.sub("", text)))
        tot_raw += raw
        tot_prose += prose
        print(f"{p.name:<34} raw={raw:<4} prose={prose:<4}")
    if len(sys.argv) > 2:
        print("-" * 52)
        print(f"{'TOTAL':<34} raw={tot_raw:<4} prose={tot_prose:<4}")


if __name__ == "__main__":
    main()
