#!/usr/bin/env python3
"""Print the line number of every section/subsection/paragraph marker."""
import re
import sys

L = open(sys.argv[1], encoding='utf-8').read().split('\n')
pat = re.compile(r'\\(section|subsection|paragraph)\{')
for i, l in enumerate(L, 1):
    if pat.match(l):
        print(f'{i:5d}  {l[:78]}')
print(f'      TOTAL LINES {len(L)}')
