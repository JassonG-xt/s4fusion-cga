#!/usr/bin/env python3
"""Dump every paragraph of main.tex with its index and opening words.

Gives a stable handle for content edits: the index is only used to look up the
paragraph, and the replacement script prints the opening words it actually hits.
"""
import re
import sys

MAIN = sys.argv[1]
t = open(MAIN, encoding='utf-8').read()
paras = t.split('\n\n')
BANNER = re.compile(r'^\s*(?:%.*?\n\s*)+')
for i, p in enumerate(paras):
    c = ' '.join(BANNER.sub('', p).split())
    if not c:
        continue
    print(f'{i:4d} {len(c.split()):4d}w  {c[:96]}')
