#!/usr/bin/env python3
"""Per-section prose word counts, measured correctly.

Splits the raw body on section/subsection delimiters FIRST, then strips float
environments, lists, comments and markup inside each chunk, so the section
labels survive the split. (An earlier version stripped markup first, which
removed the delimiters and collapsed everything into one bucket.)
"""
import re
import sys

path = sys.argv[1]
t = open(path, encoding='utf-8').read()
body = t[t.index(r'\begin{document}'):t.index(r'\bibliography{refs}')]

SPLIT = re.compile(r'(\\(?:section|subsection)\{[^}]*\})')
FLOATS = ('table*', 'table', 'figure*', 'figure', 'tabular', 'equation',
          'align', 'abstract')
LISTS = ('enumerate', 'itemize')


def strip_envs(chunk, envs):
    for env in envs:
        chunk = re.sub(r'\\begin\{' + re.escape(env) + r'\}.*?\\end\{'
                       + re.escape(env) + r'\}', ' ', chunk, flags=re.S)
    return chunk


def count(chunk, keep_lists=False):
    chunk = strip_envs(chunk, FLOATS if keep_lists else FLOATS + LISTS)
    chunk = re.sub(r'%.*', ' ', chunk)
    chunk = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', chunk)
    chunk = re.sub(r'[\\$&_{}~^]', ' ', chunk)
    return len([w for w in chunk.split() if re.search(r'[A-Za-z]', w)])


parts = SPLIT.split(body)
label, total, rows = '(front matter + abstract)', 0, []
i = 1
front = count(parts[0])
total += front
rows.append(('(front matter + abstract)', front))
while i < len(parts) - 1:
    name = re.sub(r'\\(?:section|subsection)\{', '', parts[i]).rstrip('}')
    c = count(parts[i + 1])
    total += c
    rows.append((name, c))
    i += 2

width = max(len(n) for n, _ in rows)
for n, c in rows:
    print(f'{c:6d}  {n}')
print(f'{total:6d}  TOTAL (prose only; floats, lists and the abstract excluded)')
print(f'{count(body, keep_lists=True):6d}  TOTAL (prose + list items; '
      f'floats and the abstract excluded)')
