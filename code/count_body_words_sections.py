#!/usr/bin/env python3
"""Per-section prose word counts for the manuscript."""
import re
import sys

t = open(sys.argv[1], encoding='utf-8').read()
b = t.index(r'\begin{document}')
e = t.index(r'\bibliography{refs}')
body = t[b:e]
for env in ('table*', 'table', 'figure*', 'figure', 'tabular',
            'equation', 'align', 'enumerate', 'itemize', 'abstract'):
    body = re.sub(r'\\begin\{' + re.escape(env) + r'\}.*?\\end\{'
                  + re.escape(env) + r'\}', ' ', body, flags=re.S)
body = re.sub(r'%.*', ' ', body)

tokens = re.split(r'(\\section\{[^}]*\}|\\subsection\{[^}]*\}|\\paragraph\{)',
                  body)
cur, counts, order = '(front)', {}, []
i = 0
while i < len(tokens):
    s = tokens[i]
    if s.startswith(('\\section{', '\\subsection{', '\\paragraph{')):
        nm = re.sub(r'\\(section|subsection|paragraph)\{', '', s).rstrip('}')
        cur = nm[:44]
        if cur not in counts:
            order.append(cur)
        counts.setdefault(cur, 0)
    else:
        s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', s)
        s = re.sub(r'[\\$&_{}~^]', ' ', s)
        w = len([x for x in s.split() if re.search(r'[A-Za-z]', x)])
        counts[cur] = counts.get(cur, 0) + w
    i += 1

tot = 0
for k in order:
    v = counts.get(k, 0)
    if v >= 60:
        print(f'{v:6d}  {k}')
    tot += v
print(f'{tot:6d}  TOTAL')
