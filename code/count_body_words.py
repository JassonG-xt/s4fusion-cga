#!/usr/bin/env python3
"""Crude body word count for the manuscript (prose paragraphs only).

Drops the preamble, the bibliography call, every float and list environment,
comments and LaTeX markup, then counts whitespace-separated tokens that contain
at least one letter. The point is to check the revision's length target, not to
reproduce a publisher's count.
"""
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
body = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', body)
body = re.sub(r'[\\$&_{}~^]', ' ', body)
words = [w for w in body.split() if re.search(r'[A-Za-z]', w)]
print(f'{sys.argv[1]}: {len(words)} prose words '
      f'(floats, lists and the abstract excluded)')
