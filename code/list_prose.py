#!/usr/bin/env python3
"""List the prose paragraphs of main.tex: non-blank lines that are not LaTeX
commands, with their line number, word count and opening words.

Used to drive line-number-based compression batches.
"""
import re
import sys

L = open(sys.argv[1], encoding='utf-8').read().split('\n')
CMD = re.compile(r'\\(begin|end|section|subsection|paragraph|caption|label|'
                 r'toprule|midrule|botrule|item|bmhead|bibliography|'
                 r'documentclass|usepackage|newtheorem|theoremstyle|'
                 r'raggedbottom|includegraphics|resizebox|setlength|footnotetext|'
                 r'keywords|author|affil|title|maketitle|backmatter|abstract|'
                 r'centering|small|footnotesize|scriptsize|tiny|normalsize|'
                 r'tabular|hline|cline|bmhead|vspace|hspace|noindent|newline)')
for i, l in enumerate(L, 1):
    s = l.strip()
    if not s or s.startswith('\\') or s.startswith('%'):
        continue
    if CMD.match(s):
        continue
    nparen = s.count('{') - s.count('}')
    if nparen != 0:
        continue
    w = len(s.split())
    print(f'{i:5d} {w:4d}w  {s[:74]}')
