#!/usr/bin/env python3
"""Non-destructive structural health check of the manuscript.

Reports the things that batch editing can break: figure/table/list counts, the
equation bodies of Section 3.2, duplicated paragraphs, and the audit strings.
Nothing is written.
"""
import re
import sys
from collections import Counter

MAIN = sys.argv[1]
t = open(MAIN, encoding='utf-8').read()

print(f'bytes {len(t)}  CR {t.count(chr(13))}  lines {t.count(chr(10)) + 1}')
for env in ('figure', 'figure*', 'table', 'table*', 'tabular', 'enumerate',
            'itemize', 'equation', 'align', 'abstract'):
    print(f'  {env:11s} {t.count(chr(92) + "begin{" + env + "}")}')

print('equation bodies:')
for eq in ('eq:gate', 'eq:commit', 'eq:kappa', 'eq:commitloss', 'eq:M'):
    m = re.search(r'\\begin\{equation\}\\label\{' + re.escape(eq) + r'\}\s*(.*?)\s*\\end\{equation\}',
                  t, re.S)
    body = ' '.join(m.group(1).split())[:70] if m else '*** MISSING ***'
    print(f'  {eq:12s} {body}')

paras = [p for p in t.split('\n\n') if p.strip()]
norm = [' '.join(p.split()) for p in paras]
dup = [k for k, v in Counter(norm).items() if v > 1]
print(f'paragraphs {len(paras)}   exact duplicates {len(dup)}')
for d in dup:
    print('   DUP:', d[:80])

checks = {
    'five-symbol definitions': 'five symbols with distinct referents',
    'near-binary selector term': 'trained near-binary selector',
    'image-level units': 'image as the unit of observation',
    'cluster table': 'tab:cluster',
    'controls table': 'tab:controls',
    'per-seed table': 'tab:perseed',
    'selector table': 'tab:selector',
    'colour table': 'tab:color',
    'timeline table': 'tab:timeline',
    'C3 unrounded': '-1.0043',
    'uniform negation': 'does not claim that committing everywhere does at least as well',
    'active at inference': 'active at inference',
    'unstable prototype': 'unstable experimental prototype',
    'CGA-v1': 'CGA-v1',
    'scoped case': 'single-testbed',
    'four propositions': 'Four propositions',
}
print('audit strings:')
for k, v in checks.items():
    print(f'  {"OK " if v in t else "MISS"} {k}')
