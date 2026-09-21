#!/usr/bin/env python3
"""Remove the back-matter declaration blocks from the manuscript source.

Why: the declarations sat between the Conclusion and the References, so the
compiled PDF carried two pages whose only content was the declarations (page 38
ended up holding nothing but its page number) -- i.e. the gap the user asked to
close. A PDF-level redaction cannot reflow, so the only way to get a continuous
document is to remove the blocks from the source and rebuild.

The removed text is archived verbatim next to the manuscript so it can be
restored; \backmatter itself is kept, because the bibliography's styling depends
on it. The one body-side cross-reference to the removed section ("see Data
availability", inside the provenance timeline table) is reworded.
"""
import re
import sys

MAIN = sys.argv[1] if len(sys.argv) > 1 else 'main.tex'
ARCHIVE = sys.argv[2] if len(sys.argv) > 2 else 'declarations.tex'

src = open(MAIN, encoding='utf-8').read()

start = src.index('\\backmatter')
bib_banner = src.index('%% Bibliography (sn-mathphys-num, numeric citations)')
# rewind to the start of the comment banner that precedes \bibliography{refs}
banner_start = src.rindex('%%==========', 0, bib_banner)

backmatter_block = src[start:banner_start]
removed = backmatter_block
body_after = src[start:start + len('\\backmatter')]

if '\\bmhead' not in removed:
    raise SystemExit('no \\bmhead found between \\backmatter and the bibliography')

with open(ARCHIVE, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write('% Declaration blocks removed from main.tex on 2026-09-21 at the\n'
             '% author\'s request, to remove the blank pages between the\n'
             '% Conclusion and the References. Kept verbatim for restoration:\n'
             '% re-insert the \\bmhead blocks immediately after \\backmatter.\n\n')
    fh.write(removed.strip() + '\n')

new = src[:start] + '\\backmatter\n\n' + src[banner_start:]

# the timeline table pointed at the section that has just been removed
old_ref = '--- (a snapshot; see Data availability)'
new_ref = '--- (a snapshot of the released repository)'
if old_ref in new:
    new = new.replace(old_ref, new_ref)
else:
    raise SystemExit('the dangling cross-reference was not found as expected')

open(MAIN, 'w', encoding='utf-8', newline='\n').write(new)

print(f'archived  {len(removed.split())} words to {ARCHIVE}')
print(f'bmhead in file now : {new.count(chr(92) + "bmhead")}')
print(f'backmatter kept    : {"\\\\backmatter" in new}')
print(f'bibliography kept  : {"\\\\bibliography{refs}" in new}')
print(f'end document kept  : {"\\\\end{document}" in new}')
print(f'dangling ref fixed : {new_ref in new}')
