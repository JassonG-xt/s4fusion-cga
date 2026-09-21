#!/usr/bin/env python3
"""Verification checklist for a no-reflow PDF region deletion.

Implements the skill's hard checklist: page count, untouched pages compared
text-for-text, the retained tail of the start page and the retained head of the
end page, zero hits for every deleted phrase, outline count and page targets,
links, and a full-page render pass.
"""
import sys

import fitz

SRC, DST = sys.argv[1], sys.argv[2]
START_PAGE, END_PAGE = 37, 39          # 1-based span that was edited
ZONE = {37: (209.83, 634.89), 38: (67.88, 634.89), 39: (67.88, 106.18)}
DELETED = ['Data availability', 'Code availability', 'Author contribution',
           'Funding', 'Competing interests', 'Ethics approval',
           'Use of large language models']
# "Data availability" also appears as a cross-reference inside a table on
# page 31, which is outside the requested region and must survive.
CROSSREF_OK = {31: 'Data availability'}
ok = True


def check(label, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {label}{(' :: ' + detail) if detail else ''}")


a, b = fitz.open(SRC), fitz.open(DST)

check('page count unchanged', a.page_count == b.page_count,
      f'{a.page_count} -> {b.page_count}')

# untouched pages before the span and after it
same_before = [i + 1 for i in range(START_PAGE - 1)
               if a[i].get_text() != b[i].get_text()]
same_after = [i + 1 for i in range(END_PAGE, a.page_count)
              if a[i].get_text() != b[i].get_text()]
check('pages before the span identical', not same_before, str(same_before))
check('pages after the span identical', not same_after, str(same_after))

# start page: every original block ABOVE the deletion zone must survive verbatim
zone_top = ZONE[START_PAGE][0]
kept_blocks = [' '.join(blk[4].split()) for blk in a[START_PAGE - 1].get_text('blocks')
               if blk[3] <= zone_top]
kept_text = ' '.join(b[START_PAGE - 1].get_text().split())
missing = [s for s in kept_blocks if s not in kept_text]
check('start page keeps the Conclusion tail verbatim',
      not missing, f'{len(kept_blocks)} blocks checked, missing={len(missing)}')
for s in kept_blocks:
    print(f'      retained block ends: ...{s[-70:]!r}')

# end page: text from the "References" heading to the page end must match exactly
os_, od = a[END_PAGE - 1].get_text(), b[END_PAGE - 1].get_text()
tail_a = os_[os_.index('References'):].strip()
check('end page retains References onward byte-for-byte',
      od.strip() == tail_a, f'{len(tail_a)} chars')

# every deleted phrase must be gone, except a cross-reference outside the region
for phrase in DELETED:
    hits = [i + 1 for i in range(b.page_count) if phrase in b[i].get_text()]
    leftover = [h for h in hits if CROSSREF_OK.get(h) != phrase]
    detail = (f'still on pages {leftover}' if leftover
              else f'{len(hits)} legitimate cross-reference(s) outside the region')
    check(f'removed: {phrase}', not leftover, detail)

# outline
ta, tb = a.get_toc(), b.get_toc()
check('outline has exactly the 7 declaration entries fewer',
      len(tb) == len(ta) - 7, f'{len(ta)} -> {len(tb)}')
kept_pages_a = [t[2] for t in ta if t[1] in [x[1] for x in tb]]
check('kept outline entries keep their page targets',
      [t[2] for t in tb] == kept_pages_a)

# links: none may remain inside the deleted zones
in_zone = 0
for pno, (z0, z1) in ZONE.items():
    zr = fitz.Rect(0, z0, b[0].rect.width, z1)
    in_zone += sum(1 for l in b[pno - 1].get_links()
                   if fitz.Rect(l['from']).intersects(zr))
check('no links left inside the deleted zones', in_zone == 0, f'{in_zone} remaining')
la = sum(len(a[i].get_links()) for i in range(a.page_count))
lb = sum(len(b[i].get_links()) for i in range(b.page_count))
check('total links reduced', lb < la, f'{la} -> {lb} (references kept their DOIs)')

# render every page to catch file corruption
try:
    for i in range(b.page_count):
        b[i].get_pixmap(dpi=40)
    check('every page renders', True)
except Exception as exc:                                  # pragma: no cover
    check('every page renders', False, repr(exc))

# blank-page report for the human
for i in range(START_PAGE - 1, END_PAGE):
    txt = b[i].get_text().strip()
    print(f'  page {i + 1} retains {len(txt)} chars: {txt[:70]!r}')

print('\nRESULT:', 'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
